"""engy-miner — a self-contained verifiable-inference miner in one module.

Runs next to a stock sglang serve. It dials the gateway, and for every request
the gateway routes to it: runs the generation on the serve, builds a compact
proof of the per-token final hidden states (the proof method is TOPLOC — a
locality-sensitive hash of the activations), and returns the completion plus the
proof over the same connection. A validator recomputes and compares later.

One process does everything — connect, generate, prove, answer. No local HTTP
endpoint and no extra hops.

  gateway ──(N websocket legs, one per gateway worker)──► engy-miner
                                                             │  generate on the serve
                                                             │  build the proof (TOPLOC)
                                                             └► reply: completion + proof

Run:
  GW=wss://<gateway-host>/gw MINER_KEY=<your-key> \
  python engy_miner.py \
      --checkpoint /data/models/<MODEL> \
      --serve-url  http://127.0.0.1:8000        # one, or comma-separated for several
Requires: each sglang serve started with --enable-return-hidden-states.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import os
import random
import socket
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

# Set BEFORE torch imports, so an operator does not have to. The proof build is a
# tiny LSH: unset, torch spawns a 64-thread OpenMP team inside every generation
# thread, which is pure contention at our concurrency. Both remain overridable.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import numpy as np
import requests
import torch
import websockets
from toploc import build_proofs_base64
from transformers import AutoTokenizer


# ---- gateway wire protocol (inlined so this file is self-contained; frames are
# plain JSON dicts with a "type" key). Must match the gateway's pool protocol.
class P:
    HELLO = "hello"
    RESPONSE = "response"
    CHUNK = "chunk"          # miner -> pool: {corr_id, delta} (streaming)
    HEARTBEAT = "heartbeat"
    SERVE = "serve"          # pool -> miner: {corr_id, request}
    CANCEL = "cancel"        # pool -> miner: {corr_id} buyer is gone, stop NOW
    PING = "ping"            # pool -> miner: keepalive
    RECONNECT = "reconnect"  # pool -> miner: gateway draining, re-dial
    ADMIT = "admit"
    DENY = "deny"

    @staticmethod
    def hello(miner_key, model, model_root, hw=None, capacity=None,
              worker_name="", worker_id=""):
        # worker_name names this machine among the connections sharing one
        # miner_key; a repeat HELLO with the same (key, worker_name) supersedes
        # the previous connection. Sent under both field names; a gateway that
        # doesn't use them ignores them. worker_id is this process's uuid (fresh
        # per restart) so the gateway can tell a re-dial from a new process.
        # These keep the miner admissible on the current gateway.
        f = {"type": "hello", "miner_key": miner_key, "model": model,
             "model_root": model_root, "hw": hw or {}, "versions": {},
             "capacity": capacity or {}, "worker_name": worker_name,
             "instance_id": worker_name}
        if worker_id:
            f["worker_id"] = worker_id
        return f

    @staticmethod
    def heartbeat(inflight=0, idle_seconds=0.0, capacity=None):
        f = {"type": "heartbeat", "inflight": inflight, "idle_seconds": idle_seconds}
        if capacity:
            f["capacity"] = capacity
        return f

    @staticmethod
    def chunk(corr_id, delta):
        # engy/pool/protocol.py: {corr_id, delta} only; delta is an OpenAI-style
        # dict. No commitment here -- the proof covers every generated token, so
        # it can only exist once generation ends. It rides the terminal response.
        return {"type": "chunk", "corr_id": corr_id, "delta": delta or {}}

    @staticmethod
    def response(corr_id, request_id, commitment, output=None, error=None):
        f = {"type": "response", "corr_id": corr_id, "request_id": request_id,
             "commitment": commitment, "output": output or {}}
        if error:
            f["error"] = error
        return f

# ------------------------------------------------------------------ config (env)
SERVE_URLS = [u.strip() for u in
              os.environ.get("SERVE_URL", "http://127.0.0.1:8000").split(",") if u.strip()]
CHECKPOINT = os.environ.get("CHECKPOINT", "")
MODEL = os.environ.get("MODEL", "qwen3.6-35b-a3b")
GW = os.environ.get("GW", "")          # gateway websocket URL — required, no default

TOPK = int(os.environ.get("TOPLOC_TOPK", "128"))
DBS = int(os.environ.get("TOPLOC_DBS", "32"))          # proof decode-batching size
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))  # default output cap
OUT_MAX = int(os.environ.get("MAX_OUTPUT_TOKENS", "32768"))    # 32*1024
# Chunked generation: cap output tokens per serve /generate call so the serve
# never buffers the whole hidden-states block in one response (a single
# 32k-output call balloons the serve detokenizer to ~244 GB; a 4k chunk is
# +6 GB). Each continuation re-feeds prompt+output-so-far and overlaps by one
# token so the kept hidden rows are all decode vectors (radix-cache independent).
GEN_CHUNK = int(os.environ.get("TOPLOC_GEN_CHUNK", "4096"))
# Coalesce streamed deltas: one ws frame per token is needless traffic.
STREAM_FLUSH_S = float(os.environ.get("STREAM_FLUSH_S", "0.05"))
IGNORE_EOS = os.environ.get("TOPLOC_IGNORE_EOS", "0") == "1"  # test only: force full length
HEARTBEAT_INTERVAL = 30.0                              # must stay < the gateway's 90s stale

# Capacity advertised to the gateway (it only routes to miners with capacity and
# waits up to max_request_s for the reply). max_inflight is split across the legs;
# when it is smaller than the leg count we open fewer legs instead (see _leg_plan).
CAP = {"max_inflight": int(os.environ.get("MAX_INFLIGHT", "64")),
       "max_input_tokens": int(os.environ.get("MAX_INPUT_TOKENS", "229376")),   # 224*1024
       "max_output_tokens": OUT_MAX,
       "max_request_s": float(os.environ.get("MAX_REQUEST_S", "1800.0"))}

# Our HTTP read timeout to the serve. It DEFAULTS TO the deadline we advertised,
# so it can never sit below it by accident: giving up early does not stop sglang
# (the protocol has no cancel), it just abandons work we promised to deliver and
# burns the slot for nothing.
SERVE_HTTP_S = float(os.environ.get("SERVE_HTTP_S") or CAP["max_request_s"])

# Generation (blocking serve call + proof build) runs off the event loop so the
# loop stays free to answer gateway pings.
_GEN_POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("GEN_THREADS", "256")),
                               thread_name_prefix="engy-gen")

# Aborts get their OWN pool. Submitting them to _GEN_POOL would deadlock exactly
# when they matter most: if every generation thread is busy, the abort that would
# free one queues behind the work it is trying to stop.
_ABORT_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="engy-abort")


class _Cancelled(Exception):
    """The gateway sent CANCEL for this request — unwind the worker thread.

    Not an error: the buyer disconnected and the gateway has already released
    its slot and stopped listening. We stop generating and stay quiet.
    """


class _Job:
    """Cancellation state shared between the event loop and one worker thread.

    The loop only ever SETS `flag` (and reads `serve`/`rids`); the worker thread
    only ever polls it. `flag` is a threading.Event rather than an asyncio one
    precisely because the generation runs off the loop.
    """
    __slots__ = ("flag", "serve", "rids")

    def __init__(self):
        self.flag = threading.Event()
        self.serve = None            # serve URL, once _process has picked one
        self.rids = set()            # sglang rids currently in flight for this job

    def check(self):
        """Cancellation checkpoint — call between/inside serve calls."""
        if self.flag.is_set():
            raise _Cancelled()


# corr_id -> _Job, registered in _serve BEFORE the work is queued (a CANCEL can
# arrive before the worker thread has even started) and dropped in its finally.
_JOBS: "dict[str, _Job]" = {}
_JOBS_LOCK = threading.Lock()


def _job_register(corr) -> "_Job":
    job = _Job()
    with _JOBS_LOCK:
        _JOBS[corr] = job
    return job


def _job_done(corr) -> None:
    with _JOBS_LOCK:
        _JOBS.pop(corr, None)


def _abort_rids(job) -> None:
    """Tell sglang to stop generating. Blocking — runs on _ABORT_POOL, never the
    event loop. Best-effort: the flag alone already stops us at the next
    checkpoint, this is what stops the GPU burning tokens for nobody."""
    for rid in list(job.rids):
        try:
            requests.post(job.serve + "/abort_request", timeout=10,
                          json={"rid": rid}).raise_for_status()
        except Exception as e:
            print(f"[engy-miner] abort {rid} failed: {e!r}", flush=True)


def _on_cancel(corr, tag) -> None:
    """Handle a CANCEL frame. Runs ON the event loop, so it must not block: set
    the flag synchronously (that alone bounds us to the next checkpoint even if
    the HTTP abort fails) and offload the abort itself."""
    with _JOBS_LOCK:
        job = _JOBS.get(corr)
    if job is None:
        return                       # already finished — nothing left to stop
    job.flag.set()
    n = len(job.rids)
    if job.serve and n:
        _ABORT_POOL.submit(_abort_rids, job)
    print(f"[engy-miner] {tag} CANCEL {corr} (aborting {n} serve request(s))", flush=True)


def _detect_hw():
    """Best-effort machine hardware for the HELLO frame. Never fatal, and never
    creates a CUDA context — the serve owns the GPUs, so they are read via
    nvidia-smi (a query, no context) rather than torch.cuda. HW_GPUS / HW_PAR
    still override the human-readable summary and parallelism."""
    hw = {"parallelism": os.environ.get("HW_PAR", "tp")}
    try:
        hw["host"] = socket.gethostname()
    except Exception:
        pass
    try:
        hw["cpus"] = os.cpu_count()
        hw["ram_gb"] = round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9, 1)
    except Exception:
        pass
    try:
        lines = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        if lines:
            name, mem = (c.strip() for c in lines[0].split(",")[:2])
            hw["gpu"] = name
            hw["gpu_count"] = len(lines)
            hw["gpu_mem_gb"] = round(float(mem) / 1024, 1)   # MiB -> GiB
    except Exception:
        pass
    try:
        hw["torch"] = torch.__version__
        hw["cuda"] = torch.version.cuda
    except Exception:
        pass
    hw["gpus"] = os.environ.get("HW_GPUS") or \
        f"{hw.get('gpu_count', '?')}x {hw.get('gpu', 'GPU')}"
    return hw


HW = _detect_hw()

MINER_KEY = None          # set at startup
_tokenizer = None
_model_root = None
WORKER_NAME = None             # set at startup (see _worker_name)
WORKER_ID = uuid.uuid4().hex   # this process's identity, HELLO'd on every leg


def _worker_name() -> str:
    """This machine's worker name under MINER_KEY. Several machines may share one
    key; each registers as its own named worker at the gateway. Stable across
    restarts (hostname+model+serves) so a restarted miner replaces its previous
    registration. Set ENGY_WORKER_NAME to name workers explicitly."""
    env = os.environ.get("ENGY_WORKER_NAME") or os.environ.get("ENGY_INSTANCE_ID")
    if env:
        return env
    seed = "|".join([socket.gethostname(), MODEL, ",".join(SERVE_URLS)])
    return "ins-" + hashlib.sha256(seed.encode()).hexdigest()[:12]


# ------------------------------------------------------------------ serve pool
_inflight: "dict[str, int]" = {}
_inflight_lock = threading.Lock()


def _pick_serve() -> str:
    with _inflight_lock:
        u = min(SERVE_URLS, key=lambda s: _inflight.get(s, 0))
        _inflight[u] = _inflight.get(u, 0) + 1
        return u


def _release_serve(u: str) -> None:
    with _inflight_lock:
        _inflight[u] = max(0, _inflight.get(u, 0) - 1)


# ------------------------------------------------------------------ model spec
def _model_root_of(checkpoint: str) -> str:
    """Fast checkpoint identity: sha256 over config.json + the shard manifest
    (each *.safetensors name + byte size). The proof method recomputes the
    activations, so any real weight difference is caught at verify time; the root
    just names which checkpoint this miner pinned."""
    cfg_path = os.path.join(checkpoint, "config.json")
    cfg_bytes = open(cfg_path, "rb").read() if os.path.exists(cfg_path) else b"{}"
    h = hashlib.sha256()
    h.update(cfg_bytes)
    for shard in sorted(glob.glob(os.path.join(checkpoint, "*.safetensors"))):
        h.update(os.path.basename(shard).encode())
        h.update(str(os.path.getsize(shard)).encode())
    return h.hexdigest()


# ------------------------------------------------------------------ proof
def _fin_rows(hidden_states):
    """sglang /generate returns ragged hiddens: element 0 is the prefill block
    (last row = hidden that sampled output token 0), elements 1.. are per-decode
    vectors. A full radix-cache hit returns an EMPTY prefill block, losing output
    token 0's hidden; the decode vectors then start at `offset`."""
    rows, offset, seen = [], 0, False
    for item in hidden_states:
        a = np.asarray(item, dtype=np.float32)
        if a.size == 0:
            if not seen:
                offset += 1
            continue
        seen = True
        rows.append(a[-1] if a.ndim == 2 else a)
    if not rows:
        # Every element was empty (or the serve sent none at all). Without this
        # the failure surfaces as numpy's "need at least one array to stack",
        # which says nothing about the actual cause -- and the cause is always
        # the serve, not this request.
        raise RuntimeError(
            "serve returned no hidden states. The proof cannot be built without "
            "them. Check that the sglang serve was started with "
            "--enable-return-hidden-states, and that nothing on the serve's "
            "PYTHONPATH is diverting them (a capture/trim sitecustomize will "
            "empty meta_info.hidden_states in the HTTP response). Verify with: "
            "curl $SERVE/generate -d '{\"input_ids\":[9707],\"sampling_params\":"
            "{\"max_new_tokens\":4},\"return_hidden_states\":true}'")
    return np.stack(rows), offset


# ------------------------------------------------------------------ generate + prove (blocking)
def _iter_sse_lines(r, job=None):
    """Linear-time line split over a streamed response.

    NOT requests' iter_lines(): that re-concatenates its pending buffer on every
    512-byte chunk, which is O(len^2) in the length of a single line — and the
    final /generate event carries the ENTIRE hidden-states block as one data:
    line (hundreds of MB for a long prompt). Measured on a 29k-token prompt, a
    gen thread sat at 100% CPU for 45+ minutes draining the socket at ~2 KB/s
    and the per-line cancel checkpoint never ran. Here every byte is appended,
    scanned and deleted once, and `job` is polled per network chunk so a CANCEL
    lands even in the middle of a giant line."""
    buf, scanned = bytearray(), 0        # buf[:scanned] is known newline-free
    for chunk in r.iter_content(chunk_size=1 << 20):
        if job is not None:
            job.check()
        if not chunk:
            continue
        buf += chunk
        while True:
            nl = buf.find(b"\n", scanned)
            if nl < 0:
                scanned = len(buf)
                break
            line = buf[:nl]
            if line.endswith(b"\r"):
                line = line[:-1]
            yield line.decode("utf-8", errors="replace")
            del buf[:nl + 1]
            scanned = 0
    if buf:
        yield buf.decode("utf-8", errors="replace")


def _stream_once(serve, body, emit, skip_first_token, job=None):
    """Drive one streamed /generate, handing text deltas to `emit`, and return the
    final event (complete meta_info). Deltas are coalesced by STREAM_FLUSH_S so a
    fast generation doesn't put one websocket frame on the wire per token.

    Polls `job` per SSE line AND per network chunk (inside _iter_sse_lines): with
    GEN_CHUNK at 4096 a chunk boundary can be minutes away, so a cancel that only
    took effect between chunks would leave the GPU running long after the buyer
    left."""
    base, sent, pend, last_flush, out = None, 0, [], time.time(), None

    def flush():
        nonlocal pend, last_flush
        if pend:
            emit("".join(pend))
            pend, last_flush = [], time.time()

    with requests.post(serve + "/generate", timeout=SERVE_HTTP_S, stream=True,
                       json={**body, "stream": True}) as r:
        r.raise_for_status()
        for line in _iter_sse_lines(r, job):
            if job is not None:
                job.check()           # raises _Cancelled; `with` closes the stream
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            out = ev
            txt = ev.get("text") or ""
            if skip_first_token and base is None:
                # Wait for the overlap token to land, then treat the text up to it
                # as already-delivered; only what follows is new.
                if len((ev.get("meta_info") or {}).get("output_token_logprobs") or []) < 1:
                    continue
                base = len(txt)
                continue
            if base is None:
                base = 0
            if len(txt) - base > sent:
                pend.append(txt[base + sent:])
                sent = len(txt) - base
                if (time.time() - last_flush) >= STREAM_FLUSH_S:
                    flush()
    flush()
    if out is None:
        raise RuntimeError("stream ended with no events")
    return out


def _gen_once(serve, input_ids, max_new, emit=None, skip_first_token=False, job=None):
    """One serve /generate with hidden states. Returns (oids, fin_rows, offset,
    finish_type): oids = output token ids; fin_rows = stacked per-token final
    hidden states (np, one row per token that HAS a row); offset = how many
    leading output tokens had no row (empty prefill block on a radix hit).

    With `emit` the call is streamed. sglang's /generate SSE carries the CUMULATIVE
    text on every event plus a growing meta_info, so deltas come from slicing and
    the LAST event still holds the complete hidden_states/logprobs this returns —
    streaming costs the proof nothing. Measured on an idle serve: first event at
    0.17s with return_hidden_states on, versus a whole generation of silence.

    `skip_first_token` is for continuation chunks, whose first token is the
    deliberately regenerated overlap: its text was already emitted by the previous
    chunk, so emitting starts only once that token has gone by."""
    sp = {"temperature": 0.0, "max_new_tokens": max_new}
    if IGNORE_EOS:
        sp["ignore_eos"] = True
    # An explicit rid is what makes this call abortable: /abort_request takes a
    # rid, and sglang would otherwise mint its own that we never learn.
    srid = "engy-" + uuid.uuid4().hex
    body = {"input_ids": input_ids, "sampling_params": sp, "rid": srid,
            "return_hidden_states": True, "return_logprob": True}
    if job is not None:
        job.rids.add(srid)
    try:
        if emit is None:
            # No checkpoint to poll inside a blocking POST — /abort_request is
            # what ends it early (see the except below).
            r = requests.post(serve + "/generate", timeout=SERVE_HTTP_S, json=body)
            r.raise_for_status()
            out = r.json()
        else:
            out = _stream_once(serve, body, emit, skip_first_token, job)
    except _Cancelled:
        raise
    except BaseException:
        # /abort_request ends the call one of two ways, both measured live:
        #   idle serve   -> a NORMAL response, truncated, finish_reason.type
        #                   "abort" (caught by the job.check() below);
        #   loaded serve -> the connection is torn down without a response body
        #                   and the HTTP layer raises (RemoteDisconnected) --
        #                   here, before any checkpoint can run.
        # The second is our own cancellation completing, not a serve failure;
        # reporting it as one would log noise and put an error frame on a corr
        # the gateway has already stopped listening to.
        if job is not None and job.flag.is_set():
            raise _Cancelled()
        raise
    finally:
        if job is not None:
            job.rids.discard(srid)
    if job is not None:
        # The clean-return half of the above: a truncated generation we must not
        # go on to prove as if the buyer had received it.
        job.check()
    out = out[0] if isinstance(out, list) else out
    meta = out["meta_info"]
    fin, offset = _fin_rows(meta["hidden_states"])
    oids = [int(t[1]) for t in (meta.get("output_token_logprobs") or [])]
    ftype = ((meta.get("finish_reason") or {}) or {}).get("type")
    return oids, fin, offset, ftype


def _generate_chunked(serve, prompt_ids, max_new, emit=None, job=None):
    """Generate up to max_new tokens in GEN_CHUNK-sized serve calls, collecting the
    per-token final hidden states in order. Returns (all_oids, rows_list,
    prompt_extra): all_oids = proven output tokens; rows_list = one np hidden row
    per proven token; prompt_extra = tokens folded into the prompt on chunk 1
    (only when the prompt was already radix-cached, rare).

    Continuations overlap by one token (drop the last, regenerate it greedily) so
    every KEPT row is a decode vector — never the prefill-block fin, which a full
    radix-cache hit returns empty. So the collected rows are identical to what a
    single call would have produced, just assembled from smaller responses."""
    all_oids: list = []
    rows_list: list = []
    prompt_extra: list = []
    while len(all_oids) < max_new:
        if job is not None:
            job.check()              # never START a chunk for a departed buyer
        want = min(GEN_CHUNK, max_new - len(all_oids))
        if not all_oids:
            # first chunk: input = prompt; keep the prefill fin + decode rows
            oids, fin, offset, ftype = _gen_once(serve, prompt_ids, want, emit, job=job)
            if offset:                           # prompt already cached -> fold skipped ids in
                prompt_extra = oids[:offset]
                oids = oids[offset:]
            nkeep = min(int(fin.shape[0]), len(oids))
            rows_list.extend(fin[i] for i in range(nkeep))
            all_oids.extend(oids[:nkeep])
            produced = nkeep
        else:
            # continuation: overlap by one so kept rows are all decode vectors
            inp = prompt_ids + prompt_extra + all_oids[:-1]
            oids, fin, offset, ftype = _gen_once(serve, inp, want + 1, emit,
                                                 skip_first_token=True, job=job)
            start = max(0, 1 - offset)           # row index where the NEW tokens begin
            avail = int(fin.shape[0]) - start
            kept = oids[1:1 + avail]              # skip index 0 (the regenerated dup)
            rows_list.extend(fin[start + i] for i in range(len(kept)))
            all_oids.extend(kept)
            produced = len(kept)
        if ftype != "length" or produced < want:   # EOS / short -> generation ended
            break
    return all_oids, rows_list, prompt_extra


def _build_commitment(prompt_full, all_oids, rows_list):
    """One TOPLOC proof over the whole collected hidden-state sequence."""
    n = len(rows_list)
    acts = [torch.from_numpy(np.asarray(rows_list[i], dtype=np.float32)).to(torch.bfloat16)
            for i in range(n)]
    proofs = build_proofs_base64(acts, decode_batching_size=DBS, topk=TOPK, skip_prefill=True)
    return {"model_root": _model_root, "output_token_ids": all_oids,
            "toploc": {"proofs": proofs, "topk": TOPK, "decode_batching_size": DBS,
                       "skip_prefill": True, "n_tokens": n, "prompt_ids": prompt_full}}


def _process(request: dict, emit=None, job=None):
    """One routed request end-to-end (runs in a worker thread): chat-template ->
    chunked generation with hidden states -> proof + OpenAI completion body.

    `job` carries cancellation: if the gateway says the buyer is gone this raises
    _Cancelled at the next checkpoint, and the `finally` below still returns the
    serve slot."""
    messages = [{"role": m["role"], "content": m["content"]} for m in request.get("messages", [])]
    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = [int(i) for i in _tokenizer(prompt).input_ids]
    max_new = min(int(request.get("max_tokens") or MAX_TOKENS), OUT_MAX)

    # Never clamp, never refuse: the miner's job is to finish every request the
    # gateway routes to it. Truncating a buyer's output or rejecting a long
    # prompt is not ours to decide — serve memory is managed on the serve.
    serve = _pick_serve()
    if job is not None:
        job.serve = serve            # _on_cancel needs this to address the abort
    try:
        all_oids, rows_list, prompt_extra = _generate_chunked(serve, prompt_ids,
                                                              max_new, emit, job)
    finally:
        _release_serve(serve)

    prompt_full = prompt_ids + prompt_extra
    commitment = _build_commitment(prompt_full, all_oids, rows_list)
    text = _tokenizer.decode(all_oids, skip_special_tokens=True)
    n_prompt, n = len(prompt_full), len(rows_list)
    output = {"choices": [{"index": 0, "finish_reason": "stop",
                           "message": {"role": "assistant", "content": text}}],
              "usage": {"prompt_tokens": n_prompt, "completion_tokens": n,
                        "total_tokens": n_prompt + n}}
    return commitment, output


# ------------------------------------------------------------------ gateway legs
def _worker_count() -> int:
    """N gateway worker processes; open one leg per worker so we receive all buyer
    traffic (a single-dial miner only lands on worker 0)."""
    n = os.environ.get("ENGY_GW_WORKERS")
    if n:
        return max(1, int(n))
    try:
        meta = GW.replace("wss://", "https://").replace("ws://", "http://") + "/meta"
        return max(1, int((requests.get(meta, timeout=5).json() or {}).get("workers") or 1))
    except Exception:
        return 1


def _worker_url(i: int, n: int) -> str:
    return GW if n <= 1 else f"{GW}/{i}"


def _leg_plan(n: int):
    """(legs to open, inflight each leg advertises) for `n` gateway workers.

    One leg per worker is the default — a single-dial miner only ever lands on
    worker 0. But `max_inflight // n` floors to 0 once max_inflight < n, and
    clamping that back to 1 would advertise n concurrent from a miner that can
    only run max_inflight. Below that point open max_inflight legs at 1 each
    instead: the total advertised is then exactly max_inflight, at the cost of
    reaching fewer of the gateway's workers."""
    total = max(1, CAP["max_inflight"])
    if total < n:
        return total, 1
    return n, total // n


def _leg_cap(per_leg: int) -> dict:
    c = dict(CAP)
    c["max_inflight"] = per_leg
    return c


async def _heartbeat(ws, cap):
    try:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            await ws.send(json.dumps(P.heartbeat(capacity=cap)))
    except (websockets.ConnectionClosed, asyncio.CancelledError):
        pass


async def _serve(ws, frame):
    """Stream the generation to the gateway, then close with the proof.

    Streaming is the GATEWAY's choice, signalled by `stream` on the serve frame
    (engy/pool/protocol.py serve()); without it the original single-frame path
    runs. _process is on a worker thread and must not touch the websocket, so it
    hands deltas back through an asyncio.Queue and this coroutine owns every send.

    The proof rides the terminal `response`: it covers every generated token and
    the miner reads the hidden states out of the final /generate event, so it cannot
    precede the last token. Note that once a chunk is on the wire it cannot be
    recalled — a request that streams then fails to prove has already delivered
    text, where the non-streaming path raised before the buyer saw anything."""
    corr, req = frame["corr_id"], frame["request"]
    rid = req.get("engy_request_id") or uuid.uuid4().hex
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def emit(text):                       # worker thread -> event loop
        loop.call_soon_threadsafe(q.put_nowait, {"content": text})

    streaming = bool(frame.get("stream"))
    # Registered BEFORE the work is queued: a CANCEL can arrive while the request
    # is still waiting for a free generation thread.
    job = _job_register(corr)
    fut = loop.run_in_executor(_GEN_POOL, _process, req,
                               emit if streaming else None, job)
    fut.add_done_callback(lambda _: loop.call_soon_threadsafe(q.put_nowait, None))
    try:
        while True:
            delta = await q.get()
            if delta is None:             # generation finished (or failed)
                break
            await ws.send(json.dumps(P.chunk(corr, delta)))
        commitment, output = await fut    # re-raises whatever _process raised
        await ws.send(json.dumps(P.response(corr, rid, commitment, output=output)))
    except _Cancelled:
        # Deliberately silent on the wire: the gateway released this slot and
        # popped its pending entry when it sent the CANCEL, so a response or an
        # error frame now would be addressed to nobody.
        print(f"[engy-miner] cancelled {corr}", flush=True)
    except Exception as e:
        print("[engy-miner] serve error:", repr(e), flush=True)
        try:
            await ws.send(json.dumps(P.response(corr, rid, None, error=repr(e)[:300])))
        except Exception:
            pass                          # leg already gone; the gateway will time out
    finally:
        _job_done(corr)


async def _run(ws, cap, tag):
    await ws.send(json.dumps(P.hello(MINER_KEY, MODEL, _model_root, hw=HW, capacity=cap,
                                     worker_name=WORKER_NAME, worker_id=WORKER_ID)))
    print(f"[engy-miner] {tag} HELLO sent (inflight={cap['max_inflight']}); awaiting work", flush=True)
    hb = asyncio.create_task(_heartbeat(ws, cap))
    try:
        async for msg in ws:
            frame = json.loads(msg)
            t = frame.get("type")
            if t == P.SERVE:
                asyncio.create_task(_serve(ws, frame))
            elif t == P.CANCEL:                     # buyer gone; stop generating
                _on_cancel(frame.get("corr_id"), tag)
            elif t == P.PING:
                await ws.send(json.dumps(P.heartbeat(capacity=cap)))
            elif t == P.RECONNECT:                  # gateway draining for deploy
                return
            elif t == P.ADMIT:
                print(f"[engy-miner] {tag} ADMITTED", flush=True)
            elif t == P.DENY:
                print(f"[engy-miner] {tag} DENIED:", frame.get("reason"), flush=True)
    finally:
        hb.cancel()


async def _leg(i, n, cap):
    url, tag = _worker_url(i, n), f"gw{i}"
    while True:
        try:
            async with websockets.connect(url, ping_interval=15, ping_timeout=60,
                                           close_timeout=10, max_queue=128) as ws:
                await _run(ws, cap, tag)
            print(f"[engy-miner] {tag} session ended; reconnecting", flush=True)
        except Exception as e:
            print(f"[engy-miner] {tag} disconnected:", repr(e), flush=True)
        await asyncio.sleep(0.5)


async def _serve_all(n, leg_ids, cap):
    # leg_ids is a subset of range(n); `n` still forms the per-worker URL.
    await asyncio.gather(*[_leg(i, n, cap) for i in leg_ids])


def main():
    global MINER_KEY, WORKER_NAME, SERVE_URLS, _tokenizer, _model_root
    p = argparse.ArgumentParser(description="engy-miner (verifiable-inference miner)")
    p.add_argument("--checkpoint", default=CHECKPOINT, required=not CHECKPOINT,
                   help="model checkpoint (tokenizer + model_root)")
    p.add_argument("--serve-url", default=",".join(SERVE_URLS),
                   help="one or comma-separated sglang serve URLs (least-in-flight balanced)")
    p.add_argument("--worker_name", "--worker-name", "--instance-id",
                   dest="worker_name", default=None,
                   help="this machine's worker name when several machines share "
                        "one MINER_KEY; default derives from hostname+model+serves")
    args = p.parse_args()

    if not GW:
        raise SystemExit("[engy-miner] set GW to the gateway websocket URL "
                         "(e.g. GW=wss://<gateway-host>/gw)")
    MINER_KEY = os.environ["MINER_KEY"]
    SERVE_URLS = [u.strip() for u in args.serve_url.split(",") if u.strip()]
    WORKER_NAME = args.worker_name or _worker_name()
    print(f"[engy-miner] loading tokenizer + spec from {args.checkpoint}", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    # model_root = the identity we declare to the gateway (HELLO) and stamp on every
    # proof, so it can confirm we serve the model we claim. A cheap checkpoint hash
    # (config.json + shard manifest) — the gateway computes it the SAME way (see
    # gateway toploc canonical) so the two match; the real weight check is the recompute.
    _model_root = _model_root_of(args.checkpoint)

    n = _worker_count()
    legs, per_leg = _leg_plan(n)
    # Which workers to dial when we open fewer legs than the gateway has: pick at
    # RANDOM, not 0..legs-1. Every small miner starting at leg 0 piles the whole
    # fleet onto worker 0 while the rest sit idle; a random subset spreads them.
    leg_ids = sorted(random.sample(range(n), legs)) if legs < n else list(range(n))
    cap = _leg_cap(per_leg)
    print(f"[engy-miner] dialing {legs} of {n} worker(s) of {GW} as {MODEL} "
          f"root={_model_root[:12]} serves={SERVE_URLS} leg_inflight={per_leg} "
          f"total_inflight={legs * per_leg} legs={leg_ids}", flush=True)
    print(f"[engy-miner] hw: {HW.get('gpus')} | {HW.get('gpu_mem_gb')}GB/gpu "
          f"| {HW.get('cpus')} cpu | {HW.get('ram_gb')}GB ram | host={HW.get('host')}",
          flush=True)
    asyncio.run(_serve_all(n, leg_ids, cap))


if __name__ == "__main__":
    import fcntl                                       # single instance per node
    _sing = open("/tmp/engy_miner.singleton", "w")
    try:
        fcntl.flock(_sing, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("[engy-miner] another instance is running; exiting", flush=True)
        raise SystemExit(0)
    main()
