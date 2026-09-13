r"""engy tee_miner — the gateway leg for a TEE-attested worker.

A single self-contained module. It dials the engy gateway, and for every request
the gateway routes to it, streams the generation from a local OpenAI-compatible
serve (sglang / vLLM) back over the same connection.

What makes this the *TEE* miner rather than the general one is identity. A TEE
worker does not invent who it is: the provisioning step ("declare") allocates a
`worker_provisioning` row and bakes the assigned ids into the confidential VM's
environment. This miner adopts them, so the `workers` row it registers lines up
with the row the provider declared, which is what lets activation flip that row
to `type='tee'`. A miner that mints its own uuid here registers a worker nobody
declared, activation has nothing to match, and the worker silently never becomes
a TEE worker.

  gateway ──(N websocket legs, one per gateway worker)──► tee_miner
                                                             │  stream from the local serve
                                                             └► chunk… chunk… response

Identity, read from the environment automatically:

  ENGY_MINER_KEY    the operator key this worker registers under (required)
  ENGY_WORKER_ID    provider-assigned, from declare. MUST match the
                    worker_provisioning row or activation cannot flip it
  ENGY_WORKER_NAME  this machine's name among the workers sharing one key

Configuration:

  ENGY_GW_URL          gateway websocket base   (default wss://api.engy.ai/gw)
  ENGY_MODEL           model name to advertise  (default glm-5.2)
  ENGY_SERVE_URL       local serve base url, comma-separated for several
                                                (default http://127.0.0.1:8000)
  ENGY_SERVED_MODEL    the serve's --served-model-name, if it differs
  ENGY_CHECKPOINT      checkpoint dir, for the model_root identity only
  ENGY_MAX_INFLIGHT    concurrent requests this worker accepts (default 32)
  ENGY_CONTEXT_LENGTH  advertised context window
  ENGY_READ_TIMEOUT    upstream read timeout, seconds (default 1800)
  ENGY_GPUS / ENGY_PARALLELISM / ENGY_REGION   hardware descriptors

Run:

  python tee_miner.py                       # everything from the environment
  python tee_miner.py --serve-url http://127.0.0.1:8000 --model glm-5.2

Only `websockets` and `httpx` are required beyond the standard library.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import uuid

import httpx
import websockets


# --------------------------------------------------------------- wire protocol
# Inlined so this file stands alone. Mirrors engy/pool/protocol.py; frames are
# plain JSON dicts keyed by "type".
class P:
    # miner -> gateway
    HELLO = "hello"
    CHUNK = "chunk"            # streaming delta, before the terminal RESPONSE
    RESPONSE = "response"      # terminal frame, exactly one per SERVE
    HEARTBEAT = "heartbeat"
    # gateway -> miner
    ADMIT = "admit"
    DENY = "deny"
    SERVE = "serve"
    CANCEL = "cancel"          # buyer is gone; stop generating for it now
    PING = "ping"
    RECONNECT = "reconnect"    # gateway draining for deploy; re-dial now

    @staticmethod
    def hello(miner_key, model, model_root, hw=None, capacity=None,
              worker_name="", worker_id=""):
        f = {"type": P.HELLO, "miner_key": miner_key, "model": model,
             "model_root": model_root, "hw": hw or {}, "versions": {},
             "capacity": capacity or {}, "worker_name": worker_name}
        if worker_id:
            f["worker_id"] = worker_id
        return f

    @staticmethod
    def heartbeat(inflight=0, idle_seconds=0.0, capacity=None, kv=None):
        f = {"type": P.HEARTBEAT, "inflight": inflight,
             "idle_seconds": idle_seconds}
        if capacity:
            f["capacity"] = capacity
        if kv is not None:                 # live KV-cache pressure (optional)
            f["kv"] = kv
        return f

    @staticmethod
    def chunk(corr_id, delta=None, logprobs=None):
        f = {"type": P.CHUNK, "corr_id": corr_id, "delta": delta or {}}
        if logprobs is not None:
            f["logprobs"] = logprobs
        return f

    @staticmethod
    def response(corr_id, request_id, commitment, output=None, error=None):
        f = {"type": P.RESPONSE, "corr_id": corr_id, "request_id": request_id,
             "commitment": commitment, "output": output or {}}
        if error:
            f["error"] = error
        return f


# sglang reports a payload IT could not decode as 500, but a truncated base64
# image is the BUYER's bytes, not a fault of this worker: the same request fails
# the same way on every miner. Relaying it as 500 makes the gateway emit an
# opaque 502, hides the real cause from the buyer, and charges the miss to our
# HTTP success rate at the 99% qualification gate. Narrow on purpose -- only
# messages that can originate in the request body are re-labelled, so a genuine
# engine fault still surfaces as the upstream failure it is.
_CLIENT_PAYLOAD_500 = (
    "while loading image data",
    "while loading data imagedata(",
    "while loading data audiodata(",
    "while loading data videodata(",
    "broken data stream when reading image file",
    "cannot identify image file",
)


def _client_facing_status(status: int, detail: str) -> int:
    if status == 500 and any(p in detail.lower() for p in _CLIENT_PAYLOAD_500):
        return 400
    return status


class ServeError(RuntimeError):
    """A non-2xx from the local serve, keeping the status for the gateway.

    The gateway honours a 4xx in the error frame and relays it verbatim, so a
    request the BUYER malformed (empty messages, top_p=2.0) comes back as that
    4xx instead of a generic 502 that reads as our fault -- and, more to the
    point, stops counting against the worker's HTTP success rate at the
    qualification gate.
    """

    def __init__(self, status: int, detail: str):
        super().__init__(f"serve {status}: {detail}")
        self.status = status


def _env(key, default=None):
    v = os.environ.get(key)
    return v if v not in (None, "") else default


# How long a drained leg keeps playing out work it already accepted before it is
# cut. This is a deploy-shedding bound, so it must never be shorter than the
# longest request the gateway will still wait for. Cutting early severs a
# request that has not timed out and books it as a 504 against a worker that was
# answering correctly. Size it from the longest generation this deployment
# actually serves rather than a round number; the default tracks the upstream
# read timeout, since a request the backend may still be working on is exactly
# the one a drained leg must not drop.
DRAIN_GRACE_S = float(_env("ENGY_DRAIN_GRACE_S",
                           _env("ENGY_READ_TIMEOUT", "1800")))


# -------------------------------------------------------------------- identity
def _resolve_miner_key(explicit: str | None) -> str:
    """The operator key this worker registers under. On a TEE box it is injected
    into the VM environment; MINER_KEY is accepted as the legacy spelling."""
    key = explicit or _env("ENGY_MINER_KEY") or _env("MINER_KEY")
    if not key:
        sys.exit("tee_miner: no miner key. Set ENGY_MINER_KEY (or pass "
                 "--miner-key). On a provisioned TEE worker this is injected "
                 "into the VM environment.")
    return key


def _resolve_worker_id(explicit: str | None) -> tuple[str, bool]:
    """The worker_id this process HELLOs with.

    A TEE worker adopts the PROVIDER-ASSIGNED id (explicit arg, else
    ENGY_WORKER_ID injected at declare) so its `workers` row matches its
    `worker_provisioning` row. Falling back to a minted uuid keeps the miner
    serving, but it will never be activated as a TEE worker, so the caller is
    told which of the two happened.
    """
    wid = explicit or _env("ENGY_WORKER_ID")
    if wid:
        return wid, True
    return uuid.uuid4().hex, False


def _resolve_worker_name(explicit: str | None, model: str,
                         serve_urls: list[str]) -> str:
    """This machine's worker name under one miner key. Stable across restarts so
    a restarted miner supersedes its own previous registration instead of
    briefly coexisting with the dead one."""
    name = (explicit or _env("ENGY_WORKER_NAME") or _env("ENGY_INSTANCE_ID"))
    if name:
        return name
    seed = "|".join([socket.gethostname(), model, ",".join(serve_urls)])
    return "tee-" + hashlib.sha256(seed.encode()).hexdigest()[:12]


# -------------------------------------------------------------------- hardware
def _detect_hw() -> dict:
    """Best-effort machine description for the HELLO frame. Never fatal, and
    never creates a CUDA context: the serve owns the GPUs, so they are read via
    nvidia-smi (a query) rather than by importing torch."""
    hw = {}
    for key, fn in (("host", socket.gethostname),
                    ("cpus", os.cpu_count)):
        try:
            hw[key] = fn()
        except Exception:
            pass
    try:
        hw["ram_gb"] = round(
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9, 1)
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
            hw["gpu_mem_gb"] = round(float(mem) / 1024, 1)
    except Exception:
        pass
    hw["gpus"] = _env("ENGY_GPUS") or \
        f"{hw.get('gpu_count', '?')}x {hw.get('gpu', 'GPU')}"
    if _env("ENGY_PARALLELISM"):
        hw["parallelism"] = _env("ENGY_PARALLELISM")
    if _env("ENGY_REGION"):
        hw["region"] = _env("ENGY_REGION")
    hw["tee"] = _detect_tee()
    return hw


def _detect_tee() -> dict:
    """Report the confidential-computing posture we can observe from inside the
    guest. Advisory only: the gateway trusts the attestation flow, never this."""
    tee = {}
    try:
        out = subprocess.run(["nvidia-smi", "conf-compute", "-f"],
                             capture_output=True, text=True, timeout=5).stdout
        if "ON" in out.upper():
            tee["gpu_cc"] = "on"
        elif out.strip():
            tee["gpu_cc"] = "off"
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "conf-compute", "-mgm"],
                             capture_output=True, text=True, timeout=5).stdout
        if ":" in out:
            tee["gpu_cc_multigpu"] = out.split(":", 1)[1].strip()
    except Exception:
        pass
    return tee


# ------------------------------------------------------------------ model root
def _model_root(checkpoint: str) -> str:
    """Cheap checkpoint identity: sha256 over config.json plus the shard
    manifest (each *.safetensors name and byte size). It names which checkpoint
    this worker pinned; it is not itself a weight proof."""
    if not checkpoint or not os.path.isdir(checkpoint):
        return _env("ENGY_MODEL_ROOT") or "tee-root"
    cfg = os.path.join(checkpoint, "config.json")
    h = hashlib.sha256()
    h.update(open(cfg, "rb").read() if os.path.exists(cfg) else b"{}")
    for shard in sorted(glob.glob(os.path.join(checkpoint, "*.safetensors"))):
        h.update(os.path.basename(shard).encode())
        h.update(str(os.path.getsize(shard)).encode())
    return h.hexdigest()


# ----------------------------------------------------------------------- load
class Load:
    """In-flight count shared across every leg of this process. A per-leg
    counter would report only that leg's share, which is exactly the partial
    view the gateway already has."""

    def __init__(self):
        self.inflight = 0
        self.last_idle = time.time()
        self._channels: set = set()

    def attach(self, ws):
        self._channels.add(ws)

    def detach(self, ws):
        self._channels.discard(ws)

    def start(self):
        self.inflight += 1

    def done(self):
        self.inflight = max(0, self.inflight - 1)
        if self.inflight == 0:
            self.last_idle = time.time()

    def idle_seconds(self) -> float:
        return 0.0 if self.inflight else time.time() - self.last_idle

    async def broadcast(self) -> None:
        """Push the true count to EVERY leg the moment it changes.

        The periodic heartbeat is far too coarse to admit against. A reported
        count is authoritative only while it is fresh; once it goes stale the
        gateway falls back to its own partial per-leg view. On a 30s timer that
        fallback is the common case, so admission ends up running against the
        very view this counter exists to replace.

        Reporting only to the leg that handled the request is not enough
        either. Each leg lands on a different gateway worker, and a request
        finishing on one frees capacity that a caller queued on another should
        get — a completion that other worker can never observe directly. Queued
        callers are released when a report DROPS, so without a broadcast a
        freed slot wakes nobody and the caller waits out its own timeout
        against capacity that is already idle.

        `capacity` and `kv` are both omitted. The gateway merges capacity
        across heartbeats and only re-reads KV when the frame actually carries
        it, so leaving them out preserves the last good values rather than
        clobbering them — and `kv_load()` costs two GETs per serve, which has no
        business on the request path. KV keeps riding the periodic frame.

        Best-effort per leg. This runs on the request path and must never fail
        the request it describes, so a dead leg is dropped, never raised.
        """
        for ws in list(self._channels):
            try:
                await ws.send(json.dumps(P.heartbeat(
                    inflight=self.inflight,
                    idle_seconds=self.idle_seconds())))
            except Exception:
                self._channels.discard(ws)


# ------------------------------------------------------------------ the lanes
# Sampling and tooling params the gateway may forward, passed straight through so
# a buyer's temperature/top_p/stop/tools/response_format/seed take effect.
# reasoning_effort and chat_template_kwargs BOTH gate thinking on GLM-5.2.
_PASSTHROUGH = ("temperature", "top_p", "n", "stop", "presence_penalty",
                "frequency_penalty", "seed", "logit_bias", "logprobs",
                "top_logprobs", "response_format", "tools", "tool_choice",
                "parallel_tool_calls", "user", "top_k", "min_p",
                "repetition_penalty", "stop_token_ids", "chat_template_kwargs",
                "reasoning_effort")

# Meaningful only on the raw text-completions lane. `echo` is the one that
# matters: with echo=true plus logprobs, sglang returns per-token logprobs over
# the PROMPT tokens, which is the point of the lane.
_TEXT_PASSTHROUGH = ("echo", "suffix", "best_of")


# ---- chat-template dialects ----
# One image serves the whole fleet, but the templates behind it are NOT
# interchangeable: a quirk worked around for one model is a regression on
# another, so every entry is opt-in per model rather than unconditional.
#
# * `effort` -- Qwen3.8's template accepts only xhigh (its default AND maximum) /
#   medium / low and `raise_exception`s a 400 on anything else, so OpenAI's
#   "high" and GLM's "max" have to fold onto xhigh. The prod gateway normalizes
#   every buyer effort into exactly that pair, so without this map a Qwen3.8
#   worker 400s most reasoning traffic. DeepSeek-V4 must NOT get the mapping:
#   its ladder is a preamble sglang injects (`encoding_dsv4`), and an effort the
#   active profile does not list is silently DROPPED to the profile default --
#   "xhigh" there does not 400, it quietly answers at the default instead.
# * `system_first_only` -- Qwen3.8 hard-rejects a `system` message anywhere but
#   position 0 ("System message must be at the beginning"), which 400s clients
#   that inject mid-conversation system turns (Claude Code after an auto-compact).
#   Demoting those to `user` rescues the request there, but on a template that
#   accepts them it rewrites the prompt for no reason, so it stays opt-in.
#
# Every mapping must be IDEMPOTENT: the gateway may already speak the model's
# dialect, and the miner has to be a no-op then rather than translating twice.
_DIALECTS = {
    "qwen3.8": {"effort": {"high": "xhigh", "max": "xhigh"},
                "system_first_only": True},
}
# Neutral: forward what the buyer asked for, rewrite nothing. Any model without
# a proven template quirk belongs here -- including DeepSeek-V4, whose efforts
# are all valid upstream values.
_DEFAULT_DIALECT = {"effort": {}, "system_first_only": False}


def resolve_dialect(model: str | None, override: str | None = None) -> dict:
    """The template dialect for the model this miner serves, matched on the engy
    model id (`qwen3.8-27b` -> the `qwen3.8` entry).

    `override` (--dialect / ENGY_TEMPLATE_DIALECT) names one explicitly, for a
    model id the table does not know yet. An unrecognised name resolves to the
    neutral default rather than raising: a miner that refuses to start is worse
    than one that forwards the buyer's request unmodified."""
    if override:
        return _DIALECTS.get(override.strip().lower(), _DEFAULT_DIALECT)
    name = (model or "").strip().lower()
    for prefix, dialect in _DIALECTS.items():
        if name.startswith(prefix):
            return dialect
    return _DEFAULT_DIALECT


def _raise_with_body(r) -> None:
    """httpx raises on 4xx/5xx with only the status line and an MDN link, which
    hides what the serve actually said. Surface the body."""
    if r.status_code < 400:
        return
    try:
        detail = r.text
    except Exception:
        detail = ""
    raise ServeError(_client_facing_status(r.status_code, detail), detail[:1000])


def _merge_tool_calls(acc: list, deltas: list) -> list:
    """Accumulate streamed tool-call fragments by index. OpenAI streams
    `function.arguments` as partial strings across chunks, so the pieces must be
    concatenated rather than overwritten or the call arrives truncated."""
    out = list(acc)
    for d in deltas or []:
        i = d.get("index", 0)
        while len(out) <= i:
            out.append({"index": len(out), "type": "function",
                        "function": {"name": "", "arguments": ""}})
        cur = out[i]
        if d.get("id"):
            cur["id"] = d["id"]
        if d.get("type"):
            cur["type"] = d["type"]
        fn = d.get("function") or {}
        if fn.get("name"):
            cur["function"]["name"] = fn["name"]
        if fn.get("arguments"):
            cur["function"]["arguments"] += fn["arguments"]
    return out


def is_text_completion(request: dict) -> bool:
    """True if this is a raw text completion rather than a chat request.

    Keyed on `prompt` being present rather than on a flag the gateway sets, so
    there is exactly one source of truth. `prompt` may be a string or a list of
    token ids. An empty-string prompt is still a prompt, so test for None."""
    return request.get("prompt") is not None and not request.get("messages")


def is_span_scoring(request: dict) -> bool:
    """True if the caller wants logprobs for a SPAN, not the whole prompt.

    `logprob_start_len` is a native /generate parameter; the OpenAI
    /v1/completions surface has no equivalent, which is why `echo` there is
    all-or-nothing. That all-or-nothing behaviour is what makes long-prefix
    scoring expensive: `echo` materialises a logits tensor over the WHOLE
    prompt, so its cost scales with the prefix and can exceed the memory a
    backend has left. Scoring a span costs scored_tokens x vocab x 4 instead,
    which is bounded by the span rather than by the prompt."""
    return request.get("logprob_start_len") is not None


def _span_logprobs(meta: dict) -> dict:
    """Map /generate's `input_token_logprobs` to the OpenAI logprobs shape.

    sglang returns [logprob, token_id, token_text] triples, one per scored
    token.

    We deliberately do NOT ask for `return_text_in_logprobs`. Measured on the
    live engine, that flag COLLAPSES the array: scoring 5 tokens returns 5
    entries with a null text field, but with the flag it returns ONE entry whose
    text is the whole span concatenated. That destroys the per-token breakdown a
    scorer exists to read, and it fails as a plausible-looking 200.

    So the logprobs come back with their TOKEN IDS instead of text, which is the
    stronger alignment anyway for a caller that sent input_ids. `tokens` and
    `text_offset` are omitted rather than faked."""
    triples = meta.get("input_token_logprobs") or []
    return {"token_logprobs": [t[0] if len(t) > 0 else None for t in triples],
            "token_ids": [t[1] if len(t) > 1 else None for t in triples]}


def _assistant_message(msg: dict) -> dict:
    """Preserve `tool_calls` (essential for agentic clients) and
    `reasoning_content`; drop upstream extras.

    `content` is passed through verbatim INCLUDING when empty because the model
    spent its whole budget thinking. Falling back to the reasoning text there
    would publish the model's raw chain of thought to end users as if it were
    the answer, silently and in the dangerous direction."""
    out = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning:
        out["reasoning_content"] = reasoning
    return out


def _clean_usage(raw: dict) -> dict:
    u = {"prompt_tokens": int(raw.get("prompt_tokens") or 0),
         "completion_tokens": int(raw.get("completion_tokens") or 0)}
    u["total_tokens"] = int(raw.get("total_tokens") or
                            u["prompt_tokens"] + u["completion_tokens"])
    details = raw.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        u["prompt_tokens_details"] = {
            "cached_tokens": int(details["cached_tokens"])}
    # Thinking tokens, a SUBSET of completion_tokens. sglang reports this FLAT
    # on `usage.reasoning_tokens` (its own UsageInfo field); the gateway edge
    # translates it into OpenAI's nested `completion_tokens_details` before a
    # buyer sees it, so the upstream dialect is kept here and converted once,
    # in one place. Dropping it here is what made the count invisible: a live
    # qwen3.8-27b answers 53 of 67 completion tokens on reasoning_effort=medium
    # and none of it reached the buyer.
    #
    # Absent stays absent -- a serve with no reasoning parser reports nothing,
    # and "we do not know" must not be forwarded as a zero.
    reasoning = raw.get("reasoning_tokens")
    if reasoning is None:
        nested = raw.get("completion_tokens_details")
        if isinstance(nested, dict):
            reasoning = nested.get("reasoning_tokens")
    if reasoning is not None:
        u["reasoning_tokens"] = int(reasoning)
    return u


def _chat_completion(model, message, usage, finish_reason="stop",
                     logprobs=None) -> dict:
    if isinstance(message, str):
        message = {"role": "assistant", "content": message}
    choice = {"index": 0, "finish_reason": finish_reason, "message": message}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:12],
            "object": "chat.completion", "model": model,
            "choices": [choice], "usage": usage}


def _text_completion(model, text, usage, finish_reason="stop",
                     logprobs=None) -> dict:
    """The logprobs block is passed through VERBATIM. On an echo request sglang
    fills token_logprobs/tokens/text_offset over the prompt tokens with a
    leading None for the first token, which has no predecessor to be
    conditioned on. That None is meaningful and must survive, so nothing here
    filters or re-indexes the array."""
    choice = {"index": 0, "finish_reason": finish_reason,
              "text": text if isinstance(text, str) else str(text)}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return {"id": "cmpl-" + uuid.uuid4().hex[:12], "object": "text_completion",
            "model": model, "choices": [choice], "usage": usage}


# --------------------------------------------------------------------- backend
class Serve:
    """The local OpenAI-compatible serve, least-in-flight balanced across urls.

    Three lanes, picked from the request rather than from a gateway flag:
      chat            -> /v1/chat/completions   (streamed)
      text completion -> /v1/completions        (whole prompt, echo lane)
      span scoring    -> /generate              (native, logprob_start_len)
    """

    def __init__(self, urls: list[str], served_model: str, read_timeout: float,
                 dialect: dict | None = None):
        self.urls = urls
        self.served_model = served_model
        self.read_timeout = read_timeout
        self.dialect = dialect or _DEFAULT_DIALECT
        self._n = {u: 0 for u in urls}
        self._down: set[str] = set()
        self._mtt: dict = {}   # per-serve KV pool size, cached (fixed per serve)

    def _pick(self) -> str:
        pool = [u for u in self.urls if u not in self._down] or self.urls
        u = min(pool, key=lambda s: self._n.get(s, 0))
        self._n[u] = self._n.get(u, 0) + 1
        return u

    async def probe(self, timeout: float = 8.0) -> str:
        """Backend liveness, as one of 'ok' | 'dead' | 'timeout'.

        The three outcomes are genuinely different and the caller must treat
        them differently:

        - 'dead'    the connection is refused/reset, or /health answers non-200.
                    Unambiguous, and both come back in milliseconds: the process
                    is gone, or sglang is telling us its scheduler/detokenizer
                    is hung. Act immediately.
        - 'timeout' /health accepted the connection and did not answer in time.
                    This is NOT proof of anything on its own. sglang's /health
                    round-trips through the detokenizer, so it also stalls when
                    the detokenizer is merely behind, and measured on prod
                    kimi-k3 the overwhelming majority of those clear by
                    themselves: 58 of 69 stalls on one worker lasted 13-19 s
                    (a single missed probe) while /get_load never missed a beat.
                    Only a stall that PERSISTS means the backend is unusable —
                    which is why sglang's own detokenizer watchdog waits 20 s
                    before it complains.
        - 'ok'      answered 200.

        Also refreshes the down-set `_pick` routes around; a timing-out serve is
        left routable, since the caller decides when a stall has gone on long
        enough to matter."""
        down: set[str] = set()
        worst = "ok"
        async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout, connect=4.0)) as c:
            for u in self.urls:
                try:
                    state = ("ok" if (await c.get(self._root(u) + "/health")
                                      ).status_code == 200 else "dead")
                except httpx.TimeoutException:
                    state = "timeout"
                except (httpx.NetworkError, httpx.RemoteProtocolError):
                    state = "dead"
                except Exception:
                    state = "timeout"      # unknown: treat as inconclusive
                if state == "dead":
                    down.add(u)
                if state != "ok" and worst == "ok":
                    worst = state
                elif state == "dead":
                    worst = "dead"
        self._down = down
        if len(down) < len(self.urls) and worst == "dead":
            worst = "ok"                   # another serve can still take work
        return worst

    async def scheduler_alive(self, timeout: float = 3.0) -> bool:
        """Is the SCHEDULER answering, even though /health did not?

        /health round-trips through the detokenizer, so it stalls whenever the
        detokenizer is merely behind — see `probe`. /get_load is served off the
        scheduler's own state and does not, which is why it "never missed a
        beat" through the kimi-k3 stalls: measured on an IDLE Ornith-1.5 serve
        /get_load answered in 1.2 ms against /health's 1.0 s, ~800x apart.

        True  -> at least one serve answered /get_load 200: the process is up
                 and scheduling, so a /health stall is queueing, not a wedge.
        False -> every serve failed to answer. That includes a serve too old to
                 expose /get_load at all (404), which deliberately keeps the
                 pre-existing withdraw-on-two-timeouts behaviour for it rather
                 than inventing health it cannot see.

        Never raises; only called on the timeout path, never on every probe."""
        try:
            async with httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout, connect=2.0)) as c:
                for u in self.urls:
                    try:
                        r = await c.get(self._root(u) + "/get_load")
                        if r.status_code == 200 and isinstance(r.json(), list):
                            return True
                    except Exception:
                        continue
        except Exception:
            pass
        return False

    def _release(self, u: str):
        self._n[u] = max(0, self._n.get(u, 0) - 1)

    async def _max_total(self, c, root: str):
        """KV pool size of the serve at `root` in TOKENS, cached (fixed per serve).
        sglang reports `max_total_num_tokens` PER DCP RANK, while /get_load's
        num_tokens is GLOBAL, so the true pool is max_total_num_tokens * dcp_size
        (dcp_size defaults to 1 → no change for non-DCP serves)."""
        if root not in self._mtt:
            try:
                r = await c.get(root + "/get_server_info")
                r.raise_for_status()
                info = r.json()
                mtt = int(info.get("max_total_num_tokens") or 0)
                dcp = int(info.get("dcp_size") or 1) or 1
                self._mtt[root] = (mtt * dcp) or None
            except Exception:
                self._mtt[root] = None
        return self._mtt[root]

    async def kv_load(self):
        """Live KV-cache pressure across the local serve(s): tokens occupied vs
        the pool size, summed over every scheduler of every serve url, from
        sglang `/get_load` + `max_total_num_tokens`. Cheap (two GETs) and never
        raises. Returns None if no serve exposes `/get_load` (older sglang /
        vLLM) — the gateway then judges this worker on in-flight count alone."""
        used = reqs = waiting = pending = ranks = 0
        limit = 0
        got = False
        try:
            timeout = httpx.Timeout(4.0, connect=3.0)
            async with httpx.AsyncClient(timeout=timeout) as c:
                for u in self.urls:
                    root = self._root(u)
                    try:
                        r = await c.get(root + "/get_load")
                        r.raise_for_status()
                        load = r.json()
                    except Exception:
                        continue
                    if not isinstance(load, list) or not load:
                        continue
                    mtt = await self._max_total(c, root)
                    for x in load:
                        if not isinstance(x, dict):
                            continue
                        used += int(x.get("num_tokens", 0) or 0)
                        reqs += int(x.get("num_reqs", 0) or 0)
                        waiting += int(x.get("num_waiting_reqs", 0) or 0)
                        pending += int(x.get("num_pending_tokens", 0) or 0)
                        ranks += 1
                        if mtt:
                            limit += int(mtt)
                    got = True
        except Exception:
            return None
        if not got:
            return None
        kv = {"used": used, "reqs": reqs, "waiting": waiting,
              "pending": pending, "ranks": ranks}
        if limit:
            kv["limit"] = limit
            kv["frac"] = round(used / limit, 4) if limit else None
        return kv

    @staticmethod
    def _root(url: str) -> str:
        """Base without a trailing /v1: /generate does not live under /v1."""
        return url[:-3].rstrip("/") if url.endswith("/v1") else url.rstrip("/")

    @staticmethod
    def _v1(url: str) -> str:
        return url.rstrip("/") if url.endswith("/v1") else url.rstrip("/") + "/v1"

    def _chat_body(self, request: dict, *, stream: bool) -> dict:
        # Scrub for templates that reject a `system` message anywhere but
        # position 0 -- see `_DIALECTS`. Off unless this model needs it: on a
        # template that accepts mid-conversation system turns, demoting them
        # silently rewrites the prompt.
        msgs = request.get("messages", []) or []
        if self.dialect["system_first_only"] and any(
                m.get("role") == "system" for m in msgs[1:]):
            msgs = [m if i == 0 or m.get("role") != "system"
                    else {**m, "role": "user"}
                    for i, m in enumerate(msgs)]
        body = {"model": self.served_model or request.get("model"),
                "messages": msgs}
        if request.get("max_tokens"):
            body["max_tokens"] = request["max_tokens"]
        for k in _PASSTHROUGH:
            if request.get(k) is not None:
                body[k] = request[k]
        # Translate the effort AFTER the passthrough copy, so the dialect maps
        # the value that is actually on the wire. Unmapped efforts (and every
        # effort on the neutral dialect) go upstream untouched.
        effort = self.dialect["effort"].get(body.get("reasoning_effort"))
        if effort:
            body["reasoning_effort"] = effort
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body

    def _text_body(self, request: dict) -> dict:
        """Body for the RAW text-completions lane.

        The contract is byte fidelity: `prompt` is forwarded EXACTLY as the
        buyer sent it, no chat template and no normalisation, because the caller
        rendered the template itself and is scoring a span at a known offset.
        A list-of-ints prompt stays pre-tokenised.

        `max_tokens: 0` is preserved rather than dropped: echo-scoring asks for
        no new tokens at all, so the truthiness idiom used on the chat lane
        would wrongly treat 0 as absent."""
        body = {"model": self.served_model or request.get("model"),
                "prompt": request.get("prompt")}
        if request.get("max_tokens") is not None:
            body["max_tokens"] = request["max_tokens"]
        for k in _PASSTHROUGH + _TEXT_PASSTHROUGH:
            if request.get(k) is not None:
                body[k] = request[k]
        # top_logprobs is the chat-lane spelling; the text lane carries the
        # count in `logprobs` itself, so never forward both.
        body.pop("top_logprobs", None)
        return body

    def _generate_body(self, request: dict) -> dict:
        """Translate a span-scoring completion into a native /generate call.

        `prompt` maps to `text` for a string and `input_ids` for token ids,
        keeping pre-tokenised input pre-tokenised, which is the whole reason a
        scorer sends ids. `return_text_in_logprobs` is deliberately unset; see
        _span_logprobs."""
        p = request.get("prompt")
        body: dict = {
            "sampling_params": {
                "max_new_tokens": request.get("max_tokens") or 1,
                "temperature": request.get("temperature", 0),
            },
            "return_logprob": True,
            "logprob_start_len": int(request["logprob_start_len"]),
        }
        if isinstance(p, list) and p and all(isinstance(x, int) for x in p):
            body["input_ids"] = p
        else:
            body["text"] = p
        for k in ("top_p", "top_k", "min_p", "stop", "seed",
                  "repetition_penalty", "frequency_penalty",
                  "presence_penalty"):
            if request.get(k) is not None:
                body["sampling_params"][k] = request[k]
        return body

    async def complete(self, request: dict) -> dict:
        """The two non-streaming lanes: span scoring and raw text completion."""
        url = self._pick()
        model = request.get("model")
        timeout = httpx.Timeout(self.read_timeout, connect=10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                if is_span_scoring(request):
                    r = await c.post(self._root(url) + "/generate",
                                     json=self._generate_body(request))
                    _raise_with_body(r)
                    j = r.json()
                    meta = j.get("meta_info") or {}
                    usage = {
                        "prompt_tokens": meta.get("prompt_tokens") or 0,
                        "completion_tokens": meta.get("completion_tokens") or 0}
                    usage["total_tokens"] = (usage["prompt_tokens"]
                                             + usage["completion_tokens"])
                    # Native /generate reports the prefix-cache hit as a FLAT
                    # `cached_tokens`, not the nested OpenAI shape the other
                    # lanes give. Without re-nesting it the gateway sees no
                    # cache hit and bills the whole prefix at the full input
                    # rate, which is exactly backwards for span scoring, the
                    # one workload that re-sends an identical multi-thousand
                    # token prefix on every request.
                    cached = meta.get("cached_tokens")
                    if cached is not None:
                        usage["prompt_tokens_details"] = {
                            "cached_tokens": int(cached)}
                    fr = meta.get("finish_reason")
                    finish = (fr.get("type") if isinstance(fr, dict) else fr) or "stop"
                    return _text_completion(model, j.get("text") or "", usage,
                                            finish, _span_logprobs(meta))

                r = await c.post(self._v1(url) + "/completions",
                                 json=self._text_body(request))
                _raise_with_body(r)
                j = r.json()
            choice = (j.get("choices") or [{}])[0]
            return _text_completion(model, choice.get("text") or "",
                                    _clean_usage(j.get("usage") or {}),
                                    choice.get("finish_reason") or "stop",
                                    choice.get("logprobs"))
        finally:
            self._release(url)

    async def stream(self, request: dict):
        """Yield ("delta", delta_dict) as the serve produces tokens, then
        ("done", output) once with the assembled OpenAI-shaped body."""
        url = self._pick()
        body = self._chat_body(request, stream=True)
        text_parts: list[str] = []
        usage: dict = {}
        finish = "stop"
        timeout = httpx.Timeout(self.read_timeout, connect=10.0)
        reasoning_parts: list[str] = []
        tool_calls: list = []
        logprobs_seen = None
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                async with c.stream("POST", self._v1(url) + "/chat/completions",
                                    json=body) as r:
                    if r.status_code >= 400:
                        detail = (await r.aread()).decode("utf-8", "replace")
                        raise ServeError(
                            _client_facing_status(r.status_code, detail),
                            detail[:600])
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            ev = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if ev.get("usage"):
                            usage = ev["usage"]
                        for ch in ev.get("choices") or []:
                            d = ch.get("delta") or {}
                            out = {}
                            if d.get("content"):
                                text_parts.append(d["content"])
                                out["content"] = d["content"]
                            # Reasoning deltas keep the stream from going silent
                            # while the model thinks; tool_calls are essential
                            # for agentic clients.
                            if d.get("reasoning_content"):
                                reasoning_parts.append(d["reasoning_content"])
                                out["reasoning_content"] = d["reasoning_content"]
                            if d.get("tool_calls"):
                                tool_calls = _merge_tool_calls(
                                    tool_calls, d["tool_calls"])
                                out["tool_calls"] = d["tool_calls"]
                            if ch.get("logprobs") is not None:
                                logprobs_seen = ch["logprobs"]
                            if out:
                                # choice-level logprobs ride beside the delta so
                                # a streaming buyer gets them per chunk, not
                                # just on the terminal frame
                                yield "delta", {"delta": out,
                                                "logprobs": ch.get("logprobs")}
                            if ch.get("finish_reason"):
                                finish = ch["finish_reason"]
        finally:
            self._release(url)

        msg = {"role": "assistant", "content": "".join(text_parts)}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_parts:
            msg["reasoning_content"] = "".join(reasoning_parts)
        yield "done", _chat_completion(
            request.get("model"), _assistant_message(msg),
            _clean_usage(usage) if usage else {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            finish, logprobs_seen)


# --------------------------------------------------------------------- session
async def _serve_one(ws, frame, backend: Serve, load: Load):
    corr = frame["corr_id"]
    request = frame.get("request") or {}
    rid = request.get("engy_request_id") or uuid.uuid4().hex
    load.start()
    try:
        # Report the acquire from INSIDE the try: `load.start()` has already
        # incremented, so anything raised here must still reach `finally` or the
        # slot leaks.
        await load.broadcast()
        if is_text_completion(request):
            # Both raw-completion lanes answer in one shot. Streaming a span
            # score is meaningless: max_tokens is 0 and the payload is the
            # logprob array, not tokens.
            output = await backend.complete(request)
            await ws.send(json.dumps(P.response(corr, rid, {}, output=output)))
        else:
            async for kind, payload in backend.stream(request):
                if kind == "delta":
                    await ws.send(json.dumps(P.chunk(
                        corr, payload["delta"],
                        logprobs=payload.get("logprobs"))))
                else:
                    await ws.send(json.dumps(
                        P.response(corr, rid, {}, output=payload)))
    except asyncio.CancelledError:
        # The buyer is gone. Unwinding here tears down the upstream stream,
        # which makes the engine abort the generation and free the KV slot.
        raise
    except Exception as e:
        try:
            await ws.send(json.dumps(P.response(
                corr, rid, {}, error={"message": str(e)[:500],
                                      "type": type(e).__name__,
                                      "status": getattr(e, "status", None)})))
        except Exception:
            pass
        print(f"[tee_miner] serve error: {e!r}", flush=True)
    finally:
        # MUST be in `finally`: a slot leaked on the error path would shrink the
        # advertised capacity permanently, and the count only ever drifts one
        # way, so the miner would slowly strangle itself.
        load.done()
        await load.broadcast()


async def _heartbeat(ws, interval: float, load: Load, cap: dict, backend=None):
    try:
        while True:
            await asyncio.sleep(interval)
            kv = await backend.kv_load() if backend is not None else None
            await ws.send(json.dumps(P.heartbeat(
                inflight=load.inflight, idle_seconds=load.idle_seconds(),
                capacity=cap, kv=kv)))
    except (websockets.ConnectionClosed, asyncio.CancelledError):
        pass


async def _retire(ws, hb, serving: dict, load: Load, tag: str,
                  grace: float = DRAIN_GRACE_S):
    """A drained leg keeps serving what it already accepted, then closes.

    It stays attached on purpose: the gateway worker behind it must keep
    receiving load reports until the last request lands. Closing at the drain
    instant is what 504s every in-flight request.

    `grace` bounds that wait, and is sized from the longest generation this
    deployment serves rather than picked as a round number — see
    DRAIN_GRACE_S. Anything below it hands the same 504 back on a slower path:
    the request is cut while the gateway is still waiting for it.
    """
    deadline = time.time() + grace
    try:
        while serving and time.time() < deadline:
            await asyncio.sleep(0.5)
    finally:
        hb.cancel()
        load.detach(ws)
        try:
            await ws.close()
        except Exception:
            pass
        left = len(serving)
        print(f"[tee_miner] {tag} retired"
              + (f" with {left} still in flight" if left else ""), flush=True)


async def _session(ws, cfg, cap, load: Load, tag: str) -> bool:
    """One leg. Returns True if the gateway asked us to rotate (RECONNECT), in
    which case this leg is handed to _retire and the caller must NOT close it."""
    load.attach(ws)
    await ws.send(json.dumps(P.hello(
        cfg["miner_key"], cfg["model"], cfg["model_root"], hw=cfg["hw"],
        capacity=cap, worker_name=cfg["worker_name"],
        worker_id=cfg["worker_id"])))
    print(f"[tee_miner] {tag} HELLO sent (max_inflight={cap['max_inflight']})",
          flush=True)
    hb = asyncio.create_task(
        _heartbeat(ws, cfg.get("hb_s", 30.0), load, cap, cfg["backend"]))
    serving: dict[str, asyncio.Task] = {}
    drained = False
    try:
        async for msg in ws:
            frame = json.loads(msg)
            t = frame.get("type")
            if t == P.SERVE:
                corr = frame["corr_id"]
                task = asyncio.create_task(
                    _serve_one(ws, frame, cfg["backend"], load))
                serving[corr] = task
                task.add_done_callback(lambda _t, c=corr: serving.pop(c, None))
            elif t == P.CANCEL:
                task = serving.get(frame.get("corr_id") or "")
                if task is not None:
                    task.cancel()
            elif t == P.PING:
                kv = await cfg["backend"].kv_load()
                await ws.send(json.dumps(P.heartbeat(
                    inflight=load.inflight, idle_seconds=load.idle_seconds(),
                    capacity=cap, kv=kv)))
            elif t == P.RECONNECT:
                # Gateway is draining for a deploy. Return at once so the caller
                # re-dials onto the new colour, but do NOT close this socket:
                # requests still streaming on it would 504 the instant it shut.
                drained = True
                break
            elif t == P.ADMIT:
                print(f"[tee_miner] {tag} ADMITTED", flush=True)
            elif t == P.DENY:
                print(f"[tee_miner] {tag} DENIED: {frame.get('reason')}",
                      flush=True)
                break
    except websockets.ConnectionClosed:
        pass
    finally:
        if drained:
            # Hand the socket to _retire; the caller re-dials at once.
            asyncio.create_task(_retire(ws, hb, serving, load, tag))
        else:
            hb.cancel()
            load.detach(ws)
            lost = len(serving)
            for task in list(serving.values()):
                task.cancel()
            if lost:
                # The gateway is still waiting on every one of these; each
                # becomes a 504 the instant this socket goes. This is the only
                # place that number exists — the gateway logs the 504 but not
                # that a whole leg took the request down with it.
                print(f"[tee_miner] {tag} dropping {lost} in-flight request(s)",
                      flush=True)
    return drained


# WebSocket close codes worth naming in a log line. 1006 is the one that matters
# most here: it is synthesised locally when the transport died WITHOUT a close
# frame — the peer never got to say goodbye. A gateway that keeps writing SERVE
# frames into such a socket sees the writes succeed (they land in the kernel
# buffer) and only discovers the leg is gone when its own keepalive gives up,
# 20-40 s later. Every request dispatched in that window 504s having never
# reached the serve.
_CLOSE_NAMES = {
    1000: "normal", 1001: "going away", 1002: "protocol error",
    1003: "unsupported data", 1005: "no status",
    1006: "abnormal - transport died, no close frame",
    1007: "invalid payload", 1008: "policy violation",
    1009: "message too big", 1010: "extension negotiation failed",
    1011: "server error or keepalive timeout", 1012: "service restart",
    1013: "try again later", 1015: "TLS failure",
}


def _close_note(ws) -> str:
    """Human-readable close code + reason, for a log line that can be triaged
    without a packet capture."""
    if ws is None:
        return "never connected"
    code = getattr(ws, "close_code", None)
    if code is None:
        return "close code unknown"
    reason = (getattr(ws, "close_reason", "") or "").strip()
    note = f"close {code} ({_CLOSE_NAMES.get(code, 'unspecified')})"
    return f"{note} {reason!r}" if reason else note


async def _leg(i: int, n: int, cfg, cap, load: Load):
    url = cfg["gw"] if n <= 1 else f"{cfg['gw']}/{i}"
    tag = f"gw{i}"
    while True:
        ws = None
        try:
            ws = await websockets.connect(
                url, ping_interval=15, ping_timeout=60,
                close_timeout=10, max_queue=128,
                # websockets caps inbound frames at 1 MiB by default, which is
                # smaller than a long-context request: ~156k prompt tokens is
                # already ~1 MiB of JSON. The oversized frame closes the
                # session, so the gateway books the worker as timed out and
                # answers 504 for a request the serve never even received.
                max_size=64 * 1024 * 1024)
            drained = await _session(ws, cfg, cap, load, tag)
            if drained:
                # _retire owns the socket now. Re-dial immediately so the
                # replacement leg lands on the new colour without a gap.
                print(f"[tee_miner] {tag} draining; re-dialing now", flush=True)
                continue
            await ws.close()
            # `except websockets.ConnectionClosed: pass` in _session swallows a
            # clean 1000, an abnormal 1006 and a server-side 1011 alike, so a
            # bare "session ended" cannot tell a normal rotation from a dropped
            # connection. That gap is why a 504 had to be diagnosed from a
            # duration histogram instead of from the reason. Print the code.
            print(f"[tee_miner] {tag} session ended; {_close_note(ws)}; "
                  f"reconnecting", flush=True)
        except Exception as e:
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
            print(f"[tee_miner] {tag} disconnected: {e!r}; {_close_note(ws)}",
                  flush=True)
        await asyncio.sleep(0.5)


def _split_capacity(capacity: dict, n: int) -> dict:
    """Per-leg capacity for an n-leg dial: each leg enforces its share of
    max_inflight. Ceil, so the legs together never advertise less than the
    real total."""
    if n <= 1 or not capacity.get("max_inflight"):
        return capacity
    cap = dict(capacity)
    cap["max_inflight"] = -(-int(cap["max_inflight"]) // n)
    return cap


def _fetch_worker_count(gw: str) -> int | None:
    """One leg per gateway worker process, so we receive all buyer traffic; a
    single-dial miner only ever lands on worker 0.

    Returns None (not 1) when /gw/meta is unreachable, so the supervisor KEEPS
    the current count on a transient failure instead of collapsing a live
    multi-worker miner to a single leg — dropping legs 1..N-1 504s their
    in-flight streams and sheds capacity (measured on the reference miner:
    under load the sole /gw/meta responder CPU-saturates, the GET times out,
    and the fallback-to-1 flapping cost ~60% of fleet throughput)."""
    n = _env("ENGY_GW_WORKERS")
    if n:
        return max(1, int(n))
    try:
        meta = gw.replace("wss://", "https://").replace("ws://", "http://") + "/meta"
        r = httpx.get(meta, timeout=5.0)
        r.raise_for_status()
        return max(1, min(int((r.json() or {}).get("workers") or 1), 64))
    except Exception:
        return None


# How often the supervisor re-reads /gw/meta so a gateway worker-count change
# (a scale event) is picked up without a miner restart, and re-probes the
# serve's /health.
META_RECHECK_S = 5.0
# Backend health gate: an OOM/hang is detected within ~META_RECHECK_S and every
# leg drops so the gateway routes buyers elsewhere; _HEALTH_OK_STREAK
# consecutive good probes are required before reconnecting, to ride out a
# flapping/relaunching backend.
_HEALTH_OK_STREAK = 2
# Consecutive TIMED-OUT probes before withdrawing. 1 is too trigger-happy: on
# prod kimi-k3, 58 of 69 /health stalls were a single missed probe (13-19 s)
# that cleared itself, while the real faults — a hung detokenizer — ran 34 s and
# 72 s. At the 8 s timeout + 5 s interval cadence, 2 means ~26 s, just past
# sglang's own 20 s detokenizer watchdog.
_HEALTH_FAIL_STREAK = 2
# ...and then only if the SCHEDULER has gone quiet too. A stall that /get_load
# rides out still withdraws once it has lasted this long, so a genuinely wedged
# serve — the known O(N^2) detokenizer hang keeps scheduling while it serves
# nobody — is still caught. 120 s is ~4.6x the old trip point and past every
# self-clearing stall on record (13-19 s typical, 72 s worst), so it is a
# backstop, not the primary detector.
_HEALTH_STALL_MAX_S = 120.0
# While riding a stall out, say so at most this often.
_HEALTH_NOTE_S = 30.0


class _HealthGate:
    """Should the miner be withdrawn right now? One probe outcome per tick.

    The three probe outcomes are not symmetric:

    - 'dead'    connection refused/reset or /health non-200. Unambiguous, and
                back in milliseconds — withdraw on the first one.
    - 'ok'      clears the gate; reconnect after _HEALTH_OK_STREAK of them.
    - 'timeout' NOT proof of anything. sglang's /health round-trips through the
                DETOKENIZER, so it stalls whenever the detokenizer is behind,
                which a healthy worker does routinely. Withdrawing on two of
                them cancels every leg, and the gateway books each in-flight
                request as `504 miner timeout or disconnect`: all 5 of the 504s
                the Ornith canary served across 1183 requests were this, with
                no backend fault behind any of them (the largest prompt in any
                failure window was 379 tokens, and a 1421-request load test
                never tripped it once).

    So on a timeout, ask the scheduler directly via /get_load before believing
    /health. If it answers, the backend is alive and busy, not dead: stay up
    and let the in-flight work finish. Only a stall where /get_load has gone
    quiet TOO, or one that outlasts _HEALTH_STALL_MAX_S however lively
    /get_load looks, is treated as a fault."""

    def __init__(self, backend, *, clock=time.monotonic):
        self._backend = backend
        self._clock = clock
        self.down = False
        self.ok_streak = 0
        self.fail_streak = 0
        self._stall_since = None
        self._last_note = 0.0

    async def update(self, state: str) -> bool:
        """Fold in one probe outcome; returns True while legs must stay down."""
        if state == "ok":
            self.ok_streak += 1
            self.fail_streak = 0
            self._stall_since = None
            self._last_note = 0.0
            if self.down and self.ok_streak >= _HEALTH_OK_STREAK:
                self.down = False
                print("[tee_miner] backend RECOVERED — reconnecting legs",
                      flush=True)
            return self.down
        self.ok_streak = 0
        self.fail_streak += 1
        now = self._clock()
        if self._stall_since is None:
            self._stall_since = now
        if self.down:
            return True      # already withdrawn; recovery is ok_streak's job
        why = await self._verdict(state, now - self._stall_since)
        if why:
            self.down = True
            print("[tee_miner] backend UNHEALTHY (%s) — disconnecting all legs"
                  % why, flush=True)
        return self.down

    async def _verdict(self, state: str, stalled: float):
        """A reason to withdraw, or None to stay up. Consults /get_load ONLY
        once /health has timed out _HEALTH_FAIL_STREAK times running."""
        if state == "dead":
            return "dead"
        if self.fail_streak < _HEALTH_FAIL_STREAK:
            return None
        if not await self._backend.scheduler_alive():
            return ("/health timeout x%d for %.0fs and /get_load silent too"
                    % (self.fail_streak, stalled))
        if stalled >= _HEALTH_STALL_MAX_S:
            return ("/health timeout x%d — /get_load still answers, but %.0fs "
                    "of stall is a wedge" % (self.fail_streak, stalled))
        self._note(stalled)
        return None

    def _note(self, stalled: float):
        now = self._clock()
        if self._last_note and now - self._last_note < _HEALTH_NOTE_S:
            return
        self._last_note = now
        print("[tee_miner] /health stalled %.0fs (timeout x%d) but /get_load "
              "answers — scheduler alive, staying up"
              % (stalled, self.fail_streak), flush=True)


async def _supervise(cfg, capacity: dict, load: Load):
    """Leg supervisor, ported from the reference miner.

    Two jobs beyond keeping N legs dialed:
    - Backend health gate (_HealthGate): when the serve OOMs or wedges,
      DISCONNECT every leg so the gateway stops routing buyers here (the
      alternative is advertising a dead backend and 502ing everything). A
      /health stall alone is not that — it is confirmed against /get_load
      first, since cancelling legs 504s whatever they were serving. Reconnect
      only after _HEALTH_OK_STREAK consecutive good probes.
    - Incremental leg management keyed by worker INDEX: a gateway scale event
      adds legs for new workers and drops legs for removed ones, but never
      touches an existing leg — cancelling a live leg 504s its streams.
    """
    backend: Serve = cfg["backend"]
    legs: dict[int, asyncio.Task] = {}
    cur_n = 0
    # Startup bootstrap: retry /gw/meta before committing, so leg 0 is created
    # with the split for the REAL worker count. Existing legs keep their
    # capacity, so a wrong first split would stick.
    fetched = None
    for _ in range(3):
        fetched = await asyncio.to_thread(_fetch_worker_count, cfg["gw"])
        if fetched is not None:
            break
        await asyncio.sleep(1.0)
    gate = _HealthGate(backend)
    try:
        while True:
            if await gate.update(await backend.probe()):
                if legs:
                    for t in legs.values():
                        t.cancel()
                    await asyncio.gather(*legs.values(), return_exceptions=True)
                    legs.clear()
                    cur_n = 0
                await asyncio.sleep(META_RECHECK_S)
                continue
            # KEEP the current count on a transient meta failure (None); fall
            # back to 1 only at startup, when no count has ever resolved.
            n = fetched if fetched is not None else (cur_n or 1)
            if n != cur_n or any(t.done() for t in legs.values()):
                cap = _split_capacity(capacity, n)
                for i in range(n):              # add missing / dead legs
                    if i not in legs or legs[i].done():
                        legs[i] = asyncio.create_task(_leg(i, n, cfg, cap, load))
                for i in [i for i in legs if i >= n]:   # drop removed workers
                    legs[i].cancel()
                    legs.pop(i)
                if cur_n and n != cur_n:
                    print(f"[tee_miner] worker count {cur_n} -> {n} "
                          f"({'added' if n > cur_n else 'removed'} legs, "
                          "existing kept)", flush=True)
                cur_n = n
            await asyncio.sleep(META_RECHECK_S)
            fetched = await asyncio.to_thread(_fetch_worker_count, cfg["gw"])
    finally:
        for t in legs.values():
            t.cancel()
        await asyncio.gather(*legs.values(), return_exceptions=True)


# ------------------------------------------------------------------------ main
def main(argv=None):
    p = argparse.ArgumentParser(
        description="engy tee_miner (gateway leg for a TEE-attested worker)")
    p.add_argument("--gw", default=_env("ENGY_GW_URL", "wss://api.engy.ai/gw"))
    p.add_argument("--model", default=_env("ENGY_MODEL", "glm-5.2"))
    p.add_argument("--miner-key", "--miner_key", dest="miner_key", default=None,
                   help="default $ENGY_MINER_KEY, injected on a TEE worker")
    p.add_argument("--worker-id", "--worker_id", dest="worker_id", default=None,
                   help="provider-assigned worker_id from declare; default "
                        "$ENGY_WORKER_ID. Must match the worker_provisioning "
                        "row or the worker cannot be activated as TEE")
    p.add_argument("--worker-name", "--worker_name", dest="worker_name",
                   default=None, help="default $ENGY_WORKER_NAME")
    p.add_argument("--serve-url", dest="serve_url",
                   default=_env("ENGY_SERVE_URL", "http://127.0.0.1:8000"),
                   help="local serve base url, comma-separated for several")
    p.add_argument("--served-model", dest="served_model",
                   default=_env("ENGY_SERVED_MODEL"),
                   help="the serve's --served-model-name, if it differs")
    p.add_argument("--checkpoint", default=_env("ENGY_CHECKPOINT", ""),
                   help="checkpoint dir, for the model_root identity only")
    p.add_argument("--max-inflight", type=int,
                   default=int(_env("ENGY_MAX_INFLIGHT", "32")))
    p.add_argument("--context-length", type=int,
                   default=int(_env("ENGY_CONTEXT_LENGTH", "0")) or None)
    p.add_argument("--read-timeout", type=float,
                   default=float(_env("ENGY_READ_TIMEOUT", "1800")))
    p.add_argument("--modalities", default=_env("ENGY_MODALITIES", "text"),
                   help="comma-separated input modalities the serve accepts, "
                        "e.g. 'text,image' for a VLM serve (default: text)")
    p.add_argument("--dialect", default=_env("ENGY_TEMPLATE_DIALECT"),
                   help="force a chat-template dialect (see _DIALECTS) instead "
                        "of matching on the model id")
    p.add_argument("--require-tee-identity", action="store_true",
                   default=_env("ENGY_REQUIRE_TEE_IDENTITY") == "1",
                   help="exit rather than serve without a provider-assigned "
                        "ENGY_WORKER_ID")
    args = p.parse_args(argv)

    miner_key = _resolve_miner_key(args.miner_key)
    worker_id, assigned = _resolve_worker_id(args.worker_id)
    serve_urls = [u.strip() for u in args.serve_url.split(",") if u.strip()]
    worker_name = _resolve_worker_name(args.worker_name, args.model, serve_urls)

    if not assigned:
        msg = ("tee_miner: ENGY_WORKER_ID is not set, so this process minted "
               f"{worker_id}. It will serve, but it registers a worker no "
               "provisioning row declared, so activation cannot flip it to "
               "type='tee'.")
        if args.require_tee_identity:
            sys.exit(msg.replace("It will serve, but it", "Refusing to start: it"))
        print("[tee_miner] WARNING: " + msg, flush=True)

    cap = {"max_inflight": args.max_inflight,
           # Rides undivided past _split_capacity: the gateway routes on the
           # per-leg share but records the total on the worker, and denies the
           # HELLO outright ("capacity.max_inflight_total not declared").
           "max_inflight_total": args.max_inflight}
    if args.context_length:
        cap["context_length"] = args.context_length
    mods = [s.strip() for s in (args.modalities or "").split(",") if s.strip()]
    if mods and mods != ["text"]:       # text-only is the registry default
        cap["input_modalities"] = mods
    # Shape limits (max_input_tokens / max_output_tokens / max_request_s) are
    # operator-owned per-model spec, injected by the gateway from the models
    # table. Self-reported values for those keys are ignored, so we omit them.

    cfg = {
        "gw": args.gw.rstrip("/"),
        "model": args.model,
        "miner_key": miner_key,
        "worker_id": worker_id,
        "worker_name": worker_name,
        "model_root": _model_root(args.checkpoint),
        "hw": _detect_hw(),
        "backend": Serve(serve_urls, args.served_model, args.read_timeout,
                         resolve_dialect(args.model, args.dialect)),
        "hb_s": float(_env("ENGY_HEARTBEAT_S", "30")),
    }

    # The supervisor splits max_inflight per leg (each leg advertises its SHARE,
    # not the whole thing — advertising the full value on every leg tells the
    # gateway this worker can take n times what it can), re-reads the worker
    # count for scale events, and health-gates the legs on the serve's /health.
    load = Load()
    print(f"[tee_miner] supervising legs of {cfg['gw']} as {cfg['model']} "
          f"worker={worker_name} worker_id={worker_id}"
          f"{'' if assigned else ' (MINTED, not provider-assigned)'} "
          f"root={cfg['model_root'][:12]} serves={serve_urls} "
          f"total_inflight={cap['max_inflight']} "
          f"modalities={mods or ['text']}", flush=True)
    hw = cfg["hw"]
    print(f"[tee_miner] hw: {hw.get('gpus')} | {hw.get('gpu_mem_gb')}GB/gpu | "
          f"{hw.get('cpus')} cpu | {hw.get('ram_gb')}GB ram | "
          f"cc={hw.get('tee', {}).get('gpu_cc', '?')} "
          f"multigpu={hw.get('tee', {}).get('gpu_cc_multigpu', '?')}",
          flush=True)

    asyncio.run(_supervise(cfg, cap, load))


if __name__ == "__main__":
    main()
