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
    def heartbeat(inflight=0, idle_seconds=0.0, capacity=None):
        f = {"type": P.HEARTBEAT, "inflight": inflight,
             "idle_seconds": idle_seconds}
        if capacity:
            f["capacity"] = capacity
        return f

    @staticmethod
    def chunk(corr_id, delta=None):
        return {"type": P.CHUNK, "corr_id": corr_id, "delta": delta or {}}

    @staticmethod
    def response(corr_id, request_id, commitment, output=None, error=None):
        f = {"type": P.RESPONSE, "corr_id": corr_id, "request_id": request_id,
             "commitment": commitment, "output": output or {}}
        if error:
            f["error"] = error
        return f


def _env(key, default=None):
    v = os.environ.get(key)
    return v if v not in (None, "") else default


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


# --------------------------------------------------------------------- backend
class Serve:
    """The local OpenAI-compatible serve, least-in-flight balanced across urls."""

    def __init__(self, urls: list[str], served_model: str, read_timeout: float):
        self.urls = urls
        self.served_model = served_model
        self.read_timeout = read_timeout
        self._n = {u: 0 for u in urls}

    def _pick(self) -> str:
        u = min(self.urls, key=lambda s: self._n.get(s, 0))
        self._n[u] = self._n.get(u, 0) + 1
        return u

    def _release(self, u: str):
        self._n[u] = max(0, self._n.get(u, 0) - 1)

    async def stream(self, request: dict):
        """Yield ("delta", text) as the serve produces tokens, then
        ("done", output) once with the assembled OpenAI-shaped body."""
        url = self._pick()
        body = dict(request)
        body["model"] = self.served_model or request.get("model")
        body["stream"] = True
        body.setdefault("stream_options", {"include_usage": True})
        text_parts: list[str] = []
        usage: dict = {}
        finish = "stop"
        timeout = httpx.Timeout(self.read_timeout, connect=10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                async with c.stream("POST", url + "/v1/chat/completions",
                                    json=body) as r:
                    if r.status_code >= 400:
                        detail = (await r.aread()).decode("utf-8", "replace")
                        raise RuntimeError(
                            f"serve {r.status_code}: {detail[:600]}")
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
                            piece = (ch.get("delta") or {}).get("content")
                            if piece:
                                text_parts.append(piece)
                                yield "delta", piece
                            if ch.get("finish_reason"):
                                finish = ch["finish_reason"]
        finally:
            self._release(url)

        text = "".join(text_parts)
        yield "done", {
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": text}}],
            "usage": usage or {"prompt_tokens": 0,
                               "completion_tokens": 0, "total_tokens": 0},
        }


# --------------------------------------------------------------------- session
async def _serve_one(ws, frame, backend: Serve, load: Load):
    corr = frame["corr_id"]
    request = frame.get("request") or {}
    rid = request.get("engy_request_id") or uuid.uuid4().hex
    load.start()
    try:
        async for kind, payload in backend.stream(request):
            if kind == "delta":
                await ws.send(json.dumps(P.chunk(corr, {"content": payload})))
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
                                      "type": type(e).__name__})))
        except Exception:
            pass
        print(f"[tee_miner] serve error: {e!r}", flush=True)
    finally:
        load.done()


async def _heartbeat(ws, interval: float, load: Load, cap: dict):
    try:
        while True:
            await asyncio.sleep(interval)
            await ws.send(json.dumps(P.heartbeat(
                inflight=load.inflight, idle_seconds=load.idle_seconds(),
                capacity=cap)))
    except (websockets.ConnectionClosed, asyncio.CancelledError):
        pass


async def _retire(ws, hb, serving: dict, load: Load, tag: str,
                  grace: float = 900.0):
    """A drained leg keeps serving what it already accepted, then closes.

    It stays attached on purpose: the gateway worker behind it must keep
    receiving load reports until the last request lands. Closing at the drain
    instant is what 504s every in-flight request.
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
    hb = asyncio.create_task(_heartbeat(ws, 30.0, load, cap))
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
                await ws.send(json.dumps(P.heartbeat(
                    inflight=load.inflight, idle_seconds=load.idle_seconds(),
                    capacity=cap)))
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
            for task in list(serving.values()):
                task.cancel()
    return drained


async def _leg(i: int, n: int, cfg, cap, load: Load):
    url = cfg["gw"] if n <= 1 else f"{cfg['gw']}/{i}"
    tag = f"gw{i}"
    while True:
        ws = None
        try:
            ws = await websockets.connect(
                url, ping_interval=15, ping_timeout=60,
                close_timeout=10, max_queue=128)
            drained = await _session(ws, cfg, cap, load, tag)
            if drained:
                # _retire owns the socket now. Re-dial immediately so the
                # replacement leg lands on the new colour without a gap.
                print(f"[tee_miner] {tag} draining; re-dialing now", flush=True)
                continue
            await ws.close()
            print(f"[tee_miner] {tag} session ended; reconnecting", flush=True)
        except Exception as e:
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
            print(f"[tee_miner] {tag} disconnected: {e!r}", flush=True)
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


def _worker_count(gw: str) -> int:
    """One leg per gateway worker process, so we receive all buyer traffic; a
    single-dial miner only ever lands on worker 0."""
    n = _env("ENGY_GW_WORKERS")
    if n:
        return max(1, int(n))
    try:
        meta = gw.replace("wss://", "https://").replace("ws://", "http://") + "/meta"
        r = httpx.get(meta, timeout=5.0)
        return max(1, int((r.json() or {}).get("workers") or 1))
    except Exception:
        return 1


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

    cap = {"max_inflight": args.max_inflight}
    if args.context_length:
        cap["context_length"] = args.context_length
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
        "backend": Serve(serve_urls, args.served_model, args.read_timeout),
    }

    n = _worker_count(cfg["gw"])
    # Each leg advertises its SHARE of max_inflight, not the whole thing. One
    # leg per gateway worker, so advertising the full value on every leg tells
    # the gateway this worker can take n times what it can, and the surplus
    # arrives as a burst the engine has no slots for. Ceil, so the legs together
    # never advertise less than the real total.
    cap = _split_capacity(cap, n)
    load = Load()
    print(f"[tee_miner] dialing {n} leg(s) of {cfg['gw']} as {cfg['model']} "
          f"worker={worker_name} worker_id={worker_id}"
          f"{'' if assigned else ' (MINTED, not provider-assigned)'} "
          f"root={cfg['model_root'][:12]} serves={serve_urls} "
          f"leg_inflight={cap['max_inflight']}", flush=True)
    hw = cfg["hw"]
    print(f"[tee_miner] hw: {hw.get('gpus')} | {hw.get('gpu_mem_gb')}GB/gpu | "
          f"{hw.get('cpus')} cpu | {hw.get('ram_gb')}GB ram | "
          f"cc={hw.get('tee', {}).get('gpu_cc', '?')} "
          f"multigpu={hw.get('tee', {}).get('gpu_cc_multigpu', '?')}",
          flush=True)

    async def _run_all():
        await asyncio.gather(*[_leg(i, n, cfg, cap, load) for i in range(n)])

    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
