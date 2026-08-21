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

    def __init__(self, urls: list[str], served_model: str, read_timeout: float):
        self.urls = urls
        self.served_model = served_model
        self.read_timeout = read_timeout
        self._n = {u: 0 for u in urls}
        self._down: set[str] = set()

    def _pick(self) -> str:
        pool = [u for u in self.urls if u not in self._down] or self.urls
        u = min(pool, key=lambda s: self._n.get(s, 0))
        self._n[u] = self._n.get(u, 0) + 1
        return u

    async def healthy(self, timeout: float = 8.0) -> bool:
        """Can at least one serve actually decode? sglang's /health returns
        non-200 when the scheduler/detokenizer is hung and refuses the
        connection when the process died (OOM crash) — the real signal a fake
        heartbeat lacks. Also refreshes the down-set `_pick` routes around."""
        down: set[str] = set()
        async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout, connect=4.0)) as c:
            for u in self.urls:
                try:
                    ok = (await c.get(self._root(u) + "/health")).status_code == 200
                except Exception:
                    ok = False
                if not ok:
                    down.add(u)
        self._down = down
        return len(down) < len(self.urls)

    def _release(self, u: str):
        self._n[u] = max(0, self._n.get(u, 0) - 1)

    @staticmethod
    def _root(url: str) -> str:
        """Base without a trailing /v1: /generate does not live under /v1."""
        return url[:-3].rstrip("/") if url.endswith("/v1") else url.rstrip("/")

    @staticmethod
    def _v1(url: str) -> str:
        return url.rstrip("/") if url.endswith("/v1") else url.rstrip("/") + "/v1"

    def _chat_body(self, request: dict, *, stream: bool) -> dict:
        body = {"model": self.served_model or request.get("model"),
                "messages": request.get("messages", [])}
        if request.get("max_tokens"):
            body["max_tokens"] = request["max_tokens"]
        for k in _PASSTHROUGH:
            if request.get(k) is not None:
                body[k] = request[k]
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


async def _supervise(cfg, capacity: dict, load: Load):
    """Leg supervisor, ported from the reference miner.

    Two jobs beyond keeping N legs dialed:
    - Backend health gate: when the serve OOMs or wedges, DISCONNECT every leg
      so the gateway stops routing buyers here (the alternative is advertising
      a dead backend and 502ing everything). Reconnect only after
      _HEALTH_OK_STREAK consecutive good probes.
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
    backend_down = False
    ok_streak = 0
    try:
        while True:
            ok = await backend.healthy()
            if ok:
                ok_streak += 1
                if backend_down and ok_streak >= _HEALTH_OK_STREAK:
                    backend_down = False
                    print("[tee_miner] backend RECOVERED — reconnecting legs",
                          flush=True)
            else:
                ok_streak = 0
                if not backend_down:
                    backend_down = True
                    print("[tee_miner] backend UNHEALTHY (OOM/unreachable) — "
                          "disconnecting all legs", flush=True)
            if backend_down:
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
        "backend": Serve(serve_urls, args.served_model, args.read_timeout),
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
