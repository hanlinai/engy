"""First-chunk usage: the prompt side (prompt_tokens + cached_tokens) rides
the FIRST chunk frame so the gateway can send exact Msh-Usage-Cache-Write-*
stream headers. Every other chunk is unchanged, and the final usage still
comes from the serve's include_usage summary only."""
import asyncio
import json

from miner import tee_miner as tm

MSGS = {"messages": [{"role": "user", "content": "hi"}]}


def _serve(model="glm-5.2"):
    return tm.Serve(["http://x/v1"], "/model", 60.0, tm.resolve_dialect(model, None))


def _sse(*events):
    return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) \
        + b"data: [DONE]\n\n"


def _cu(pt, ct, cached):
    """A continuous_usage_stats per-chunk usage block (sglang OpenAI wire)."""
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
            "prompt_tokens_details": {"cached_tokens": cached}}


def _run(monkeypatch, sse, headers=None, model="glm-5.2"):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return tm.httpx.Response(200, content=sse, headers={
            "content-type": "text/event-stream", **(headers or {})})

    real = tm.httpx.AsyncClient

    def client(*a, **kw):
        return real(*a, transport=tm.httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(tm.httpx, "AsyncClient", client)

    async def run():
        return [x async for x in _serve(model).stream(dict(MSGS))]

    return seen, asyncio.run(run())


FINAL = {"prompt_tokens": 4550, "completion_tokens": 2, "total_tokens": 4552,
         "prompt_tokens_details": {"cached_tokens": 4096}}


def test_stream_asks_for_continuous_usage_and_keeps_include_usage():
    body = _serve()._chat_body(dict(MSGS), stream=True)
    assert body["stream_options"] == {"include_usage": True,
                                      "continuous_usage_stats": True}
    assert "stream_options" not in _serve()._chat_body(dict(MSGS), stream=False)


def test_only_the_first_delta_carries_the_prompt_side(monkeypatch):
    sse = _sse(
        {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"delta": {"reasoning_content": "th"}}], "usage": _cu(4550, 1, 4096)},
        {"choices": [{"delta": {"content": "ok"}}], "usage": _cu(4550, 2, 4096)},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": FINAL})
    seen, out = _run(monkeypatch, sse)
    assert seen["stream_options"]["continuous_usage_stats"] is True
    deltas = [p for k, p in out if k == "delta"]
    assert deltas[0]["usage"] == {"prompt_tokens": 4550,
                                  "prompt_tokens_details": {"cached_tokens": 4096}}
    assert deltas[0]["delta"] == {"reasoning_content": "th"}
    assert all("usage" not in d for d in deltas[1:])
    kind, done = out[-1]
    assert kind == "done" and done["usage"] == tm._clean_usage(FINAL)


def test_per_chunk_usage_never_becomes_the_final_usage(monkeypatch):
    # a stream cut before the summary keeps the old answer (no usage), not the
    # last continuous snapshot
    sse = _sse({"choices": [{"delta": {"content": "ok"}}], "usage": _cu(10, 1, 0)})
    _, out = _run(monkeypatch, sse)
    assert out[-1][1]["usage"] == {"prompt_tokens": 0, "completion_tokens": 0,
                                   "total_tokens": 0}


def test_k3_wire_usage_headers_feed_the_first_chunk(monkeypatch):
    # Kimi-K3 wire: no per-chunk usage; counts come as response headers
    sse = _sse({"choices": [{"delta": {"content": "ok"}}]},
               {"choices": [{"delta": {"content": "!"}}]},
               {"choices": [{"delta": {}, "finish_reason": "stop",
                             "usage": FINAL}]},
               {"choices": [], "usage": FINAL})
    _, out = _run(monkeypatch, sse, model="kimi-k3.1-pro",
                  headers={"Msh-Usage-Prompt-Tokens": "4550",
                           "Msh-Usage-Cached-Tokens": "0"})
    deltas = [p for k, p in out if k == "delta"]
    assert deltas[0]["usage"] == {"prompt_tokens": 4550,
                                  "prompt_tokens_details": {"cached_tokens": 0}}
    assert "usage" not in deltas[1]
    assert out[-1][1]["usage"] == tm._clean_usage(FINAL)


def test_no_counts_means_no_usage_key(monkeypatch):
    # an older serve / no --enable-cache-report: nothing is guessed
    monkeypatch.setattr(tm, "_CACHED_ZERO_DEFAULT", False)
    sse = _sse({"choices": [{"delta": {"content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1,
                          "prompt_tokens_details": None}},
               {"choices": [], "usage": FINAL})
    _, out = _run(monkeypatch, sse)
    assert "usage" not in out[0][1]


def test_header_helpers():
    assert tm._early_usage_from_headers({"X-Usage-Prompt-Tokens": "7",
                                         "x-usage-cached-tokens": "3"}) == {
        "prompt_tokens": 7, "prompt_tokens_details": {"cached_tokens": 3}}
    assert tm._early_usage_from_headers({"X-Usage-Prompt-Tokens": "7"}) is None
    assert tm._early_usage_from_headers({"content-type": "text/event-stream"}) is None
    assert tm._early_usage(0, 0) is None and tm._early_usage("x", 1) is None


class _WS:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(json.loads(payload))


class _Backend:
    async def stream(self, request):
        yield "delta", {"delta": {"content": "a"}, "logprobs": None,
                        "usage": {"prompt_tokens": 5,
                                  "prompt_tokens_details": {"cached_tokens": 0}}}
        yield "delta", {"delta": {"content": "b"}, "logprobs": None}
        yield "done", {"id": "x", "choices": [], "usage": {}}


def test_serve_one_forwards_usage_on_the_first_chunk_frame_only():
    ws = _WS()
    load = tm.Load()
    load.attach(ws)
    asyncio.run(tm._serve_one(ws, {"corr_id": "c1", "request": dict(MSGS)},
                              _Backend(), load))
    chunks = [f for f in ws.sent if f["type"] == tm.P.CHUNK]
    assert chunks[0]["usage"] == {"prompt_tokens": 5,
                                  "prompt_tokens_details": {"cached_tokens": 0}}
    assert "usage" not in chunks[1]
    assert "usage" not in tm.P.chunk("c", {"content": "x"})


# ENGY_CACHED_ZERO_DEFAULT: sglang 0.5.20 with --enable-cache-report omits
# prompt_tokens_details entirely on a 0 cache hit (b300-tee-17, 2026-11-25).
# On (our boxes): that absence is a 0 read. Off (repo default): unknown, as for
# a serve with no cache accounting at all.

def _omitted(pt, ct):
    """sglang 0.5.20 --enable-cache-report usage on a 0-hit request."""
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}


def test_cached_zero_default_is_off_unless_env_set():
    import importlib
    import os
    saved = os.environ.pop("ENGY_CACHED_ZERO_DEFAULT", None)
    try:
        assert importlib.reload(tm)._CACHED_ZERO_DEFAULT is False
        for v, want in (("1", True), ("0", False), ("", False)):
            os.environ["ENGY_CACHED_ZERO_DEFAULT"] = v
            assert importlib.reload(tm)._CACHED_ZERO_DEFAULT is want
    finally:
        os.environ.pop("ENGY_CACHED_ZERO_DEFAULT", None)
        if saved is not None:
            os.environ["ENGY_CACHED_ZERO_DEFAULT"] = saved
        importlib.reload(tm)


def test_clean_usage_omitted_details(monkeypatch):
    monkeypatch.setattr(tm, "_CACHED_ZERO_DEFAULT", True)
    assert tm._clean_usage(_omitted(4550, 2))["prompt_tokens_details"] == {
        "cached_tokens": 0}
    # details present but without cached_tokens: filled, other keys kept
    assert tm._clean_usage({"prompt_tokens": 10, "prompt_tokens_details": {
        "cache_write_tokens": 4}})["prompt_tokens_details"] == {
        "cached_tokens": 0, "cache_write_tokens": 4}
    # a reported count always wins
    assert tm._clean_usage(FINAL)["prompt_tokens_details"] == {"cached_tokens": 4096}
    # no prompt, nothing to claim
    assert "prompt_tokens_details" not in tm._clean_usage({})
    monkeypatch.setattr(tm, "_CACHED_ZERO_DEFAULT", False)
    assert "prompt_tokens_details" not in tm._clean_usage(_omitted(4550, 2))
    assert tm._clean_usage(FINAL)["prompt_tokens_details"] == {"cached_tokens": 4096}


def test_early_usage_from_chunk_omitted_details(monkeypatch):
    monkeypatch.setattr(tm, "_CACHED_ZERO_DEFAULT", True)
    want = {"prompt_tokens": 4550, "prompt_tokens_details": {"cached_tokens": 0}}
    assert tm._early_usage_from_chunk(_omitted(4550, 1)) == want
    assert tm._early_usage_from_chunk(
        {"prompt_tokens": 4550, "prompt_tokens_details": None}) == want
    assert tm._early_usage_from_chunk(
        {"prompt_tokens": 4550, "prompt_tokens_details": {}}) == want
    assert tm._early_usage_from_chunk(_cu(4550, 1, 4096))["prompt_tokens_details"] \
        == {"cached_tokens": 4096}
    assert tm._early_usage_from_chunk({"prompt_tokens": 0}) is None
    assert tm._early_usage_from_chunk(None) is None
    monkeypatch.setattr(tm, "_CACHED_ZERO_DEFAULT", False)
    assert tm._early_usage_from_chunk(_omitted(4550, 1)) is None
    assert tm._early_usage_from_chunk(
        {"prompt_tokens": 4550, "prompt_tokens_details": {}}) is None


def test_stream_zero_hit_with_omitted_details_on(monkeypatch):
    monkeypatch.setattr(tm, "_CACHED_ZERO_DEFAULT", True)
    sse = _sse({"choices": [{"delta": {"content": "ok"}}], "usage": _omitted(4550, 1)},
               {"choices": [{"delta": {"content": "!"}}], "usage": _omitted(4550, 2)},
               {"choices": [{"delta": {}, "finish_reason": "stop"}]},
               {"choices": [], "usage": _omitted(4550, 2)})
    _, out = _run(monkeypatch, sse)
    deltas = [p for k, p in out if k == "delta"]
    assert deltas[0]["usage"] == {"prompt_tokens": 4550,
                                  "prompt_tokens_details": {"cached_tokens": 0}}
    assert "usage" not in deltas[1]
    assert out[-1][1]["usage"]["prompt_tokens_details"] == {"cached_tokens": 0}


def test_stream_zero_hit_with_omitted_details_off(monkeypatch):
    monkeypatch.setattr(tm, "_CACHED_ZERO_DEFAULT", False)
    sse = _sse({"choices": [{"delta": {"content": "ok"}}], "usage": _omitted(4550, 1)},
               {"choices": [], "usage": _omitted(4550, 2)})
    _, out = _run(monkeypatch, sse)
    assert "usage" not in out[0][1]
    assert "prompt_tokens_details" not in out[-1][1]["usage"]
