"""tee_miner load reporting and drain grace.

Both behaviours are invisible in a single-process test of the miner alone —
they only matter through the gateway's admission path — so these pin the two
properties that path reads: that an in-flight transition reaches EVERY leg
immediately, and that a drained leg outlives any request its backend may still
be working on.
"""
import asyncio
import time
import json
import types
import inspect

import pytest

from miner import tee_miner as tm


class FakeWS:
    """Collects the JSON frames a leg would have sent."""

    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    async def send(self, payload):
        if self.fail:
            raise ConnectionResetError("leg is gone")
        self.sent.append(json.loads(payload))

    def heartbeats(self):
        return [f for f in self.sent if f["type"] == tm.P.HEARTBEAT]


class FakeBackend:
    async def stream(self, request):
        yield "delta", {"delta": {"content": "hi"}, "logprobs": None}
        yield "done", {"id": "chatcmpl-x", "choices": [], "usage": {}}


CHAT = {"corr_id": "c1",
        "request": {"messages": [{"role": "user", "content": "hi"}]}}


def test_broadcast_reaches_every_leg_not_just_the_one_serving():
    """Queued callers are released off a DROPPING report, and a caller can be
    queued on any gateway worker — so every leg has to hear it."""
    legs = [FakeWS(), FakeWS(), FakeWS()]
    load = tm.Load()
    for ws in legs:
        load.attach(ws)

    load.start()
    asyncio.run(load.broadcast())

    for ws in legs:
        assert [h["inflight"] for h in ws.heartbeats()] == [1]


def _serve_and_flush(ws, req, backend, load):
    """Run one serve, then let the background broadcast it scheduled finish
    (reports are kicked off the request path, so asyncio.run would otherwise
    return before they are sent)."""
    async def go():
        await tm._serve_one(ws, req, backend, load)
        while load._bcast_task is not None and not load._bcast_task.done():
            await load._bcast_task
    asyncio.run(go())


def test_serve_one_reports_the_transition_at_both_ends():
    """Acquire AND release. Reporting only the acquire would leave a count that
    never comes back down until the next periodic frame."""
    ws = FakeWS()
    load = tm.Load()
    load.attach(ws)

    _serve_and_flush(ws, CHAT, FakeBackend(), load)

    # Reports are coalesced: a transition that is already over when the send
    # runs is folded into the next one, but the last report is always the truth.
    sent = [h["inflight"] for h in ws.heartbeats()]
    assert sent and sent[-1] == 0
    assert load.inflight == 0


def test_broadcast_omits_capacity_and_kv_so_the_last_good_values_survive():
    """The gateway merges capacity across heartbeats and re-reads KV only when
    a frame carries it, so omitting both preserves them. Sampling KV here would
    also put two GETs per serve on the request path."""
    ws = FakeWS()
    load = tm.Load()
    load.attach(ws)

    asyncio.run(load.broadcast())

    frame = ws.heartbeats()[0]
    assert "capacity" not in frame
    assert "kv" not in frame


def test_a_dead_leg_is_dropped_never_raised():
    """This runs on the request path — it must not fail the request it
    describes, and the live legs must still get the report."""
    dead, live = FakeWS(fail=True), FakeWS()
    load = tm.Load()
    load.attach(dead)
    load.attach(live)

    asyncio.run(load.broadcast())

    assert len(live.heartbeats()) == 1
    assert dead not in load._channels


def test_serve_one_still_reports_the_release_when_the_backend_fails():
    """A slot leaked on the error path shrinks advertised capacity
    permanently: the count only ever drifts one way."""
    class Broken:
        async def stream(self, request):
            raise RuntimeError("serve 500: boom")
            yield  # pragma: no cover - makes this an async generator

    ws = FakeWS()
    load = tm.Load()
    load.attach(ws)

    _serve_and_flush(ws, CHAT, Broken(), load)

    sent = [h["inflight"] for h in ws.heartbeats()]
    assert sent and sent[-1] == 0


def test_drain_grace_outlives_the_upstream_read_timeout():
    """The two knobs have to move together. A leg cut while its backend is
    still allowed to be generating books a 504 against a worker that was
    answering correctly — the same failure the retire path exists to prevent,
    just on a slower path."""
    assert tm.DRAIN_GRACE_S >= float(tm._env("ENGY_READ_TIMEOUT", "1800"))


def test_heartbeat_refreshes_kv_inside_the_gateway_kv_ttl():
    """KV rides only on the periodic heartbeat, and the gateway fails its KV
    gate open on a report older than ENGY_KV_TTL_S (15 s by default). A period
    at or past that TTL leaves a full serve admitted for part of every cycle."""
    gateway_kv_ttl_s = 15.0
    assert tm.HEARTBEAT_S < gateway_kv_ttl_s


def test_retire_defaults_to_the_configured_grace():
    import inspect
    default = inspect.signature(tm._retire).parameters["grace"].default
    assert default == tm.DRAIN_GRACE_S


class ClosedWS:
    """A socket after the fact: only the close code and reason survive."""

    def __init__(self, code=None, reason=""):
        self.close_code = code
        self.close_reason = reason


def test_close_note_names_the_dropped_transport_apart_from_a_clean_close():
    # The whole point: 1006 (transport died, no close frame) and 1000 (peer
    # said goodbye) both surface as websockets.ConnectionClosed, which _session
    # swallows. If the log cannot tell them apart, a leg that took requests
    # down with it is indistinguishable from a normal rotation.
    clean = tm._close_note(ClosedWS(1000))
    dropped = tm._close_note(ClosedWS(1006))
    assert "1000" in clean and "normal" in clean
    assert "1006" in dropped and "transport died" in dropped
    assert clean != dropped


def test_close_note_carries_the_servers_reason_when_there_is_one():
    note = tm._close_note(ClosedWS(1011, "keepalive ping timeout"))
    assert "1011" in note
    assert "keepalive ping timeout" in note


def test_close_note_never_raises_on_a_socket_that_never_opened():
    # _leg's error path logs this for `ws is None` (connect itself failed) and
    # for a socket closed without a code; a logging helper must not be the
    # thing that kills the reconnect loop.
    assert tm._close_note(None) == "never connected"
    assert "unknown" in tm._close_note(ClosedWS(None))


def test_close_note_still_reports_a_code_it_has_no_name_for():
    note = tm._close_note(ClosedWS(4321))
    assert "4321" in note and "unspecified" in note


class FakeInfoClient:
    """Stands in for the httpx client `_max_total` probes /get_server_info with.

    `script` is consumed one entry per call: an Exception is raised (the serve
    is not listening yet), a dict is returned as the JSON body.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def get(self, url):
        self.calls += 1
        item = self.script.pop(0) if self.script else self.script_last
        self.script_last = item
        if isinstance(item, Exception):
            raise item
        return FakeInfoResponse(item)


class FakeInfoResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


OK_INFO = {"max_total_num_tokens": 1436312, "dcp_size": 1}


# named apart from the dialect block's `_serve(model, override)` below: two
# helpers of the same name silently shadow, and the survivor decides which
# Serve every caller gets.
def _kv_serve():
    return tm.Serve(["http://127.0.0.1:30001/v1"], "/model", 60.0)


def test_pool_size_is_cached_after_one_successful_probe():
    """The pool is fixed for the life of a serve, so a success must not be
    re-fetched on every heartbeat."""
    s = _kv_serve()
    c = FakeInfoClient([OK_INFO])
    got = [asyncio.run(s._max_total(c, "http://127.0.0.1:30001")) for _ in range(5)]
    assert got == [1436312] * 5
    assert c.calls == 1


def test_a_failed_probe_is_retried_rather_than_cached_forever():
    """The regression this fixes: the miner is started alongside its serve, so
    the first probe lands while sglang is still loading weights. Caching that
    failure left the worker reporting kv `used`/`reqs` with no `limit` for the
    whole life of the process."""
    s = _kv_serve()
    c = FakeInfoClient([ConnectionRefusedError("still loading"), OK_INFO])
    assert asyncio.run(s._max_total(c, "http://127.0.0.1:30001")) is None
    s._mtt_retry.clear()                      # stand in for the backoff elapsing
    assert asyncio.run(s._max_total(c, "http://127.0.0.1:30001")) == 1436312
    assert c.calls == 2


def test_a_failure_backs_off_instead_of_probing_every_heartbeat():
    """A serve that genuinely has no /get_server_info (older sglang, vLLM) must
    not be re-probed several times a second."""
    s = _kv_serve()
    c = FakeInfoClient([ConnectionRefusedError("no such endpoint")])
    for _ in range(10):
        assert asyncio.run(s._max_total(c, "http://127.0.0.1:30001")) is None
    assert c.calls == 1


def test_an_answer_with_no_usable_pool_size_also_backs_off():
    """A 200 carrying max_total_num_tokens=0 is not a pool size; treat it like a
    failure rather than caching 0 and reporting frac=inf."""
    s = _kv_serve()
    c = FakeInfoClient([{"max_total_num_tokens": 0, "dcp_size": 1}])
    assert asyncio.run(s._max_total(c, "http://127.0.0.1:30001")) is None
    assert s._mtt.get("http://127.0.0.1:30001") is None
    assert "http://127.0.0.1:30001" in s._mtt_retry


def test_dcp_size_multiplies_the_per_rank_pool():
    """sglang reports max_total_num_tokens PER DCP RANK while /get_load's
    num_tokens is global."""
    s = _kv_serve()
    c = FakeInfoClient([{"max_total_num_tokens": 100, "dcp_size": 8}])
    assert asyncio.run(s._max_total(c, "http://127.0.0.1:30001")) == 800


def test_each_serve_is_probed_and_backed_off_independently():
    """A host runs two serves; one being slow must not blind the other."""
    s = tm.Serve(["http://127.0.0.1:30001/v1", "http://127.0.0.1:30002/v1"],
                 "/model", 60.0)
    c = FakeInfoClient([OK_INFO, ConnectionRefusedError("not up")])
    assert asyncio.run(s._max_total(c, "http://127.0.0.1:30001")) == 1436312
    assert asyncio.run(s._max_total(c, "http://127.0.0.1:30002")) is None
    assert "http://127.0.0.1:30001" not in s._mtt_retry
    assert "http://127.0.0.1:30002" in s._mtt_retry


# ---- vLLM: no /get_load, so KV pressure comes from its Prometheus /metrics ----

# The shape prometheus_client renders for a data-parallel (2-engine) vLLM V1
# serve: labels sorted, one sample per engine, the pool size only as labels of
# the cache_config_info gauge.
VLLM_DP2 = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="deepseek-v4.1-flash"} 5.0
vllm:num_requests_running{engine="1",model_name="deepseek-v4.1-flash"} 3.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="deepseek-v4.1-flash"} 2.0
vllm:num_requests_waiting{engine="1",model_name="deepseek-v4.1-flash"} 0.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="deepseek-v4.1-flash",reason="capacity"} 2.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="deepseek-v4.1-flash"} 0.5
vllm:kv_cache_usage_perc{engine="1",model_name="deepseek-v4.1-flash"} 0.25
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",cache_dtype="auto",enable_prefix_caching="True",engine="0",gpu_memory_utilization="0.9",num_cpu_blocks="None",num_gpu_blocks="10000",prefix_caching_hash_algo="sha256"} 1.0
vllm:cache_config_info{block_size="16",cache_dtype="auto",enable_prefix_caching="True",engine="1",gpu_memory_utilization="0.9",num_cpu_blocks="None",num_gpu_blocks="10000",prefix_caching_hash_algo="sha256"} 1.0
"""

# Older vLLM: the pre-V1 gauge name, no engine label, no pool size published.
VLLM_V0 = """\
vllm:gpu_cache_usage_perc{model_name="glm-5.2"} 0.875
vllm:num_requests_running{model_name="glm-5.2"} 7.0
vllm:num_requests_waiting{model_name="glm-5.2"} 1.0
"""

SGLANG_METRICS = """\
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name="glm-5.2"} 0.42
sglang:num_running_reqs{model_name="glm-5.2"} 7.0
"""


def test_vllm_metrics_give_tokens_used_against_the_pool_summed_over_engines():
    kv = tm._vllm_kv_from_metrics(VLLM_DP2)
    # 2 engines x 10000 blocks x 16 tokens; 0.5 and 0.25 of each.
    assert kv == {"used": 80000 + 40000, "limit": 320000, "frac": 0.375,
                  "reqs": 8, "waiting": 2, "ranks": 2}


def test_older_vllm_without_a_pool_size_still_reports_how_full_it_is():
    """The gate reads frac; a serve that cannot size its pool must not be
    reported as empty, which would admit it while it is 87.5% full."""
    kv = tm._vllm_kv_from_metrics(VLLM_V0)
    assert kv["frac"] == 0.875
    assert kv["limit"] == 0 and kv["used"] == 0
    assert kv["reqs"] == 7 and kv["waiting"] == 1 and kv["ranks"] == 1


def test_a_pool_size_of_none_before_profiling_falls_back_to_the_fraction():
    text = VLLM_DP2.replace('num_gpu_blocks="10000"', 'num_gpu_blocks="None"')
    kv = tm._vllm_kv_from_metrics(text)
    assert kv["limit"] == 0
    assert kv["frac"] == 0.375                  # mean of 0.5 and 0.25


def test_metrics_that_are_not_vllm_are_not_mistaken_for_it():
    assert tm._vllm_kv_from_metrics(SGLANG_METRICS) is None
    assert tm._vllm_kv_from_metrics("") is None


def _routes(monkeypatch, routes):
    """Point every httpx client the miner opens at `routes`: {path: (status, body)}."""
    def handler(request):
        status, body = routes.get(request.url.path, (404, "not found"))
        if isinstance(body, str):
            return tm.httpx.Response(status, text=body)
        return tm.httpx.Response(status, json=body)

    real = tm.httpx.AsyncClient

    def client(*a, **kw):
        return real(*a, transport=tm.httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(tm.httpx, "AsyncClient", client)


def test_kv_load_falls_back_to_vllm_metrics_when_there_is_no_get_load(monkeypatch):
    _routes(monkeypatch, {"/metrics": (200, VLLM_DP2)})
    kv = asyncio.run(_kv_serve().kv_load())
    assert kv["frac"] == 0.375 and kv["limit"] == 320000 and kv["used"] == 120000
    assert kv["reqs"] == 8 and kv["waiting"] == 2 and kv["ranks"] == 2


def test_kv_load_reports_a_fraction_only_vllm_serve(monkeypatch):
    _routes(monkeypatch, {"/metrics": (200, VLLM_V0)})
    kv = asyncio.run(_kv_serve().kv_load())
    assert kv["frac"] == 0.875 and "limit" not in kv


def test_kv_load_still_prefers_sglang_get_load(monkeypatch):
    _routes(monkeypatch, {
        "/get_load": (200, [{"num_tokens": 500, "num_reqs": 2,
                             "num_waiting_reqs": 0, "num_pending_tokens": 0}]),
        "/get_server_info": (200, {"max_total_num_tokens": 1000}),
        "/metrics": (200, VLLM_DP2)})
    kv = asyncio.run(_kv_serve().kv_load())
    assert kv["used"] == 500 and kv["limit"] == 1000 and kv["frac"] == 0.5


SGLANG_V1_LOADS = {"loads": [{"dp_rank": 0, "num_running_reqs": 6,
                               "num_waiting_reqs": 4, "num_used_tokens": 10613473,
                               "max_total_num_tokens": 11536384,
                               "token_usage": 0.92}]}


def test_kv_load_falls_back_to_sglang_v1_loads_when_get_load_is_gone(monkeypatch):
    """Newer sglang 404s /get_load and serves /v1/loads; the engine queue
    (waiting) must still reach the gateway, and the pool is not rescaled by
    dcp_size (it is already global on these serves)."""
    _routes(monkeypatch, {"/v1/loads": (200, SGLANG_V1_LOADS),
                          "/get_server_info": (200, {"max_total_num_tokens": 11536384,
                                                     "dcp_size": 8})})
    kv = asyncio.run(_kv_serve().kv_load())
    assert kv["limit"] == 11536384 and kv["frac"] == 0.92
    assert kv["waiting"] == 4 and kv["reqs"] == 6 and kv["ranks"] == 1


def test_sglang_v1_loads_wins_over_vllm_metrics(monkeypatch):
    _routes(monkeypatch, {"/v1/loads": (200, SGLANG_V1_LOADS),
                          "/metrics": (200, VLLM_DP2)})
    kv = asyncio.run(_kv_serve().kv_load())
    assert kv["limit"] == 11536384


def test_kv_load_is_none_for_a_serve_with_neither(monkeypatch):
    _routes(monkeypatch, {"/metrics": (200, SGLANG_METRICS)})
    assert asyncio.run(_kv_serve().kv_load()) is None
# ---------------------------------------------------------- template dialects
# The prod gateway normalizes every buyer `reasoning_effort` onto "high"/"max",
# and Qwen3.8's chat template 400s on both. These pin the translation, and pin
# just as hard that it stays OFF for every other model -- on DeepSeek-V4 the
# same mapping would be silently discarded upstream and answer at the default.


def _serve(model=None, override=None):
    return tm.Serve(["http://x/v1"], "/model", 60.0,
                    tm.resolve_dialect(model, override))


def test_dialect_matches_on_the_engy_model_id_prefix():
    assert tm.resolve_dialect("qwen3.8-27b")["system_first_only"] is True
    assert tm.resolve_dialect("deepseek-v4-flash-0731") is tm._DEFAULT_DIALECT
    assert tm.resolve_dialect("glm-5.2") is tm._DEFAULT_DIALECT
    assert tm.resolve_dialect(None) is tm._DEFAULT_DIALECT


def test_unknown_dialect_override_falls_back_instead_of_raising():
    # A miner that refuses to start is worse than one that forwards unmodified.
    assert tm.resolve_dialect("glm-5.2", "nope") is tm._DEFAULT_DIALECT
    assert tm.resolve_dialect("glm-5.2", "qwen3.8")["system_first_only"] is True


def test_qwen_effort_folds_onto_xhigh_and_is_idempotent():
    s = _serve("qwen3.8-27b")
    msgs = [{"role": "user", "content": "hi"}]
    for sent, want in [("high", "xhigh"), ("max", "xhigh"), ("xhigh", "xhigh"),
                       ("medium", "medium"), ("low", "low")]:
        body = s._chat_body({"messages": msgs, "reasoning_effort": sent},
                            stream=False)
        assert body["reasoning_effort"] == want, sent
    # No effort on the wire must not invent one.
    assert "reasoning_effort" not in s._chat_body({"messages": msgs},
                                                  stream=False)


def test_other_models_keep_the_effort_the_buyer_asked_for():
    msgs = [{"role": "user", "content": "hi"}]
    for model in ("deepseek-v4-flash-0731", "glm-5.2", None):
        s = _serve(model)
        for eff in ("high", "max", "low"):
            body = s._chat_body({"messages": msgs, "reasoning_effort": eff},
                                stream=False)
            assert body["reasoning_effort"] == eff, (model, eff)


def test_qwen_demotes_only_mid_conversation_system_turns():
    msgs = [{"role": "system", "content": "S0"},
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "S-mid"}]
    body = _serve("qwen3.8-27b")._chat_body({"messages": msgs}, stream=False)
    assert body["messages"][0]["role"] == "system"     # position 0 survives
    assert body["messages"][2]["role"] == "user"       # the mid one is demoted
    assert body["messages"][2]["content"] == "S-mid"   # content is untouched
    assert msgs[2]["role"] == "system"                 # caller's list unmutated


def test_demotion_is_off_for_templates_that_accept_mid_system():
    msgs = [{"role": "system", "content": "S0"},
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "S-mid"}]
    body = _serve("deepseek-v4-flash-0731")._chat_body({"messages": msgs},
                                                       stream=False)
    assert body["messages"][2]["role"] == "system"


def test_dialect_does_not_disturb_the_rest_of_the_body():
    s = _serve("qwen3.8-27b")
    body = s._chat_body({"messages": [{"role": "user", "content": "hi"}],
                         "max_tokens": 99, "temperature": 0.3, "tools": [1]},
                        stream=True)
    assert body["model"] == "/model"
    assert body["max_tokens"] == 99 and body["temperature"] == 0.3
    assert body["tools"] == [1]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True,
                                      "continuous_usage_stats": True}


# ---- prefix continuation: the keys the gateway's prefill.py sets ----

PREFILL = {"messages": [{"role": "user", "content": "2+3? answer in JSON"},
                        {"role": "assistant", "content": '{"answer": '}],
           "continue_final_message": True, "add_generation_prompt": False,
           "chat_template_kwargs": {"thinking": False},
           "reasoning_effort": "none"}


def test_chat_lane_forwards_the_prefill_keys():
    # Dropped here, sglang renders the prefix as a closed turn and opens a new
    # one: kimi-k3 re-answered '{"answer": 5}' instead of continuing with '5}'.
    for model in ("kimi-k3", "qwen3.8-27b", None):
        body = _serve(model)._chat_body(PREFILL, stream=True)
        assert body["continue_final_message"] is True, model
        # False is a value, not an absence: vLLM needs it on the wire.
        assert body["add_generation_prompt"] is False, model
        assert body["messages"][-1] == PREFILL["messages"][-1], model


def test_prefill_keys_are_not_invented_for_an_ordinary_request():
    body = _serve("kimi-k3")._chat_body(
        {"messages": [{"role": "user", "content": "hi"}]}, stream=True)
    assert "continue_final_message" not in body
    assert "add_generation_prompt" not in body


def test_text_lane_never_carries_the_prefill_keys():
    # A raw prompt has no trailing assistant turn to continue.
    body = _serve("kimi-k3")._text_body({"prompt": "abc", **PREFILL})
    assert "continue_final_message" not in body
    assert "add_generation_prompt" not in body


def test_prefill_keys_reach_the_serve_on_the_wire(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        sse = (b'data: {"choices":[{"delta":{"content":"5}"},'
               b'"finish_reason":"stop"}]}\n\n'
               b'data: {"choices":[],"usage":{"prompt_tokens":42,'
               b'"completion_tokens":3,"total_tokens":45}}\n\n'
               b'data: [DONE]\n\n')
        return tm.httpx.Response(200, content=sse,
                                 headers={"content-type": "text/event-stream"})

    real = tm.httpx.AsyncClient

    def client(*a, **kw):
        return real(*a, transport=tm.httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(tm.httpx, "AsyncClient", client)

    async def run():
        return [x async for x in _serve("kimi-k3").stream(dict(PREFILL))]

    out = asyncio.run(run())
    assert seen["continue_final_message"] is True
    assert seen["add_generation_prompt"] is False
    assert seen["messages"][-1] == {"role": "assistant", "content": '{"answer": '}
    kind, done = out[-1]
    assert kind == "done"
    assert done["choices"][0]["message"]["content"] == "5}"
# ---- usage: the thinking-token count the gateway translates for buyers ----

# verbatim from a prod qwen3.8-27b serve (m5-27:30001), reasoning_effort=medium
SGLANG_USAGE = {"prompt_tokens": 22, "total_tokens": 89, "completion_tokens": 67,
                "prompt_tokens_details": None, "reasoning_tokens": 53}


def test_clean_usage_keeps_the_flat_count_sglang_actually_sends():
    """Dropping it here is what made thinking invisible to buyers: the count
    exists upstream, and the miner's allowlist threw it away before the
    gateway could translate it."""
    assert tm._clean_usage(SGLANG_USAGE)["reasoning_tokens"] == 53


def test_clean_usage_also_accepts_the_openai_nested_spelling():
    u = tm._clean_usage({"prompt_tokens": 1, "completion_tokens": 9,
                         "completion_tokens_details": {"reasoning_tokens": 4}})
    assert u["reasoning_tokens"] == 4


def test_clean_usage_omits_it_when_the_serve_never_counted_it():
    """A serve with no reasoning parser. "Unknown" must not become a zero."""
    assert "reasoning_tokens" not in tm._clean_usage({"prompt_tokens": 5,
                                                      "completion_tokens": 7})


def test_clean_usage_keeps_a_genuine_zero():
    u = tm._clean_usage({"prompt_tokens": 5, "completion_tokens": 7,
                         "reasoning_tokens": 0})
    assert u["reasoning_tokens"] == 0


def test_clean_usage_still_drops_upstream_extras():
    """The allowlist exists so provider cost/byok fields never reach billing."""
    u = tm._clean_usage({**SGLANG_USAGE, "cost": 0.12, "provider": "openrouter"})
    assert set(u) == {"prompt_tokens", "completion_tokens", "total_tokens",
                      "reasoning_tokens"}


def test_clean_usage_leaves_the_billed_numbers_alone():
    """Thinking tokens are a subset of completion_tokens — already billed."""
    u = tm._clean_usage(SGLANG_USAGE)
    assert (u["prompt_tokens"], u["completion_tokens"], u["total_tokens"]) == (22, 67, 89)


# --- the backend health gate ------------------------------------------------
#
# The gate decides whether a /health stall means the serve is unusable. Getting
# that wrong is expensive in one direction only: withdrawing cancels every leg
# and the gateway books each in-flight request as a 504, so a stall must be
# CONFIRMED against the scheduler (/get_load) before it costs anyone a request.


class FakeServe:
    """Just the two calls the gate makes on a backend."""

    def __init__(self, scheduler=True):
        self.scheduler = scheduler
        self.load_probes = 0

    async def scheduler_alive(self, timeout: float = 3.0) -> bool:
        self.load_probes += 1
        return self.scheduler


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def tick(self, dt):
        self.t += dt
        return self.t


def drive(gate, states, clock=None, dt=13.0):
    """Feed probe outcomes at the real ~13 s cadence (8 s timeout + 5 s sleep)."""
    out = []
    for s in states:
        out.append(asyncio.run(gate.update(s)))
        if clock:
            clock.tick(dt)
    return out


def test_health_stall_does_not_withdraw_while_the_scheduler_answers():
    """The Ornith canary bug: two /health timeouts used to cancel every leg.
    /get_load answering means the process is scheduling — a busy backend, not a
    dead one — so the legs must stay up and the in-flight work must survive."""
    serve = FakeServe(scheduler=True)
    clock = Clock()
    gate = tm._HealthGate(serve, clock=clock)
    assert drive(gate, ["timeout"] * 6, clock) == [False] * 6
    assert gate.down is False


def test_a_single_missed_probe_never_even_asks_the_scheduler():
    """/get_load is consulted on the timeout path only, and only once the
    streak is long enough to matter — one missed probe stays free."""
    serve = FakeServe(scheduler=True)
    gate = tm._HealthGate(serve, clock=Clock())
    drive(gate, ["timeout", "ok", "ok", "timeout", "ok"])
    assert serve.load_probes == 0


def test_a_stall_the_scheduler_cannot_confirm_still_withdraws():
    """Both channels silent is the unambiguous case the gate exists for."""
    serve = FakeServe(scheduler=False)
    clock = Clock()
    gate = tm._HealthGate(serve, clock=clock)
    assert drive(gate, ["timeout", "timeout"], clock) == [False, True]
    assert serve.load_probes == 1      # asked once, at the streak, not before


def test_dead_withdraws_on_the_first_probe_without_asking():
    """Connection refused / non-200 needs no second opinion."""
    serve = FakeServe(scheduler=True)
    gate = tm._HealthGate(serve, clock=Clock())
    assert drive(gate, ["dead"]) == [True]
    assert serve.load_probes == 0


def test_a_stall_that_outlasts_the_backstop_withdraws_anyway():
    """A wedged serve (the O(N^2) detokenizer hang) keeps answering /get_load
    while serving nobody, so a stall that simply never ends must still trip."""
    serve = FakeServe(scheduler=True)
    clock = Clock()
    gate = tm._HealthGate(serve, clock=clock)
    down = drive(gate, ["timeout"] * 12, clock)
    assert down[0] is False and True in down
    tripped_at = down.index(True) * 13.0
    assert tripped_at >= tm._HEALTH_STALL_MAX_S
    assert tripped_at < tm._HEALTH_STALL_MAX_S + 13.0   # and not much later


def test_the_stall_clock_restarts_after_a_good_probe():
    """Two separate stalls either side of a healthy probe are not one long one,
    so they must not add up into a withdrawal."""
    serve = FakeServe(scheduler=True)
    clock = Clock()
    gate = tm._HealthGate(serve, clock=clock)
    drive(gate, ["timeout"] * 8, clock)
    drive(gate, ["ok"], clock)
    assert drive(gate, ["timeout"] * 8, clock) == [False] * 8


def test_recovery_still_needs_a_streak_of_good_probes():
    serve = FakeServe(scheduler=False)
    gate = tm._HealthGate(serve, clock=Clock())
    drive(gate, ["timeout", "timeout"])
    assert gate.down is True
    assert drive(gate, ["ok"]) == [True]                # one is not enough
    assert drive(gate, ["ok"]) == [False]


def test_a_withdrawn_gate_stops_re_probing_the_scheduler():
    """Once the legs are down there is nothing left to protect, so the extra
    GET stops until /health comes back."""
    serve = FakeServe(scheduler=False)
    gate = tm._HealthGate(serve, clock=Clock())
    drive(gate, ["timeout"] * 6)
    assert serve.load_probes == 1


def test_the_log_tells_a_ridden_out_stall_apart_from_a_withdrawal(capsys):
    """Whoever reads miner.log next has to be able to see which happened."""
    clock = Clock()
    gate = tm._HealthGate(FakeServe(scheduler=True), clock=clock)
    drive(gate, ["timeout", "timeout"], clock)
    kept = capsys.readouterr().out
    assert "staying up" in kept and "UNHEALTHY" not in kept

    gate = tm._HealthGate(FakeServe(scheduler=False), clock=Clock())
    drive(gate, ["timeout", "timeout"])
    gone = capsys.readouterr().out
    assert "UNHEALTHY" in gone and "/get_load silent" in gone


def test_the_ridden_out_stall_is_logged_at_most_once_a_note_interval(capsys):
    """A long stall must stay visible in the log without flooding it."""
    clock = Clock()
    gate = tm._HealthGate(FakeServe(scheduler=True), clock=clock)
    drive(gate, ["timeout"] * 11, clock, dt=1.0)     # 11 s of stall, 1 s apart
    notes = [ln for ln in capsys.readouterr().out.splitlines()
             if "staying up" in ln]
    assert len(notes) == 1

    clock = Clock()
    gate = tm._HealthGate(FakeServe(scheduler=True), clock=clock)
    drive(gate, ["timeout"] * 8, clock, dt=tm._HEALTH_NOTE_S + 1)
    notes = [ln for ln in capsys.readouterr().out.splitlines()
             if "staying up" in ln]
    assert 2 <= len(notes) <= 8      # paced, not silenced


SECRET = "CONFIDENTIAL-BUYER-PARAGRAPH-7731"  # no quotes: repr() would escape them


def test_a_lone_surrogate_no_longer_fails_the_request():
    """httpx's `json=` raised UnicodeEncodeError on half an emoji, so the buyer
    got an error. Now the half becomes U+FFFD and a whole emoji is untouched."""
    body = {"messages": [{"role": "user", "content": "hi \ud83d and \U0001f600"}]}
    out = json.loads(tm._json_body(body).decode("utf-8"))
    assert out["messages"][0]["content"] == "hi � and \U0001f600"


def test_a_clean_body_is_sent_byte_for_byte_as_utf8():
    body = {"messages": [{"role": "user", "content": "héllo 世界"}], "max_tokens": 8}
    assert json.loads(tm._json_body(body)) == body


def test_a_unicode_error_is_logged_without_the_request():
    """repr() of a UnicodeEncodeError carries the whole string it failed on,
    which was the buyer's JSON body: tens of KB of prompt in the miner log."""
    payload = json.dumps({"messages": [{"role": "user",
                                        "content": SECRET * 1000 + "\ud83d"}]},
                         ensure_ascii=False)
    try:
        payload.encode("utf-8")
    except UnicodeEncodeError as e:
        err = e
    line = tm._log_safe_error(err)
    assert SECRET not in line and '"role"' not in line
    assert line.startswith("UnicodeEncodeError") and len(line) < 200


def test_an_engine_validation_error_is_logged_without_the_echoed_input():
    e = tm.ServeError(422, '{"detail":[{"type":"string_type","loc":["body",'
                           '"messages",0,"content"],"msg":"Input should be a '
                           'valid string","input":"' + SECRET + '"}]}')
    line = tm._log_safe_error(e)
    assert SECRET not in line and "status=422" in line


def test_serve_one_never_prints_the_request_on_failure(capsys):
    class Unencodable:
        async def stream(self, request):
            json.dumps(request, ensure_ascii=False).encode("utf-8")
            yield  # pragma: no cover

    frame = {"corr_id": "c1", "request": {"messages": [
        {"role": "user", "content": SECRET + "\ud83d"}]}}
    asyncio.run(tm._serve_one(FakeWS(), frame, Unencodable(), tm.Load()))
    out = capsys.readouterr().out
    assert "serve error: UnicodeEncodeError" in out
    assert SECRET not in out


# ------------------------------------------------------------- gateway doors
LB, CF = "wss://lb.example/gw", "wss://cf.example/gw"


class DoorClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_each_door_carries_an_equal_share_of_the_legs():
    homes = tm.Doors.parse(f"{LB},{CF}", seed="w1").homes(8)
    assert sorted(homes) == [CF] * 4 + [LB] * 4


def test_a_door_listed_twice_carries_twice_the_share():
    homes = tm.Doors.parse(f"{LB},{LB},{LB},{CF}", seed="w1").homes(8)
    assert homes.count(LB) == 6 and homes.count(CF) == 2


def test_the_shuffle_is_stable_for_a_worker_and_differs_between_workers():
    """Stable, so a restart puts the legs back on the same doors. Different, so
    across the fleet every gateway process gets legs through every door,
    instead of leg 0 always arriving through the first one listed."""
    spec = f"{LB},{CF}"
    assert tm.Doors.parse(spec, "w1").homes(8) == tm.Doors.parse(spec, "w1").homes(8)
    plans = [tm.Doors.parse(spec, f"w{k}").homes(8) for k in range(40)]
    for leg in range(8):
        assert {plan[leg] for plan in plans} == {LB, CF}


def test_one_door_keeps_the_old_single_url_behaviour():
    doors = tm.Doors.parse(LB + "/")
    assert doors.homes(8) == [LB] * 8
    assert doors.pick(LB) == LB


def test_two_failed_opens_bench_a_door_and_its_legs_move_over():
    clock = DoorClock()
    doors = tm.Doors.parse(f"{LB},{CF}", clock=clock)
    doors.failed(LB)
    assert doors.pick(LB) == LB          # one miss is not enough
    doors.failed(LB)
    assert doors.pick(LB) == CF          # benched: the leg dials the other door
    assert doors.pick(CF) == CF          # legs already home on CF are unmoved
    clock.t += tm._DOOR_BENCH_S
    assert doors.pick(LB) == LB          # bench over: back home on the next dial


def test_an_open_clears_the_strikes_against_a_door():
    doors = tm.Doors.parse(f"{LB},{CF}", clock=DoorClock())
    doors.failed(LB)
    doors.opened(LB)
    doors.failed(LB)
    assert doors.pick(LB) == LB


def test_a_door_that_stays_down_is_benched_longer_each_time():
    clock = DoorClock()
    doors = tm.Doors.parse(f"{LB},{CF}", clock=clock)
    benches = []
    for _ in range(7):
        doors.failed(LB)
        doors.failed(LB)
        benches.append(doors._until[LB] - clock.t)
        clock.t = doors._until[LB]
    assert benches[:3] == [tm._DOOR_BENCH_S, 2 * tm._DOOR_BENCH_S,
                           4 * tm._DOOR_BENCH_S]
    assert max(benches) == tm._DOOR_BENCH_MAX_S


def test_with_every_door_benched_the_leg_dials_the_one_back_soonest():
    clock = DoorClock()
    doors = tm.Doors.parse(f"{LB},{CF}", clock=clock)
    doors.failed(CF), doors.failed(CF)
    clock.t += 10
    doors.failed(LB), doors.failed(LB)
    assert doors.pick(LB) == CF


def test_meta_asks_open_doors_first():
    clock = DoorClock()
    doors = tm.Doors.parse(f"{LB},{CF}", clock=clock)
    assert doors.by_preference() == [LB, CF]
    doors.failed(LB), doors.failed(LB)
    assert doors.by_preference() == [CF, LB]


def test_worker_count_falls_through_to_a_door_that_answers(monkeypatch):
    asked = []

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"workers": 8}

    def get(url, timeout):
        asked.append(url)
        if "lb.example" in url:
            raise OSError("no route")
        return R()

    monkeypatch.delenv("ENGY_GW_WORKERS", raising=False)
    monkeypatch.setattr(tm.httpx, "get", get)
    assert tm._fetch_worker_count([LB, CF]) == 8
    assert asked == ["https://lb.example/gw/meta", "https://cf.example/gw/meta"]
    assert tm._fetch_worker_count(LB) is None       # one door, unreachable


def _run_leg(monkeypatch, connect, dials=6, spec=f"{LB},{CF}"):
    """Drive one leg through `dials` dial attempts and return the URLs it used."""
    urls = []

    async def fake_connect(url, **kw):
        urls.append(url)
        if len(urls) > dials:
            raise asyncio.CancelledError
        return await connect(url)

    async def no_sleep(_):
        pass

    async def session(ws, cfg, cap, load, tag):
        raise ConnectionResetError("cut by the edge")

    monkeypatch.setattr(tm.websockets, "connect", fake_connect)
    monkeypatch.setattr(tm.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(tm, "_session", session)
    doors = tm.Doors.parse(spec, seed="w1")
    leg = doors.homes(8).index(LB)
    cfg = {"doors": doors}
    try:
        asyncio.run(tm._leg(leg, 8, cfg, {}, tm.Load()))
    except asyncio.CancelledError:
        pass
    return [u.rsplit("/", 1)[0] for u in urls[:dials]], leg


class OpenWS:
    close_code = 1006
    close_reason = ""

    async def close(self):
        pass


def test_a_leg_whose_door_will_not_open_moves_to_the_other_door(monkeypatch):
    async def connect(url):
        if "lb.example" in url:
            raise OSError("timed out during opening handshake")
        return OpenWS()

    doors, _ = _run_leg(monkeypatch, connect)
    assert doors[:2] == [LB, LB]
    assert set(doors[2:]) == {CF}


def test_sessions_the_edge_cuts_do_not_move_a_leg_off_its_door(monkeypatch):
    async def connect(url):
        return OpenWS()

    doors, leg = _run_leg(monkeypatch, connect)
    assert doors == [LB] * 6


# ------------------------------------------------------------- default doors
def test_the_default_splits_legs_half_lb_half_cloudflare_shuffled_per_worker():
    lb, cf = "wss://lb.engy.ai/gw", "wss://api.engy.ai/gw"
    homes = {w: tm.Doors.parse(tm.DEFAULT_GW, seed=w).homes(8) for w in ("w1", "w2", "w3", "w4")}
    for h in homes.values():
        assert h.count(lb) == 4 and h.count(cf) == 4
    assert len({tuple(h) for h in homes.values()}) > 1           # order differs across workers
    assert tm.Doors.parse(tm.DEFAULT_GW, seed="w1").homes(8) == homes["w1"]  # stable per worker
    assert tm.Doors.parse(tm.DEFAULT_GW, seed="w1").backups == []


def test_the_default_moves_a_dead_doors_legs_to_the_other_door():
    lb, cf = "wss://lb.engy.ai/gw", "wss://api.engy.ai/gw"
    doors = tm.Doors.parse(tm.DEFAULT_GW, seed="w1", clock=DoorClock())
    doors.failed(lb), doors.failed(lb)
    assert doors.pick(lb) == cf and doors.pick(cf) == cf


# ------------------------------------------------------------- backup doors


def test_a_backup_carries_no_legs_while_the_primary_opens():
    doors = tm.Doors.parse(f"{LB}|{CF}", seed="w1", clock=DoorClock())
    assert doors.homes(8) == [LB] * 8
    assert doors.pick(LB) == LB


def test_legs_use_the_backup_only_while_the_primary_is_benched():
    clock = DoorClock()
    doors = tm.Doors.parse(f"{LB}|{CF}", clock=clock)
    doors.failed(LB), doors.failed(LB)
    assert doors.pick(LB) == CF
    clock.t += tm._DOOR_BENCH_S
    assert doors.pick(LB) == LB          # back on the LB at the next dial


def test_a_second_primary_is_tried_before_the_backup():
    other = "wss://lb2.example/gw"
    doors = tm.Doors.parse(f"{LB},{other}|{CF}", clock=DoorClock())
    doors.failed(LB), doors.failed(LB)
    assert doors.pick(LB) == other
    doors.failed(other), doors.failed(other)
    assert doors.pick(LB) == CF


def test_failures_landing_after_the_bench_do_not_lengthen_it():
    """Eight legs dial at once; the two that bench the door are followed by six
    more failures from the same outage. Counting them would double the bench
    three times over before a single retry."""
    clock = DoorClock()
    doors = tm.Doors.parse(f"{LB}|{CF}", clock=clock)
    for _ in range(8):
        doors.failed(LB)
    assert doors._until[LB] - clock.t == tm._DOOR_BENCH_S


def test_meta_asks_the_primary_first_and_the_backup_while_it_is_benched():
    doors = tm.Doors.parse(f"{LB}|{CF}", clock=DoorClock())
    assert doors.by_preference() == [LB, CF]
    doors.failed(LB), doors.failed(LB)
    assert doors.by_preference() == [CF, LB]


def test_a_backup_with_no_primary_is_just_the_door():
    doors = tm.Doors.parse(f"|{CF}")
    assert doors.homes(8) == [CF] * 8 and doors.backups == []


def test_a_leg_falls_back_to_the_backup_when_the_lb_will_not_open(monkeypatch):
    async def connect(url):
        if "lb.example" in url:
            raise OSError("cannot resolve")
        return OpenWS()

    doors, _ = _run_leg(monkeypatch, connect, spec=f"{LB}|{CF}")
    assert doors[:2] == [LB, LB]
    assert set(doors[2:]) == {CF}


# ---- engine prefill backlog (2026-11-19) ----

def _v1(unc, tok, busy_us, ts):
    return {"loads": [{**SGLANG_V1_LOADS["loads"][0],
                       "num_waiting_uncached_tokens": unc,
                       "total_prefill_uncached_tokens": tok,
                       "total_prefill_busy_us": busy_us, "timestamp": ts}]}


def test_v1_loads_carries_backlog_and_measured_rate(monkeypatch):
    seq = [_v1(15_757, 100_000, 1_000_000, 1.0), _v1(0, 200_000, 7_000_000, 31.0)]

    def handler(request):
        if request.url.path == "/v1/loads":
            return tm.httpx.Response(200, json=seq.pop(0))
        return tm.httpx.Response(404, text="nf")
    real = tm.httpx.AsyncClient
    monkeypatch.setattr(tm.httpx, "AsyncClient",
                        lambda *a, **kw: real(*a, transport=tm.httpx.MockTransport(handler), **kw))
    s = _kv_serve()
    kv1 = asyncio.run(s.kv_load())
    kv2 = asyncio.run(s.kv_load())
    assert kv1["uncached"] == 15_757 and "pf_tps" not in kv1
    assert kv2["uncached"] == 0 and kv2["pf_tps"] == 16_666 and kv2["pf_busy"] == 0.2
    assert kv2["frac"] == 0.92 and kv2["pf_ranks"] == 1


def _dp(n, tok, busy_s, ts, busy_ranks=None):
    busy_ranks = range(n) if busy_ranks is None else busy_ranks
    return [{"dp_rank": r, "num_waiting_uncached_tokens": 0,
             "total_prefill_uncached_tokens": tok if r in busy_ranks else 0,
             "total_prefill_busy_us": int(busy_s * 1e6) if r in busy_ranks else 0,
             "timestamp": ts} for r in range(n)]


def test_dp_ranks_prefilling_together_sum_their_rates():
    # dp8, every rank prefills 10k tok in the same 1 s: the serve did 80k tok/s.
    # Each rank's busy counter adds that same second, so Σtok/Σbusy says 10k.
    _, st = tm.prefill_backlog_fields(_dp(8, 0, 0, 100.0), None)
    f, _ = tm.prefill_backlog_fields(_dp(8, 10_000, 1.0, 101.0), st)
    assert f["pf_tps"] == 80_000 and f["pf_busy"] == 1.0 and f["pf_ranks"] == 8


def test_dp_one_busy_rank_reports_its_rate_and_share():
    _, st = tm.prefill_backlog_fields(_dp(8, 0, 0, 100.0), None)
    f, _ = tm.prefill_backlog_fields(_dp(8, 2_500, 0.5, 102.0, busy_ranks={3}), st)
    assert f["pf_tps"] == 5_000 and f["pf_busy"] == 0.25


def test_dp_rank_keys_survive_reordering():
    _, st = tm.prefill_backlog_fields(_dp(2, 0, 0, 0.0), None)
    f, _ = tm.prefill_backlog_fields(
        list(reversed(_dp(2, 6_000, 1.0, 1.0, busy_ranks={0}))), st)
    assert f["pf_tps"] == 6_000


def test_v1_loads_dp_serve_reports_summed_rate_and_ranks(monkeypatch):
    base = SGLANG_V1_LOADS["loads"][0]
    seq = [{"loads": [{**base, **x} for x in _dp(2, 0, 0, 1.0)]},
           {"loads": [{**base, **x} for x in _dp(2, 17_000, 1.0, 31.0)]}]

    def handler(request):
        if request.url.path == "/v1/loads":
            return tm.httpx.Response(200, json=seq.pop(0))
        return tm.httpx.Response(404, text="nf")
    real = tm.httpx.AsyncClient
    monkeypatch.setattr(tm.httpx, "AsyncClient",
                        lambda *a, **kw: real(*a, transport=tm.httpx.MockTransport(handler), **kw))
    s = _kv_serve()
    asyncio.run(s.kv_load())
    kv = asyncio.run(s.kv_load())
    assert kv["pf_tps"] == 34_000 and kv["pf_ranks"] == 2


def test_older_v1_loads_has_no_backlog_keys(monkeypatch):
    _routes(monkeypatch, {"/v1/loads": (200, SGLANG_V1_LOADS)})
    kv = asyncio.run(_kv_serve().kv_load())
    assert "uncached" not in kv and "pf_tps" not in kv


def test_prefill_fields_rate_is_busy_time_based_and_resets_safely():
    f, st = tm.prefill_backlog_fields([{"num_waiting_uncached_tokens": 5,
                                        "total_prefill_uncached_tokens": 0,
                                        "total_prefill_busy_us": 0,
                                        "timestamp": 0.0}], None)
    assert f == {"uncached": 5, "pf_ranks": 1}
    f, st = tm.prefill_backlog_fields([{"total_prefill_uncached_tokens": 17_000,
                                        "total_prefill_busy_us": 1_000_000,
                                        "timestamp": 10.0}], st)
    assert f["pf_tps"] == 17_000 and f["pf_busy"] == 0.1
    f, st = tm.prefill_backlog_fields([{"total_prefill_uncached_tokens": 10,
                                        "total_prefill_busy_us": 10,
                                        "timestamp": 20.0}], st)
    assert "pf_busy" not in f and f["pf_tps"] == 17_000   # restart: no bogus rate


def test_backlog_push_decision():
    assert tm.backlog_moved(None, 0) and not tm.backlog_moved(1, None)
    assert tm.backlog_moved(0, 1) and tm.backlog_moved(1, 0)
    assert not tm.backlog_moved(10_000, 15_000, min_tokens=8192)
    assert tm.backlog_moved(10_000, 30_000, min_tokens=8192)


def test_backlog_watch_pushes_to_every_leg(monkeypatch):
    class WS:
        def __init__(self):
            self.sent = []

        async def send(self, m):
            self.sent.append(json.loads(m))

    class B:
        n = 0

        async def kv_load(self):
            B.n += 1
            return {"used": 1, "uncached": 0 if B.n == 1 else 300_000}

    async def go():
        load = tm.Load()
        a, b = WS(), WS()
        load.attach(a), load.attach(b)
        t = asyncio.create_task(tm._backlog_watch(load, B(), interval=0.01))
        for _ in range(100):
            if len(a.sent) >= 2:
                break
            await asyncio.sleep(0.01)
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
        return a, b
    a, b = asyncio.run(go())
    assert [f["kv"]["uncached"] for f in a.sent[:2]] == [0, 300_000]
    assert len(b.sent) >= 2


# ------------------------------------------------------- name-only identity
class _AdmitWS:
    def __init__(self, *frames):
        self.sent = []
        self._frames = [json.dumps(f) for f in frames]

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._frames:
            raise StopAsyncIteration
        return self._frames.pop(0)


class _NoBackend:
    async def kv_load(self):
        return None


def _name_cfg(**kw):
    return {"miner_key": "k", "model": "m", "model_root": "r", "hw": {},
            "worker_name": "w1", "backend": _NoBackend(), "hb_s": 1000, **kw}


def _run_session(ws, cfg):
    return asyncio.run(tm._session(ws, cfg, {"max_inflight": 4}, tm.Load(), "gw0"))


def test_hello_carries_a_provided_worker_id():
    assert tm.P.hello("k", "m", "r", worker_name="w1",
                      worker_id="w-9")["worker_id"] == "w-9"


def test_hello_without_worker_id_has_only_the_name():
    hello = tm.P.hello("k", "m", "r", worker_name="w1")
    assert "worker_id" not in hello and hello["worker_name"] == "w1"


def test_session_hello_carries_cfg_worker_id():
    ws = _AdmitWS({"type": tm.P.ADMIT, "worker_id": "w-1"})
    _run_session(ws, _name_cfg(worker_id="w-9"))
    assert ws.sent[0]["worker_id"] == "w-9" and ws.sent[0]["worker_name"] == "w1"


def test_session_hello_without_cfg_worker_id_omits_it():
    ws = _AdmitWS({"type": tm.P.ADMIT})
    _run_session(ws, _name_cfg())
    assert "worker_id" not in ws.sent[0]


def test_admit_worker_id_is_printed_once_per_process(monkeypatch, capsys):
    monkeypatch.setattr(tm, "_worker_id_announced", False)
    for _ in range(3):
        _run_session(_AdmitWS({"type": tm.P.ADMIT, "worker_id": "w-1"}), _name_cfg())
    out = capsys.readouterr().out
    assert out.count("worker_id=w-1 worker_name=w1") == 1


def test_admit_without_worker_id_prints_unknown(monkeypatch, capsys):
    monkeypatch.setattr(tm, "_worker_id_announced", False)
    _run_session(_AdmitWS({"type": tm.P.ADMIT}), _name_cfg())
    assert "worker_id=unknown worker_name=w1" in capsys.readouterr().out


def _main_cfg(monkeypatch, argv):
    seen = []

    async def stop(cfg, cap, load):
        seen.append(cfg)
    monkeypatch.setattr(tm, "_supervise", stop)
    monkeypatch.setattr(tm, "_detect_hw", lambda: {})
    monkeypatch.setenv("ENGY_MINER_KEY", "k")
    monkeypatch.setenv("ENGY_WORKER_NAME", "w1")
    tm.main(argv)
    return seen[0]


@pytest.mark.parametrize("argv", [["--worker-id", "x"], ["--worker-id=x"],
                                  ["--worker_id", "x"]])
def test_main_uses_worker_id_flag(monkeypatch, argv):
    monkeypatch.delenv("ENGY_WORKER_ID", raising=False)
    cfg = _main_cfg(monkeypatch, argv)
    assert cfg["worker_id"] == "x" and cfg["worker_name"] == "w1"


def test_main_uses_worker_id_env(monkeypatch):
    monkeypatch.setenv("ENGY_WORKER_ID", "e")
    assert _main_cfg(monkeypatch, [])["worker_id"] == "e"


def test_main_flag_beats_env_worker_id(monkeypatch):
    monkeypatch.setenv("ENGY_WORKER_ID", "e")
    assert _main_cfg(monkeypatch, ["--worker-id", "x"])["worker_id"] == "x"


def test_main_without_worker_id_mints_none(monkeypatch):
    monkeypatch.delenv("ENGY_WORKER_ID", raising=False)
    cfg = _main_cfg(monkeypatch, [])
    assert not cfg.get("worker_id")


# ------------------------------------------------------- default worker name
GPU_UUIDS = ["GPU-7c1e0a52-9f3d-4b8e-a1c2-000000000002",
             "GPU-7c1e0a52-9f3d-4b8e-a1c2-000000000001"]
SIG12 = "097f2c051f8f"


def _smi(monkeypatch, uuids=GPU_UUIDS, fail=False):
    def run(cmd, **kw):
        if fail:
            raise FileNotFoundError("nvidia-smi")
        assert cmd[0] == "nvidia-smi" and "--query-gpu=uuid" in cmd
        return types.SimpleNamespace(stdout="\n".join(uuids) + "\n", returncode=0)
    monkeypatch.setattr(tm.subprocess, "run", run)


@pytest.fixture
def _noname(monkeypatch):
    for k in ("ENGY_WORKER_NAME", "ENGY_INSTANCE_ID", "ENGY_SLOT"):
        monkeypatch.delenv(k, raising=False)


def test_default_name_is_model_at_hardware_sig(monkeypatch, _noname):
    _smi(monkeypatch)
    assert tm._resolve_worker_name(None, "kimi-k3", []) == f"kimi-k3@{SIG12}"
    assert tm._resolve_worker_name(None, " kimi-k3 ", []) == f"kimi-k3@{SIG12}"


@pytest.mark.parametrize("slot,suffix", [("1", "-1"), ("2", "-2"), ("0", ""),
                                         ("", "")])
def test_default_name_slot_suffix(monkeypatch, _noname, slot, suffix):
    _smi(monkeypatch)
    monkeypatch.setenv("ENGY_SLOT", slot)
    assert tm._resolve_worker_name(None, "kimi-k3", []) == \
        f"kimi-k3@{SIG12}{suffix}"


def test_explicit_name_wins_over_default(monkeypatch, _noname):
    _smi(monkeypatch)
    monkeypatch.setenv("ENGY_SLOT", "1")
    assert tm._resolve_worker_name("mine", "kimi-k3", []) == "mine"
    monkeypatch.setenv("ENGY_WORKER_NAME", "envname")
    assert tm._resolve_worker_name(None, "kimi-k3", []) == "envname"


def test_default_name_falls_back_with_one_warning(monkeypatch, _noname, capsys):
    _smi(monkeypatch, fail=True)
    name = tm._resolve_worker_name(None, "kimi-k3", ["http://x"])
    assert name.startswith("tee-") and len(name) == 16
    assert capsys.readouterr().out.count("WARNING") == 1


def test_default_name_falls_back_on_empty_uuid_list(monkeypatch, _noname):
    _smi(monkeypatch, uuids=[])
    assert tm._resolve_worker_name(None, "kimi-k3", []).startswith("tee-")


def test_door_seeding_is_stable_for_the_same_worker_name():
    a = tm.Doors.parse(tm.DEFAULT_GW, seed="w1").homes(8)
    assert a == tm.Doors.parse(tm.DEFAULT_GW, seed="w1").homes(8)


# ---- engines behind a front (PD: sgl-router) ----

def _host_routes(monkeypatch, routes):
    """{(host, path): (status, body)}; a missing host -> connection refused."""
    def handler(request):
        key = (request.url.host + ":" + str(request.url.port), request.url.path)
        hosts = {h for h, _ in routes}
        if key[0] not in hosts:
            raise tm.httpx.ConnectError("refused", request=request)
        status, body = routes.get(key, (404, "not found"))
        if isinstance(body, str):
            return tm.httpx.Response(status, text=body)
        return tm.httpx.Response(status, json=body)
    real = tm.httpx.AsyncClient
    monkeypatch.setattr(tm.httpx, "AsyncClient",
                        lambda *a, **kw: real(*a, transport=tm.httpx.MockTransport(handler), **kw))


RTR, PRE, DEC = "127.0.0.1:30000", "10.0.0.1:8000", "127.0.0.1:8000"


def _pd_serve():
    return tm.Serve([f"http://{RTR}"], "/model", 60.0,
                    health_urls=[f"http://{PRE}", f"http://{DEC}"],
                    load_urls=[f"http://{DEC}", f"http://{PRE}"])


def test_pd_probe_ok_only_when_router_and_every_engine_are_healthy(monkeypatch):
    ok = {(h, "/health"): (200, "") for h in (RTR, PRE, DEC)}
    _host_routes(monkeypatch, ok)
    assert asyncio.run(_pd_serve().probe()) == "ok"


def test_pd_probe_dead_engine_behind_an_always_200_router_is_dead(monkeypatch):
    # sgl-router answers /health 200 whatever its engines do; the prefill
    # engine is gone. A plain multi-serve would say "ok" (another serve can take
    # work); behind a router every engine is required.
    _host_routes(monkeypatch, {(RTR, "/health"): (200, ""), (DEC, "/health"): (200, "")})
    s = _pd_serve()
    assert asyncio.run(s.probe()) == "dead"
    assert s._down == set()          # the router itself stays routable


def test_pd_probe_unhealthy_engine_is_dead(monkeypatch):
    _host_routes(monkeypatch, {(RTR, "/health"): (200, ""), (PRE, "/health"): (503, ""),
                               (DEC, "/health"): (200, "")})
    assert asyncio.run(_pd_serve().probe()) == "dead"


def test_pd_probe_router_down_is_dead(monkeypatch):
    _host_routes(monkeypatch, {(PRE, "/health"): (200, ""), (DEC, "/health"): (200, "")})
    assert asyncio.run(_pd_serve().probe()) == "dead"


def test_pd_kv_load_reads_the_engines_not_the_router(monkeypatch):
    v1 = SGLANG_V1_LOADS
    _host_routes(monkeypatch, {(RTR, "/v1/loads"): (501, "nope"),
                               (DEC, "/v1/loads"): (200, v1), (PRE, "/v1/loads"): (200, v1)})
    kv = asyncio.run(_pd_serve().kv_load())
    one = asyncio.run(tm.Serve([f"http://{DEC}"], "/model", 60.0).kv_load())
    assert kv["ranks"] == 2 * one["ranks"] and kv["limit"] == 2 * one["limit"]


def test_plain_serve_unchanged_without_health_or_load_urls():
    s = tm.Serve(["http://a:1", "http://b:2"], "/m", 60.0)
    assert s.health_urls == [] and s.load_urls == ["http://a:1", "http://b:2"]


# ---- decode rate + PD per-role figures (2026-11-27) ----

def _eng(mode=None, gen=None, run=0, unc=0, tok=0, busy_us=0, ts=1.0, rank=0):
    x = {**SGLANG_V1_LOADS["loads"][0], "dp_rank": rank, "num_running_reqs": run,
         "num_waiting_uncached_tokens": unc, "total_prefill_uncached_tokens": tok,
         "total_prefill_busy_us": busy_us, "timestamp": ts}
    if gen is not None:
        x["gen_throughput"] = gen
    if mode is not None:
        x["disaggregation"] = {"mode": mode, "prefill_bootstrap_queue_reqs": 0}
    return x


def test_decode_fields_sum_ranks_and_omit_when_absent():
    f = tm.decode_fields([_eng(gen=40.5, run=2, rank=0), _eng(gen=60.0, run=3, rank=1)])
    assert f == {"gen_tps": 100.5, "gen_reqs": 5}
    assert tm.decode_fields([_eng()]) == {}                 # older serve: omit
    assert tm.decode_fields([_eng(gen="x")]) == {}          # unparseable: omit
    assert tm.decode_fields([_eng(gen=0.0)]) == {"gen_tps": 0.0, "gen_reqs": 0}


def test_engine_role_prefers_sglang_mode():
    assert tm.engine_role([_eng(mode="prefill")]) == "prefill"
    assert tm.engine_role([_eng(mode="decode")]) == "decode"
    assert tm.engine_role([_eng(mode="null")]) == "null"
    assert tm.engine_role([_eng()]) is None
    assert tm.engine_role([_eng()], "decode") == "decode"
    assert tm.engine_role([_eng(mode="null")], "prefill") == "null"   # loads wins


def test_looks_like_prefill_heuristic():
    assert tm.looks_like_prefill([_eng(gen=0.0, tok=5_000)])
    assert not tm.looks_like_prefill([_eng(gen=30.0, tok=5_000)])     # decoding
    assert not tm.looks_like_prefill([_eng(gen=0.0)])                 # no work
    assert not tm.looks_like_prefill([_eng(tok=5_000)])               # no gen key


def test_single_engine_reports_gen_tps_and_no_pd(monkeypatch):
    _routes(monkeypatch, {"/v1/loads": (200, {"loads": [_eng(gen=90.0, run=6, unc=7)]})})
    kv = asyncio.run(_kv_serve().kv_load())
    assert kv["gen_tps"] == 90.0 and kv["gen_reqs"] == 6 and kv["uncached"] == 7
    assert "pd" not in kv and kv["reqs"] == 6


def test_single_engine_without_gen_throughput_omits_gen(monkeypatch):
    _routes(monkeypatch, {"/v1/loads": (200, SGLANG_V1_LOADS)})
    kv = asyncio.run(_kv_serve().kv_load())
    assert "gen_tps" not in kv and "gen_reqs" not in kv and "pd" not in kv


def _pd_seq(monkeypatch, dec_loads, pre_loads, info=None):
    seq = {DEC: list(dec_loads), PRE: list(pre_loads)}
    info = info or {}

    def handler(request):
        h = f"{request.url.host}:{request.url.port}"
        if request.url.path == "/v1/loads" and h in seq:
            return tm.httpx.Response(200, json={"loads": seq[h].pop(0)})
        if request.url.path == "/get_server_info" and h in info:
            return tm.httpx.Response(200, json=info[h])
        return tm.httpx.Response(404, text="nf")
    real = tm.httpx.AsyncClient
    monkeypatch.setattr(tm.httpx, "AsyncClient",
                        lambda *a, **kw: real(*a, transport=tm.httpx.MockTransport(handler), **kw))


def test_pd_reports_prefill_backlog_from_prefill_and_decode_from_decode(monkeypatch):
    _pd_seq(monkeypatch,
            dec_loads=[[_eng("decode", gen=120.0, run=8, unc=999, tok=0, ts=1.0)],
                       [_eng("decode", gen=120.0, run=8, unc=999, tok=0, ts=31.0)]],
            pre_loads=[[_eng("prefill", gen=0.0, run=1, unc=50_000, tok=0, ts=1.0)],
                       [_eng("prefill", gen=0.0, run=1, unc=40_000, tok=100_000,
                             busy_us=5_000_000, ts=31.0)]])
    s = _pd_serve()
    asyncio.run(s.kv_load())
    kv = asyncio.run(s.kv_load())
    assert kv["pd"] is True
    assert kv["uncached"] == 40_000                  # decode's 999 not added
    assert kv["pf_tps"] == 20_000 and kv["pf_ranks"] == 1
    assert kv["gen_tps"] == 120.0 and kv["gen_reqs"] == 8
    assert kv["reqs"] == 9                           # old field unchanged: all engines


def test_pd_role_from_server_info_when_loads_has_no_mode(monkeypatch):
    _pd_seq(monkeypatch, dec_loads=[[_eng(gen=0.0, run=0)]],
            pre_loads=[[_eng(gen=0.0, unc=3_000)]],
            info={PRE: {"disaggregation_mode": "prefill"},
                  DEC: {"disaggregation_mode": "decode"}})
    kv = asyncio.run(_pd_serve().kv_load())
    # idle decode engine is still "decode" (explicit), not mistaken for prefill
    assert kv["pd"] is True and kv["uncached"] == 3_000
    assert kv["gen_tps"] == 0.0 and kv["gen_reqs"] == 0


def test_pd_heuristic_fallback_without_any_mode(monkeypatch):
    _pd_seq(monkeypatch, dec_loads=[[_eng(gen=75.0, run=5)]],
            pre_loads=[[_eng(gen=0.0, unc=2_000, tok=10_000)]])
    kv = asyncio.run(_pd_serve().kv_load())
    assert kv["pd"] is True and kv["uncached"] == 2_000
    assert kv["gen_tps"] == 75.0 and kv["gen_reqs"] == 5


def test_two_unified_serves_are_not_pd(monkeypatch):
    _pd_seq(monkeypatch, dec_loads=[[_eng("null", gen=50.0, run=2, unc=10)]],
            pre_loads=[[_eng("null", gen=0.0, run=0, unc=20, tok=10_000)]])
    kv = asyncio.run(_pd_serve().kv_load())
    assert "pd" not in kv and kv["uncached"] == 30
    assert kv["gen_tps"] == 50.0 and kv["gen_reqs"] == 2


# ---- /v1/loads preferred over /get_load (2026-11-27) ----
# sglang 0.5.20 (kimi-k3 B300s) answers BOTH endpoints. Asking /get_load first
# meant the backlog (uncached/pf_*) and decode (gen_*) fields never reached the
# gateway, so its prewarm idle gate saw "no signal" on those boxes forever.

_REAL_ASYNC_CLIENT = tm.httpx.AsyncClient


def _counting_routes(monkeypatch, routes):
    hits = []

    def handler(request):
        hits.append(request.url.path)
        status, body = routes.get(request.url.path, (404, "not found"))
        if isinstance(body, str):
            return tm.httpx.Response(status, text=body)
        return tm.httpx.Response(status, json=body)
    real = _REAL_ASYNC_CLIENT               # not a previous patch: re-patchable
    monkeypatch.setattr(tm.httpx, "AsyncClient",
                        lambda *a, **kw: real(*a, transport=tm.httpx.MockTransport(handler), **kw))
    return hits


# Same live state seen through both endpoints, as 216.137.78.93 serves it.
_BOTH_GET_LOAD = [{"dp_rank": 0, "num_reqs": 6, "num_waiting_reqs": 4,
                   "num_tokens": 10_613_473, "num_pending_tokens": 0}]
_BOTH_V1 = {"loads": [{**SGLANG_V1_LOADS["loads"][0],
                       "num_waiting_uncached_tokens": 15_757,
                       "total_prefill_uncached_tokens": 100_000,
                       "total_prefill_busy_us": 1_000_000, "timestamp": 1.0,
                       "gen_throughput": 88.5}]}


def test_serve_answering_both_uses_v1_loads_and_keeps_kv_numbers(monkeypatch):
    # Old path (/get_load only): the reference numbers.
    _counting_routes(monkeypatch, {
        "/get_load": (200, _BOTH_GET_LOAD),
        "/get_server_info": (200, {"max_total_num_tokens": 11536384 // 8,
                                   "dcp_size": 8})})
    old = asyncio.run(_kv_serve().kv_load())
    hits = _counting_routes(monkeypatch, {
        "/get_load": (200, _BOTH_GET_LOAD),
        "/v1/loads": (200, _BOTH_V1),
        "/get_server_info": (200, {"max_total_num_tokens": 11536384 // 8,
                                   "dcp_size": 8})})
    kv = asyncio.run(_kv_serve().kv_load())
    assert kv["uncached"] == 15_757 and kv["gen_tps"] == 88.5 and kv["gen_reqs"] == 6
    for k in ("limit", "reqs", "waiting", "ranks", "pending"):
        assert kv[k] == old[k], k
    assert abs(kv["used"] - old["used"]) <= 1000 and kv["frac"] == old["frac"]
    assert hits == ["/v1/loads"]            # one probe; /get_load not also asked


def test_serve_with_only_get_load_keeps_old_behaviour(monkeypatch):
    hits = _counting_routes(monkeypatch, {
        "/get_load": (200, [{"num_tokens": 500, "num_reqs": 2,
                             "num_waiting_reqs": 1, "num_pending_tokens": 7}]),
        "/get_server_info": (200, {"max_total_num_tokens": 100, "dcp_size": 10})})
    kv = asyncio.run(_kv_serve().kv_load())
    assert kv == {"used": 500, "reqs": 2, "waiting": 1, "pending": 7,
                  "ranks": 1, "limit": 1000, "frac": 0.5}
    assert hits[:2] == ["/v1/loads", "/get_load"] and "/metrics" not in hits


def test_serve_with_only_v1_loads_is_unchanged(monkeypatch):
    hits = _counting_routes(monkeypatch, {"/v1/loads": (200, _BOTH_V1)})
    kv = asyncio.run(_kv_serve().kv_load())
    assert kv["limit"] == 11536384 and kv["frac"] == 0.92 and kv["used"] == 10_613_473
    assert kv["reqs"] == 6 and kv["waiting"] == 4 and kv["pending"] == 0
    assert kv["uncached"] == 15_757 and kv["gen_tps"] == 88.5
    assert hits == ["/v1/loads"]


def test_a_slow_leg_never_blocks_the_request_path():
    """11-25: broadcast was awaited on the request path, sequentially and with
    no timeout, so one backpressured leg stalled every new request before it
    reached the serve (buyers gave up at ~15 s with no first token). kick()
    must return at once, and a stuck leg must not starve the live ones."""
    class Stuck(FakeWS):
        async def send(self, frame):
            await asyncio.sleep(3600)

    stuck, live = Stuck(), FakeWS()
    load = tm.Load()
    load.attach(stuck)
    load.attach(live)

    async def go():
        t0 = time.monotonic()
        load.start()
        load.kick()
        assert time.monotonic() - t0 < 0.05
        await asyncio.wait_for(load._bcast_task, tm.BCAST_SEND_TIMEOUT_S + 2)

    asyncio.run(go())
    assert [h["inflight"] for h in live.heartbeats()] == [1]
    assert stuck in load._channels  # slow, not dead: kept for the next report
