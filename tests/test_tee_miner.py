"""tee_miner load reporting and drain grace.

Both behaviours are invisible in a single-process test of the miner alone —
they only matter through the gateway's admission path — so these pin the two
properties that path reads: that an in-flight transition reaches EVERY leg
immediately, and that a drained leg outlives any request its backend may still
be working on.
"""
import asyncio
import json

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


def test_serve_one_reports_the_transition_at_both_ends():
    """Acquire AND release. Reporting only the acquire would leave a count that
    never comes back down until the next periodic frame."""
    ws = FakeWS()
    load = tm.Load()
    load.attach(ws)

    asyncio.run(tm._serve_one(ws, CHAT, FakeBackend(), load))

    assert [h["inflight"] for h in ws.heartbeats()] == [1, 0]
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

    asyncio.run(tm._serve_one(ws, CHAT, Broken(), load))

    assert [h["inflight"] for h in ws.heartbeats()] == [1, 0]


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
    assert body["stream_options"] == {"include_usage": True}


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
