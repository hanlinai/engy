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
