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
