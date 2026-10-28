"""tee_miner proactive leg rotation (LEG_ROTATE_S).

The GCP LB cuts every gateway WebSocket at exactly 24 h, and legs that connect
together expire together (10-27: one wave per gateway process; a stream cut by
it resumed onto a sibling leg that died 2 s later). Each leg now retires itself
at a random age without cutting a stream: idle -> close and re-dial; busy ->
hand new requests back with a retryable 503 until the in-flight ones finish."""
import asyncio
import json
import random

from miner import tee_miner as tm


class QueueWS:
    """A leg's socket: frames pushed in are read by _session; frames it sends
    are collected; close() ends the read loop like a closed socket."""

    def __init__(self):
        self.sent = []
        self.inbox = asyncio.Queue()
        self.closed = False

    def push(self, frame):
        self.inbox.put_nowait(json.dumps(frame))

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def close(self):
        if not self.closed:
            self.closed = True
            self.inbox.put_nowait(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        m = await self.inbox.get()
        if m is None:
            raise StopAsyncIteration
        return m

    def responses(self):
        return {f["corr_id"]: f for f in self.sent if f["type"] == tm.P.RESPONSE}


class SlowBackend:
    def __init__(self, delay):
        self.delay = delay

    async def stream(self, request):
        await asyncio.sleep(self.delay)
        yield "delta", {"delta": {"content": "hi"}, "logprobs": None}
        yield "done", {"id": "chatcmpl-x", "choices": [], "usage": {}}

    async def kv_load(self):
        return None


def _cfg(backend):
    return {"miner_key": "k", "model": "kimi-k3", "model_root": "r", "hw": {},
            "worker_name": "w", "worker_id": "wid", "backend": backend,
            "hb_s": 1000}


def _serve(corr, rid=None):
    req = {"messages": [{"role": "user", "content": "hi"}]}
    if rid:
        req["engy_request_id"] = rid
    return {"type": tm.P.SERVE, "corr_id": corr, "request": req}


def test_lifetime_is_random_within_the_window(monkeypatch):
    monkeypatch.setattr(tm, "LEG_ROTATE_S", 1000.0)
    monkeypatch.setattr(tm, "LEG_ROTATE_JITTER_S", 100.0)
    rng = random.Random(7)
    vals = [tm.leg_lifetime(rng) for _ in range(200)]
    assert all(1000.0 <= v <= 1100.0 for v in vals)
    assert len({round(v, 3) for v in vals}) > 150


def test_rotation_off_at_zero(monkeypatch):
    monkeypatch.setattr(tm, "LEG_ROTATE_S", 0.0)
    assert tm.leg_lifetime() is None


def test_defaults_retire_well_before_the_24h_cut():
    worst = (tm.LEG_ROTATE_S + tm.LEG_ROTATE_JITTER_S
             + tm.LEG_ROTATE_IDLE_WAIT_S + tm.LEG_ROTATE_DRAIN_S)
    assert tm.LEG_ROTATE_S > 0
    assert worst < 24 * 3600 - 1800


def test_idle_leg_rotates():
    async def s():
        ws = QueueWS()
        out = await asyncio.wait_for(tm._session(
            ws, _cfg(SlowBackend(0)), {"max_inflight": 4}, tm.Load(), "gw0",
            rotate_after=0.05), 3.0)
        assert out is False and ws.closed      # normal end; _leg re-dials
    asyncio.run(s())


def test_busy_leg_hands_new_requests_back_and_finishes_inflight(monkeypatch):
    monkeypatch.setattr(tm, "LEG_ROTATE_IDLE_WAIT_S", 0.05)
    monkeypatch.setattr(tm, "LEG_ROTATE_DRAIN_S", 5.0)

    async def s():
        ws = QueueWS()
        sess = asyncio.create_task(tm._session(
            ws, _cfg(SlowBackend(0.6)), {"max_inflight": 4}, tm.Load(), "gw0",
            rotate_after=0.05))
        ws.push(_serve("c1"))
        await asyncio.sleep(0.3)               # rotation due, c1 still running
        ws.push(_serve("c2", rid="r2"))
        assert await asyncio.wait_for(sess, 3.0) is False
        got = ws.responses()
        assert got["c2"]["error"] == tm._ROTATING_ERROR and got["c2"]["request_id"] == "r2"
        assert "error" not in got["c1"]        # the stream already here completed
        assert ws.closed
    asyncio.run(s())


def test_leg_ending_first_cancels_its_rotation():
    async def s():
        ws = QueueWS()
        sess = asyncio.create_task(tm._session(
            ws, _cfg(SlowBackend(0)), {"max_inflight": 4}, tm.Load(), "gw0",
            rotate_after=30.0))
        await asyncio.sleep(0.05)
        await ws.close()                       # the gateway side goes away
        assert await asyncio.wait_for(sess, 3.0) is False
        await asyncio.sleep(0)
        stray = [t for t in asyncio.all_tasks()
                 if t is not asyncio.current_task() and not t.done()
                 and "_LegRotation" in repr(t.get_coro())]
        assert not stray
    asyncio.run(s())
