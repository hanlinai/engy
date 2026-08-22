"""ChainClient: a persistent chain connection reused across ticks.

The light validator used to open a fresh bt.Subtensor on every tick and never
close it (validator.chain.open_chain), so the websockets to the finney
entrypoint accumulated until the endpoint rate-limited the IP (HTTP 429) and
the validator could no longer set weights. ChainClient holds one connection,
takes a fresh metagraph snapshot on it each tick, reconnects only on failure,
and closes cleanly — so the socket count stays at one instead of leaking.
"""
import sys
import types

import pytest

from validator.chain import ChainClient, ChainView


class _FakeMeta:
    def __init__(self, hotkeys):
        self.hotkeys = list(hotkeys)


class _FakeSub:
    def __init__(self, network, script):
        self.network = network
        self.script = script
        self.closed = False
        script["constructed"] += 1
        # bittensor's Subtensor exposes .substrate.close() and .close(); both
        # must be called to actually tear down the async websocket.
        self.substrate = types.SimpleNamespace(close=self._substrate_close)

    def _substrate_close(self):
        self.script["substrate_closed"] += 1

    def close(self):
        self.closed = True
        self.script["closed"] += 1

    def metagraph(self, netuid):
        self.script["metagraph_calls"] += 1
        if self.script["fail_metagraph"] > 0:
            self.script["fail_metagraph"] -= 1
            raise RuntimeError("server rejected WebSocket connection: HTTP 429")
        return _FakeMeta(self.script["hotkeys"])

    def get_current_block(self):
        return self.script["block"]


def _install_fake_bt(monkeypatch, **overrides):
    """Swap in a fake bittensor whose Subtensor records how many were built."""
    script = {
        "constructed": 0, "substrate_closed": 0, "closed": 0,
        "metagraph_calls": 0, "fail_metagraph": 0,
        "hotkeys": ["5Aaa", "5Bbb"], "block": 100,
    }
    script.update(overrides)
    subs = []

    def _factory(*, network):
        sub = _FakeSub(network, script)
        subs.append(sub)
        return sub

    fake_bt = types.ModuleType("bittensor")
    fake_bt.Subtensor = _factory
    monkeypatch.setitem(sys.modules, "bittensor", fake_bt)
    return script, subs


def test_reuses_one_connection_across_multiple_opens(monkeypatch):
    # The whole fix: N ticks, ONE websocket. Fresh snapshot each time.
    script, _ = _install_fake_bt(monkeypatch)
    c = ChainClient()
    v1 = c.open_chain(network="finney", netuid=53)
    v2 = c.open_chain(network="finney", netuid=53)
    assert script["constructed"] == 1
    assert script["metagraph_calls"] == 2
    assert v1.hotkeys == ["5Aaa", "5Bbb"] and v1.block == 100
    assert v2.sub is v1.sub


def test_close_closes_both_layers_and_is_idempotent(monkeypatch):
    script, _ = _install_fake_bt(monkeypatch)
    c = ChainClient()
    c.open_chain(network="finney", netuid=53)
    c.close()
    assert script["substrate_closed"] == 1 and script["closed"] == 1
    c.close()  # nothing held any more — must not raise or double-close
    assert script["closed"] == 1


def test_reconnects_after_a_snapshot_failure(monkeypatch):
    # A stale/rate-limited connection surfaces as a metagraph error. The client
    # drops it (closing it) and reconnects, and the retry succeeds — leaving one
    # clean connection in hand rather than propagating a transient failure.
    script, subs = _install_fake_bt(monkeypatch, fail_metagraph=1)
    c = ChainClient()
    v = c.open_chain(network="finney", netuid=53)
    assert script["constructed"] == 2      # reconnected once
    assert subs[0].closed is True          # the failed connection was closed
    assert v.sub is subs[1]                # holding the fresh one


def test_does_not_leak_when_the_chain_stays_unreachable(monkeypatch):
    # The 429 outage: every snapshot fails. open_chain must raise so the loop
    # logs "open failed" and skips the tick — but it must leave NO live
    # connection behind. Every socket it opened is closed; nothing is held.
    # This is the property whose absence caused the production outage.
    script, subs = _install_fake_bt(monkeypatch, fail_metagraph=99)
    c = ChainClient()
    with pytest.raises(Exception):
        c.open_chain(network="finney", netuid=53)
    assert script["constructed"] == script["closed"]  # every socket closed
    assert all(s.closed for s in subs)
    assert c._sub is None                             # nothing held
