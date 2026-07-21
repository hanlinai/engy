import hashlib
import json
import time

import httpx
import pytest
from substrateinterface import Keypair

from validator import chain as chain_mod
from validator.state import (
    read_state, last_applied, last_submit_block, cached_weights,
)
from validator.sync import epoch_message, MAX_RESPONSE_BYTES
from validator.validator import (
    tick, _heartbeat_age, _watchdog_check, stall_limit, healthcheck,
)

GENESIS = 1784505600
IDX = 12
END = GENESIS + 13 * 604800
NOW = float(END + 3600)
MASTER = Keypair.create_from_uri("//Alice")


def _result_json(**over):
    result = {
        "epoch_end": END, "epoch_index": IDX, "epoch_start": END - 604800,
        "miners": [], "netuid": 53, "params": {}, "v": 1,
        "weights": [["5Aaa", 65535]],
    }
    result.update(over)
    return json.dumps(result, sort_keys=True, separators=(",", ":"))


def _payload(*, weights_top_level=None):
    rj = _result_json()
    digest = hashlib.sha256(rj.encode("utf-8")).hexdigest()
    return {
        "v": 1, "netuid": 53, "epoch_index": IDX,
        "epoch_start": END - 604800, "epoch_end": END, "digest": digest,
        "result_json": rj,
        "weights": weights_top_level if weights_top_level is not None else [["5Aaa", 65535]],
        "signed_hotkey": MASTER.ss58_address,
        "signature": MASTER.sign(epoch_message(53, IDX, digest).encode()).hex(),
        "signed_at": END + 610,
    }


def _payload_with(weights):
    """A genuine payload carrying an arbitrary verified weight vector."""
    rj = json.dumps({"epoch_end": END, "epoch_index": IDX,
                     "epoch_start": END - 604800, "miners": [], "netuid": 53,
                     "params": {}, "v": 1, "weights": weights},
                    sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(rj.encode("utf-8")).hexdigest()
    return {"v": 1, "netuid": 53, "epoch_index": IDX,
            "epoch_start": END - 604800, "epoch_end": END, "digest": digest,
            "result_json": rj, "weights": weights,
            "signed_hotkey": MASTER.ss58_address,
            "signature": MASTER.sign(epoch_message(53, IDX, digest).encode()).hex(),
            "signed_at": END + 610}


def _cfg(tmp_path):
    return {"api": "https://engy.example", "master_hotkey": MASTER.ss58_address,
            "netuid": 53, "genesis": GENESIS, "network": "finney",
            "wallet": "w", "wallet_hotkey": "hk", "poll_s": 300,
            "state_file": str(tmp_path / "state.json"),
            "heartbeat_file": str(tmp_path / "heartbeat.json")}


class FakeChain:
    """Stands in for validator.chain, recording what reached the chain."""

    def __init__(self, hotkeys=("5Aaa",), block=5000, ok=True, open_raises=False):
        self.hotkeys = list(hotkeys)
        self.block = block
        self.ok = ok
        self.open_raises = open_raises
        self.submitted = []
        self.opens = 0

    def open_chain(self, *, network, netuid):
        self.opens += 1
        if self.open_raises:
            raise RuntimeError("subtensor unreachable")
        return chain_mod.ChainView(sub=object(), hotkeys=self.hotkeys,
                                   block=self.block)

    resolve_uids = staticmethod(chain_mod.resolve_uids)
    skipped_hotkeys = staticmethod(chain_mod.skipped_hotkeys)
    dropped_weight_share = staticmethod(chain_mod.dropped_weight_share)

    def set_weights(self, view, *, wallet, wallet_hotkey, netuid, uids, ws):
        self.submitted.append((uids, ws))
        return self.ok


def _client(payload):
    return httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=payload)))


def _dead_client():
    return httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(503)))


# ── submission scheduling ────────────────────────────────────────

def test_new_epoch_applies_and_records_every_state_field(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    assert tick(cfg, now=NOW, client=_client(_payload()), chain=fake) == "applied"
    assert fake.submitted == [([0], [65535])]
    s = read_state(cfg["state_file"])
    assert last_applied(s) == IDX
    assert last_submit_block(s) == 5000
    assert cached_weights(s) == [["5Aaa", 65535]]


def test_same_epoch_holds_until_the_resubmit_interval(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    tick(cfg, now=NOW, client=_client(_payload()), chain=fake)
    # 99 blocks later, past the pre-gate but short of the interval
    fake.block = 5099
    out = tick(cfg, now=NOW + 1000, client=_client(_payload()), chain=fake)
    assert out == "skipped:too-soon"
    assert len(fake.submitted) == 1


def test_same_epoch_resubmits_at_100_blocks(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    tick(cfg, now=NOW, client=_client(_payload()), chain=fake)
    fake.block = 5100
    out = tick(cfg, now=NOW + 1200, client=_client(_payload()), chain=fake)
    assert out == "resubmitted"
    assert fake.submitted == [([0], [65535]), ([0], [65535])]
    s = read_state(cfg["state_file"])
    assert last_submit_block(s) == 5100      # advances
    assert last_applied(s) == IDX            # unchanged — same epoch


def test_the_pregate_avoids_opening_a_connection_when_clearly_too_early(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    tick(cfg, now=NOW, client=_client(_payload()), chain=fake)
    assert fake.opens == 1
    out = tick(cfg, now=NOW + 300, client=_client(_payload()), chain=fake)
    assert out == "skipped:too-soon"
    assert fake.opens == 1                   # no second connection


def test_a_missing_block_number_falls_back_to_the_wall_clock(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=None)
    assert tick(cfg, now=NOW, client=_client(_payload()), chain=fake) == "applied"
    assert last_submit_block(read_state(cfg["state_file"])) is None
    # 1199s later: still short of the 1200s fallback interval
    assert tick(cfg, now=NOW + 1199, client=_client(_payload()),
                chain=fake) == "skipped:too-soon"
    assert tick(cfg, now=NOW + 1200, client=_client(_payload()),
                chain=fake) == "resubmitted"


# ── failure containment ──────────────────────────────────────────

def test_submit_failure_advances_no_state_field(tmp_path):
    # The invariant that makes the whole schedule safe: advancing
    # last_submit_block on a failure would make the loop believe it had just
    # submitted and wait a full interval before retrying, amplifying a
    # transient chain error into 20 minutes of silence.
    cfg = _cfg(tmp_path)
    fake = FakeChain(ok=False)
    assert tick(cfg, now=NOW, client=_client(_payload()), chain=fake) == "submit-failed"
    assert read_state(cfg["state_file"]) == {}

    fake.ok = True
    assert tick(cfg, now=NOW + 1, client=_client(_payload()), chain=fake) == "applied"


def test_a_failed_resubmit_retries_on_the_next_poll_not_a_full_interval_later(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    tick(cfg, now=NOW, client=_client(_payload()), chain=fake)

    fake.block, fake.ok = 5100, False
    assert tick(cfg, now=NOW + 1200, client=_client(_payload()),
                chain=fake) == "submit-failed"
    assert last_submit_block(read_state(cfg["state_file"])) == 5000  # not advanced

    # one poll later (300s, 25 blocks) the retry goes through
    fake.block, fake.ok = 5125, True
    assert tick(cfg, now=NOW + 1500, client=_client(_payload()),
                chain=fake) == "resubmitted"


def test_an_unreachable_chain_is_contained_and_leaves_state_alone(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(open_raises=True)
    assert tick(cfg, now=NOW, client=_client(_payload()), chain=fake) == "chain-failed"
    assert read_state(cfg["state_file"]) == {}
    assert _heartbeat_age(cfg["heartbeat_file"], now=NOW) == 0.0


def test_evil_top_level_weights_never_reach_submit(tmp_path):
    # The core security property: a compromised coordination layer swaps the
    # display-only top-level `weights` field, but the daemon must submit the
    # weights extracted from the verified result_json, not the evil field.
    cfg = _cfg(tmp_path)
    fake = FakeChain(hotkeys=["5Aaa"])
    p = _payload(weights_top_level=[["5Evil", 65535]])
    out = tick(cfg, now=NOW, client=_client(p), chain=fake)
    # 5Evil is not on chain, so had the evil field been used there would be no
    # uid to submit at all — reaching "applied" proves result_json won.
    assert out == "applied"
    assert fake.submitted == [([0], [65535])]


def test_bad_signature_never_reaches_chain(tmp_path):
    p = _payload()
    p["signature"] = "00" * 64
    fake = FakeChain()
    out = tick(_cfg(tmp_path), now=NOW, client=_client(p), chain=fake)
    assert out == "rejected:signature"
    assert fake.submitted == [] and fake.opens == 0


def test_fetch_failure_is_contained(tmp_path):
    assert tick(_cfg(tmp_path), now=NOW, client=_dead_client(),
                chain=FakeChain()) == "fetch-failed"


def test_non_dict_response_is_rejected_not_raised(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=[1, 2, 3])))
    out = tick(_cfg(tmp_path), now=NOW, client=client, chain=FakeChain())
    assert out == "rejected:malformed"


def test_oversized_response_is_treated_as_fetch_failure(tmp_path):
    huge = b"[" + b"1" * (MAX_RESPONSE_BYTES + 1) + b"]"
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, content=huge,
                                   headers={"content-type": "application/json"})))
    out = tick(_cfg(tmp_path), now=NOW, client=client, chain=FakeChain())
    assert out == "fetch-failed"


# ── on-chain identity check (logs, never blocks) ─────────────────

def test_an_unregistered_hotkey_is_dropped_and_logged_never_blocking(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    fake = FakeChain(hotkeys=["5Aaa"])
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=_payload_with(
            [["5Aaa", 40000], ["5Gone", 25535]]))))
    assert tick(cfg, now=NOW, client=client, chain=fake) == "applied"
    assert fake.submitted == [([0], [65535])]        # renormalized onto 5Aaa
    out = capsys.readouterr().out
    assert "5Gone" in out and "39.0%" in out


# ── liveness (heartbeat + watchdog) ──────────────────────────────

def test_heartbeat_is_written_on_every_outcome_not_just_applied(tmp_path):
    # The state file only moves when a submission lands, so it is useless as a
    # liveness signal. The heartbeat must advance on every tick, including the
    # ones that skip, reject, or fail.
    cfg = _cfg(tmp_path)
    hb = cfg["heartbeat_file"]

    tick(cfg, now=NOW, client=_client(_payload()), chain=FakeChain())
    assert _heartbeat_age(hb, now=NOW) == 0.0

    # a skipped tick 100s later still proves the loop is alive
    tick(cfg, now=NOW + 100, client=_client(_payload()), chain=FakeChain())
    assert _heartbeat_age(hb, now=NOW + 100) == 0.0

    # ...and a fetch failure too
    tick(cfg, now=NOW + 200, client=_dead_client(), chain=FakeChain())
    assert _heartbeat_age(hb, now=NOW + 200) == 0.0


def test_heartbeat_age_grows_when_the_loop_stops_ticking(tmp_path):
    cfg = _cfg(tmp_path)
    tick(cfg, now=NOW, client=_client(_payload()), chain=FakeChain())
    assert _heartbeat_age(cfg["heartbeat_file"], now=NOW + 3600) == 3600.0


def test_missing_or_corrupt_heartbeat_reads_as_unknown(tmp_path):
    assert _heartbeat_age(str(tmp_path / "nope"), now=NOW) is None
    corrupt = tmp_path / "hb"
    corrupt.write_text("not json")
    assert _heartbeat_age(str(corrupt), now=NOW) is None


def test_stall_limit_leaves_room_for_a_slow_tick(tmp_path):
    # A tick that is merely slow (chain submit retrying) must not trip the
    # watchdog — only a loop that has missed several polls in a row.
    assert stall_limit(600) > 600
    assert stall_limit(600) == 1800
    # a very short poll interval still gets an absolute floor
    assert stall_limit(10) == 900


def test_watchdog_exits_only_after_the_stall_limit(tmp_path):
    cfg = _cfg(tmp_path)
    hb = cfg["heartbeat_file"]
    tick(cfg, now=NOW, client=_client(_payload()), chain=FakeChain())

    exits = []
    # healthy: one poll interval later, well inside the limit
    _watchdog_check(hb, poll_s=600, now=NOW + 600, exit_fn=lambda code: exits.append(code))
    assert exits == []
    # wedged: past the stall limit → force a non-zero exit so the container
    # restart policy can recover it
    _watchdog_check(hb, poll_s=600, now=NOW + 1801, exit_fn=lambda code: exits.append(code))
    assert exits == [1]


def test_watchdog_ignores_a_missing_heartbeat_at_startup(tmp_path):
    # before the first tick completes there is no heartbeat; that is not a stall
    exits = []
    _watchdog_check(str(tmp_path / "nope"), poll_s=600, now=NOW,
                    exit_fn=lambda code: exits.append(code))
    assert exits == []


def test_healthcheck_exits_nonzero_on_a_stale_heartbeat(tmp_path, monkeypatch, capsys):
    hb = tmp_path / "heartbeat.json"
    monkeypatch.setenv("ENGY_SN53_API", "https://engy.example")
    monkeypatch.setenv("ENGY_SN53_MASTER_HOTKEY", MASTER.ss58_address)
    monkeypatch.setenv("ENGY_SN53_HEARTBEAT_FILE", str(hb))
    monkeypatch.setenv("ENGY_SN53_POLL_S", "600")

    # no heartbeat yet → unhealthy
    with pytest.raises(SystemExit) as e:
        healthcheck()
    assert e.value.code != 0

    hb.write_text(json.dumps({"ts": time.time()}))
    healthcheck()  # fresh → exits normally
    assert "ok" in capsys.readouterr().out

    hb.write_text(json.dumps({"ts": time.time() - 5000}))
    with pytest.raises(SystemExit) as e:
        healthcheck()
    assert e.value.code != 0


# ── provider-outage fallback ─────────────────────────────────────

def test_a_provider_outage_keeps_resubmitting_the_cached_vector(tmp_path):
    # Without this the validator goes silent through the outage and passes
    # activity_cutoff, dropping out of consensus for the rest of the epoch.
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    tick(cfg, now=NOW, client=_client(_payload()), chain=fake)

    fake.block = 5100
    out = tick(cfg, now=NOW + 1200, client=_dead_client(), chain=fake)
    assert out == "resubmitted:cached"
    assert fake.submitted == [([0], [65535]), ([0], [65535])]
    assert last_submit_block(read_state(cfg["state_file"])) == 5100


def test_the_cache_still_respects_the_resubmit_interval(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    tick(cfg, now=NOW, client=_client(_payload()), chain=fake)
    fake.block = 5050
    assert tick(cfg, now=NOW + 1000, client=_dead_client(),
                chain=fake) == "skipped:too-soon"
    assert len(fake.submitted) == 1


def test_a_stale_epoch_during_rollover_falls_back_to_the_cache(tmp_path):
    # After a new epoch opens but before the provider publishes it, the served
    # payload is rejected as stale. Keep submitting instead of going silent.
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    tick(cfg, now=NOW, client=_client(_payload()), chain=fake)

    fake.block = 5100
    rolled = NOW + 604800          # one epoch later; IDX is now two epochs old
    out = tick(cfg, now=rolled, client=_client(_payload()), chain=fake)
    assert out == "resubmitted:cached"
    assert len(fake.submitted) == 2


def test_no_cache_and_no_payload_submits_nothing(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain()
    assert tick(cfg, now=NOW, client=_dead_client(), chain=fake) == "fetch-failed"
    assert fake.submitted == [] and fake.opens == 0


def test_a_recovered_provider_takes_over_from_the_cache(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    tick(cfg, now=NOW, client=_client(_payload()), chain=fake)
    fake.block = 5100
    tick(cfg, now=NOW + 1200, client=_dead_client(), chain=fake)

    # next epoch publishes: the fresh vector wins and applies immediately
    nxt = NOW + 604800
    p = _payload_with([["5Bbb", 65535]])
    p["epoch_index"] = IDX + 1
    rj = json.loads(p["result_json"]); rj["epoch_index"] = IDX + 1
    p["result_json"] = json.dumps(rj, sort_keys=True, separators=(",", ":"))
    p["digest"] = hashlib.sha256(p["result_json"].encode()).hexdigest()
    p["signature"] = MASTER.sign(epoch_message(53, IDX + 1, p["digest"]).encode()).hex()

    fake.hotkeys = ["5Aaa", "5Bbb"]
    assert tick(cfg, now=nxt, client=_client(p), chain=fake) == "applied"
    s = read_state(cfg["state_file"])
    assert last_applied(s) == IDX + 1
    assert cached_weights(s) == [["5Bbb", 65535]]


def test_a_corrupt_cache_is_not_submitted(tmp_path):
    cfg = _cfg(tmp_path)
    with open(cfg["state_file"], "w") as f:
        json.dump({"last_applied": IDX, "last_submit_block": 5000,
                   "last_submit_ts": NOW, "cached_weights": "garbage"}, f)
    fake = FakeChain(block=5100)
    assert tick(cfg, now=NOW + 1200, client=_dead_client(),
                chain=fake) == "fetch-failed"
    assert fake.submitted == []
