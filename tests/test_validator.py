import hashlib
import json
import time

import httpx
import pytest
from substrateinterface import Keypair

from validator.sync import epoch_message, MAX_RESPONSE_BYTES
from validator.validator import (
    tick, _last_applied, _heartbeat_age, _watchdog_check, stall_limit, healthcheck,
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


def _cfg(tmp_path):
    return {"api": "https://engy.example", "master_hotkey": MASTER.ss58_address,
            "netuid": 53, "genesis": GENESIS, "network": "finney",
            "wallet": "w", "wallet_hotkey": "hk", "poll_s": 600,
            "state_file": str(tmp_path / "state.json"),
            "heartbeat_file": str(tmp_path / "heartbeat.json")}


def _client(payload):
    return httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=payload)))


def test_applied_then_deduplicated(tmp_path):
    cfg = _cfg(tmp_path)
    submitted = []
    ok = tick(cfg, now=NOW, client=_client(_payload()),
              submit_fn=lambda c, w: submitted.append(w) or True)
    assert ok == "applied"
    assert submitted == [[["5Aaa", 65535]]]
    assert json.load(open(cfg["state_file"])) == {"last_applied": IDX}
    # second tick with the same payload: already applied, no resubmission
    assert tick(cfg, now=NOW, client=_client(_payload()),
                submit_fn=lambda c, w: submitted.append(w) or True) == "rejected:already-applied"
    assert len(submitted) == 1


def test_evil_top_level_weights_never_reach_submit(tmp_path):
    # The core security property: a compromised coordination layer swaps the
    # display-only top-level `weights` field, but the daemon must submit the
    # weights extracted from the verified result_json, not the evil field.
    cfg = _cfg(tmp_path)
    p = _payload(weights_top_level=[["5Evil", 65535]])
    submitted = []
    out = tick(cfg, now=NOW, client=_client(p),
               submit_fn=lambda c, w: submitted.append(w) or True)
    assert out == "applied"
    assert submitted == [[["5Aaa", 65535]]]
    assert submitted != [p["weights"]]


def test_bad_signature_never_reaches_chain(tmp_path):
    p = _payload()
    p["signature"] = "00" * 64
    called = []
    out = tick(_cfg(tmp_path), now=NOW, client=_client(p),
               submit_fn=lambda c, w: called.append(1) or True)
    assert out == "rejected:signature" and called == []


def test_fetch_failure_is_contained(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(503)))
    assert tick(_cfg(tmp_path), now=NOW, client=client,
                submit_fn=lambda c, w: True) == "fetch-failed"


def test_submit_failure_does_not_advance_state(tmp_path):
    cfg = _cfg(tmp_path)
    out = tick(cfg, now=NOW, client=_client(_payload()), submit_fn=lambda c, w: False)
    assert out == "submit-failed"
    # state not written → next tick retries the same epoch
    out = tick(cfg, now=NOW, client=_client(_payload()), submit_fn=lambda c, w: True)
    assert out == "applied"


def test_non_dict_response_is_rejected_not_raised(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=[1, 2, 3])))
    out = tick(_cfg(tmp_path), now=NOW, client=client, submit_fn=lambda c, w: True)
    assert out == "rejected:malformed"


def test_oversized_response_is_treated_as_fetch_failure(tmp_path):
    huge = b"[" + b"1" * (MAX_RESPONSE_BYTES + 1) + b"]"
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, content=huge,
                                    headers={"content-type": "application/json"})))
    out = tick(_cfg(tmp_path), now=NOW, client=client, submit_fn=lambda c, w: True)
    assert out == "fetch-failed"


def test_last_applied_survives_corrupt_state_file(tmp_path):
    state = tmp_path / "state.json"
    # non-int last_applied
    state.write_text(json.dumps({"last_applied": "twelve"}))
    assert _last_applied(str(state)) is None
    # bool last_applied (bool is a subclass of int — must not pass through)
    state.write_text(json.dumps({"last_applied": True}))
    assert _last_applied(str(state)) is None
    # non-dict top-level JSON
    state.write_text(json.dumps([1, 2, 3]))
    assert _last_applied(str(state)) is None
    # invalid JSON entirely
    state.write_text("not json at all")
    assert _last_applied(str(state)) is None
    # a genuinely valid file still round-trips
    state.write_text(json.dumps({"last_applied": 7}))
    assert _last_applied(str(state)) == 7


# ── liveness (heartbeat + watchdog) ──────────────────────────────

def test_heartbeat_is_written_on_every_outcome_not_just_applied(tmp_path):
    # The state file only moves once a week, when an epoch is applied, so it is
    # useless as a liveness signal. The heartbeat must advance on every tick,
    # including the ones that reject or fail.
    cfg = _cfg(tmp_path)
    hb = cfg["heartbeat_file"]

    tick(cfg, now=NOW, client=_client(_payload()), submit_fn=lambda c, w: True)
    assert _heartbeat_age(hb, now=NOW) == 0.0

    # a rejected tick 100s later still proves the loop is alive
    tick(cfg, now=NOW + 100, client=_client(_payload()), submit_fn=lambda c, w: True)
    assert _heartbeat_age(hb, now=NOW + 100) == 0.0

    # ...and a fetch failure too
    dead = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(503)))
    tick(cfg, now=NOW + 200, client=dead, submit_fn=lambda c, w: True)
    assert _heartbeat_age(hb, now=NOW + 200) == 0.0


def test_heartbeat_age_grows_when_the_loop_stops_ticking(tmp_path):
    cfg = _cfg(tmp_path)
    tick(cfg, now=NOW, client=_client(_payload()), submit_fn=lambda c, w: True)
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
    tick(cfg, now=NOW, client=_client(_payload()), submit_fn=lambda c, w: True)

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
