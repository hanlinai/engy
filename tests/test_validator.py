import json

import httpx
from substrateinterface import Keypair

from engy_sn53.sync import epoch_message
from engy_sn53.validator import tick

GENESIS = 1784505600
IDX = 12
END = GENESIS + 13 * 604800
NOW = float(END + 3600)
MASTER = Keypair.create_from_uri("//Alice")
DIGEST = "cd" * 32


def _payload():
    return {
        "v": 1, "netuid": 53, "epoch_index": IDX,
        "epoch_start": END - 604800, "epoch_end": END, "digest": DIGEST,
        "weights": [["5Aaa", 65535]],
        "signed_hotkey": MASTER.ss58_address,
        "signature": MASTER.sign(epoch_message(53, IDX, DIGEST).encode()).hex(),
        "signed_at": END + 610,
    }


def _cfg(tmp_path):
    return {"api": "https://engy.example", "master_hotkey": MASTER.ss58_address,
            "netuid": 53, "genesis": GENESIS, "network": "finney",
            "wallet": "w", "wallet_hotkey": "hk", "poll_s": 600,
            "state_file": str(tmp_path / "state.json")}


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
