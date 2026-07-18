import hashlib
import json

import httpx
from substrateinterface import Keypair

from validator.sync import epoch_index, epoch_message, verify_payload, fetch_weights

GENESIS = 1784505600
NETUID = 53
IDX = 12
START = GENESIS + IDX * 604800
END = START + 604800
NOW = float(END + 3600)          # inside epoch 13 → freshly finalized epoch is 12

MASTER = Keypair.create_from_uri("//Alice")


def _result_json(**over):
    result = {
        "epoch_end": END, "epoch_index": IDX, "epoch_start": START,
        "miners": [], "netuid": NETUID, "params": {}, "v": 1,
        "weights": [["5Aaa", 65535]],
    }
    result.update(over)
    return json.dumps(result, sort_keys=True, separators=(",", ":"))


def payload(*, result_json=None, digest=None, signature=None, **over):
    rj = result_json if result_json is not None else _result_json()
    d = digest if digest is not None else hashlib.sha256(rj.encode("utf-8")).hexdigest()
    sig = signature if signature is not None else MASTER.sign(epoch_message(NETUID, IDX, d).encode()).hex()
    base = {
        "v": 1, "netuid": NETUID, "epoch_index": IDX,
        "epoch_start": START, "epoch_end": END,
        "digest": d,
        "result_json": rj,
        "weights": [["5Aaa", 65535]],
        "signed_hotkey": MASTER.ss58_address,
        "signature": sig,
        "signed_at": END + 610,
    }
    base.update(over)
    return base


def verify(p, **over):
    kw = dict(master_hotkey=MASTER.ss58_address, netuid=NETUID,
              genesis=GENESIS, now=NOW, last_applied=None)
    kw.update(over)
    return verify_payload(p, **kw)


def test_epoch_math_matches_spec_example():
    assert epoch_index(START, GENESIS) == 12
    assert epoch_message(53, 12, "d3ad") == "engy-sn53:epoch:v1:53:12:d3ad"


def test_accepts_a_genuine_fresh_payload():
    ok, reason, weights = verify(payload())
    assert ok, reason
    assert weights == [["5Aaa", 65535]]


def test_evil_top_level_weights_are_ignored_verified_weights_come_from_result_json():
    # A compromised coordination layer swaps the display-only top-level
    # `weights` field while leaving result_json/digest/signature intact.
    # The payload must still verify (those are untouched), but the weights
    # actually returned for submission must come from result_json, NOT the
    # tampered top-level field.
    p = payload(weights=[["5Evil", 65535]])
    ok, reason, weights = verify(p)
    assert ok, reason
    assert weights == [["5Aaa", 65535]]
    assert weights != p["weights"]


def test_tampered_result_json_with_stale_digest_is_rejected():
    genuine = payload()
    tampered_rj = genuine["result_json"][:-1] + ("1" if genuine["result_json"][-1] != "1" else "2")
    p = dict(genuine, result_json=tampered_rj)  # digest/signature left as-is (old)
    assert verify(p) == (False, "digest", None)


def test_result_json_with_mismatched_epoch_index_is_malformed():
    rj = _result_json(epoch_index=IDX + 1)
    digest = hashlib.sha256(rj.encode("utf-8")).hexdigest()
    sig = MASTER.sign(epoch_message(NETUID, IDX, digest).encode()).hex()
    p = payload(result_json=rj, digest=digest, signature=sig)
    assert verify(p) == (False, "malformed", None)


def test_result_json_with_mismatched_netuid_is_malformed():
    rj = _result_json(netuid=99)
    digest = hashlib.sha256(rj.encode("utf-8")).hexdigest()
    sig = MASTER.sign(epoch_message(NETUID, IDX, digest).encode()).hex()
    p = payload(result_json=rj, digest=digest, signature=sig)
    assert verify(p) == (False, "malformed", None)


def test_missing_result_json_is_malformed():
    p = payload()
    del p["result_json"]
    assert verify(p) == (False, "malformed", None)


def test_non_string_result_json_is_malformed():
    p = payload()
    p["result_json"] = 12345
    assert verify(p) == (False, "malformed", None)


def test_rejects_bad_signature():
    assert verify(payload(signature="00" * 64)) == (False, "signature", None)


def test_rejects_wrong_master_key():
    other = Keypair.create_from_uri("//Bob")
    rj = _result_json()
    digest = hashlib.sha256(rj.encode("utf-8")).hexdigest()
    p = payload(signed_hotkey=other.ss58_address,
                signature=other.sign(epoch_message(NETUID, IDX, digest).encode()).hex())
    # signed_hotkey in the payload is irrelevant — only the pinned key counts
    assert verify(p) == (False, "signature", None)


def test_rejects_wrong_netuid_and_version():
    assert verify(payload(netuid=99))[1] == "netuid"
    assert verify(payload(v=2))[1] == "version"


def test_rejects_stale_epoch():
    # a replayed epoch-11 payload while epoch 13 is running
    old = 11
    rj = _result_json(epoch_index=old, epoch_start=START - 604800, epoch_end=START)
    digest = hashlib.sha256(rj.encode("utf-8")).hexdigest()
    p = payload(epoch_index=old, result_json=rj, digest=digest,
                signature=MASTER.sign(epoch_message(NETUID, old, digest).encode()).hex())
    assert verify(p) == (False, "stale-epoch", None)


def test_rejects_already_applied():
    assert verify(payload(), last_applied=12) == (False, "already-applied", None)
    ok, reason, weights = verify(payload(), last_applied=11)
    assert ok, reason


def test_rejects_malformed_signature_non_hex():
    assert verify(payload(signature="zz")) == (False, "signature", None)


def test_rejects_malformed_signature_wrong_length():
    assert verify(payload(signature="00")) == (False, "signature", None)


def test_fetch_weights_hits_the_v1_route():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json=payload())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = fetch_weights("https://engy.example", client=client)
    assert seen["url"] == "https://engy.example/api/subnet/v1/weights/latest"
    assert p["epoch_index"] == IDX


def test_non_dict_payload_is_malformed():
    for bad in ([1, 2, 3], "nope", 5, None):
        ok, reason, weights = verify_payload(bad, master_hotkey=MASTER.ss58_address, netuid=NETUID,
                                    genesis=GENESIS, now=NOW, last_applied=None)
        assert ok is False and reason == "malformed" and weights is None


def test_rejects_bytes_wrapped_signature():
    # substrate-interface's Keypair.verify silently retries with <Bytes>...</Bytes>
    # wrapping (as used by polkadot{.js}extension-signed messages). The
    # cross-repo contract is raw-bytes-only signing — a signature over the
    # wrapped message must be rejected, not accepted via fallback retry.
    rj = _result_json()
    digest = hashlib.sha256(rj.encode("utf-8")).hexdigest()
    msg = epoch_message(NETUID, IDX, digest)
    wrapped = b"<Bytes>" + msg.encode() + b"</Bytes>"
    sig = MASTER.sign(wrapped).hex()
    assert verify(payload(signature=sig)) == (False, "signature", None)


def test_rejects_bool_epoch_index():
    assert verify(payload(epoch_index=True))[0] is False
    assert verify(payload(epoch_index=False))[0] is False


def test_rejects_missing_weights_in_result_json():
    rj_dict = json.loads(_result_json())
    del rj_dict["weights"]
    rj = json.dumps(rj_dict, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(rj.encode("utf-8")).hexdigest()
    sig = MASTER.sign(epoch_message(NETUID, IDX, digest).encode()).hex()
    p = payload(result_json=rj, digest=digest, signature=sig)
    assert verify(p) == (False, "malformed", None)


def test_rejects_non_list_weights_in_result_json():
    rj = _result_json(weights="nope")
    digest = hashlib.sha256(rj.encode("utf-8")).hexdigest()
    sig = MASTER.sign(epoch_message(NETUID, IDX, digest).encode()).hex()
    p = payload(result_json=rj, digest=digest, signature=sig)
    assert verify(p) == (False, "malformed", None)


def test_rejects_weights_with_non_pair_entry_in_result_json():
    rj = _result_json(weights=[["5Aaa", 100], "not-a-pair"])
    digest = hashlib.sha256(rj.encode("utf-8")).hexdigest()
    sig = MASTER.sign(epoch_message(NETUID, IDX, digest).encode()).hex()
    p = payload(result_json=rj, digest=digest, signature=sig)
    assert verify(p) == (False, "malformed", None)


def test_rejects_weights_with_out_of_range_value_in_result_json():
    rj = _result_json(weights=[["5Aaa", 70000]])
    digest = hashlib.sha256(rj.encode("utf-8")).hexdigest()
    sig = MASTER.sign(epoch_message(NETUID, IDX, digest).encode()).hex()
    p = payload(result_json=rj, digest=digest, signature=sig)
    assert verify(p) == (False, "malformed", None)


def test_accepts_well_formed_weights():
    rj = _result_json(weights=[["5Aaa", 0], ["5Bbb", 65535]])
    digest = hashlib.sha256(rj.encode("utf-8")).hexdigest()
    sig = MASTER.sign(epoch_message(NETUID, IDX, digest).encode()).hex()
    p = payload(result_json=rj, digest=digest, signature=sig)
    ok, reason, weights = verify(p)
    assert ok, reason
    assert weights == [["5Aaa", 0], ["5Bbb", 65535]]
