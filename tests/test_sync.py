import json

import httpx
from substrateinterface import Keypair

from engy_sn53.sync import epoch_index, epoch_message, verify_payload, fetch_weights

GENESIS = 1784505600
NETUID = 53
IDX = 12
START = GENESIS + IDX * 604800
END = START + 604800
NOW = float(END + 3600)          # inside epoch 13 → freshly finalized epoch is 12

MASTER = Keypair.create_from_uri("//Alice")
DIGEST = "ab" * 32


def payload(**over):
    base = {
        "v": 1, "netuid": NETUID, "epoch_index": IDX,
        "epoch_start": START, "epoch_end": END,
        "digest": DIGEST,
        "weights": [["5Aaa", 51916], ["5Bbb", 13619]],
        "signed_hotkey": MASTER.ss58_address,
        "signature": MASTER.sign(epoch_message(NETUID, IDX, DIGEST).encode()).hex(),
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
    ok, reason = verify(payload())
    assert ok, reason


def test_rejects_bad_signature():
    assert verify(payload(signature="00" * 64)) == (False, "signature")


def test_rejects_wrong_master_key():
    other = Keypair.create_from_uri("//Bob")
    p = payload(signed_hotkey=other.ss58_address,
                signature=other.sign(epoch_message(NETUID, IDX, DIGEST).encode()).hex())
    # signed_hotkey in the payload is irrelevant — only the pinned key counts
    assert verify(p) == (False, "signature")


def test_rejects_wrong_netuid_and_version():
    assert verify(payload(netuid=99))[1] == "netuid"
    assert verify(payload(v=2))[1] == "version"


def test_rejects_stale_epoch():
    # a replayed epoch-11 payload while epoch 13 is running
    old = 11
    p = payload(epoch_index=old,
                signature=MASTER.sign(epoch_message(NETUID, old, DIGEST).encode()).hex())
    assert verify(p) == (False, "stale-epoch")


def test_rejects_already_applied():
    assert verify(payload(), last_applied=12) == (False, "already-applied")
    ok, _ = verify(payload(), last_applied=11)
    assert ok


def test_rejects_malformed_signature_non_hex():
    assert verify(payload(signature="zz")) == (False, "signature")


def test_rejects_malformed_signature_wrong_length():
    assert verify(payload(signature="00")) == (False, "signature")


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
        ok, reason = verify_payload(bad, master_hotkey=MASTER.ss58_address, netuid=NETUID,
                                    genesis=GENESIS, now=NOW, last_applied=None)
        assert ok is False and reason == "malformed"


def test_rejects_bytes_wrapped_signature():
    # substrate-interface's Keypair.verify silently retries with <Bytes>...</Bytes>
    # wrapping (as used by polkadot{.js}extension-signed messages). The
    # cross-repo contract is raw-bytes-only signing — a signature over the
    # wrapped message must be rejected, not accepted via fallback retry.
    msg = epoch_message(NETUID, IDX, DIGEST)
    wrapped = b"<Bytes>" + msg.encode() + b"</Bytes>"
    sig = MASTER.sign(wrapped).hex()
    assert verify(payload(signature=sig)) == (False, "signature")


def test_rejects_bool_epoch_index():
    assert verify(payload(epoch_index=True))[0] is False
    assert verify(payload(epoch_index=False))[0] is False


def test_rejects_missing_weights():
    p = payload()
    del p["weights"]
    assert verify(p) == (False, "malformed")


def test_rejects_non_list_weights():
    assert verify(payload(weights="nope")) == (False, "malformed")


def test_rejects_weights_with_non_pair_entry():
    assert verify(payload(weights=[["5Aaa", 100], "not-a-pair"])) == (False, "malformed")


def test_rejects_weights_with_out_of_range_value():
    assert verify(payload(weights=[["5Aaa", 70000]])) == (False, "malformed")


def test_accepts_well_formed_weights():
    ok, reason = verify(payload(weights=[["5Aaa", 0], ["5Bbb", 65535]]))
    assert ok, reason
