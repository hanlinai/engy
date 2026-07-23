import hashlib
import json

import httpx
from bittensor_wallet import Keypair

from validator import heartbeat as hb

HK = Keypair.create_from_uri("//LightHK")
NETUID = 53
PATH = "/api/subnet/v1/validator/heartbeat"


def test_canonical_message_matches_the_cross_repo_contract():
    body = b'{"v":1}'
    bh = hashlib.sha256(body).hexdigest()
    assert hb.canonical_message(53, "post", PATH, 1700, body) == \
        f"engy-sn53:api:v1:53:POST:{PATH}:1700:{bh}"


def test_build_body_liveness_only_when_no_epoch():
    assert hb.build_body("engy-lv 1", None, "d") == {"v": 1, "version": "engy-lv 1"}


def test_build_body_epoch_without_digest_reports_no_claim():
    assert hb.build_body("v", 12, None) == {"v": 1, "version": "v", "synced_epoch": 12}


def test_build_body_epoch_with_digest_attaches_recomputed():
    assert hb.build_body("v", 12, "abc") == {
        "v": 1, "version": "v", "synced_epoch": 12,
        "recomputed": {"epoch_index": 12, "digest": "abc"}}


def test_signed_headers_verify_against_the_pinned_scheme():
    body = json.dumps(hb.build_body("v", 12, "abc"), separators=(",", ":")).encode()
    headers = hb.signed_headers(HK, NETUID, "POST", PATH, body, ts=1700)
    assert headers["X-Validator-Hotkey"] == HK.ss58_address
    msg = hb.canonical_message(NETUID, "POST", PATH, 1700, body)
    assert HK.verify(msg.encode(), bytes.fromhex(headers["X-Validator-Sig"]))


def test_post_heartbeat_signs_and_sends_and_returns_true():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.raw_path.decode()
        seen["body"] = json.loads(request.content)
        seen["hotkey"] = request.headers["X-Validator-Hotkey"]
        # the signature must verify over the EXACT path+body the provider sees
        ts = int(request.headers["X-Validator-Ts"])
        msg = hb.canonical_message(NETUID, "POST", seen["path"], ts, request.content)
        seen["sig_ok"] = HK.verify(
            msg.encode(), bytes.fromhex(request.headers["X-Validator-Sig"]))
        return httpx.Response(200, json={"ok": True, "server_ts": 1.0})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ok = hb.post_heartbeat("https://engy.example", HK, NETUID, version="engy-lv 1",
                           synced_epoch=12, digest="abc", client=client)
    assert ok is True
    assert seen["path"] == PATH
    assert seen["sig_ok"] is True
    assert seen["hotkey"] == HK.ss58_address
    assert seen["body"] == {"v": 1, "version": "engy-lv 1", "synced_epoch": 12,
                            "recomputed": {"epoch_index": 12, "digest": "abc"}}


def test_post_heartbeat_returns_false_on_non_200_without_raising():
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(401, json={"error": "bad signature"})))
    assert hb.post_heartbeat("https://engy.example", HK, NETUID, version="v",
                             synced_epoch=None, digest=None, client=client) is False


def test_post_heartbeat_returns_false_on_network_error_without_raising():
    def boom(request):
        raise httpx.ConnectError("down")
    client = httpx.Client(transport=httpx.MockTransport(boom))
    assert hb.post_heartbeat("https://engy.example", HK, NETUID, version="v",
                             synced_epoch=7, digest="d", client=client) is False


def test_emit_reports_the_last_submitted_epoch_and_digest_from_state(tmp_path, monkeypatch):
    from validator import validator as v
    from validator.state import write_state
    state_file = str(tmp_path / "state.json")
    write_state(state_file, last_applied=12, last_submit_block=100,
                last_submit_ts=1.0, cached_weights=[["5Aaa", 65535]],
                cached_digest="deadbeef")
    captured = {}
    monkeypatch.setattr(v._heartbeat, "post_heartbeat",
                        lambda *a, **k: captured.update(k) or True)
    v.emit_provider_heartbeat(
        {"api": "https://e", "netuid": 53, "state_file": state_file},
        object(), "engy-lv 1")
    assert captured == {"version": "engy-lv 1", "synced_epoch": 12, "digest": "deadbeef"}


def test_emit_is_a_noop_when_the_wallet_keypair_is_unavailable(tmp_path, monkeypatch):
    from validator import validator as v
    calls = []
    monkeypatch.setattr(v._heartbeat, "post_heartbeat",
                        lambda *a, **k: calls.append(1))
    v.emit_provider_heartbeat(
        {"api": "x", "netuid": 53, "state_file": str(tmp_path / "s.json")},
        None, "v")
    assert calls == []
