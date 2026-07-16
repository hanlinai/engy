"""Fetch and verify the master-signed weight payload (spec §8, §10).

Trust model: the ONLY root of trust is the master hotkey pinned in local
config. The payload's own `signed_hotkey` field is display metadata — never
verify against it. Replay protection: the epoch index is inside the signed
message, and only `current_epoch − 1` (the last completed epoch) is accepted.
"""
from __future__ import annotations

import json

import httpx

EPOCH_S = 604800
MAX_RESPONSE_BYTES = 1024 * 1024  # 1 MiB cap on the weights payload


def epoch_index(ts: float, genesis: int) -> int:
    return int((ts - genesis) // EPOCH_S)


def epoch_message(netuid: int, epoch_index: int, digest_hex: str) -> str:
    return f"engy-sn53:epoch:v1:{netuid}:{epoch_index}:{digest_hex}"


def verify_payload(payload: dict, *, master_hotkey: str, netuid: int, genesis: int,
                   now: float, last_applied: int | None) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "malformed"
    if payload.get("v") != 1:
        return False, "version"
    if payload.get("netuid") != netuid:
        return False, "netuid"
    idx = payload.get("epoch_index")
    digest = payload.get("digest")
    sig = payload.get("signature", "")
    if (not isinstance(idx, int) or isinstance(idx, bool)
            or not isinstance(digest, str) or not isinstance(sig, str)):
        return False, "signature"
    try:
        import sr25519
        from substrateinterface import Keypair
        pub = Keypair(ss58_address=master_hotkey).public_key
        raw_sig = bytes.fromhex(sig)
        message = epoch_message(netuid, idx, digest).encode()
        if len(raw_sig) != 64 or not sr25519.verify(raw_sig, message, pub):
            return False, "signature"
    except (ValueError, TypeError):
        return False, "signature"
    if idx != epoch_index(now, genesis) - 1:
        return False, "stale-epoch"
    if last_applied is not None and idx <= last_applied:
        return False, "already-applied"
    w = payload.get("weights")
    if (not isinstance(w, list) or
            not all(isinstance(p, list) and len(p) == 2 and isinstance(p[0], str)
                    and isinstance(p[1], int) and not isinstance(p[1], bool)
                    and 0 <= p[1] <= 65535 for p in w)):
        return False, "malformed"
    return True, "ok"


def fetch_weights(api_base: str, timeout: float = 30.0, client: httpx.Client | None = None) -> dict:
    own = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        url = f"{api_base.rstrip('/')}/api/subnet/v1/weights/latest"
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            cl = resp.headers.get("content-length")
            if cl is not None and int(cl) > MAX_RESPONSE_BYTES:
                raise httpx.HTTPError(f"response too large ({cl} bytes)")
            body = bytearray()
            for chunk in resp.iter_bytes():
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise httpx.HTTPError("response exceeded size cap")
            return json.loads(bytes(body))
    finally:
        if own:
            client.close()
