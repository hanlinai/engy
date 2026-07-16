"""Fetch and verify the master-signed weight payload (spec §8, §10).

Trust model: the ONLY root of trust is the master hotkey pinned in local
config. The payload's own `signed_hotkey` field is display metadata — never
verify against it. Replay protection: the epoch index is inside the signed
message, and only `current_epoch − 1` (the last completed epoch) is accepted.
"""
from __future__ import annotations

import httpx

EPOCH_S = 604800


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
    if not isinstance(idx, int) or not isinstance(digest, str) or not isinstance(sig, str):
        return False, "signature"
    try:
        from substrateinterface import Keypair
        kp = Keypair(ss58_address=master_hotkey)
        if not kp.verify(epoch_message(netuid, idx, digest).encode(), bytes.fromhex(sig)):
            return False, "signature"
    except (ValueError, TypeError):
        return False, "signature"
    if idx != epoch_index(now, genesis) - 1:
        return False, "stale-epoch"
    if last_applied is not None and idx <= last_applied:
        return False, "already-applied"
    return True, "ok"


def fetch_weights(api_base: str, timeout: float = 30.0, client: httpx.Client | None = None) -> dict:
    own = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        resp = client.get(f"{api_base.rstrip('/')}/api/subnet/v1/weights/latest")
        resp.raise_for_status()
        return resp.json()
    finally:
        if own:
            client.close()
