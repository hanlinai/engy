"""Fetch and verify the master-signed weight payload (spec §8, §10).

Trust model: the ONLY root of trust is the master hotkey pinned in local
config. The payload's own `signed_hotkey` field is display metadata — never
verify against it. Replay protection: the epoch index is inside the signed
message, and only `current_epoch − 1` (the last completed epoch) is accepted.

Binding: the master signs `sha256(result_json)`, not the top-level `weights`
field. The weight vector actually submitted MUST be extracted from the
verified `result_json` bytes — the top-level `weights` field is untrusted
display metadata and is never used for submission.
"""
from __future__ import annotations

import hashlib
import json

import httpx

EPOCH_S = 604800
MAX_RESPONSE_BYTES = 1024 * 1024  # 1 MiB cap on the weights payload


def epoch_index(ts: float, genesis: int) -> int:
    return int((ts - genesis) // EPOCH_S)


def epoch_message(netuid: int, epoch_index: int, digest_hex: str) -> str:
    return f"engy-sn53:epoch:v1:{netuid}:{epoch_index}:{digest_hex}"


def _well_formed_weights(w) -> bool:
    return (isinstance(w, list) and
            all(isinstance(p, list) and len(p) == 2 and isinstance(p[0], str)
                and isinstance(p[1], int) and not isinstance(p[1], bool)
                and 0 <= p[1] <= 65535 for p in w))


def verify_payload(payload: dict, *, master_hotkey: str, netuid: int, genesis: int,
                   now: float, last_applied: int | None) -> tuple[bool, str, list | None]:
    """Verify a fetched payload and return (ok, reason, weights).

    `weights` (only set when ok) is the weight vector extracted from the
    VERIFIED `result_json` bytes — never the top-level `payload["weights"]`
    field, which is display metadata a compromised coordination layer could
    forge independently of the signature.
    """
    if not isinstance(payload, dict):
        return False, "malformed", None
    if payload.get("v") != 1:
        return False, "version", None
    if payload.get("netuid") != netuid:
        return False, "netuid", None

    result_json = payload.get("result_json")
    if not isinstance(result_json, str):
        return False, "malformed", None

    # Binding step: the served digest must equal the hash of the served bytes.
    local_digest = hashlib.sha256(result_json.encode("utf-8")).hexdigest()
    if local_digest != payload.get("digest"):
        return False, "digest", None

    idx = payload.get("epoch_index")
    sig = payload.get("signature", "")
    try:
        import sr25519
        from substrateinterface import Keypair
        pub = Keypair(ss58_address=master_hotkey).public_key
        raw_sig = bytes.fromhex(sig)
        # Verify against local_digest (recomputed), NOT payload["digest"] —
        # we already know they're equal at this point, but this makes the
        # binding explicit: sign what we hashed, not what we were told.
        message = epoch_message(netuid, idx, local_digest).encode()
        if len(raw_sig) != 64 or not sr25519.verify(raw_sig, message, pub):
            return False, "signature", None
    except (ValueError, TypeError):
        return False, "signature", None

    try:
        result = json.loads(result_json)
    except ValueError:
        return False, "malformed", None
    if (not isinstance(result, dict)
            or result.get("epoch_index") != idx
            or result.get("netuid") != netuid
            or not _well_formed_weights(result.get("weights"))):
        return False, "malformed", None
    weights = result["weights"]

    if idx != epoch_index(now, genesis) - 1:
        return False, "stale-epoch", None
    if last_applied is not None and idx <= last_applied:
        return False, "already-applied", None

    return True, "ok", weights


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
