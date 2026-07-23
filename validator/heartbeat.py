"""Emit the light validator's liveness heartbeat to the provider (provider
spec 2026-07-24).

Unlike the master, a light validator reports the **consensus digest** it
verified and submitted on chain, so the provider's /admin can reconcile it
against the master-signed digest. This is NOT an independent rescore — the light
validator trusts the master's scoring and only checks the signature
(validator/sync.py). The consensus digest is therefore "the master-signed result
I actually put on chain": sha256 over the bytes it fetched and verified against
the pinned master key. That still catches a fork, a wrong master-key pin, or a
serve path handing this validator a different result than the provider's own
latest-finalized epoch — it does not catch a master that scored wrong (nothing
on the light side can, by design).

Request signing is the same sr25519 scheme the provider verifies and the master
uses (engy-validator/engy_validator/signing.py):
    engy-sn53:api:v1:<netuid>:<METHOD>:<path>:<unix_ts>:<sha256(body)_hex>
signed with THIS validator's wallet hotkey — its own identity, distinct from
the master key the payload signature is checked against.

Best-effort throughout: signing needs the wallet hotkey and the network can be
down; neither may ever disturb the submit loop, whose own heartbeat file is the
real watchdog signal. `post_heartbeat` returns a bool and never raises.
"""
from __future__ import annotations

import hashlib
import json
import time

import httpx

HEARTBEAT_PATH = "/api/subnet/v1/validator/heartbeat"


def canonical_message(netuid: int, method: str, path: str, ts: int,
                      body: bytes = b"") -> str:
    return (f"engy-sn53:api:v1:{netuid}:{method.upper()}:{path}:{ts}:"
            f"{hashlib.sha256(body).hexdigest()}")


def signed_headers(keypair, netuid: int, method: str, path: str,
                   body: bytes = b"", *, ts: int | None = None) -> dict[str, str]:
    ts = int(time.time()) if ts is None else ts
    msg = canonical_message(netuid, method, path, ts, body)
    return {
        "X-Validator-Hotkey": keypair.ss58_address,
        "X-Validator-Ts": str(ts),
        "X-Validator-Sig": keypair.sign(msg.encode()).hex(),
        "Content-Type": "application/json",
    }


def build_body(version: str, synced_epoch: int | None,
               digest: str | None) -> dict:
    """The heartbeat JSON. `consensus` (the consensus digest) is attached only
    when we have both the epoch and its digest — the provider files a
    reconciliation claim for exactly that (epoch, digest) pair. A standby
    submission (no scored vector) reports liveness and the epoch, but no digest
    to reconcile."""
    body: dict = {"v": 1, "version": version}
    if synced_epoch is not None:
        body["synced_epoch"] = synced_epoch
        if digest:
            body["consensus"] = {"epoch_index": synced_epoch, "digest": digest}
    return body


def load_hotkey_keypair(wallet: str, wallet_hotkey: str):
    """The wallet hotkey Keypair used to SIGN heartbeats (this validator's own
    identity). Lazy bittensor import so the verify/test paths never need the
    chain extra."""
    import bittensor as bt
    return bt.Wallet(name=wallet, hotkey=wallet_hotkey).hotkey


def post_heartbeat(api_base: str, keypair, netuid: int, *, version: str,
                   synced_epoch: int | None, digest: str | None,
                   client: httpx.Client | None = None,
                   timeout: float = 10.0) -> bool:
    """POST the signed heartbeat. Returns True on a 200, False on any failure —
    never raises, so the caller's submit loop is never disturbed."""
    body = json.dumps(build_body(version, synced_epoch, digest),
                      separators=(",", ":")).encode()
    headers = signed_headers(keypair, netuid, "POST", HEARTBEAT_PATH, body)
    own = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        url = f"{api_base.rstrip('/')}{HEARTBEAT_PATH}"
        resp = client.post(url, content=body, headers=headers)
        if resp.status_code != 200:
            print(f"[heartbeat] rejected: {resp.status_code} "
                  f"{resp.text[:120]}", flush=True)
            return False
        return True
    except (httpx.HTTPError, ValueError) as e:
        print(f"[heartbeat] send failed: {str(e)[:120]}", flush=True)
        return False
    finally:
        if own:
            client.close()
