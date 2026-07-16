"""The light-validator loop: poll engy.web, verify, set weights (spec §2).

Failure posture (spec §10): on ANY failure — API down, bad signature, stale
epoch, chain error — do nothing. The chain persists the last submitted
weights, so inaction is always the safe move.
"""
from __future__ import annotations

import json
import os
import sys
import time

import httpx

from .chain import submit
from .sync import fetch_weights, verify_payload


def load_config() -> dict:
    env = os.environ
    for req in ("ENGY_SN53_API", "ENGY_SN53_MASTER_HOTKEY"):
        if not env.get(req):
            sys.exit(f"{req} is required")
    return {
        "api": env["ENGY_SN53_API"],
        "master_hotkey": env["ENGY_SN53_MASTER_HOTKEY"],
        "netuid": int(env.get("ENGY_SN53_NETUID", "53")),
        "genesis": int(env.get("ENGY_SN53_GENESIS_TS", "1784505600")),
        "network": env.get("ENGY_SN53_NETWORK", "finney"),
        "wallet": env.get("ENGY_SN53_WALLET", "default"),
        "wallet_hotkey": env.get("ENGY_SN53_WALLET_HOTKEY", "default"),
        "poll_s": int(env.get("ENGY_SN53_POLL_S", "600")),
        "state_file": env.get("ENGY_SN53_STATE_FILE",
                              os.path.expanduser("~/.engy-sn53/state.json")),
    }


def _last_applied(path: str) -> int | None:
    try:
        with open(path) as f:
            return json.load(f).get("last_applied")
    except (OSError, ValueError):
        return None


def tick(cfg: dict, *, now: float, client: httpx.Client | None = None,
         submit_fn=None) -> str:
    submit_fn = submit_fn or submit
    try:
        payload = fetch_weights(cfg["api"], client=client)
    except (httpx.HTTPError, ValueError) as e:
        print(f"[sync] fetch failed: {e}", flush=True)
        return "fetch-failed"

    ok, reason = verify_payload(
        payload, master_hotkey=cfg["master_hotkey"], netuid=cfg["netuid"],
        genesis=cfg["genesis"], now=now, last_applied=_last_applied(cfg["state_file"]))
    if not ok:
        print(f"[sync] payload rejected: {reason}", flush=True)
        return f"rejected:{reason}"

    if not submit_fn(cfg, payload["weights"]):
        return "submit-failed"

    os.makedirs(os.path.dirname(cfg["state_file"]), exist_ok=True)
    with open(cfg["state_file"], "w") as f:
        json.dump({"last_applied": payload["epoch_index"]}, f)
    print(f"[sync] applied epoch {payload['epoch_index']} "
          f"({len(payload['weights'])} hotkeys)", flush=True)
    return "applied"


def main() -> None:
    cfg = load_config()
    print(f"[engy-sn53-validator] api={cfg['api']} netuid={cfg['netuid']} "
          f"master={cfg['master_hotkey']}", flush=True)
    try:
        while True:
            tick(cfg, now=time.time())
            time.sleep(cfg["poll_s"])
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
