"""The light-validator loop: poll the engy provider API, verify, set weights (spec §2).

Failure posture (spec §10): on ANY failure — API down, bad signature, stale
epoch, chain error — do nothing. The chain persists the last submitted
weights, so inaction is always the safe move.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

import httpx

from .chain import submit
from .sync import fetch_weights, verify_payload


def load_config() -> dict:
    env = os.environ
    for req in ("ENGY_SN53_API", "ENGY_SN53_MASTER_HOTKEY"):
        if not env.get(req):
            sys.exit(f"{req} is required")
    state_file = env.get("ENGY_SN53_STATE_FILE",
                         os.path.expanduser("~/.engy-sn53/state.json"))
    return {
        "api": env["ENGY_SN53_API"],
        "master_hotkey": env["ENGY_SN53_MASTER_HOTKEY"],
        "netuid": int(env.get("ENGY_SN53_NETUID", "53")),
        "genesis": int(env.get("ENGY_SN53_GENESIS_TS", "1784505600")),
        "network": env.get("ENGY_SN53_NETWORK", "finney"),
        "wallet": env.get("ENGY_SN53_WALLET", "default"),
        "wallet_hotkey": env.get("ENGY_SN53_WALLET_HOTKEY", "default"),
        "poll_s": int(env.get("ENGY_SN53_POLL_S", "600")),
        "state_file": state_file,
        "heartbeat_file": env.get(
            "ENGY_SN53_HEARTBEAT_FILE",
            os.path.join(os.path.dirname(state_file) or ".", "heartbeat.json")),
    }


def _last_applied(path: str) -> int | None:
    try:
        with open(path) as f:
            data = json.load(f)
        v = data.get("last_applied") if isinstance(data, dict) else None
        return v if isinstance(v, int) and not isinstance(v, bool) else None
    except (OSError, ValueError, AttributeError, TypeError):
        return None


# ── Liveness ─────────────────────────────────────────────────────
#
# `restart: unless-stopped` only recovers a process that *exits*. A loop wedged
# inside a chain call stays "running" forever while silently submitting
# nothing. The state file is no help as a liveness signal — it only advances
# when an epoch is applied, i.e. once a week. So every tick stamps a heartbeat
# regardless of outcome, a watchdog thread turns a stall into an exit, and the
# container HEALTHCHECK reads the same file for `docker ps` visibility.

WATCHDOG_INTERVAL_S = 30
MIN_STALL_LIMIT_S = 900


def stall_limit(poll_s: int) -> int:
    """How long without a completed tick counts as wedged.

    Three poll intervals, floored at 15 min: a merely slow tick (a chain submit
    retrying) must not trip the watchdog — only a loop that has missed several
    polls in a row.
    """
    return max(poll_s * 3, MIN_STALL_LIMIT_S)


def _write_heartbeat(path: str, now: float) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"ts": now}, f)
    except OSError as e:  # never let a heartbeat failure kill the loop
        print(f"[health] heartbeat write failed: {e}", flush=True)


def _heartbeat_age(path: str, *, now: float) -> float | None:
    """Seconds since the last completed tick, or None if unknown."""
    try:
        with open(path) as f:
            ts = json.load(f).get("ts")
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(ts, (int, float)) or isinstance(ts, bool):
        return None
    return max(0.0, float(now) - float(ts))


def _watchdog_check(path: str, *, poll_s: int, now: float, exit_fn=os._exit) -> None:
    age = _heartbeat_age(path, now=now)
    if age is None:
        return  # no heartbeat yet (startup) — not a stall
    limit = stall_limit(poll_s)
    if age > limit:
        print(f"[health] no completed tick in {age:.0f}s (limit {limit}s) — "
              f"exiting so the restart policy can recover", flush=True)
        exit_fn(1)


def _start_watchdog(cfg: dict) -> None:
    def loop():
        while True:
            time.sleep(WATCHDOG_INTERVAL_S)
            _watchdog_check(cfg["heartbeat_file"], poll_s=cfg["poll_s"],
                            now=time.time())

    threading.Thread(target=loop, daemon=True, name="watchdog").start()


def healthcheck() -> None:
    """Container HEALTHCHECK entrypoint: exit 0 if the loop is still ticking."""
    cfg = load_config()
    age = _heartbeat_age(cfg["heartbeat_file"], now=time.time())
    limit = stall_limit(cfg["poll_s"])
    if age is None:
        sys.exit("no heartbeat yet")
    if age > limit:
        sys.exit(f"stale heartbeat: {age:.0f}s > {limit}s")
    print(f"ok (last tick {age:.0f}s ago)")


def tick(cfg: dict, *, now: float, client: httpx.Client | None = None,
         submit_fn=None) -> str:
    """Run one poll cycle, stamping the heartbeat however it turns out.

    The heartbeat records "the loop completed a cycle", not "weights were
    applied" — a rejected or failed tick is still proof of life, and under the
    spec's do-nothing failure posture those are the normal case.
    """
    try:
        return _run_tick(cfg, now=now, client=client, submit_fn=submit_fn)
    finally:
        _write_heartbeat(cfg["heartbeat_file"], now)


def _run_tick(cfg: dict, *, now: float, client: httpx.Client | None = None,
              submit_fn=None) -> str:
    submit_fn = submit_fn or submit
    try:
        payload = fetch_weights(cfg["api"], client=client)
    except (httpx.HTTPError, ValueError) as e:
        print(f"[sync] fetch failed: {e}", flush=True)
        return "fetch-failed"

    ok, reason, weights = verify_payload(
        payload, master_hotkey=cfg["master_hotkey"], netuid=cfg["netuid"],
        genesis=cfg["genesis"], now=now, last_applied=_last_applied(cfg["state_file"]))
    if not ok:
        print(f"[sync] payload rejected: {reason}", flush=True)
        return f"rejected:{reason}"

    if not submit_fn(cfg, weights):
        return "submit-failed"

    os.makedirs(os.path.dirname(cfg["state_file"]) or ".", exist_ok=True)
    with open(cfg["state_file"], "w") as f:
        json.dump({"last_applied": payload["epoch_index"]}, f)
    print(f"[sync] applied epoch {payload['epoch_index']} "
          f"({len(weights)} hotkeys)", flush=True)
    return "applied"


def main() -> None:
    cfg = load_config()
    print(f"[engy-sn53-validator] api={cfg['api']} netuid={cfg['netuid']} "
          f"master={cfg['master_hotkey']}", flush=True)
    print(f"[health] heartbeat={cfg['heartbeat_file']} "
          f"stall_limit={stall_limit(cfg['poll_s'])}s", flush=True)
    _start_watchdog(cfg)
    try:
        while True:
            try:
                tick(cfg, now=time.time())
            except Exception as e:
                print(f"[sync] tick error: {e}", flush=True)
            time.sleep(cfg["poll_s"])
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
