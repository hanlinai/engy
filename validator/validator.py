"""The light-validator loop: poll the engy provider API, verify, set weights (spec §2).

Failure posture (spec §10): never submit anything unverified. A bad signature,
a stale epoch or a malformed payload submits nothing at all.

That posture is *not* extended to silence. A validator whose last_update
exceeds the subnet's activity_cutoff (~5000 blocks) is treated as inactive and
drops out of Yuma consensus, so submitting once per epoch and then waiting
would forfeit most of the epoch's dividends. The verified vector is therefore
resubmitted every RESUBMIT_BLOCKS for the whole epoch, and a provider outage
falls back to the last vector this validator actually put on chain — the same
weights either way, but still counted as alive.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

import httpx

from . import chain as _chain
from .schedule import RESUBMIT_BLOCKS, pregate_skip, should_submit
from .state import (
    cached_weights, last_applied, last_submit_block, last_submit_ts,
    read_state, write_state,
)
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
        "network": env.get("ENGY_SN53_NETWORK", "finney"),
        "wallet": env.get("ENGY_SN53_WALLET", "default"),
        "wallet_hotkey": env.get("ENGY_SN53_WALLET_HOTKEY", "default"),
        "poll_s": int(env.get("ENGY_SN53_POLL_S", "300")),
        "resubmit_blocks": int(env.get("ENGY_SN53_RESUBMIT_BLOCKS",
                                       str(RESUBMIT_BLOCKS))),
        "state_file": state_file,
        "heartbeat_file": env.get(
            "ENGY_SN53_HEARTBEAT_FILE",
            os.path.join(os.path.dirname(state_file) or ".", "heartbeat.json")),
    }


# ── Liveness ─────────────────────────────────────────────────────
#
# `restart: unless-stopped` only recovers a process that *exits*. A loop wedged
# inside a chain call stays "running" forever while silently submitting
# nothing. The state file is no help as a liveness signal — it only advances on
# a successful submit, and most ticks legitimately skip because the resubmit
# interval is several polls long. So every tick stamps a heartbeat regardless
# of outcome, a watchdog thread turns a stall into an exit, and the container
# HEALTHCHECK reads the same file for `docker ps` visibility.

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
         chain=None) -> str:
    """Run one poll cycle, stamping the heartbeat however it turns out.

    The heartbeat records "the loop completed a cycle", not "weights were
    applied" — a skipped or failed tick is still proof of life, and most ticks
    now legitimately skip: the resubmit interval is longer than the poll.
    """
    try:
        return _run_tick(cfg, now=now, client=client, chain=chain)
    finally:
        _write_heartbeat(cfg["heartbeat_file"], now)


def _resolve_weights(cfg: dict, state: dict, *, now: float,
                     client: httpx.Client | None) -> tuple[list | None, int | None, bool, str | None]:
    """Pick the vector to submit: a freshly verified one, else the cache.

    Returns (weights, epoch_index, from_cache, failure). `failure` is a tick
    return code when nothing could be resolved, and None otherwise.

    The cache holds the last vector this validator actually put on chain, so
    falling back to it changes nothing about what miners receive — weights are
    constant within an epoch. It exists so a provider outage cannot take the
    validator off chain: stopping would keep the same weights on chain while
    costing us our own consensus membership.
    """
    failure = "fetch-failed"
    try:
        payload = fetch_weights(cfg["api"], client=client)
    except (httpx.HTTPError, ValueError) as e:
        print(f"[sync] fetch failed: {e}", flush=True)
    else:
        ok, reason, weights, idx = verify_payload(
            payload, master_hotkey=cfg["master_hotkey"], netuid=cfg["netuid"])
        if ok:
            return weights, idx, False, None
        print(f"[sync] payload rejected: {reason}", flush=True)
        failure = f"rejected:{reason}"

    applied = last_applied(state)
    cached = cached_weights(state)
    if cached is None or applied is None:
        return None, None, False, failure
    print(f"[sync] no usable payload ({failure}) — resubmitting cached epoch "
          f"{applied} vector ({len(cached)} hotkeys)", flush=True)
    return cached, applied, True, None


def _run_tick(cfg: dict, *, now: float, client: httpx.Client | None = None,
              chain=None) -> str:
    chain = chain or _chain
    state = read_state(cfg["state_file"])
    applied = last_applied(state)

    weights, epoch, from_cache, failure = _resolve_weights(
        cfg, state, now=now, client=client)
    if failure is not None:
        return failure

    is_new_epoch = applied is None or epoch > applied
    interval = cfg.get("resubmit_blocks", RESUBMIT_BLOCKS)
    if not is_new_epoch and pregate_skip(now=now, last_submit_ts=last_submit_ts(state),
                                         interval_blocks=interval):
        return "skipped:too-soon"

    try:
        view = chain.open_chain(network=cfg["network"], netuid=cfg["netuid"])
    except Exception as e:
        # Broad on purpose: connecting can fail in as many ways as bittensor
        # has dependencies. The type name distinguishes a real outage from a
        # local bug.
        print(f"[chain] open failed ({type(e).__name__}: {e})", flush=True)
        return "chain-failed"

    previous_block = last_submit_block(state)
    if not should_submit(epoch_index=epoch, last_applied=applied,
                         current_block=view.block, last_submit_block=previous_block,
                         now=now, last_submit_ts=last_submit_ts(state),
                         interval_blocks=interval):
        return "skipped:too-soon"

    dropped = chain.skipped_hotkeys(weights, view.hotkeys)
    if dropped:
        share = chain.dropped_weight_share(weights, view.hotkeys)
        print(f"[chain] {len(dropped)} payload hotkey(s) not registered on chain, "
              f"holding {share:.1%} of weight — dropped: {', '.join(dropped)}",
              flush=True)

    uids, ws = chain.resolve_uids(weights, view.hotkeys)
    if not uids or sum(ws) == 0:
        print("[chain] no payload hotkey is registered on chain; keeping last weights",
              flush=True)
        return "submit-failed"

    if not chain.set_weights(view, wallet=cfg["wallet"],
                             wallet_hotkey=cfg["wallet_hotkey"],
                             netuid=cfg["netuid"], uids=uids, ws=ws):
        return "submit-failed"

    # Only now, and all four together: a partial advance would make the next
    # tick believe it had already submitted.
    write_state(cfg["state_file"], last_applied=epoch, last_submit_block=view.block,
                last_submit_ts=now, cached_weights=weights)

    gap = ("?" if view.block is None or previous_block is None
           else view.block - previous_block)
    print(f"[sync] epoch {epoch}: submitted {len(uids)} uids "
          f"({gap} blocks since last submit)", flush=True)
    if from_cache:
        return "resubmitted:cached"
    return "applied" if is_new_epoch else "resubmitted"


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
