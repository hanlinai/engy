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

The same reasoning covers a payload that verifies but lands nothing, which is
what a provider pointing at unregistered hotkeys produces: it reaches the chain
step and would otherwise submit an empty vector, i.e. go silent for exactly the
reason above. `_standby_vector` degrades that to the cache, then to a burn.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from importlib.metadata import PackageNotFoundError, version as _pkg_version

import httpx

from . import chain as _chain
from . import heartbeat as _heartbeat
from .chain import U16, EXPECTED_OWNER_HOTKEY, burn_target
from .schedule import BLOCK_S, RESUBMIT_BLOCKS, pregate_skip, should_submit
from .state import (
    cached_digest, cached_weights, last_applied, last_submit_block,
    last_submit_ts, read_state, write_state,
)
from .sync import fetch_weights, verify_payload


# Protocol constants: identical for everyone on the subnet, so an operator
# should not have to know them. Both stay overridable — staging points at its
# own provider with its own master key — but set them as a pair: the master
# hotkey only signs for the provider that holds it, and a mismatch rejects
# every payload.
DEFAULT_API = "https://provider.engy.ai"
DEFAULT_MASTER_HOTKEY = "5DXSBCCKH5ENuyHFNaAvtaMfbhEEWpjSJB4rzc4mJfsc1uvJ"


def load_config() -> dict:
    env = os.environ
    # The wallet is the one thing that is genuinely this operator's. Defaulting
    # it to bittensor's "default" naming would let a misconfigured validator
    # sign with whatever wallet happens to carry that name — a different key
    # than intended, discovered only from on-chain behaviour.
    for req in ("ENGY_SN53_WALLET", "ENGY_SN53_WALLET_HOTKEY"):
        if not env.get(req):
            sys.exit(f"{req} is required (your bittensor wallet/hotkey name)")
    state_file = env.get("ENGY_SN53_STATE_FILE",
                         os.path.expanduser("~/.engy-sn53/state.json"))
    return {
        "api": env.get("ENGY_SN53_API") or DEFAULT_API,
        "master_hotkey": env.get("ENGY_SN53_MASTER_HOTKEY") or DEFAULT_MASTER_HOTKEY,
        "netuid": int(env.get("ENGY_SN53_NETUID", "53")),
        "network": env.get("ENGY_SN53_NETWORK", "finney"),
        "wallet": env["ENGY_SN53_WALLET"],
        "wallet_hotkey": env["ENGY_SN53_WALLET_HOTKEY"],
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
                     client: httpx.Client | None) -> tuple[list | None, int | None, str | None, bool, str | None]:
    """Pick the vector to submit: a freshly verified one, else the cache.

    Returns (weights, epoch_index, digest, from_cache, failure). `digest` is the
    master-signed digest of the chosen vector (the verified payload's, or the
    cached one on a fallback) — reported in the liveness heartbeat, never a
    submission input. `failure` is a tick return code when nothing could be
    resolved, and None otherwise.

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
            # payload["digest"] is verified equal to sha256(result_json) inside
            # verify_payload, so it is the digest of the bytes we accepted.
            return weights, idx, payload.get("digest"), False, None
        print(f"[sync] payload rejected: {reason}", flush=True)
        failure = f"rejected:{reason}"

    applied = last_applied(state)
    cached = cached_weights(state)
    if cached is None or applied is None:
        return None, None, None, False, failure
    print(f"[sync] no usable payload ({failure}) — resubmitting cached epoch "
          f"{applied} vector ({len(cached)} hotkeys)", flush=True)
    return cached, applied, cached_digest(state), True, None


def _lands(chain, weights: list | None, hotkeys_on_chain: list[str]) -> bool:
    """Whether this vector would put any non-zero weight on chain."""
    if not weights:
        return False
    uids, ws = chain.resolve_uids(weights, hotkeys_on_chain)
    return bool(uids) and sum(ws) > 0


def _standby_vector(chain, state: dict, hotkeys_on_chain: list[str], *,
                    from_cache: bool) -> tuple[list, str] | tuple[None, None]:
    """What to submit when the provider's vector lands nothing, and which kind.

    Submitting nothing is the one failure mode the resubmit machinery exists to
    prevent: once last_update passes activity_cutoff the chain treats this
    validator as inactive and drops its weights from consensus, so a provider
    that emits unregistered hotkeys would cost us the epoch's dividends on top
    of its own bug.

    The cache comes first — it is the last vector this validator actually
    landed, so resubmitting it keeps miners on the weights they already had.
    Only when there is no landable cache (nothing has ever been submitted, or
    its miners have since deregistered) do we burn, which changes the
    distribution and is therefore the last resort rather than the first.
    """
    if not from_cache:  # already tried, and it is what got us here
        cached = cached_weights(state)
        if _lands(chain, cached, hotkeys_on_chain):
            print(f"[chain] standby: resubmitting the last vector this "
                  f"validator landed ({len(cached)} hotkeys)", flush=True)
            return cached, "standby:cached"
    if EXPECTED_OWNER_HOTKEY in hotkeys_on_chain:
        print(f"[chain] standby: no landable cache — burning to "
              f"{EXPECTED_OWNER_HOTKEY} to stay inside activity_cutoff",
              flush=True)
        return [[EXPECTED_OWNER_HOTKEY, U16]], "standby:burn"
    return None, None


def _run_tick(cfg: dict, *, now: float, client: httpx.Client | None = None,
              chain=None) -> str:
    chain = chain or _chain
    state = read_state(cfg["state_file"])
    applied = last_applied(state)

    weights, epoch, digest, from_cache, failure = _resolve_weights(
        cfg, state, now=now, client=client)
    if failure is not None:
        return failure

    is_new_epoch = applied is None or epoch > applied
    interval = cfg.get("resubmit_blocks", RESUBMIT_BLOCKS)
    if not is_new_epoch and pregate_skip(now=now, last_submit_ts=last_submit_ts(state),
                                         interval_blocks=interval):
        # Most ticks land here. Say so: with a 300s poll and a ~24min interval,
        # four ticks in five legitimately skip, and a silent skip makes a
        # healthy validator indistinguishable from a wedged one in the log.
        due_in = interval * BLOCK_S - (now - (last_submit_ts(state) or now))
        print(f"[tick] epoch {epoch} already submitted; next resubmit in "
              f"~{max(0, due_in) / 60:.0f} min (not contacting the chain yet)",
              flush=True)
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
        waited = (view.block - previous_block
                  if view.block is not None and previous_block is not None else None)
        detail = (f"{interval - waited} more blocks" if waited is not None
                  else "waiting on the wall clock (block number unavailable)")
        print(f"[tick] epoch {epoch} already submitted at block {previous_block}; "
              f"resubmit in {detail}", flush=True)
        return "skipped:too-soon"

    target = burn_target(weights)
    if target is not None and target != EXPECTED_OWNER_HOTKEY:
        print(f"[chain] burn goes to {target}, expected owner "
              f"{EXPECTED_OWNER_HOTKEY} — submitting anyway; check the "
              f"provider's owner_hotkey", flush=True)

    dropped = chain.skipped_hotkeys(weights, view.hotkeys)
    if dropped:
        share = chain.dropped_weight_share(weights, view.hotkeys)
        print(f"[chain] {len(dropped)} payload hotkey(s) not registered on chain, "
              f"holding {share:.1%} of weight — dropped: {', '.join(dropped)}",
              flush=True)

    standby = None
    uids, ws = chain.resolve_uids(weights, view.hotkeys)
    if not uids or sum(ws) == 0:
        # Distinct from a chain error: the payload is fine, but none of it
        # lands. For a burn this is what a wrong provider owner_hotkey looks
        # like, so name that rather than leave a generic failure in the log.
        who = target or ", ".join(hk for hk, _ in weights[:5])
        print(f"[chain] none of the payload's hotkeys are registered on chain "
              f"({who}). For a burn this usually means the provider's "
              f"owner_hotkey is wrong.", flush=True)
        fallback, standby = _standby_vector(chain, state, view.hotkeys,
                                            from_cache=from_cache)
        if fallback is None:
            print("[chain] no standby vector available — submitting nothing, "
                  "keeping last weights", flush=True)
            return "submit-failed"
        uids, ws = chain.resolve_uids(fallback, view.hotkeys)

    if not chain.set_weights(view, wallet=cfg["wallet"],
                             wallet_hotkey=cfg["wallet_hotkey"],
                             netuid=cfg["netuid"], uids=uids, ws=ws):
        return "submit-failed"

    # Only now, and all four together: a partial advance would make the next
    # tick believe it had already submitted.
    #
    # A standby submission advances only the anchor. It marks the epoch seen so
    # the resubmit interval gates the next attempt, but leaves cached_weights
    # alone: the standby vector is a liveness stopgap, not this epoch's result,
    # and caching it would let a later provider outage resubmit it as if it had
    # been scored.
    write_state(cfg["state_file"], last_applied=epoch, last_submit_block=view.block,
                last_submit_ts=now,
                cached_weights=cached_weights(state) if standby else weights,
                cached_digest=cached_digest(state) if standby else digest)

    gap = ("?" if view.block is None or previous_block is None
           else view.block - previous_block)
    print(f"[sync] epoch {epoch}: submitted {len(uids)} uids "
          f"({gap} blocks since last submit)", flush=True)
    if standby:
        return standby
    if from_cache:
        return "resubmitted:cached"
    return "applied" if is_new_epoch else "resubmitted"


def _version() -> str:
    try:
        return f"engy-lv {_pkg_version('engy-sn53')}"
    except PackageNotFoundError:
        return "engy-lv unknown"


def emit_provider_heartbeat(cfg: dict, keypair, version: str) -> None:
    """Report liveness + the digest this validator is running on chain to the
    provider's /admin view. Reads what was last submitted from state, so it
    reflects on-chain truth (not merely what was just fetched). Best-effort:
    post_heartbeat never raises, and a None keypair (wallet unavailable) skips
    it entirely — the local heartbeat file remains the real watchdog signal."""
    if keypair is None:
        return
    state = read_state(cfg["state_file"])
    _heartbeat.post_heartbeat(
        cfg["api"], keypair, cfg["netuid"], version=version,
        synced_epoch=last_applied(state), digest=cached_digest(state))


def main() -> None:
    cfg = load_config()
    print(f"[engy-sn53-validator] api={cfg['api']} netuid={cfg['netuid']} "
          f"master={cfg['master_hotkey']}", flush=True)
    print(f"[health] heartbeat={cfg['heartbeat_file']} "
          f"stall_limit={stall_limit(cfg['poll_s'])}s", flush=True)
    _start_watchdog(cfg)
    version = _version()
    # The signing key is this validator's own wallet hotkey — loaded once, and
    # optional: a wallet that cannot be opened disables only the provider
    # heartbeat, never the submit loop.
    try:
        hb_keypair = _heartbeat.load_hotkey_keypair(cfg["wallet"], cfg["wallet_hotkey"])
    except Exception as e:
        print(f"[heartbeat] wallet load failed ({type(e).__name__}: {e}); "
              f"provider heartbeat disabled", flush=True)
        hb_keypair = None
    try:
        while True:
            try:
                tick(cfg, now=time.time())
            except Exception as e:
                print(f"[sync] tick error: {e}", flush=True)
            emit_provider_heartbeat(cfg, hb_keypair, version)
            time.sleep(cfg["poll_s"])
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
