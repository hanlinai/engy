"""Chain access for the light validator: read the metagraph and block, submit
weights. Everything bittensor is lazy-imported so tests and the verify path
run without the chain extra.

Split into open/submit rather than one call because the loop needs the current
block number to decide *whether* to submit, and both reads must come from the
same connection. Arguments are explicit rather than a config blob so each
piece is independently testable and reusable outside this package.
"""
from __future__ import annotations

from dataclasses import dataclass

U16 = 65535

# The provider's configured owner_hotkey — where a burn pays out. Registered
# on netuid 53 at uid 229. Note this is deliberately NOT the chain's
# SubnetOwnerHotkey (5F2HTUq…, uid 161); it is the same key as the master
# hotkey that signs epoch results.
#
# A tripwire for a misconfigured owner_hotkey on the provider, which otherwise
# fails silently: a burn to an unregistered hotkey resolves to no uid, so the
# validator submits nothing and just looks idle.
#
# Warns, never rejects. v0.2.0 shipped the chain's SubnetOwnerHotkey here,
# inferred from chain state rather than confirmed against the provider's
# config — wrong, and a blocking check would have made that guess take every
# third-party validator offline on the first burn.
EXPECTED_OWNER_HOTKEY = "5DXSBCCKH5ENuyHFNaAvtaMfbhEEWpjSJB4rzc4mJfsc1uvJ"


def burn_target(weights: list[list]) -> str | None:
    """The sole recipient when this vector is a burn, else None.

    A burn is one hotkey holding all the weight (the provider emits
    `[[owner, 65535]]` when an epoch has no billed traffic). Zero-weight rows
    are ignored — a vector can carry scored-but-zero miners alongside it.
    """
    nonzero = [hk for hk, w in weights if w > 0]
    return nonzero[0] if len(nonzero) == 1 else None


@dataclass
class ChainView:
    """One connection's worth of chain state, taken at a single moment."""
    sub: object
    hotkeys: list[str]
    block: int | None


def _valid_block(b) -> int | None:
    """A block number, or None if the node returned something unusable.

    bool is excluded explicitly: it subclasses int, and True would otherwise
    read as block 1 and make every interval look enormous.
    """
    return b if isinstance(b, int) and not isinstance(b, bool) and b > 0 else None


def open_chain(*, network: str, netuid: int) -> ChainView:
    """Connect and read the metagraph and current block. Raises on failure."""
    import bittensor as bt
    sub = bt.Subtensor(network=network)
    meta = sub.metagraph(netuid)
    return ChainView(sub=sub, hotkeys=list(meta.hotkeys), block=_read_block(sub))


def _read_block(sub) -> int | None:
    """The block number is a scheduling nicety, not a prerequisite — a failure
    here degrades to the wall-clock fallback instead of losing the tick."""
    try:
        return _valid_block(sub.get_current_block())
    except Exception as e:
        print(f"[chain] get_current_block failed ({type(e).__name__}: {e}) — "
              f"falling back to wall clock", flush=True)
        return None


def resolve_uids(weights: list[list], hotkeys_on_chain: list[str]) -> tuple[list[int], list[int]]:
    uid_by_hotkey = {hk: uid for uid, hk in enumerate(hotkeys_on_chain)}
    present = sorted((uid_by_hotkey[hk], w) for hk, w in weights if hk in uid_by_hotkey)
    if not present:
        return [], []
    total = sum(w for _, w in present)
    if total == 0:
        return [uid for uid, _ in present], [0 for _ in present]
    scaled = [(uid, w * U16 // total) for uid, w in present]
    gap = U16 - sum(w for _, w in scaled)
    top = max(range(len(scaled)), key=lambda i: (scaled[i][1], -scaled[i][0]))
    scaled[top] = (scaled[top][0], scaled[top][1] + gap)
    return [uid for uid, _ in scaled], [w for _, w in scaled]


def skipped_hotkeys(weights: list[list], hotkeys_on_chain: list[str]) -> list[str]:
    registered = set(hotkeys_on_chain)
    return [hk for hk, _ in weights if hk not in registered]


def dropped_weight_share(weights: list[list], hotkeys_on_chain: list[str]) -> float:
    """Fraction of total weight held by hotkeys not registered on chain.

    Share, not count: dropping ten zero-weight strangers is nothing, dropping
    one hotkey holding 40% reshapes the whole distribution. This never blocks a
    submission — it is logged so an operator can see it happening.
    """
    total = sum(w for _, w in weights)
    if total == 0:
        return 0.0
    registered = set(hotkeys_on_chain)
    return sum(w for hk, w in weights if hk not in registered) / total


def _extrinsic_result(res) -> tuple[bool, str]:
    """Whether the chain accepted the extrinsic, and why not if it did not.

    bittensor 10.x returns an ExtrinsicResponse. It mimics a tuple —
    ``res[0]`` is the success flag, ``len(res)`` is 2 — but it is NOT a tuple
    instance and defines no ``__bool__``, so ``bool(res)`` is True for a
    REJECTED extrinsic just as surely as an accepted one.

    Reading it that way turns every rejection into a recorded success: state
    advances, the loop waits out a full resubmit interval, and no weights ever
    reach the chain while the validator looks perfectly healthy. Observed in
    production — `set_weights(1 uids) -> True` against a uid whose on-chain
    weights stayed empty and whose last_update never moved, because the hotkey
    had no validator permit.

    The message matters as much as the flag: "No validator permit" is a
    problem an operator can act on, where a bare False is not.
    """
    success = getattr(res, "success", None)
    if success is not None:
        return bool(success), str(getattr(res, "message", "") or "")
    if isinstance(res, tuple):                      # older SDKs: (bool, msg)
        return bool(res[0]), str(res[1]) if len(res) > 1 else ""
    return bool(res), ""


def set_weights(view: ChainView, *, wallet: str, wallet_hotkey: str, netuid: int,
                uids: list[int], ws: list[int]) -> bool:
    """Submit. False on any failure — the caller must not advance state."""
    try:
        import bittensor as bt
        wal = bt.Wallet(name=wallet, hotkey=wallet_hotkey)
        res = view.sub.set_weights(wallet=wal, netuid=netuid, uids=uids, weights=ws,
                                   version_key=0, wait_for_inclusion=True)
        ok, why = _extrinsic_result(res)
        print(f"[chain] set_weights({len(uids)} uids) -> {ok}"
              f"{f' ({why})' if why else ''}", flush=True)
        return ok
    except Exception as e:
        # Broad on purpose: bittensor raises too varied a set for narrow
        # catching to be robust. The type name keeps a local AttributeError
        # from reading as a chain outage.
        print(f"[chain] set_weights failed ({type(e).__name__}: {e})", flush=True)
        return False
