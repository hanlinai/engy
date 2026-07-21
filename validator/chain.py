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
    sub = bt.subtensor(network=network)
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


def set_weights(view: ChainView, *, wallet: str, wallet_hotkey: str, netuid: int,
                uids: list[int], ws: list[int]) -> bool:
    """Submit. False on any failure — the caller must not advance state."""
    try:
        import bittensor as bt
        wal = bt.wallet(name=wallet, hotkey=wallet_hotkey)
        ok = view.sub.set_weights(wallet=wal, netuid=netuid, uids=uids, weights=ws,
                                  version_key=0, wait_for_inclusion=True)
        # newer SDKs return (bool, msg)
        ok = ok[0] if isinstance(ok, tuple) else bool(ok)
        print(f"[chain] set_weights({len(uids)} uids) -> {ok}", flush=True)
        return ok
    except Exception as e:
        # Broad on purpose: bittensor raises too varied a set for narrow
        # catching to be robust. The type name keeps a local AttributeError
        # from reading as a chain outage.
        print(f"[chain] set_weights failed ({type(e).__name__}: {e})", flush=True)
        return False
