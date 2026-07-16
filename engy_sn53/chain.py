"""set_weights for the light validator. bittensor is lazy-imported so tests
and the verify path run without the chain extra."""
from __future__ import annotations

U16 = 65535


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


def submit(cfg: dict, weights: list[list]) -> bool:
    try:
        import bittensor as bt
        sub = bt.subtensor(network=cfg["network"])
        meta = sub.metagraph(cfg["netuid"])
        uids, ws = resolve_uids(weights, list(meta.hotkeys))
        if not uids:
            print("[chain] no payload hotkey is registered on chain; keeping last weights",
                  flush=True)
            return False
        wallet = bt.wallet(name=cfg["wallet"], hotkey=cfg["wallet_hotkey"])
        ok = sub.set_weights(wallet=wallet, netuid=cfg["netuid"], uids=uids, weights=ws,
                             version_key=0, wait_for_inclusion=True)
        ok = ok[0] if isinstance(ok, tuple) else bool(ok)
        print(f"[chain] set_weights({len(uids)} uids) -> {ok}", flush=True)
        return ok
    except Exception as e:
        print(f"[chain] set_weights failed: {e}", flush=True)
        return False
