"""When to submit weights. Pure decisions — no IO, no chain, no clock reads.

The validator submits once per epoch and would then go silent for the rest of
it. Under Yuma consensus a validator whose `last_update` is older than the
subnet's `activity_cutoff` (~5000 blocks) counts as inactive: its weights drop
out of consensus and it earns nothing. On-chain weights do not expire — what
expires is the chain's evidence that this validator is alive. So the same
vector is resubmitted on a tempo-scale cadence for the whole epoch.

The block number is authoritative. Wall-clock time is used only to skip
opening a chain connection when it is clearly too early (`pregate_skip`), and
as a fallback when the block RPC is unavailable.

Failure directions are asymmetric: a missed submission costs consensus
membership, an extra one costs at most a `weights_rate_limit` rejection. Every
ambiguous case below therefore resolves toward submitting.
"""
from __future__ import annotations

# SN53's on-chain weights_rate_limit is 100 blocks (verified on finney), and
# the interval must clear it with room to spare rather than merely equal it.
# Our anchor is the block read *before* submitting, while the chain anchors on
# the block the extrinsic actually executed in — always the later of the two.
# At exactly 100 the first attempt of every cycle is refused as
# SettingWeightsTooFast, recovering only on the next poll: self-healing, but it
# burns an extrinsic and logs a failure every ~20 minutes, which is precisely
# the kind of recurring false alarm that hides a real one.
#
# The margin has to cover the *inclusion lag*, not the poll interval: poll
# granularity only ever delays a submission, and late is always safe. 120
# tolerates a 20-block (~4 min) lag, far beyond what wait_for_inclusion
# normally sees, and is still ~40x inside activity_cutoff (5000).
RESUBMIT_BLOCKS = 120      # ≈24 min at 12s blocks
BLOCK_S = 12               # nominal block time, for wall-clock fallback only
PREGATE_FRACTION = 0.8     # pre-gate skips below 80% of the interval

# The chain hyperparameter this interval must stay above. Kept as a named
# constant so the margin is asserted in tests rather than assumed.
WEIGHTS_RATE_LIMIT = 100


def should_submit(*, epoch_index: int, last_applied: int | None,
                  current_block: int | None, last_submit_block: int | None,
                  now: float, last_submit_ts: float | None,
                  interval_blocks: int = RESUBMIT_BLOCKS) -> bool:
    """Submit a new epoch at once; resubmit the current one every interval."""
    if last_applied is None or epoch_index > last_applied:
        return True
    if epoch_index < last_applied:
        return False       # defensive; freshness already rejects stale epochs
    if current_block is None or last_submit_block is None:
        return _clock_says_due(now=now, last_submit_ts=last_submit_ts,
                               interval_blocks=interval_blocks)
    return current_block - last_submit_block >= interval_blocks


def _clock_says_due(*, now: float, last_submit_ts: float | None,
                    interval_blocks: int) -> bool:
    """Wall-clock fallback for when the block number is unavailable."""
    if last_submit_ts is None:
        return True
    elapsed = now - last_submit_ts
    # elapsed < 0 means the clock jumped backward, not that we just submitted
    return elapsed < 0 or elapsed >= interval_blocks * BLOCK_S


def pregate_skip(*, now: float, last_submit_ts: float | None,
                 interval_blocks: int = RESUBMIT_BLOCKS) -> bool:
    """True when it is clearly too early to bother opening a chain connection.

    Only consulted for a resubmission of the epoch already applied — a new
    epoch always goes straight to the chain. Deliberately conservative: it may
    only skip ticks that `should_submit` would refuse anyway, so a
    misjudgement here can never delay a due submission.
    """
    if last_submit_ts is None:
        return False
    elapsed = now - last_submit_ts
    return 0 <= elapsed < interval_blocks * BLOCK_S * PREGATE_FRACTION
