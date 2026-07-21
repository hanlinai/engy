# Light validator: resubmit weights every 100 blocks

Date: 2026-07-21

## Background

The light validator submits weights **once per epoch** and then goes silent.
`verify_payload` rejects any epoch it has already applied
(`validator/sync.py:93`), so after one successful `set_weights` the loop does
nothing until the next epoch opens — a week in prod.

That breaks under Yuma consensus. A validator whose `last_update` is older
than the subnet's `activity_cutoff` (default 5000 blocks ≈ 16.7 h) is treated
as inactive: its weights drop out of consensus and it earns no dividends. With
a weekly epoch the validator is live for the first ~17 hours and dead for the
remaining six days. Shortening the epoch to a day would not fix it — 24 h
still exceeds the cutoff.

On-chain weights themselves do not expire. What expires is the chain's
evidence that this validator is still alive. The fix is therefore not to
compute anything new, but to **keep re-submitting the same vector** on a
tempo-scale cadence.

Confirm SN53's actual `activity_cutoff` and `weights_rate_limit` on chain
before rollout; the numbers above are Bittensor defaults.

## Design

### 1. Separate payload authenticity from submission scheduling

`verify_payload` currently mixes two questions: "is this payload genuine?" and
"should I submit right now?". The second one moves out.

`last_applied` is dropped from `verify_payload` entirely. It is redundant for
security: `validator/sync.py:91` already pins the accepted epoch to exactly
`current - 1`, so any older epoch is rejected as `stale-epoch` before the
`last_applied` check is ever reached. Its only live function today is
"don't submit twice", which is precisely the behaviour being reversed.

New signature:

```
verify_payload(payload, *, master_hotkey, netuid, genesis, now)
    -> (ok, reason, weights, epoch_index)
```

It checks version, netuid, digest binding, signature, `result_json` internal
cross-checks, weight well-formedness, and epoch freshness. Nothing else.

Replay protection remains covered by `tests/test_sync.py:132`
(`test_rejects_stale_epoch`).

### 2. Scheduling

State file gains two fields and the cached vector (§4), written atomically as
one document:

```json
{
  "last_applied": 12,
  "last_submit_block": 4823910,
  "last_submit_ts": 1784601234.5,
  "cached_weights": [["5F...", 4210], ...]
}
```

The decision is a pure function, no IO, exhaustively testable:

```python
RESUBMIT_BLOCKS = 100   # override: ENGY_SN53_RESUBMIT_BLOCKS

def should_submit(*, epoch_index, last_applied, current_block,
                  last_submit_block, interval_blocks):
    if last_applied is None or epoch_index > last_applied:
        return True                                   # new epoch — submit now
    if epoch_index < last_applied:
        return False                                  # rollback (defensive)
    return current_block - last_submit_block >= interval_blocks
```

Tick flow:

1. Fetch + verify payload. No chain contact.
2. **Wall-clock pre-gate**: same epoch and
   `now - last_submit_ts < interval_blocks * 12 * 0.8` (≈16 min) → skip
   without opening a chain connection. Pure cost saving; the test is
   conservative and can only skip early, never late.
3. Past the pre-gate, open one `bt.subtensor(network)` and take both
   `get_current_block()` and `metagraph(netuid)` from it.
4. `should_submit()` decides on the **block number**. Wall clock is used for
   the pre-gate and the RPC-failure fallback only — never as the authority.
5. Re-run `resolve_uids` against the freshly-pulled metagraph, log dropped
   hotkeys and their weight share, then `set_weights`.

If `get_current_block()` is unavailable or returns a non-positive / non-int
value, fall back to wall clock: submit when
`now - last_submit_ts >= interval_blocks * 12`. The failure directions are
asymmetric — a missed submission costs consensus membership, an extra one
costs at most one `weights_rate_limit` rejection.

**`resolve_uids` must be re-run on every submit, never cached.** The weight
vector is stable within an epoch but the hotkey→uid mapping is not: a
deregistered miner's uid is reassigned to a different hotkey. Mapping by
hotkey (already correct today) means a reassigned uid cannot inherit the old
hotkey's weight, but the mapping must be recomputed to stay accurate.

### 3. Poll interval 600s → 300s

`ENGY_SN53_POLL_S` drops to 300 (`validator/validator.py:36`), matching
engy-validator's `ENGY_SN53_WEIGHTS_POLL_S` (`engy_validator/config.py:87`)
and removing one difference to reconcile when the two are merged.

`POLL_S` must stay well below the resubmit interval or scheduling precision is
lost to poll granularity. At 300s vs 1200s the margin is 4×, so resubmits land
between blocks 100–125 — two orders of magnitude inside `activity_cutoff`.

`stall_limit` (`validator/validator.py:74`, `max(poll_s * 3, 900)`) evaluates
to 900s at the new poll rate, its floor value. The formula is unchanged.

Also update: `.env.validator.example:29` and the comment at lines 27-28, which
justifies slow polling with "the weight vector changes at most once per weekly
epoch". Poll now drives resubmit scheduling too, so that reasoning no longer
holds. Sync `docker/docker-compose.validator*.yml` if the value is set there.

### 4. Provider-outage fallback: cached vector

Submission depends on a verified payload, so a provider outage longer than
`activity_cutoff` would drop the validator out of consensus. This is a new
exposure created by the change: previously a multi-hour outage was harmless
because the validator only needed to submit once per epoch.

The last successfully submitted weight vector is cached in the state file.
When a fetch or verification fails and a cached vector exists, the cached
vector is resubmitted on the same 100-block cadence. Because weights are
constant within an epoch, resubmitting the cache is identical to resubmitting
a freshly fetched payload — no new semantics.

`cached_weights` is written atomically together with `last_applied`,
`last_submit_block` and `last_submit_ts`, and only after `set_weights`
succeeds (§6). It therefore always holds the vector belonging to
`last_applied` — the last thing this validator actually put on chain — and
needs no epoch field of its own. A tick whose payload verifies uses the fresh
vector directly; the cache is consulted only when fetch or verification
fails.

This also hardens epoch rollover. In the window after a new epoch opens but
before the provider publishes it, the served payload is rejected as
`stale-epoch`; the validator now falls back to the cache and keeps submitting
instead of going silent.

**Resubmission from cache is unbounded in time.** Halting protects nobody: the
chain retains the last submitted weights either way, so miners receive an
identical distribution whether we keep submitting or stop. The only effect of
stopping is that this validator leaves consensus and forfeits its own
dividends. A prolonged outage does freeze the distribution at the last good
epoch, which is a real risk — the mitigation is alerting and human
intervention, not automatic self-removal. Document this and alert on it.

### 5. Burn

No validator-side handling, by design.

Burn is settled provider-side: with no billed traffic, or all scores zero, the
full 65535 goes to the owner hotkey (`docs/SN53_ONE_PAGER.md:183-184`). It
reaches the validator as an ordinary row in the signed `result_json` —
`[owner_hotkey, 65535]` — which `resolve_uids` already normalizes correctly as
a single-entry vector. No special branch is needed.

The validator neither knows nor should know the owner hotkey.
`ENGY_SN53_OWNER_HOTKEY` is listed in engy-validator's `DEPRECATED_ENV`
(`engy_validator/config.py:14`) and its presence is a hard startup failure.
Letting a validator identify the owner and decide to burn locally would be an
unsigned local decision taken independently per validator — a consensus split.
Burn must arrive signed or not at all.

### 6. Error handling

**Core invariant: a failed submission advances no state field.**

This is the primary risk the new scheduling introduces. Advancing
`last_submit_block` on a failed `set_weights` would make the validator believe
it had just submitted and wait a full 100 blocks before retrying — amplifying
a transient chain error into 20 minutes of silence. `last_applied`,
`last_submit_block` and `last_submit_ts` are written together, only after
`set_weights` returns True. On failure nothing is written and the next poll
(300s) retries immediately. This gets a dedicated regression test.

**Atomic state writes.** `validator/validator.py:164-166` writes the state file
in place; a crash mid-write leaves truncated JSON. Adopt engy-validator's
tmp-file + `os.replace` (`engy_validator/weights_main.py:25-30`). `_last_applied`
tolerates a corrupt file by returning None, but that also discards
`last_submit_block` and the cached vector.

**Layered but still broad exception handling.** `validator/chain.py:49` keeps
its blanket `except Exception` — bittensor raises too varied a set for narrow
catching to be robust — but `open_chain` and `set_weights` get separate try
blocks so logs distinguish "cannot reach chain" from "chain rejected the
submission". Log `type(e).__name__`: without it an `AttributeError` reads as a
chain outage and misdirects debugging.

**No backoff.** A 300s retry is already gentle; exponential backoff would only
slow recovery.

Edge cases:

- `get_current_block()` returns a non-int or non-positive value → treat as
  unavailable, use the wall-clock fallback.
- Negative elapsed time in the pre-gate (backward clock jump) → do not skip.
  An extra chain connection is cheaper than silent stalling.
- `set_weights` hangs → heartbeat goes unstamped → watchdog `os._exit(1)` at
  900s → container restarts, reads back state, sees `last_submit_block`
  unadvanced, retries. If the extrinsic actually landed, the duplicate is
  refused by `weights_rate_limit` and the next poll recovers. Safe, because
  chain state is authoritative.

**Logging.** Every submission logs epoch, blocks elapsed since last submit, uid
count, and dropped hotkeys with their weight share. Dropped weight share
matters most: per §7 it never blocks a submission, so the log is the only
signal.

### 7. On-chain identity check

Before each submission the freshly-pulled metagraph is compared against the
hotkeys in the verified result. Hotkeys absent from the metagraph are logged
with their weight share, then dropped and the remainder renormalized — the
current behaviour. **The check never blocks a submission.**

Accepted trade-off: if a metagraph fetch ever returns a partial result, the
validator will concentrate weight onto the surviving hotkeys rather than
stopping. The log records it; someone has to read the log.

### 8. `chain.py` interface

Split and de-cfg-ified into explicit keyword arguments, both to make each piece
independently testable and to leave a reusable interface for engy-validator:

```
open_chain(*, network, netuid)          -> (subtensor, hotkeys, current_block)
resolve_uids(weights, hotkeys_on_chain) -> (uids, ws)          # unchanged
set_weights(subtensor, *, wallet, wallet_hotkey, netuid, uids, ws) -> bool
```

### 9. Tick return codes

`applied` / `resubmitted` / `resubmitted:cached` / `skipped:too-soon` /
`fetch-failed` / `rejected:<reason>` / `chain-failed` / `submit-failed`

## Testing

Pure functions, no chain, no IO:

- `should_submit` table: new epoch; same epoch at 99 and 100 blocks; epoch
  rollback; `current_block` None; `last_submit_block` None.
- Wall-clock pre-gate: either side of the threshold; negative elapsed.
- Dropped-hotkey weight-share computation.
- `resolve_uids`: existing tests unchanged.

Tick level, extending the existing `submit_fn` injection pattern to a fake
chain:

- New epoch → submits; all state fields advance together.
- Same epoch, interval not elapsed → no submission; state unchanged.
- Same epoch, 100 blocks elapsed → resubmits; `last_submit_block` advances,
  `last_applied` unchanged.
- **Submission fails → no state field changes; next tick retries immediately.**
- `open_chain` raises → `chain-failed`; state unchanged; heartbeat still
  stamped.
- Block number unavailable → wall-clock fallback path.
- Fetch fails with a cached vector present → `resubmitted:cached`.
- Fetch fails with no cache → no submission.

Existing tests to change:

- `tests/test_validator.py:68` — `rejected:already-applied` becomes
  `skipped:too-soon`.
- `tests/test_sync.py:142` — `test_rejects_already_applied` is deleted; the
  parameter no longer exists and its intent is covered by
  `test_rejects_stale_epoch`.

## Out of scope

**engy-validator gets no changes in this work.** Its `weights_main.py` has the
same once-per-epoch defect (`engy_validator/weights_main.py:62`), but the
scheduling code here is written in a reusable shape — explicit arguments, no
cfg blob, pure decision functions — so a later PR can point engy-validator at
this package rather than porting the logic twice.

**engy-validator's missing digest binding is tracked separately.**
`engy_validator/weights_main.py:38-41` verifies the master signature over the
server-supplied `result_digest` and never checks that `result_json` hashes to
it; line 71 then reads weights from that unverified `result_json`. A
compromised provider API could replay a genuine `(epoch_index, result_digest,
signature)` triple while substituting arbitrary weights. engy guards this at
`validator/sync.py:59-62`. This is more urgent than the scheduling work but is
a security fix on a different repo, and it cannot ship until
`sha256(result_json) == result_digest` is confirmed to hold against real
provider payloads — encoding drift (canonical JSON vs. re-serialized) would
make the check reject everything. Filed separately so it does not block or get
blocked by this change.
