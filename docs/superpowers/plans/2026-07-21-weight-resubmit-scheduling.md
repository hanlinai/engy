# Weight Resubmit Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the light validator resubmit its weight vector every ~100 blocks so it stays inside `activity_cutoff` and never drops out of Yuma consensus mid-epoch.

**Architecture:** Payload authenticity (`sync.py`) is separated from submission scheduling (new `schedule.py`, pure functions). Scheduling state moves to a new `state.py` with atomic writes. `chain.py` is split into `open_chain` / `resolve_uids` / `set_weights` with explicit keyword arguments so `validator.py` can read the chain block number *before* deciding to submit, and so engy-validator can reuse the same functions later. The last successfully submitted vector is cached so a provider outage cannot take the validator off chain.

**Tech Stack:** Python 3.11+, httpx, substrate-interface, bittensor (lazy-imported, `chain` extra), pytest.

**Spec:** `docs/superpowers/specs/2026-07-21-weight-resubmit-scheduling-design.md`

## Global Constraints

- Run tests with `.venv/bin/python -m pytest` — bare `python` is not on PATH and system `python3` lacks the deps.
- Everything `bittensor` stays lazy-imported inside function bodies. The verify path and all tests must run without the `chain` extra installed.
- `RESUBMIT_BLOCKS = 100`, `BLOCK_S = 12`, `PREGATE_FRACTION = 0.8`.
- **A failed submission advances no state field.** `last_applied`, `last_submit_block`, `last_submit_ts`, `cached_weights` are written together, atomically, only after `set_weights` returns True.
- Every broad `except Exception` logs `type(e).__name__` alongside the message.
- The validator never learns the owner hotkey. Burn arrives as an ordinary row in the signed `result_json`; no burn-specific branch anywhere.
- Failure directions are asymmetric: when a check is ambiguous, prefer submitting over staying silent.
- Baseline before starting: 45 tests pass.

---

## File Structure

**Create:**
- `validator/schedule.py` — pure scheduling decisions. No IO, no chain, no clock reads (`now` is always a parameter).
- `validator/state.py` — read/write the scheduling state file atomically, with per-field type validation.
- `tests/test_schedule.py`
- `tests/test_state.py`

**Modify:**
- `validator/sync.py` — `verify_payload` drops `last_applied`, returns `epoch_index`; `_well_formed_weights` becomes public.
- `validator/chain.py` — split into `ChainView` / `open_chain` / `set_weights`, de-cfg-ified; add `dropped_weight_share`.
- `validator/validator.py` — tick rewritten around the new flow; `_last_applied` removed (moves to `state.py`); `POLL_S` default 600 → 300.
- `tests/test_sync.py`, `tests/test_validator.py`, `tests/test_chain.py`
- `.env.validator.example`, `docker/docker-compose.validator*.yml`

---

### Task 1: `verify_payload` stops owning the resubmit decision

`last_applied` is redundant for security — `validator/sync.py:91` already pins the accepted epoch to exactly `current - 1`, so anything older is rejected as `stale-epoch` before the `last_applied` check is reached. Its only live function is "don't submit twice", which is the behaviour being reversed.

**Files:**
- Modify: `validator/sync.py:32-36` (make `_well_formed_weights` public), `validator/sync.py:39-96` (signature + return)
- Test: `tests/test_sync.py:47-51` (helper), `tests/test_sync.py:142-146` (delete)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `verify_payload(payload, *, master_hotkey: str, netuid: int, genesis: int, now: float) -> tuple[bool, str, list | None, int | None]` — returns `(ok, reason, weights, epoch_index)`. `weights` and `epoch_index` are non-None only when `ok`.
  - `well_formed_weights(w) -> bool` — public rename of `_well_formed_weights`.

- [ ] **Step 1: Update the test helper and delete the obsolete test**

In `tests/test_sync.py`, replace the `verify` helper (lines 47-51) with:

```python
def verify(p, **over):
    kw = dict(master_hotkey=MASTER.ss58_address, netuid=NETUID,
              genesis=GENESIS, now=NOW)
    kw.update(over)
    return verify_payload(p, **kw)
```

Delete `test_rejects_already_applied` entirely (lines 142-146). Its intent — replay protection — is covered by `test_rejects_stale_epoch` at line 132.

- [ ] **Step 2: Update every existing assertion to the 4-tuple**

Every `verify(...)` assertion in `tests/test_sync.py` compares against a 3-tuple. Rejections gain a trailing `None`; the success case gains the epoch index. Mechanically:

- `(False, "<reason>", None)` → `(False, "<reason>", None, None)`
- `test_accepts_a_genuine_fresh_payload` (line 59) — unpack four values and assert the index:

```python
def test_accepts_a_genuine_fresh_payload():
    ok, reason, weights, idx = verify(payload())
    assert (ok, reason, weights, idx) == (True, "ok", [["5Aaa", 65535]], IDX)
```

- `test_evil_top_level_weights_are_ignored_verified_weights_come_from_result_json` (line 65) and `test_accepts_well_formed_weights` (line 228) unpack four values the same way.

- [ ] **Step 3: Add a test pinning the new signature**

Append to `tests/test_sync.py`:

```python
def test_verify_payload_no_longer_takes_last_applied():
    # Scheduling moved out of verification: an epoch already on chain still
    # verifies, because deciding whether to resubmit is the tick's job now.
    import inspect
    assert "last_applied" not in inspect.signature(verify_payload).parameters
    ok, _, _, idx = verify(payload())
    assert ok and idx == IDX
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sync.py -q`
Expected: FAIL — `TypeError: verify_payload() got an unexpected keyword argument` is gone but assertions compare 3-tuples to 4-tuples, plus `test_verify_payload_no_longer_takes_last_applied` fails on the `last_applied` parameter still being present.

- [ ] **Step 5: Make `_well_formed_weights` public**

In `validator/sync.py`, rename lines 32-36:

```python
def well_formed_weights(w) -> bool:
    return (isinstance(w, list) and
            all(isinstance(p, list) and len(p) == 2 and isinstance(p[0], str)
                and isinstance(p[1], int) and not isinstance(p[1], bool)
                and 0 <= p[1] <= 65535 for p in w))
```

Update its one call site at line 87 from `_well_formed_weights(...)` to `well_formed_weights(...)`.

- [ ] **Step 6: Change the signature and return type**

In `validator/sync.py`, change the `verify_payload` definition (line 39) to:

```python
def verify_payload(payload: dict, *, master_hotkey: str, netuid: int, genesis: int,
                   now: float) -> tuple[bool, str, list | None, int | None]:
    """Verify a fetched payload and return (ok, reason, weights, epoch_index).

    `weights` (only set when ok) is the weight vector extracted from the
    VERIFIED `result_json` bytes — never the top-level `payload["weights"]`
    field, which is display metadata a compromised coordination layer could
    forge independently of the signature.

    Scheduling is NOT decided here. Whether an already-applied epoch should be
    resubmitted is the tick's decision (validator/schedule.py); this function
    answers only "is this payload genuine and fresh?". Replay protection is
    unaffected: the epoch is pinned to exactly `current - 1` below, so an older
    epoch is rejected as stale before anything else can look at it.
    """
```

Add `None` as a fourth element to every `return False, "<reason>", None` in the body — lines 49, 51, 53, 57, 62, 76, 78, 83, 88, 92. Then delete the `last_applied` check (lines 93-94) entirely. Change the final return (line 96) to:

```python
    return True, "ok", weights, idx
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sync.py -q`
Expected: PASS (one test fewer than before — `test_rejects_already_applied` is gone, `test_verify_payload_no_longer_takes_last_applied` is new).

`tests/test_validator.py` will fail at this point — Task 5 fixes it. Confirm the failure is confined to that file:

Run: `.venv/bin/python -m pytest tests/ -q 2>&1 | tail -5`
Expected: failures only in `tests/test_validator.py`.

- [ ] **Step 8: Commit**

```bash
git add validator/sync.py tests/test_sync.py
git commit -m "refactor(sync): verify_payload returns epoch_index, drops last_applied

Verification answers 'is this payload genuine' only; whether an
already-applied epoch should be resubmitted becomes the tick's decision.
Replay protection is unchanged - the epoch is already pinned to exactly
current-1, so last_applied was redundant."
```

---

### Task 2: `validator/state.py` — atomic scheduling state

`validator/validator.py:164-166` writes the state file in place; a crash mid-write leaves truncated JSON. The file also grows from one field to four.

**Files:**
- Create: `validator/state.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Consumes: `well_formed_weights` from Task 1.
- Produces:
  - `read_state(path: str) -> dict` — `{}` on missing/corrupt/non-dict.
  - `last_applied(state: dict) -> int | None`
  - `last_submit_block(state: dict) -> int | None`
  - `last_submit_ts(state: dict) -> float | None`
  - `cached_weights(state: dict) -> list | None`
  - `write_state(path: str, *, last_applied: int, last_submit_block: int | None, last_submit_ts: float, cached_weights: list) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_state.py`:

```python
import json

from validator.state import (
    read_state, write_state, last_applied, last_submit_block, last_submit_ts,
    cached_weights,
)

WEIGHTS = [["5Aaa", 40000], ["5Bbb", 25535]]


def test_round_trips_every_field(tmp_path):
    p = str(tmp_path / "state.json")
    write_state(p, last_applied=12, last_submit_block=4823910,
                last_submit_ts=1784601234.5, cached_weights=WEIGHTS)
    s = read_state(p)
    assert last_applied(s) == 12
    assert last_submit_block(s) == 4823910
    assert last_submit_ts(s) == 1784601234.5
    assert cached_weights(s) == WEIGHTS


def test_write_is_atomic_leaving_no_temp_file(tmp_path):
    p = str(tmp_path / "state.json")
    write_state(p, last_applied=1, last_submit_block=2, last_submit_ts=3.0,
                cached_weights=WEIGHTS)
    assert [f.name for f in tmp_path.iterdir()] == ["state.json"]


def test_write_creates_missing_parent_directory(tmp_path):
    p = str(tmp_path / "deep" / "nested" / "state.json")
    write_state(p, last_applied=1, last_submit_block=2, last_submit_ts=3.0,
                cached_weights=WEIGHTS)
    assert last_applied(read_state(p)) == 1


def test_a_truncated_file_reads_as_empty_not_raises(tmp_path):
    p = tmp_path / "state.json"
    p.write_text('{"last_applied": 12, "cached_w')
    s = read_state(str(p))
    assert s == {}
    assert last_applied(s) is None and cached_weights(s) is None


def test_missing_file_reads_as_empty(tmp_path):
    assert read_state(str(tmp_path / "nope.json")) == {}


def test_non_dict_top_level_reads_as_empty(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps([1, 2, 3]))
    assert read_state(str(p)) == {}


def test_each_field_validates_its_own_type_independently(tmp_path):
    # One poisoned field must not discard the others: losing last_submit_block
    # to a bad value should still leave last_applied usable.
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"last_applied": 12, "last_submit_block": "nope",
                             "last_submit_ts": None, "cached_weights": WEIGHTS}))
    s = read_state(str(p))
    assert last_applied(s) == 12
    assert last_submit_block(s) is None
    assert last_submit_ts(s) is None
    assert cached_weights(s) == WEIGHTS


def test_bools_are_not_accepted_as_ints(tmp_path):
    # bool subclasses int; True must not read as epoch 1
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"last_applied": True, "last_submit_block": False}))
    s = read_state(str(p))
    assert last_applied(s) is None and last_submit_block(s) is None


def test_malformed_cached_weights_are_rejected(tmp_path):
    p = tmp_path / "state.json"
    for bad in ("nope", [["5Aaa"]], [["5Aaa", 70000]], [["5Aaa", "40000"]], [[1, 2]]):
        p.write_text(json.dumps({"cached_weights": bad}))
        assert cached_weights(read_state(str(p))) is None, bad


def test_ints_are_accepted_for_last_submit_ts(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"last_submit_ts": 1784601234}))
    assert last_submit_ts(read_state(str(p))) == 1784601234.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_state.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'validator.state'`

- [ ] **Step 3: Write the implementation**

Create `validator/state.py`:

```python
"""Scheduling state, persisted atomically.

The file records what this validator last put on chain: which epoch, at which
block, at what wall-clock time, and the exact vector. All four move together
or not at all — a partial advance would make the loop believe it had submitted
when it had not, and then wait a full resubmit interval before retrying.

Every field validates independently on read. A single poisoned field must not
discard the rest; the loop degrades further with each field it loses, and
losing all of them only costs one redundant submission.
"""
from __future__ import annotations

import json
import os

from .sync import well_formed_weights


def read_state(path: str) -> dict:
    """The state file as a dict; {} if missing, unreadable, or not a JSON object."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _int(v) -> int | None:
    # bool subclasses int — True must never read as epoch 1
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def last_applied(state: dict) -> int | None:
    return _int(state.get("last_applied"))


def last_submit_block(state: dict) -> int | None:
    return _int(state.get("last_submit_block"))


def last_submit_ts(state: dict) -> float | None:
    v = state.get("last_submit_ts")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def cached_weights(state: dict) -> list | None:
    """The last successfully submitted vector, or None if absent/malformed.

    Validated with the same predicate as a freshly verified payload: this
    vector goes straight to the chain during a provider outage, so a corrupt
    file must not become a corrupt submission.
    """
    w = state.get("cached_weights")
    return w if well_formed_weights(w) else None


def write_state(path: str, *, last_applied: int, last_submit_block: int | None,
                last_submit_ts: float, cached_weights: list) -> None:
    """Replace the state file atomically. Call only after a successful submit."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"last_applied": last_applied,
                   "last_submit_block": last_submit_block,
                   "last_submit_ts": last_submit_ts,
                   "cached_weights": cached_weights}, f)
    os.replace(tmp, path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_state.py -q`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add validator/state.py tests/test_state.py
git commit -m "feat(state): atomic scheduling state with per-field validation

Four fields advance together or not at all. tmp+os.replace so a crash
mid-write cannot leave truncated JSON, and each field validates on read
so one poisoned value does not discard the others."
```

---

### Task 3: `validator/schedule.py` — pure scheduling decisions

**Files:**
- Create: `validator/schedule.py`
- Test: `tests/test_schedule.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `RESUBMIT_BLOCKS = 100`, `BLOCK_S = 12`, `PREGATE_FRACTION = 0.8`
  - `should_submit(*, epoch_index: int, last_applied: int | None, current_block: int | None, last_submit_block: int | None, now: float, last_submit_ts: float | None, interval_blocks: int = RESUBMIT_BLOCKS) -> bool`
  - `pregate_skip(*, now: float, last_submit_ts: float | None, interval_blocks: int = RESUBMIT_BLOCKS) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schedule.py`:

```python
from validator.schedule import (
    should_submit, pregate_skip, RESUBMIT_BLOCKS, BLOCK_S, PREGATE_FRACTION,
)

TS = 1784601234.0
INTERVAL_S = RESUBMIT_BLOCKS * BLOCK_S          # 1200
PREGATE_S = INTERVAL_S * PREGATE_FRACTION       # 960


def sub(**over):
    kw = dict(epoch_index=12, last_applied=12, current_block=5000,
              last_submit_block=4900, now=TS, last_submit_ts=TS)
    kw.update(over)
    return should_submit(**kw)


def test_a_new_epoch_submits_immediately_regardless_of_block_distance():
    # Fresh results must not wait out a resubmit interval.
    assert sub(epoch_index=13, last_applied=12, current_block=4901,
               last_submit_block=4900) is True


def test_the_very_first_submission_has_no_history_to_compare():
    assert sub(last_applied=None, last_submit_block=None,
               last_submit_ts=None) is True


def test_same_epoch_resubmits_exactly_at_the_interval():
    assert sub(current_block=4900 + RESUBMIT_BLOCKS) is True


def test_same_epoch_holds_one_block_short_of_the_interval():
    assert sub(current_block=4900 + RESUBMIT_BLOCKS - 1) is False


def test_same_epoch_resubmits_past_the_interval():
    assert sub(current_block=4900 + RESUBMIT_BLOCKS + 50) is True


def test_an_epoch_older_than_what_we_applied_never_submits():
    # Defensive: verify_payload's freshness check already rejects these.
    assert sub(epoch_index=11, last_applied=12) is False


def test_missing_block_number_falls_back_to_the_wall_clock():
    # RPC unavailable: submit once the interval has elapsed in seconds.
    assert sub(current_block=None, now=TS + INTERVAL_S) is True
    assert sub(current_block=None, now=TS + INTERVAL_S - 1) is False


def test_missing_last_submit_block_also_falls_back_to_the_wall_clock():
    assert sub(last_submit_block=None, now=TS + INTERVAL_S) is True
    assert sub(last_submit_block=None, now=TS + INTERVAL_S - 1) is False


def test_fallback_with_no_timestamp_at_all_submits():
    # Nothing to compare against; submitting costs at most a rate-limit
    # rejection, staying silent costs consensus membership.
    assert sub(current_block=None, last_submit_ts=None) is True


def test_a_backward_clock_jump_submits_rather_than_stalling():
    # now < last_submit_ts: the clock moved, not the chain. Do not let a
    # negative elapsed time freeze submission until the clock catches up.
    assert sub(current_block=None, now=TS - 10_000) is True


def test_block_number_wins_over_the_wall_clock_when_both_are_available():
    # Blocks say not yet, clock says long overdue → blocks are authoritative.
    assert sub(current_block=4901, now=TS + 10 * INTERVAL_S) is False


# ── pre-gate (avoids opening a chain connection when clearly too early) ──

def test_pregate_skips_well_inside_the_interval():
    assert pregate_skip(now=TS + PREGATE_S - 1, last_submit_ts=TS) is True


def test_pregate_does_not_skip_at_the_threshold():
    assert pregate_skip(now=TS + PREGATE_S, last_submit_ts=TS) is False


def test_pregate_never_skips_without_a_timestamp():
    assert pregate_skip(now=TS, last_submit_ts=None) is False


def test_pregate_never_skips_on_a_backward_clock_jump():
    # An extra chain connection is cheaper than a silent stall.
    assert pregate_skip(now=TS - 10_000, last_submit_ts=TS) is False


def test_pregate_is_strictly_more_conservative_than_should_submit():
    # The pre-gate may only skip ticks that should_submit would refuse anyway;
    # if it ever skipped a due submission it would silently delay by one poll.
    for offset in range(0, INTERVAL_S + 60, 30):
        now = TS + offset
        if pregate_skip(now=now, last_submit_ts=TS):
            assert should_submit(epoch_index=12, last_applied=12,
                                 current_block=None, last_submit_block=None,
                                 now=now, last_submit_ts=TS) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_schedule.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'validator.schedule'`

- [ ] **Step 3: Write the implementation**

Create `validator/schedule.py`:

```python
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

RESUBMIT_BLOCKS = 100      # ≈20 min at 12s blocks, far inside activity_cutoff
BLOCK_S = 12               # nominal block time, for wall-clock fallback only
PREGATE_FRACTION = 0.8     # pre-gate skips below 80% of the interval


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_schedule.py -q`
Expected: PASS (16 tests)

- [ ] **Step 5: Commit**

```bash
git add validator/schedule.py tests/test_schedule.py
git commit -m "feat(schedule): pure resubmit-cadence decisions

Block number is authoritative; wall clock only pre-gates the chain
connection and covers RPC failure. Every ambiguous case resolves toward
submitting - a missed submission costs consensus membership, an extra
one costs at most a rate-limit rejection."
```

---

### Task 4: Split `chain.py` and report dropped weight share

`validator.py` needs the block number *before* deciding whether to submit, so the single `submit(cfg, weights)` entry point has to become two calls sharing one connection. The cfg dict goes away in favour of explicit keyword arguments — both to make each piece independently testable and to leave an interface engy-validator can reuse without adopting engy's config shape.

**Files:**
- Modify: `validator/chain.py` (whole file)
- Test: `tests/test_chain.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ChainView` dataclass with fields `sub: object`, `hotkeys: list[str]`, `block: int | None`
  - `open_chain(*, network: str, netuid: int) -> ChainView` — raises on failure
  - `resolve_uids(weights: list[list], hotkeys_on_chain: list[str]) -> tuple[list[int], list[int]]` — unchanged
  - `skipped_hotkeys(weights: list[list], hotkeys_on_chain: list[str]) -> list[str]` — unchanged
  - `dropped_weight_share(weights: list[list], hotkeys_on_chain: list[str]) -> float`
  - `set_weights(view: ChainView, *, wallet: str, wallet_hotkey: str, netuid: int, uids: list[int], ws: list[int]) -> bool` — returns False on any failure
- Removed: `submit(cfg, weights)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chain.py`:

```python
from validator.chain import ChainView, dropped_weight_share, _valid_block


def test_dropped_weight_share_is_by_weight_not_by_count():
    # Ten zero-weight strangers matter far less than one heavy one.
    weights = [["5Aaa", 60000], ["5Gone", 5535]]
    assert dropped_weight_share(weights, ["5Aaa"]) == 5535 / 65535


def test_dropped_weight_share_is_zero_when_everyone_is_registered():
    assert dropped_weight_share([["5Aaa", 65535]], ["5Aaa", "5Bbb"]) == 0.0


def test_dropped_weight_share_is_one_when_nobody_is_registered():
    assert dropped_weight_share([["5Aaa", 65535]], ["5Zzz"]) == 1.0


def test_dropped_weight_share_of_an_all_zero_vector_does_not_divide_by_zero():
    assert dropped_weight_share([["5Aaa", 0], ["5Bbb", 0]], []) == 0.0


def test_dropped_weight_share_of_an_empty_vector_is_zero():
    assert dropped_weight_share([], ["5Aaa"]) == 0.0


def test_a_burn_vector_resolves_to_the_owner_alone():
    # Burn arrives as an ordinary single row; no special-casing anywhere.
    uids, ws = resolve_uids([["5Owner", 65535]], ["5Aaa", "5Owner", "5Bbb"])
    assert (uids, ws) == ([1], [65535])
    assert dropped_weight_share([["5Owner", 65535]], ["5Aaa", "5Owner"]) == 0.0


def test_chain_view_carries_a_possibly_absent_block_number():
    v = ChainView(sub=object(), hotkeys=["5Aaa"], block=None)
    assert v.block is None and v.hotkeys == ["5Aaa"]


def test_only_positive_ints_count_as_a_block_number():
    # A bad RPC reply must degrade to the wall-clock fallback, not poison
    # the block arithmetic.
    assert _valid_block(4823910) == 4823910
    for bad in (None, 0, -1, True, False, 1.5, "4823910"):
        assert _valid_block(bad) is None, bad
```

Note: `resolve_uids` is already imported at the top of `tests/test_chain.py`; add the new names to that existing import rather than duplicating it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_chain.py -q`
Expected: FAIL — `ImportError: cannot import name 'ChainView' from 'validator.chain'`

- [ ] **Step 3: Rewrite `validator/chain.py`**

Replace the whole file:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_chain.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add validator/chain.py tests/test_chain.py
git commit -m "refactor(chain): split open/submit, add dropped weight share

The loop needs the block number to decide whether to submit, and both
reads must come from one connection. Explicit kwargs replace the cfg
blob so the pieces are testable and reusable. Dropped hotkeys are now
reported by weight share, which is what actually matters."
```

---

### Task 5: Rewrite the tick around the new flow

**Files:**
- Modify: `validator/validator.py:44-51` (delete `_last_applied`), `validator/validator.py:131-169` (tick), imports at `validator/validator.py:9-18`
- Test: `tests/test_validator.py`

**Interfaces:**
- Consumes: Tasks 1-4 — `verify_payload` (4-tuple), `read_state`/`write_state`/accessors, `should_submit`/`pregate_skip`, `open_chain`/`set_weights`/`resolve_uids`/`skipped_hotkeys`/`dropped_weight_share`.
- Produces: `tick(cfg, *, now: float, client: httpx.Client | None = None, chain=None) -> str`. `chain` is any object exposing the `validator.chain` module's functions; it defaults to that module. Return codes: `applied`, `resubmitted`, `resubmitted:cached`, `skipped:too-soon`, `fetch-failed`, `rejected:<reason>`, `chain-failed`, `submit-failed`.

Task 6 adds the cached-fallback path; this task leaves `_resolve_weights` returning the failure code when no fresh payload verifies.

- [ ] **Step 1: Replace the test fixtures**

`submit_fn` injection cannot express a block number, so tests inject a fake chain instead. In `tests/test_validator.py`, replace the imports at lines 9-12 with:

```python
from validator import chain as chain_mod
from validator.state import (
    read_state, last_applied, last_submit_block, cached_weights,
)
from validator.sync import epoch_message, MAX_RESPONSE_BYTES
from validator.validator import (
    tick, _heartbeat_age, _watchdog_check, stall_limit, healthcheck,
)
```

Add after the `_cfg` helper (line 50):

```python
class FakeChain:
    """Stands in for validator.chain, recording what reached the chain."""

    def __init__(self, hotkeys=("5Aaa",), block=5000, ok=True, open_raises=False):
        self.hotkeys = list(hotkeys)
        self.block = block
        self.ok = ok
        self.open_raises = open_raises
        self.submitted = []
        self.opens = 0

    def open_chain(self, *, network, netuid):
        self.opens += 1
        if self.open_raises:
            raise RuntimeError("subtensor unreachable")
        return chain_mod.ChainView(sub=object(), hotkeys=self.hotkeys,
                                   block=self.block)

    resolve_uids = staticmethod(chain_mod.resolve_uids)
    skipped_hotkeys = staticmethod(chain_mod.skipped_hotkeys)
    dropped_weight_share = staticmethod(chain_mod.dropped_weight_share)

    def set_weights(self, view, *, wallet, wallet_hotkey, netuid, uids, ws):
        self.submitted.append((uids, ws))
        return self.ok
```

Change `_cfg` to use the new poll default so the fixtures match production (line 48): `"poll_s": 300,`.

- [ ] **Step 2: Rewrite the affected tests**

Replace `test_applied_then_deduplicated` (lines 58-69) with:

```python
def test_new_epoch_applies_and_records_every_state_field(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    assert tick(cfg, now=NOW, client=_client(_payload()), chain=fake) == "applied"
    assert fake.submitted == [([0], [65535])]
    s = read_state(cfg["state_file"])
    assert last_applied(s) == IDX
    assert last_submit_block(s) == 5000
    assert cached_weights(s) == [["5Aaa", 65535]]


def test_same_epoch_holds_until_the_resubmit_interval(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    tick(cfg, now=NOW, client=_client(_payload()), chain=fake)
    # 99 blocks later, past the pre-gate but short of the interval
    fake.block = 5099
    out = tick(cfg, now=NOW + 1000, client=_client(_payload()), chain=fake)
    assert out == "skipped:too-soon"
    assert len(fake.submitted) == 1


def test_same_epoch_resubmits_at_100_blocks(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    tick(cfg, now=NOW, client=_client(_payload()), chain=fake)
    fake.block = 5100
    out = tick(cfg, now=NOW + 1200, client=_client(_payload()), chain=fake)
    assert out == "resubmitted"
    assert fake.submitted == [([0], [65535]), ([0], [65535])]
    s = read_state(cfg["state_file"])
    assert last_submit_block(s) == 5100      # advances
    assert last_applied(s) == IDX            # unchanged — same epoch


def test_the_pregate_avoids_opening_a_connection_when_clearly_too_early(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    tick(cfg, now=NOW, client=_client(_payload()), chain=fake)
    assert fake.opens == 1
    out = tick(cfg, now=NOW + 300, client=_client(_payload()), chain=fake)
    assert out == "skipped:too-soon"
    assert fake.opens == 1                   # no second connection
```

Replace `test_submit_failure_does_not_advance_state` (lines 102-108) with:

```python
def test_submit_failure_advances_no_state_field(tmp_path):
    # The invariant that makes the whole schedule safe: advancing
    # last_submit_block on a failure would make the loop believe it had just
    # submitted and wait a full interval before retrying, amplifying a
    # transient chain error into 20 minutes of silence.
    cfg = _cfg(tmp_path)
    fake = FakeChain(ok=False)
    assert tick(cfg, now=NOW, client=_client(_payload()), chain=fake) == "submit-failed"
    assert read_state(cfg["state_file"]) == {}

    fake.ok = True
    assert tick(cfg, now=NOW + 1, client=_client(_payload()), chain=fake) == "applied"


def test_a_failed_resubmit_retries_on_the_next_poll_not_a_full_interval_later(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    tick(cfg, now=NOW, client=_client(_payload()), chain=fake)

    fake.block, fake.ok = 5100, False
    assert tick(cfg, now=NOW + 1200, client=_client(_payload()),
                chain=fake) == "submit-failed"
    assert last_submit_block(read_state(cfg["state_file"])) == 5000  # not advanced

    # one poll later (300s, 25 blocks) the retry goes through
    fake.block, fake.ok = 5125, True
    assert tick(cfg, now=NOW + 1500, client=_client(_payload()),
                chain=fake) == "resubmitted"
```

Add:

```python
def test_an_unreachable_chain_is_contained_and_leaves_state_alone(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(open_raises=True)
    assert tick(cfg, now=NOW, client=_client(_payload()), chain=fake) == "chain-failed"
    assert read_state(cfg["state_file"]) == {}
    assert _heartbeat_age(cfg["heartbeat_file"], now=NOW) == 0.0


def test_a_missing_block_number_falls_back_to_the_wall_clock(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=None)
    assert tick(cfg, now=NOW, client=_client(_payload()), chain=fake) == "applied"
    assert last_submit_block(read_state(cfg["state_file"])) is None
    # 1199s later: still short of the 1200s fallback interval
    assert tick(cfg, now=NOW + 1199, client=_client(_payload()),
                chain=fake) == "skipped:too-soon"
    assert tick(cfg, now=NOW + 1200, client=_client(_payload()),
                chain=fake) == "resubmitted"


def test_an_unregistered_hotkey_is_dropped_and_logged_never_blocking(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    fake = FakeChain(hotkeys=["5Aaa"])
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=_payload_with(
            [["5Aaa", 40000], ["5Gone", 25535]]))))
    assert tick(cfg, now=NOW, client=client, chain=fake) == "applied"
    assert fake.submitted == [([0], [65535])]        # renormalized onto 5Aaa
    out = capsys.readouterr().out
    assert "5Gone" in out and "39.0%" in out
```

Add the payload helper next to `_payload` (after line 42):

```python
def _payload_with(weights):
    """A genuine payload carrying an arbitrary verified weight vector."""
    rj = json.dumps({"epoch_end": END, "epoch_index": IDX,
                     "epoch_start": END - 604800, "miners": [], "netuid": 53,
                     "params": {}, "v": 1, "weights": weights},
                    sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(rj.encode("utf-8")).hexdigest()
    return {"v": 1, "netuid": 53, "epoch_index": IDX,
            "epoch_start": END - 604800, "epoch_end": END, "digest": digest,
            "result_json": rj, "weights": weights,
            "signed_hotkey": MASTER.ss58_address,
            "signature": MASTER.sign(epoch_message(53, IDX, digest).encode()).hex(),
            "signed_at": END + 610}
```

In the remaining tests, replace every `submit_fn=lambda c, w: ...` with `chain=FakeChain()` (failure cases: `chain=FakeChain(ok=False)`). Affected: lines 79, 90-91, 98-99, 113, 122, 154, 158, 163, 169, 192. Delete `test_last_applied_survives_corrupt_state_file` (lines 126-142) — `_last_applied` no longer exists and `tests/test_state.py` covers its behaviour.

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_validator.py -q`
Expected: FAIL — `tick() got an unexpected keyword argument 'chain'`

- [ ] **Step 4: Rewrite the tick**

In `validator/validator.py`, replace the imports at lines 9-18 with:

```python
import json
import os
import sys
import threading
import time

import httpx

from . import chain as _chain
from .schedule import pregate_skip, should_submit
from .state import (
    cached_weights, last_applied, last_submit_block, last_submit_ts,
    read_state, write_state,
)
from .sync import fetch_weights, verify_payload
```

Delete `_last_applied` (lines 44-51) — `state.py` owns this now.

Replace `tick` and `_run_tick` (lines 131-169) with:

```python
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
                     client: httpx.Client | None) -> tuple[list | None, int | None, str | None]:
    """Pick the vector to submit: a freshly verified one, else the cache.

    Returns (weights, epoch_index, failure). `failure` is a tick return code
    when nothing could be resolved, and None otherwise. Task 6 adds the cache
    branch; for now a failed fetch or rejected payload ends the tick.
    """
    try:
        payload = fetch_weights(cfg["api"], client=client)
    except (httpx.HTTPError, ValueError) as e:
        print(f"[sync] fetch failed: {e}", flush=True)
        return None, None, "fetch-failed"

    ok, reason, weights, idx = verify_payload(
        payload, master_hotkey=cfg["master_hotkey"], netuid=cfg["netuid"],
        genesis=cfg["genesis"], now=now)
    if not ok:
        print(f"[sync] payload rejected: {reason}", flush=True)
        return None, None, f"rejected:{reason}"
    return weights, idx, None


def _run_tick(cfg: dict, *, now: float, client: httpx.Client | None = None,
              chain=None) -> str:
    chain = chain or _chain
    state = read_state(cfg["state_file"])
    applied = last_applied(state)

    weights, epoch, failure = _resolve_weights(cfg, state, now=now, client=client)
    if failure is not None:
        return failure

    is_new_epoch = applied is None or epoch > applied
    if not is_new_epoch and pregate_skip(now=now, last_submit_ts=last_submit_ts(state)):
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
                         now=now, last_submit_ts=last_submit_ts(state)):
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
    return "applied" if is_new_epoch else "resubmitted"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add validator/validator.py tests/test_validator.py
git commit -m "feat(validator): resubmit weights every 100 blocks

A validator that submits once per epoch passes activity_cutoff (~16.7h)
long before a weekly epoch closes and drops out of Yuma consensus. The
tick now resubmits the same vector on a 100-block cadence, with the
chain block number authoritative and the wall clock only pre-gating the
connection. State advances only on a successful submit."
```

---

### Task 6: Cached-vector fallback for provider outages

Submission now depends on a verified payload every ~20 minutes, so a provider outage longer than `activity_cutoff` would drop the validator out of consensus — an exposure Task 5 creates. Halting protects nobody: the chain keeps the last submitted weights either way, so miners receive an identical distribution whether we keep submitting or stop. The only effect of stopping is that this validator leaves consensus and forfeits its own dividends.

**Files:**
- Modify: `validator/validator.py` (`_resolve_weights`, `_run_tick` return)
- Test: `tests/test_validator.py`

**Interfaces:**
- Consumes: `cached_weights(state)` from Task 2.
- Produces: `_resolve_weights(cfg, state, *, now, client) -> tuple[list | None, int | None, bool, str | None]` — gains a `from_cache` flag before `failure`. New return code `resubmitted:cached`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_validator.py`:

```python
def _dead_client():
    return httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(503)))


def test_a_provider_outage_keeps_resubmitting_the_cached_vector(tmp_path):
    # Without this the validator goes silent through the outage and passes
    # activity_cutoff, dropping out of consensus for the rest of the epoch.
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    tick(cfg, now=NOW, client=_client(_payload()), chain=fake)

    fake.block = 5100
    out = tick(cfg, now=NOW + 1200, client=_dead_client(), chain=fake)
    assert out == "resubmitted:cached"
    assert fake.submitted == [([0], [65535]), ([0], [65535])]
    assert last_submit_block(read_state(cfg["state_file"])) == 5100


def test_the_cache_still_respects_the_resubmit_interval(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    tick(cfg, now=NOW, client=_client(_payload()), chain=fake)
    fake.block = 5050
    assert tick(cfg, now=NOW + 1000, client=_dead_client(),
                chain=fake) == "skipped:too-soon"
    assert len(fake.submitted) == 1


def test_a_stale_epoch_during_rollover_falls_back_to_the_cache(tmp_path):
    # After a new epoch opens but before the provider publishes it, the served
    # payload is rejected as stale. Keep submitting instead of going silent.
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    tick(cfg, now=NOW, client=_client(_payload()), chain=fake)

    fake.block = 5100
    rolled = NOW + 604800          # one epoch later; IDX is now two epochs old
    out = tick(cfg, now=rolled, client=_client(_payload()), chain=fake)
    assert out == "resubmitted:cached"
    assert len(fake.submitted) == 2


def test_no_cache_and_no_payload_submits_nothing(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain()
    assert tick(cfg, now=NOW, client=_dead_client(), chain=fake) == "fetch-failed"
    assert fake.submitted == [] and fake.opens == 0


def test_a_recovered_provider_takes_over_from_the_cache(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeChain(block=5000)
    tick(cfg, now=NOW, client=_client(_payload()), chain=fake)
    fake.block = 5100
    tick(cfg, now=NOW + 1200, client=_dead_client(), chain=fake)

    # next epoch publishes: the fresh vector wins and applies immediately
    nxt = NOW + 604800
    p = _payload_with([["5Bbb", 65535]])
    p["epoch_index"] = IDX + 1
    rj = json.loads(p["result_json"]); rj["epoch_index"] = IDX + 1
    p["result_json"] = json.dumps(rj, sort_keys=True, separators=(",", ":"))
    p["digest"] = hashlib.sha256(p["result_json"].encode()).hexdigest()
    p["signature"] = MASTER.sign(epoch_message(53, IDX + 1, p["digest"]).encode()).hex()

    fake.hotkeys = ["5Aaa", "5Bbb"]
    assert tick(cfg, now=nxt, client=_client(p), chain=fake) == "applied"
    s = read_state(cfg["state_file"])
    assert last_applied(s) == IDX + 1
    assert cached_weights(s) == [["5Bbb", 65535]]


def test_a_corrupt_cache_is_not_submitted(tmp_path):
    cfg = _cfg(tmp_path)
    with open(cfg["state_file"], "w") as f:
        json.dump({"last_applied": IDX, "last_submit_block": 5000,
                   "last_submit_ts": NOW, "cached_weights": "garbage"}, f)
    fake = FakeChain(block=5100)
    assert tick(cfg, now=NOW + 1200, client=_dead_client(),
                chain=fake) == "fetch-failed"
    assert fake.submitted == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_validator.py -q -k "cache or rollover"`
Expected: FAIL — `assert 'fetch-failed' == 'resubmitted:cached'`

- [ ] **Step 3: Add the cache branch**

In `validator/validator.py`, replace `_resolve_weights` with:

```python
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
            payload, master_hotkey=cfg["master_hotkey"], netuid=cfg["netuid"],
            genesis=cfg["genesis"], now=now)
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
```

Update the two call sites in `_run_tick`:

```python
    weights, epoch, from_cache, failure = _resolve_weights(
        cfg, state, now=now, client=client)
```

and the final return:

```python
    if from_cache:
        return "resubmitted:cached"
    return "applied" if is_new_epoch else "resubmitted"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add validator/validator.py tests/test_validator.py
git commit -m "feat(validator): resubmit the cached vector during a provider outage

Submission now depends on a verified payload every ~20 minutes, so an
outage past activity_cutoff would drop the validator out of consensus.
Halting protects nobody - the chain keeps the last submitted weights
either way, so stopping costs only our own dividends. Also covers the
rollover window where the served epoch is briefly stale."
```

---

### Task 7: Poll interval 600s → 300s, and the docs that justify it

`POLL_S` now drives resubmit scheduling, not just payload freshness, so the comment justifying slow polling is obsolete. 300s also matches engy-validator's `ENGY_SN53_WEIGHTS_POLL_S` (`engy_validator/config.py:87`), removing one difference to reconcile when the two are merged.

**Files:**
- Modify: `validator/validator.py:36`, `.env.validator.example:27-29`, `docker/docker-compose.validator.yml`, `docker/docker-compose.validator-dev.yml`, `docker/docker-compose.validator-staging.yml`, `README.md`
- Test: `tests/test_validator.py`

**Interfaces:**
- Consumes: Tasks 5-6.
- Produces: nothing new.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_validator.py`:

```python
def test_poll_default_leaves_room_under_the_resubmit_interval(monkeypatch):
    # POLL_S drives resubmit scheduling, so it must stay well below the
    # interval or precision is lost to poll granularity.
    from validator.schedule import BLOCK_S, RESUBMIT_BLOCKS
    from validator.validator import load_config
    monkeypatch.setenv("ENGY_SN53_API", "https://engy.example")
    monkeypatch.setenv("ENGY_SN53_MASTER_HOTKEY", MASTER.ss58_address)
    monkeypatch.delenv("ENGY_SN53_POLL_S", raising=False)
    poll = load_config()["poll_s"]
    assert poll == 300
    assert poll * 4 <= RESUBMIT_BLOCKS * BLOCK_S
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_validator.py::test_poll_default_leaves_room_under_the_resubmit_interval -q`
Expected: FAIL — `assert 600 == 300`

- [ ] **Step 3: Change the default**

In `validator/validator.py:36`:

```python
        "poll_s": int(env.get("ENGY_SN53_POLL_S", "300")),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS

Confirm `stall_limit` still behaves at the new rate — `max(300 * 3, 900) == 900`, its floor:

Run: `.venv/bin/python -m pytest tests/test_validator.py -q -k stall`
Expected: PASS, no change needed to the formula.

- [ ] **Step 5: Update the docs**

In `.env.validator.example`, replace lines 27-29:

```
# Poll interval in seconds. This drives two things: picking up a new epoch,
# and the resubmit cadence that keeps the validator inside activity_cutoff.
# It must stay well below the resubmit interval (100 blocks ≈ 20 min) or
# scheduling precision is lost to poll granularity.
ENGY_SN53_POLL_S=300
```

Append to the optional section:

```
# Blocks between resubmissions of the same epoch's vector. The chain treats a
# validator whose last_update exceeds activity_cutoff (~5000 blocks) as
# inactive and drops its weights from consensus, so the vector is resubmitted
# for the whole epoch rather than once. Must be >= the subnet's
# weights_rate_limit or every submission is refused.
# ENGY_SN53_RESUBMIT_BLOCKS=100
```

The compose files do not set `POLL_S` — they inherit it from the env file — so no change is needed there. Confirm that is still true:

```bash
grep -n "POLL_S" docker/docker-compose.validator*.yml
```

Expected: no output. If a hit appears, update it to 300.

`README.md:38-40` is now actively wrong — it says the state volume means "an update never re-submits an already-applied epoch", which is exactly the behaviour being reversed. Replace that sentence:

```markdown
pulls new releases automatically, so a running validator stays current without
manual intervention. Submission state lives in a named volume, so a restart
resumes the resubmit schedule instead of starting the epoch over.
```

And in the section intro at `README.md:30-33`, replace the last sentence ("and submits the same weight vector on chain") with:

```markdown
and submits that weight vector on chain, resubmitting it roughly every 100
blocks for the rest of the epoch. The chain treats a validator that has not
submitted within `activity_cutoff` as inactive and drops its weights from
consensus, so a once-per-epoch submission would leave the validator earning
nothing for most of the week.
```

- [ ] **Step 6: Full suite and commit**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS

```bash
git add validator/validator.py .env.validator.example docker/ README.md tests/test_validator.py
git commit -m "chore(config): poll every 300s, document the resubmit cadence

POLL_S now drives resubmit scheduling, not just payload freshness, so
the comment justifying slow polling no longer holds. 300s also matches
engy-validator's weights poll, removing one difference to reconcile
when the two are merged."
```

---

## Verification

After Task 7, confirm the whole thing end to end:

- [ ] `.venv/bin/python -m pytest tests/ -q` — all pass
- [ ] `grep -rn "submit_fn\|_last_applied\|_well_formed_weights" validator/ tests/` returns nothing — no stragglers from the old interfaces
- [ ] `grep -rn "last_applied" validator/sync.py` returns nothing — scheduling fully left verification
- [ ] `.venv/bin/python -c "import validator.validator, validator.chain, validator.schedule, validator.state"` — imports clean without the `chain` extra installed

## Follow-ups (not this plan)

1. **Confirm SN53's on-chain `activity_cutoff` and `weights_rate_limit`.** The 5000 / 100 figures are Bittensor defaults. If `weights_rate_limit` exceeds 100, every resubmission is refused and `ENGY_SN53_RESUBMIT_BLOCKS` must be raised above it. This is the one premise that can invalidate the design — verify before rollout.
2. **Alert on prolonged cache-only operation.** `resubmitted:cached` is unbounded in time by design, which freezes the distribution at the last good epoch during a long provider outage. The mitigation is alerting plus human intervention; nothing here emits an alert yet.
3. **engy-validator's missing digest binding.** `engy_validator/weights_main.py:38-41` verifies the master signature over the server-supplied `result_digest` and never checks that `result_json` hashes to it; line 71 then reads weights from that unverified JSON. A compromised provider could replay a genuine triple while substituting arbitrary weights. More urgent than this work, but a different repo, and it cannot ship until `sha256(result_json) == result_digest` is confirmed against real payloads.
4. **Point engy-validator at this package.** `engy_validator/weights_main.py:62` has the same once-per-epoch defect. The code here is shaped for reuse — explicit kwargs, pure decision functions — so the port should be dependency plumbing rather than a rewrite.
