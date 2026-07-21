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
