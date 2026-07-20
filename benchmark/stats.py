"""Pure aggregation over request records. No network, no IO."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

PERCENTILES = (50, 90, 99)


@dataclass
class RequestRecord:
    """One completed request attempt, successful or not.

    ``ttft`` and ``latency`` are seconds and are None when the request never
    got far enough to measure them.
    """
    ok: bool
    status: int | None
    ttft: float | None
    latency: float | None
    output_tokens: int
    error: str | None


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile. Returns None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1))))
    return ordered[rank]


def _distribution(values: list[float]) -> dict:
    return {f"p{p}": percentile(values, p) for p in PERCENTILES} | {
        "mean": sum(values) / len(values) if values else None,
    }


def summarize(records: list[RequestRecord], *, wall_s: float) -> dict:
    """Fold records into the numbers we report. Failed requests count toward
    the error rate but never toward latency or throughput."""
    ok = [r for r in records if r.ok]
    failed = [r for r in records if not r.ok]
    tokens = sum(r.output_tokens for r in ok)

    return {
        "total": len(records),
        "ok": len(ok),
        "failed": len(failed),
        "error_rate": len(failed) / len(records) if records else 0.0,
        "wall_s": wall_s,
        "completed_per_s": len(ok) / wall_s if wall_s > 0 else 0.0,
        "output_tokens_per_s": tokens / wall_s if wall_s > 0 else 0.0,
        "output_tokens": tokens,
        "ttft": _distribution([r.ttft for r in ok if r.ttft is not None]),
        "latency": _distribution([r.latency for r in ok if r.latency is not None]),
        "errors": dict(Counter(r.error for r in failed if r.error)),
    }
