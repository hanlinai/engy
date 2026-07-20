"""Command line front end for the concurrent benchmark."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from .runner import BenchConfig, run
from .stats import summarize


def build_config(argv: list[str]) -> BenchConfig:
    p = argparse.ArgumentParser(
        prog="engy-bench",
        description="Concurrent load benchmark for an OpenAI-compatible endpoint.")
    p.add_argument("--url", required=True, help="base URL, e.g. https://api.engy.ai/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default=None,
                   help="defaults to $ENGY_API_KEY")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--duration", type=float, default=30.0, help="seconds")
    p.add_argument("--warmup", type=float, default=5.0,
                   help="seconds of traffic to discard before measuring")
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--timeout", type=float, default=120.0, help="per-request seconds")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--json", dest="json_out", default=None,
                   help="write the full summary to this path")
    a = p.parse_args(argv)

    for name in ("concurrency", "prompt_tokens", "max_tokens"):
        if getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("duration", "timeout"):
        if getattr(a, name) <= 0:
            p.error(f"--{name} must be positive")
    if a.warmup < 0:
        p.error("--warmup cannot be negative")

    return BenchConfig(
        url=a.url, model=a.model,
        api_key=a.api_key or os.environ.get("ENGY_API_KEY"),
        concurrency=a.concurrency, duration=a.duration, warmup=a.warmup,
        prompt_tokens=a.prompt_tokens, max_tokens=a.max_tokens,
        timeout=a.timeout, seed=a.seed, json_out=a.json_out)


def _fmt(v: float | None, places: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{places}f}"


def format_summary(s: dict, *, concurrency: int) -> str:
    lines = [
        f"concurrency {concurrency} | {_fmt(s['wall_s'], 1)}s measured",
        f"requests    {s['total']} total, {s['ok']} ok, {s['failed']} failed "
        f"({s['error_rate'] * 100:.1f}% errors)",
        f"throughput  {s['completed_per_s']:.2f} req/s, "
        f"{s['output_tokens_per_s']:.1f} output tok/s",
        f"ttft (s)    p50 {_fmt(s['ttft']['p50'])}  p90 {_fmt(s['ttft']['p90'])}  "
        f"p99 {_fmt(s['ttft']['p99'])}",
        f"latency (s) p50 {_fmt(s['latency']['p50'])}  p90 {_fmt(s['latency']['p90'])}  "
        f"p99 {_fmt(s['latency']['p99'])}",
    ]
    if s["errors"]:
        detail = ", ".join(f"{k}×{v}" for k, v in sorted(s["errors"].items()))
        lines.append(f"errors      {detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    cfg = build_config(sys.argv[1:] if argv is None else argv)
    print(f"warming up {cfg.warmup:g}s, then measuring {cfg.duration:g}s "
          f"at concurrency {cfg.concurrency}...", flush=True)

    records, wall = asyncio.run(run(cfg))
    summary = summarize(records, wall_s=wall)
    print(format_summary(summary, concurrency=cfg.concurrency))

    if cfg.json_out:
        payload = summary | {"config": {
            "url": cfg.url, "model": cfg.model, "concurrency": cfg.concurrency,
            "duration": cfg.duration, "warmup": cfg.warmup,
            "prompt_tokens": cfg.prompt_tokens, "max_tokens": cfg.max_tokens,
            "seed": cfg.seed}}
        with open(cfg.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {cfg.json_out}")

    return 1 if summary["ok"] == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
