"""Concurrency engine: N worker coroutines hammering the endpoint for a fixed
wall-clock duration.

A worker's loop is send → consume SSE → record → send again, so exactly
``concurrency`` requests are in flight at any moment. Nothing here interprets
the numbers; that is stats.py's job.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

import httpx

from .prompts import synth_prompt
from .stats import RequestRecord


@dataclass
class BenchConfig:
    url: str
    model: str
    api_key: str | None
    concurrency: int
    duration: float
    warmup: float
    prompt_tokens: int
    max_tokens: int
    timeout: float
    seed: int
    # Output destination for the full summary. The runner ignores it; it rides
    # along here so the CLI has a single object to pass around.
    json_out: str | None = None


def _payload(cfg: BenchConfig, index: int) -> dict:
    return {
        "model": cfg.model,
        "messages": [{"role": "user", "content": synth_prompt(
            seed=cfg.seed, index=index, tokens=cfg.prompt_tokens)}],
        "max_tokens": cfg.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _headers(cfg: BenchConfig) -> dict:
    h = {"content-type": "application/json"}
    if cfg.api_key:
        h["authorization"] = f"Bearer {cfg.api_key}"
    return h


async def one_request(client: httpx.AsyncClient, cfg: BenchConfig,
                      *, index: int) -> RequestRecord:
    """Issue one streamed completion. Never raises: transport and protocol
    failures come back as a failed record so one bad request cannot end the run."""
    started = time.monotonic()
    ttft: float | None = None
    deltas = 0
    reported: int | None = None

    try:
        async with client.stream(
            "POST", f"{cfg.url.rstrip('/')}/chat/completions",
            json=_payload(cfg, index), headers=_headers(cfg),
            timeout=cfg.timeout,
        ) as response:
            if response.status_code != 200:
                await response.aread()
                return RequestRecord(ok=False, status=response.status_code,
                                     ttft=None, latency=None, output_tokens=0,
                                     error=f"http-{response.status_code}")

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                body = line[6:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except ValueError:
                    continue

                # An error can arrive inside a 200 stream — engy's gateway
                # answers auth failures that way. The status line lies; the
                # body is the truth.
                err = chunk.get("error")
                if isinstance(err, dict):
                    code = err.get("code")
                    return RequestRecord(
                        ok=False, status=response.status_code, ttft=None,
                        latency=None, output_tokens=0,
                        error=f"stream-error-{code}" if code else "stream-error")

                usage = chunk.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens"):
                    reported = usage["completion_tokens"]

                for choice in chunk.get("choices") or []:
                    if (choice.get("delta") or {}).get("content"):
                        if ttft is None:
                            ttft = time.monotonic() - started
                        deltas += 1
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        return RequestRecord(ok=False, status=None, ttft=None, latency=None,
                             output_tokens=0, error=type(e).__name__)

    tokens = reported if reported is not None else deltas
    if tokens == 0:
        # 200, no error, no content: nothing was generated, so there is no
        # latency worth reporting.
        return RequestRecord(ok=False, status=200, ttft=None, latency=None,
                             output_tokens=0, error="empty-stream")

    return RequestRecord(ok=True, status=200, ttft=ttft,
                         latency=time.monotonic() - started,
                         output_tokens=tokens, error=None)


async def run(cfg: BenchConfig, *,
              transport: httpx.AsyncBaseTransport | None = None
              ) -> tuple[list[RequestRecord], float]:
    """Drive the load and return (records, measured wall seconds).

    Requests completed during warmup are discarded — the first requests to hit
    a cold server measure queue fill and connection setup, not steady state.
    """
    records: list[RequestRecord] = []
    counter = 0
    limits = httpx.Limits(max_connections=cfg.concurrency * 2,
                          max_keepalive_connections=cfg.concurrency * 2)

    async with httpx.AsyncClient(transport=transport, limits=limits) as client:
        began = time.monotonic()
        measure_from = began + cfg.warmup
        deadline = measure_from + cfg.duration

        async def worker() -> None:
            nonlocal counter
            while time.monotonic() < deadline:
                index, counter = counter, counter + 1
                sent_at = time.monotonic()
                record = await one_request(client, cfg, index=index)
                if sent_at >= measure_from:
                    records.append(record)

        await asyncio.gather(*(worker() for _ in range(cfg.concurrency)))
        measured = time.monotonic() - max(measure_from, began)

    return records, measured
