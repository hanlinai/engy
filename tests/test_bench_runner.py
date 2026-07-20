import json

import httpx
import pytest

from benchmark.runner import BenchConfig, one_request, run


def cfg(**over):
    base = dict(url="https://api.test/v1", model="m", api_key="k",
                concurrency=2, duration=0.2, warmup=0.0, prompt_tokens=8,
                max_tokens=4, timeout=5.0, seed=1)
    base.update(over)
    return BenchConfig(**base)


def sse(*contents):
    lines = []
    for c in contents:
        chunk = {"choices": [{"delta": {"content": c}}]}
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def transport_of(handler):
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_counts_output_tokens_from_stream_deltas():
    def handler(request):
        return httpx.Response(200, content=sse("a", "b", "c"))

    async with httpx.AsyncClient(transport=transport_of(handler)) as client:
        r = await one_request(client, cfg(), index=0)

    assert r.ok
    assert r.output_tokens == 3
    assert r.ttft is not None and r.latency is not None


@pytest.mark.asyncio
async def test_prefers_usage_block_when_server_reports_it():
    def handler(request):
        body = sse("a", "b")
        usage = json.dumps({"choices": [], "usage": {"completion_tokens": 42}})
        return httpx.Response(200, content=body.replace(
            b"data: [DONE]", f"data: {usage}\n\ndata: [DONE]".encode()))

    async with httpx.AsyncClient(transport=transport_of(handler)) as client:
        r = await one_request(client, cfg(), index=0)

    assert r.output_tokens == 42


@pytest.mark.asyncio
async def test_http_error_is_recorded_not_raised():
    def handler(request):
        return httpx.Response(503, text="overloaded")

    async with httpx.AsyncClient(transport=transport_of(handler)) as client:
        r = await one_request(client, cfg(), index=0)

    assert not r.ok
    assert r.status == 503
    assert r.error == "http-503"
    assert r.output_tokens == 0


@pytest.mark.asyncio
async def test_network_failure_is_recorded_not_raised():
    def handler(request):
        raise httpx.ConnectError("refused")

    async with httpx.AsyncClient(transport=transport_of(handler)) as client:
        r = await one_request(client, cfg(), index=0)

    assert not r.ok
    assert r.status is None
    assert r.error == "ConnectError"


@pytest.mark.asyncio
async def test_request_carries_auth_model_and_stream_flag():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        seen["path"] = request.url.path
        return httpx.Response(200, content=sse("a"))

    async with httpx.AsyncClient(transport=transport_of(handler)) as client:
        await one_request(client, cfg(model="glm-5.2-fp8", max_tokens=16), index=0)

    assert seen["auth"] == "Bearer k"
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["model"] == "glm-5.2-fp8"
    assert seen["body"]["stream"] is True
    assert seen["body"]["max_tokens"] == 16


@pytest.mark.asyncio
async def test_run_stops_at_duration_and_returns_records():
    def handler(request):
        return httpx.Response(200, content=sse("a", "b"))

    records, wall = await run(cfg(duration=0.2, concurrency=3),
                              transport=transport_of(handler))

    assert records, "expected at least one completed request"
    assert all(r.ok for r in records)
    assert 0.2 <= wall < 2.0


@pytest.mark.asyncio
async def test_run_never_exceeds_the_configured_concurrency():
    state = {"in_flight": 0, "peak": 0}

    def handler(request):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        state["in_flight"] -= 1
        return httpx.Response(200, content=sse("a"))

    await run(cfg(duration=0.2, concurrency=4), transport=transport_of(handler))
    assert state["peak"] <= 4


@pytest.mark.asyncio
async def test_warmup_requests_are_excluded_from_records():
    def handler(request):
        return httpx.Response(200, content=sse("a"))

    warm, _ = await run(cfg(duration=0.1, warmup=0.2, concurrency=1),
                        transport=transport_of(handler))
    cold, _ = await run(cfg(duration=0.1, warmup=0.0, concurrency=1),
                        transport=transport_of(handler))

    # Warmup traffic must not land in the sample; the warm run measures the
    # same 0.1s window as the cold one, so it cannot have far more records.
    assert len(warm) <= len(cold) * 2


@pytest.mark.asyncio
async def test_error_object_inside_a_200_stream_is_a_failure():
    # engy's gateway answers auth failures with HTTP 200 and an error object in
    # the SSE body. Counting those as successes measures the rejection path.
    def handler(request):
        err = json.dumps({"error": {"message": "invalid api key",
                                    "type": "authentication_error", "code": 401}})
        return httpx.Response(200, content=f"data: {err}\n\n".encode())

    async with httpx.AsyncClient(transport=transport_of(handler)) as client:
        r = await one_request(client, cfg(), index=0)

    assert not r.ok
    assert r.error == "stream-error-401"
    assert r.output_tokens == 0


@pytest.mark.asyncio
async def test_stream_error_without_a_code_still_fails():
    def handler(request):
        err = json.dumps({"error": {"message": "upstream exploded"}})
        return httpx.Response(200, content=f"data: {err}\n\n".encode())

    async with httpx.AsyncClient(transport=transport_of(handler)) as client:
        r = await one_request(client, cfg(), index=0)

    assert not r.ok
    assert r.error == "stream-error"


@pytest.mark.asyncio
async def test_a_200_stream_that_yields_no_tokens_is_a_failure():
    # No error object, no content either — nothing was actually generated.
    def handler(request):
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    async with httpx.AsyncClient(transport=transport_of(handler)) as client:
        r = await one_request(client, cfg(), index=0)

    assert not r.ok
    assert r.error == "empty-stream"


@pytest.mark.asyncio
async def test_reasoning_deltas_count_as_output_tokens():
    # qwen3.6 streams its chain of thought as `reasoning_content`. Those are
    # generated tokens the user pays for and waits on, so they count.
    def handler(request):
        chunks = [{"choices": [{"delta": {"reasoning_content": "think"}}]},
                  {"choices": [{"delta": {"reasoning_content": "more"}}]},
                  {"choices": [{"delta": {"content": "answer"}}]}]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
        return httpx.Response(200, content=(body + "data: [DONE]\n\n").encode())

    async with httpx.AsyncClient(transport=transport_of(handler)) as client:
        r = await one_request(client, cfg(), index=0)

    assert r.ok
    assert r.output_tokens == 3


@pytest.mark.asyncio
async def test_ttft_measures_the_first_token_of_any_kind():
    # A reasoning model that thinks for 10s before its first content token has
    # a 10s TTFT from the user's seat, not a fast one.
    def handler(request):
        chunks = [{"choices": [{"delta": {"reasoning_content": "think"}}]},
                  {"choices": [{"delta": {"content": "answer"}}]}]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
        return httpx.Response(200, content=(body + "data: [DONE]\n\n").encode())

    async with httpx.AsyncClient(transport=transport_of(handler)) as client:
        r = await one_request(client, cfg(), index=0)

    assert r.ttft is not None
    assert r.ttft <= r.latency


@pytest.mark.asyncio
async def test_role_only_opening_chunk_is_not_a_token():
    def handler(request):
        chunks = [{"choices": [{"delta": {"role": "assistant"}}]},
                  {"choices": [{"delta": {"content": "a"}}]}]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
        return httpx.Response(200, content=(body + "data: [DONE]\n\n").encode())

    async with httpx.AsyncClient(transport=transport_of(handler)) as client:
        r = await one_request(client, cfg(), index=0)

    assert r.output_tokens == 1
