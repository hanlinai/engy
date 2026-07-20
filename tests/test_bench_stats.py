from benchmark.stats import RequestRecord, percentile, summarize


def rec(**over):
    base = dict(ok=True, status=200, ttft=0.5, latency=2.0,
                output_tokens=100, error=None)
    base.update(over)
    return RequestRecord(**base)


def test_percentile_picks_nearest_rank_value():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 50) == 30.0
    assert percentile(values, 90) == 50.0
    assert percentile(values, 0) == 10.0


def test_percentile_of_empty_is_none():
    assert percentile([], 50) is None


def test_percentile_sorts_input():
    assert percentile([50.0, 10.0, 30.0], 50) == 30.0


def test_summarize_counts_ok_and_failed():
    s = summarize([rec(), rec(), rec(ok=False, status=500, error="500")], wall_s=10.0)
    assert s["total"] == 3
    assert s["ok"] == 2
    assert s["failed"] == 1
    assert s["error_rate"] == 1 / 3


def test_summarize_reports_throughput_over_wall_clock():
    s = summarize([rec(output_tokens=100) for _ in range(4)], wall_s=10.0)
    assert s["completed_per_s"] == 0.4
    assert s["output_tokens_per_s"] == 40.0


def test_summarize_ignores_failed_requests_in_latency_stats():
    records = [rec(ttft=1.0, latency=1.0), rec(ok=False, ttft=None, latency=None,
                                               status=500, error="500")]
    s = summarize(records, wall_s=10.0)
    assert s["ttft"]["p50"] == 1.0
    assert s["latency"]["p50"] == 1.0


def test_summarize_groups_errors_by_reason():
    records = [rec(ok=False, status=500, error="500"),
               rec(ok=False, status=500, error="500"),
               rec(ok=False, status=None, error="timeout")]
    s = summarize(records, wall_s=10.0)
    assert s["errors"] == {"500": 2, "timeout": 1}


def test_summarize_of_no_records_does_not_crash():
    s = summarize([], wall_s=10.0)
    assert s["total"] == 0
    assert s["error_rate"] == 0.0
    assert s["ttft"]["p50"] is None
