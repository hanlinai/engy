import json

import pytest

from benchmark.cli import build_config, format_summary
from benchmark.stats import RequestRecord, summarize


def args(*extra):
    return ["--url", "https://api.test/v1", "--model", "m", *extra]


def test_api_key_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("ENGY_API_KEY", "from-env")
    assert build_config(args()).api_key == "from-env"


def test_explicit_api_key_beats_environment(monkeypatch):
    monkeypatch.setenv("ENGY_API_KEY", "from-env")
    assert build_config(args("--api-key", "explicit")).api_key == "explicit"


def test_concurrency_and_duration_are_parsed():
    cfg = build_config(args("--concurrency", "32", "--duration", "60"))
    assert cfg.concurrency == 32
    assert cfg.duration == 60.0


def test_rejects_non_positive_concurrency():
    with pytest.raises(SystemExit):
        build_config(args("--concurrency", "0"))


def test_summary_reports_the_headline_numbers():
    records = [RequestRecord(ok=True, status=200, ttft=0.25, latency=2.0,
                             output_tokens=100, error=None)] * 4
    text = format_summary(summarize(records, wall_s=10.0), concurrency=8)

    assert "requests" in text
    assert "0.250" in text          # p50 TTFT
    assert "40.0" in text           # output tokens/s
    assert "8" in text              # concurrency echoed back


def test_summary_surfaces_errors_when_present():
    records = [RequestRecord(ok=False, status=503, ttft=None, latency=None,
                             output_tokens=0, error="http-503")]
    text = format_summary(summarize(records, wall_s=10.0), concurrency=1)
    assert "http-503" in text
    assert "100.0%" in text


def test_summary_of_zero_successes_does_not_crash():
    records = [RequestRecord(ok=False, status=None, ttft=None, latency=None,
                             output_tokens=0, error="ConnectError")]
    text = format_summary(summarize(records, wall_s=10.0), concurrency=1)
    assert "n/a" in text


def test_json_output_round_trips(tmp_path):
    out = tmp_path / "r.json"
    summary = summarize([RequestRecord(ok=True, status=200, ttft=0.1, latency=1.0,
                                       output_tokens=10, error=None)], wall_s=1.0)
    out.write_text(json.dumps(summary))
    assert json.loads(out.read_text())["ok"] == 1
