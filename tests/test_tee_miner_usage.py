"""The standalone TEE miner must preserve cache counts supplied by its serve."""

from miner.tee_miner import _clean_usage


def test_cache_write_count_survives_tee_usage_boundary():
    usage = _clean_usage({
        "prompt_tokens": 100, "completion_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": 64, "cache_write_tokens": 32},
    })
    assert usage["prompt_tokens_details"] == {
        "cached_tokens": 64, "cache_write_tokens": 32}


def test_tee_does_not_invent_cache_write_count():
    usage = _clean_usage({"prompt_tokens": 100, "completion_tokens": 2,
                          "prompt_tokens_details": {"cached_tokens": 64}})
    assert usage["prompt_tokens_details"] == {"cached_tokens": 64}
