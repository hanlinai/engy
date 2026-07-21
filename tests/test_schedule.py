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
