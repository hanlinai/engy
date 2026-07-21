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
