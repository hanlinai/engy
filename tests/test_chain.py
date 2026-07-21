from validator.chain import (
    EXPECTED_OWNER_HOTKEY, ChainView, burn_target, dropped_weight_share,
    resolve_uids, set_weights, skipped_hotkeys, _valid_block,
)


def test_maps_hotkeys_to_uids_in_uid_order():
    uids, ws = resolve_uids([["5B", 100], ["5A", 200]], ["5A", "5B"])
    assert uids == [0, 1]
    assert ws[0] > ws[1] and sum(ws) == 65535


def test_drops_unknown_and_renormalizes():
    uids, ws = resolve_uids([["5A", 32768], ["5Gone", 32767]], ["5A"])
    assert uids == [0] and ws == [65535]


def test_all_unknown_is_empty():
    assert resolve_uids([["5Gone", 65535]], ["5A", "5B"]) == ([], [])


def test_all_zero_stays_zero():
    uids, ws = resolve_uids([["5A", 0], ["5B", 0]], ["5A", "5B"])
    assert uids == [0, 1] and ws == [0, 0]


def test_skipped_hotkeys_lists_unregistered_in_payload_order():
    assert skipped_hotkeys([["5Gone", 10], ["5A", 20], ["5Also", 30]],
                           ["5A", "5B"]) == ["5Gone", "5Also"]


def test_skipped_hotkeys_empty_when_all_registered():
    assert skipped_hotkeys([["5A", 10], ["5B", 20]], ["5A", "5B"]) == []


def test_dropped_weight_share_is_by_weight_not_by_count():
    # Ten zero-weight strangers matter far less than one heavy one.
    weights = [["5Aaa", 60000], ["5Gone", 5535]]
    assert dropped_weight_share(weights, ["5Aaa"]) == 5535 / 65535


def test_dropped_weight_share_is_zero_when_everyone_is_registered():
    assert dropped_weight_share([["5Aaa", 65535]], ["5Aaa", "5Bbb"]) == 0.0


def test_dropped_weight_share_is_one_when_nobody_is_registered():
    assert dropped_weight_share([["5Aaa", 65535]], ["5Zzz"]) == 1.0


def test_dropped_weight_share_of_an_all_zero_vector_does_not_divide_by_zero():
    assert dropped_weight_share([["5Aaa", 0], ["5Bbb", 0]], []) == 0.0


def test_dropped_weight_share_of_an_empty_vector_is_zero():
    assert dropped_weight_share([], ["5Aaa"]) == 0.0


def test_a_burn_vector_resolves_to_the_owner_alone():
    # Burn arrives as an ordinary single row; no special-casing anywhere.
    uids, ws = resolve_uids([["5Owner", 65535]], ["5Aaa", "5Owner", "5Bbb"])
    assert (uids, ws) == ([1], [65535])
    assert dropped_weight_share([["5Owner", 65535]], ["5Aaa", "5Owner"]) == 0.0


def test_chain_view_carries_a_possibly_absent_block_number():
    v = ChainView(sub=object(), hotkeys=["5Aaa"], block=None)
    assert v.block is None and v.hotkeys == ["5Aaa"]


def test_only_positive_ints_count_as_a_block_number():
    # A bad RPC reply must degrade to the wall-clock fallback, not poison
    # the block arithmetic.
    assert _valid_block(4823910) == 4823910
    for bad in (None, 0, -1, True, False, 1.5, "4823910"):
        assert _valid_block(bad) is None, bad


# ── burn target ──────────────────────────────────────────────────

def test_burn_target_names_the_sole_recipient():
    assert burn_target([["5Owner", 65535]]) == "5Owner"


def test_burn_target_ignores_zero_weight_rows():
    # A vector can carry scored-but-zero miners alongside the burn.
    assert burn_target([["5A", 0], ["5Owner", 65535], ["5B", 0]]) == "5Owner"


def test_burn_target_is_none_for_a_real_distribution():
    assert burn_target([["5A", 40000], ["5B", 25535]]) is None


def test_burn_target_is_none_for_an_empty_or_all_zero_vector():
    assert burn_target([]) is None
    assert burn_target([["5A", 0], ["5B", 0]]) is None


def test_the_expected_owner_matches_the_providers_configured_value():
    # The provider's owner_hotkey, registered on netuid 53 at uid 229 — the
    # same key that signs epoch results, NOT the chain's SubnetOwnerHotkey.
    # A burn to an unregistered hotkey resolves to no uid at all, so nothing
    # is submitted and the validator silently sets no weights.
    assert EXPECTED_OWNER_HOTKEY == "5DXSBCCKH5ENuyHFNaAvtaMfbhEEWpjSJB4rzc4mJfsc1uvJ"


# ── set_weights result handling ──────────────────────────────────

class _ExtrinsicResponse:
    """What bittensor 10.x returns. It mimics tuple access but is NOT a tuple
    instance, and defines no __bool__ — so bool(response) is True even for a
    rejected extrinsic."""

    def __init__(self, success, message=""):
        self.success, self.message = success, message

    def __getitem__(self, i):
        return (self.success, self.message)[i]

    def __len__(self):
        return 2


class _FakeSub:
    def __init__(self, result):
        self.result = result

    def set_weights(self, **kw):
        return self.result


def _submit(result, monkeypatch):
    import sys
    import types
    fake_bt = types.ModuleType("bittensor")
    fake_bt.Wallet = lambda **kw: object()
    monkeypatch.setitem(sys.modules, "bittensor", fake_bt)
    view = ChainView(sub=_FakeSub(result), hotkeys=["5A"], block=1)
    return set_weights(view, wallet="w", wallet_hotkey="hk", netuid=53,
                       uids=[0], ws=[65535])


def test_a_rejected_extrinsic_is_not_read_as_success(monkeypatch):
    # The object is truthy, so bool() reports a rejected extrinsic as a
    # successful submission: the loop advances its state, waits out a full
    # resubmit interval, and sets no weights at all while looking healthy.
    assert bool(_ExtrinsicResponse(False, "No validator permit")) is True
    assert _submit(_ExtrinsicResponse(False, "No validator permit"), monkeypatch) is False


def test_an_accepted_extrinsic_is_read_as_success(monkeypatch):
    assert _submit(_ExtrinsicResponse(True, "ok"), monkeypatch) is True


def test_the_rejection_reason_reaches_the_log(monkeypatch, capsys):
    # "No validator permit" is the difference between a problem an operator can
    # fix and an outage they cannot explain.
    _submit(_ExtrinsicResponse(False, "No validator permit"), monkeypatch)
    assert "No validator permit" in capsys.readouterr().out


def test_older_tuple_and_bare_bool_returns_still_work(monkeypatch):
    assert _submit((False, "nope"), monkeypatch) is False
    assert _submit((True, "ok"), monkeypatch) is True
    assert _submit(False, monkeypatch) is False
    assert _submit(True, monkeypatch) is True
