from engy_sn53.chain import resolve_uids, skipped_hotkeys


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
