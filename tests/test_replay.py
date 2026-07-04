"""Synthetic-data tests for the offline state replayer.

A real capture is still needed to validate Majsoul-specific visual edge cases
(kan-dora timing, exact called-tile rendering, sanma) — marked # VALIDATE in
replay.py. These tests pin the well-defined event semantics.
"""

from majsoul_eye.state.replay import Replayer, check_invariants


HERO_HAND = ["1m", "2m", "3m", "2p", "2p", "5p", "6p", "7p", "9p", "1s", "2s", "3s", "9s"]


def _events():
    return [
        {"type": "start_game", "id": 0},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "1m",
            "honba": 0,
            "kyoku": 2,          # oya=1 -> East-2
            "kyotaku": 0,
            "oya": 1,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [HERO_HAND, ["?"] * 13, ["?"] * 13, ["?"] * 13],
        },
        {"type": "tsumo", "actor": 1, "pai": "?"},
        {"type": "dahai", "actor": 1, "pai": "E", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "?"},
        {"type": "dahai", "actor": 2, "pai": "2p", "tsumogiri": False},
        # hero pons the 2p from seat 2
        {"type": "pon", "actor": 0, "target": 2, "pai": "2p", "consumed": ["2p", "2p"]},
        {"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": False},
        # seat 3 riichi
        {"type": "tsumo", "actor": 3, "pai": "?"},
        {"type": "reach", "actor": 3},
        {"type": "dahai", "actor": 3, "pai": "5s", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 3},
        {"type": "end_kyoku"},
    ]


def _run():
    rp = Replayer()
    for ev in _events():
        rp.apply(ev)
    return rp.state


def test_round_meta():
    s = _run()
    assert s.bakaze == "E" and s.kyoku == 2 and s.oya == 1
    assert s.dora_markers == ["1m"]


def test_pon_meld_and_called_tile_removed_from_river():
    s = _run()
    assert s.num_melds(0) == 1
    meld = s.melds[0][0]
    assert meld.type == "pon" and meld.from_seat == 2
    assert meld.tiles.count("2p") == 3
    assert meld.called_pai == "2p"       # identity of the sideways-rendered tile survives
    # seat 2's only discard was called away -> not visible in the 河
    assert len(s.rivers[2]) == 1 and s.rivers[2][0].called is True
    assert s.visible_river(2) == []


def test_riichi_sideways_and_stick():
    s = _run()
    assert s.reach[3] is True
    assert s.rivers[3][-1].riichi is True and s.rivers[3][-1].pai == "5s"
    assert s.scores[3] == 24000 and s.kyotaku == 1


def test_concealed_counts_and_hero_hand():
    s = _run()
    assert s.concealed_counts == [10, 13, 13, 13]
    assert len(s.hero_hand) == 10
    # hero hand + 3*melds must be a legal 13/14
    assert len(s.hero_hand) + 3 * s.num_melds(0) == 13


def test_invariants_clean():
    s = _run()
    assert check_invariants(s) == []


def test_invariant_detects_fifth_tile():
    from majsoul_eye.state.replay import RiverTile
    rp = Replayer()
    for ev in _events()[:2]:
        rp.apply(ev)
    # hero already holds one 1m and the dora marker is 1m (2 total); add 3 more
    # across rivers -> 5 of '1m', which is impossible.
    rp.state.rivers[1].extend([RiverTile(pai="1m") for _ in range(3)])
    viol = check_invariants(rp.state)
    assert any("1m" in v for v in viol)


def test_called_tile_not_double_counted():
    # Regression (real capture): a called tile lives in BOTH rivers[target]
    # (called=True) AND the caller's meld. check_invariants must count only the
    # visible 河, else it double-counts and reports a false >4 violation.
    from majsoul_eye.state.replay import Replayer, Meld, RiverTile
    rp = Replayer()
    rp.apply({"type": "start_game", "id": 0})
    s = rp.state
    s.hero_seat = 0
    s.melds[0] = [Meld(type="pon", from_seat=1, tiles=["C", "C", "C"])]
    s.rivers[1] = [RiverTile(pai="C", called=True)]   # called tile, moved to meld
    s.rivers[2] = [RiverTile(pai="C")]                # the 4th C, still on the table
    assert check_invariants(s) == []                   # 3 + 1 == 4; old checker saw 5


def test_is_deal_window_true_until_first_discard():
    # The deal-in animation (~2-3s) plays from start_kyoku until the first dahai;
    # a frame there shows an unsorted/incomplete hero hand that won't match GT.
    # is_deal_window keys off "no discard yet" (rivers all empty), which is robust
    # to the bridge bundling [start_kyoku, tsumo] into one record (last_event=='tsumo').
    from majsoul_eye.state.replay import Replayer, is_deal_window
    rp = Replayer()
    rp.apply({"type": "start_game", "id": 0})
    assert is_deal_window(rp.state) is False          # no kyoku started yet
    rp.apply(_events()[1])                             # start_kyoku
    assert is_deal_window(rp.state) is True            # dealt, no discard
    rp.apply({"type": "tsumo", "actor": 1, "pai": "?"})
    assert rp.state.last_event == "tsumo"
    assert is_deal_window(rp.state) is True            # still no discard (last_event!=start_kyoku)
    rp.apply({"type": "dahai", "actor": 1, "pai": "E", "tsumogiri": True})
    assert is_deal_window(rp.state) is False           # first discard ended the deal window


def test_is_deal_window_false_after_round_ends():
    from majsoul_eye.state.replay import is_deal_window
    s = _run()                                         # full kyoku incl. discards then end_kyoku
    assert is_deal_window(s) is False


def test_hero_tsumo_tracks_drawn_tile():
    # The hero's freshly-drawn tile renders in a separated slot on screen; the
    # labeler needs to know WHICH tile it is (hero_hand is sorted, so the draw is
    # merged and lost). drawn_tile carries it from the hero's tsumo to the next
    # hero action; it is None on every other player's turn.
    rp = Replayer()
    rp.apply({"type": "start_game", "id": 0})
    rp.apply(_events()[1])                                   # start_kyoku: hero(seat0) dealt 13
    assert rp.state.drawn_tile is None
    rp.apply({"type": "tsumo", "actor": 1, "pai": "?"})       # opponent draws -> unaffected
    assert rp.state.drawn_tile is None
    rp.apply({"type": "dahai", "actor": 1, "pai": "E", "tsumogiri": True})
    rp.apply({"type": "tsumo", "actor": 0, "pai": "4p"})      # HERO draws
    assert rp.state.drawn_tile == "4p"
    assert len(rp.state.hero_hand) == 14 and rp.state.hero_hand.count("4p") == 1
    assert rp.state.copy().drawn_tile == "4p"                 # survives snapshot (build_seq_state)
    rp.apply({"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": False})  # hero discards
    assert rp.state.drawn_tile is None
    assert len(rp.state.hero_hand) == 13


def test_drawn_tile_cleared_when_hero_calls_kan():
    # After a hero draw, declaring an ankan (instead of discarding) ends the
    # separated-draw display; drawn_tile must clear (the rinshan draw re-sets it).
    rp = Replayer()
    rp.apply({"type": "start_game", "id": 0})
    hand = ["1m", "1m", "1m", "1m", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "1s", "2s"]
    rp.apply({**_events()[1], "tehais": [hand, ["?"] * 13, ["?"] * 13, ["?"] * 13]})
    rp.apply({"type": "tsumo", "actor": 0, "pai": "3s"})
    assert rp.state.drawn_tile == "3s"
    rp.apply({"type": "ankan", "actor": 0, "consumed": ["1m", "1m", "1m", "1m"]})
    assert rp.state.drawn_tile is None


def test_leftTileCount_extracted_from_camelcase():
    # Regression (real capture): Majsoul sends 'leftTileCount', not 'left_tile_count'.
    class _Rec:
        raw_liqi = {"data": {"name": "ActionDealTile",
                             "data": {"seat": 1, "tile": "4s", "leftTileCount": 68}}}
        mjai = [{"type": "tsumo", "actor": 1, "pai": "?"}]
    rp = Replayer()
    rp.apply({"type": "start_game", "id": 0})
    rp.apply_record(_Rec())
    assert rp.state.left_tile_count == 68


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_replay OK")
