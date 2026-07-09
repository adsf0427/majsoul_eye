"""ObservedState schema + single-frame consistency checks (spec 2026-07-05 §3.1)."""
from majsoul_eye.state.observe import (
    ObservedMeld, ObservedRiverTile, ObservedState, check_observed)


def _minimal():
    return ObservedState(
        hero_hand=["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
        rivers=[[], [], [], []], melds=[[], [], [], []],
        dora_markers=["5s"], concealed_counts=[None, 13, 13, 13],
        reach=[False] * 4)


def test_clean_state_has_no_violations():
    assert check_observed(_minimal()) == []


def test_fifth_copy_flagged():
    o = _minimal()
    o.rivers[1] = [ObservedRiverTile("1m") for _ in range(4)]  # + one in hand = 5
    v = check_observed(o)
    assert any("1m" in m and "5" in m for m in v)


def test_red_five_counts_with_plain():
    o = _minimal()
    o.hero_hand = ["5m", "5m", "5m", "5mr"] + o.hero_hand[4:]
    o.rivers[2] = [ObservedRiverTile("5m")]                    # 5th 5m-kind
    assert check_observed(o)


def test_hand_size_vs_melds():
    o = _minimal()
    o.hero_hand = o.hero_hand[:12]                             # 12 + 0 melds != 13
    assert any("hand" in m for m in check_observed(o))
    o2 = _minimal()
    o2.hero_hand = o2.hero_hand[:10]
    o2.melds[0] = [ObservedMeld("pon", ["P", "P", "P"], called_pai="P", from_rel=2)]
    assert check_observed(o2) == []                            # 10 + 3*1 == 13


def test_hero_call_pending_shape_allowed():
    # hero just chi/pon'd, mandatory discard not yet made: 11 tiles + 1 meld
    # (14 accounting), no drawn slot — fully visible, a real decision point.
    o = _minimal()
    o.hero_hand = o.hero_hand[:11]
    o.melds[0] = [ObservedMeld("pon", ["P", "P", "P"], called_pai="P", from_rel=2)]
    assert check_observed(o) == []
    # ... but only for a trailing chi/pon: an "extra tile" next to an ankan
    # (which can't be awaiting a call-discard) stays a violation
    o2 = _minimal()
    o2.hero_hand = o2.hero_hand[:11]
    o2.melds[0] = [ObservedMeld("ankan", ["C", "C", "C", "C"])]
    o2.dora_markers = ["5s", "6s"]
    assert any("hand" in m for m in check_observed(o2))
    # ... and not with a drawn tile too (can't be both mid-call and mid-draw)
    o3 = _minimal()
    o3.hero_hand = o3.hero_hand[:11]
    o3.melds[0] = [ObservedMeld("pon", ["P", "P", "P"], called_pai="P", from_rel=2)]
    o3.drawn_tile = "9s"
    assert any("hand" in m for m in check_observed(o3))


def test_drawn_tile_is_extra():
    o = _minimal()
    o.drawn_tile = "9s"
    assert check_observed(o) == []                             # 13 + drawn == to-act


def test_dora_rules():
    o = _minimal()
    o.dora_markers = []
    assert any("dora" in m for m in check_observed(o))
    o2 = _minimal()
    o2.dora_markers = ["5s", "6s", "7s"]                       # 3 markers, 0 kans
    assert any("dora" in m or "kan" in m for m in check_observed(o2))
    o3 = _minimal()
    o3.hero_hand = o3.hero_hand[:10]
    o3.melds[0] = [ObservedMeld("ankan", ["C", "C", "C", "C"])]
    o3.dora_markers = ["5s", "6s"]                             # 2 markers, 1 kan: ok
    assert check_observed(o3) == []


def test_concealed_counts_cross_check():
    o = _minimal()
    o.concealed_counts = [None, 10, 13, 13]                    # seat1: 10 but 0 melds
    assert any("concealed" in m for m in check_observed(o))
    o2 = _minimal()
    o2.melds[1] = [ObservedMeld("chi", ["1s", "2s", "3s"], called_pai="2s", from_rel=3)]
    o2.concealed_counts = [None, 10, 13, 13]
    assert check_observed(o2) == []                            # 13 - 3*1 == 10 (or 11 mid-draw)


def _played_state():
    from majsoul_eye.state.replay import Replayer
    rp = Replayer()
    for ev in [
        {"type": "start_game", "id": 1},
        {"type": "start_kyoku", "bakaze": "E", "dora_marker": "1m", "honba": 1,
         "kyoku": 2, "kyotaku": 0, "oya": 1,
         "scores": [25000, 25000, 25000, 25000],
         "tehais": [["?"] * 13,
                    ["1m", "2m", "3m", "2p", "2p", "5p", "6p", "7p", "9p", "1s", "2s", "3s", "9s"],
                    ["?"] * 13, ["?"] * 13]},
        {"type": "tsumo", "actor": 1, "pai": "4p"},
        {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": False},
        {"type": "tsumo", "actor": 2, "pai": "?"},
        {"type": "dahai", "actor": 2, "pai": "2p", "tsumogiri": True},
        {"type": "pon", "actor": 1, "target": 2, "pai": "2p", "consumed": ["2p", "2p"]},
        {"type": "dahai", "actor": 1, "pai": "9p", "tsumogiri": False},
        {"type": "tsumo", "actor": 3, "pai": "?"},
        {"type": "reach", "actor": 3},
        {"type": "dahai", "actor": 3, "pai": "W", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 3},
        {"type": "tsumo", "actor": 0, "pai": "?"},
        {"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "8p"},
    ]:
        rp.apply(ev)
    return rp.state


def test_projection_relative_seats_and_zones():
    from majsoul_eye.state.observe import observed_from_board
    s = _played_state()                       # hero = abs seat 1
    o = observed_from_board(s)
    assert check_observed(o) == []
    # hero (rel 0): river [9s, 9p]; pon from rel target: target abs2 = hero+1 -> from_rel 1
    assert [t.pai for t in o.rivers[0]] == ["9s", "9p"]
    assert o.melds[0][0].type == "pon" and o.melds[0][0].from_rel == 1
    # abs2 = rel1: river had 2p but it was called away -> visible []
    assert o.rivers[1] == []
    # abs3 = rel2: riichi discard W is sideways; reach flag on
    assert [t.pai for t in o.rivers[2]] == ["W"] and o.rivers[2][0].sideways
    assert o.reach == [False, False, True, False]
    # abs0 = rel3
    assert [t.pai for t in o.rivers[3]] == ["E"]
    # hero hand excludes the fresh 8p draw
    assert o.drawn_tile == "8p" and "8p" not in o.hero_hand and len(o.hero_hand) == 10
    # HUD projection (include_hud default True): relative score order
    # rel order = [abs1, abs2, abs3, abs0]; abs3 paid 1000 for riichi
    assert o.scores == [25000, 25000, 24000, 25000]
    assert o.bakaze == "E" and o.kyoku == 2 and o.honba == 1 and o.kyotaku == 1
    assert o.seat_wind_self == "E"            # hero IS oya (kyoku 2, oya=1=hero)


def test_projection_without_hud():
    from majsoul_eye.state.observe import observed_from_board
    o = observed_from_board(_played_state(), include_hud=False)
    assert o.scores is None and o.bakaze is None and o.kyoku is None
    assert o.dora_markers == ["1m"]           # dora strip is detectable, not an HUD slot


# --- HUD x vision cross-checks (spec 2026-07-09 §3) --------------------------

def _hud_obs(**kw):
    o = ObservedState()
    o.hero_hand = ["1m"] * 4 + ["2m"] * 4 + ["3m"] * 4 + ["4m"]   # 13, none >4
    o.dora_markers = ["1p"]
    for k, v in kw.items():
        setattr(o, k, v)
    return o

assert check_observed(_hud_obs()) == []                    # HUD None -> checks dormant

# kyotaku < visible riichi count -> hard violation
bad = _hud_obs(kyotaku=0, reach=[True, False, False, False])
assert any("kyotaku" in m for m in check_observed(bad))
ok = _hud_obs(kyotaku=1, reach=[True, False, False, False])
assert not any("kyotaku" in m for m in check_observed(ok))
carry = _hud_obs(kyotaku=2, reach=[False] * 4)             # carryover only: fine
assert not any("kyotaku" in m for m in check_observed(carry))

# score conservation: sum(scores) + 1000*kyotaku == 100000
bad = _hud_obs(scores=[25000, 25000, 25000, 25000], kyotaku=1)
assert any("scores sum" in m for m in check_observed(bad))
ok = _hud_obs(scores=[24000, 25000, 25000, 25000], kyotaku=1,
              reach=[True, False, False, False])
assert not any("scores sum" in m for m in check_observed(ok))
# scores present but kyotaku unread -> conservation check stays dormant
half = _hud_obs(scores=[25000, 25000, 25000, 25000])
assert not any("scores sum" in m for m in check_observed(half))

# wall conservation: pred = 70 - sum(rivers) - n_kans - (1 if drawn)
o = _hud_obs(left_tile_count=70)
assert not any("wall count" in m for m in check_observed(o))
o = _hud_obs(left_tile_count=69)                           # +-1 tolerance
assert not any("wall count" in m for m in check_observed(o))
o = _hud_obs(left_tile_count=68)
assert any("wall count" in m for m in check_observed(o))
o = _hud_obs(left_tile_count=68, drawn_tile="9p")          # pred 69 -> |69-68|<=1
assert not any("wall count" in m for m in check_observed(o))
o = _hud_obs(left_tile_count=68)                           # a kan drops pred to 69
o.melds[1] = [ObservedMeld("ankan", ["9s", "9s", "9s", "9s"])]
assert not any("wall count" in m for m in check_observed(o))
o = _hud_obs(left_tile_count=64)
o.rivers[0] = [ObservedRiverTile("1s")] * 3
o.rivers[2] = [ObservedRiverTile("2s")] * 2                # pred = 70-5 = 65
assert not any("wall count" in m for m in check_observed(o))

print("test_observe hud cross-checks OK")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_observe OK")
