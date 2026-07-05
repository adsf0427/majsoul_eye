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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_observe OK")
