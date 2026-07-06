"""Regression test for count_dora_glow's called-tile de-dup.

A tile that was called (chi/pon/kan) stays in the discarder's river flagged
`called=True` AND appears in the caller's `Meld.tiles`. glow_eligible_tiles must
count it once (meld only), via `visible_river`. This locks the fix for the
river/meld double-count bug found in review."""

from majsoul_eye.state.replay import BoardState, RiverTile, Meld
from scripts.inspect.count_dora_glow import glow_eligible_tiles


def test_called_tile_counted_once_meld_only():
    st = BoardState()
    # seat 0 discarded 1m (kept) then 3s, which seat 1 pon'd (3s taken away)
    st.rivers[0] = [RiverTile("1m"), RiverTile("3s", called=True)]
    st.melds[1] = [Meld("pon", 0, ["3s", "3s", "3s"], called_pai="3s")]
    tiles = glow_eligible_tiles(st)
    assert tiles.count("3s") == 3      # only the 3 meld copies, NOT 4 (river excluded)
    assert "1m" in tiles              # un-called river tile still counted
    assert tiles.count("1m") == 1


def test_visible_river_only_uncalled():
    st = BoardState()
    st.rivers[2] = [RiverTile("5p"), RiverTile("7p", called=True), RiverTile("9p")]
    tiles = glow_eligible_tiles(st)
    assert tiles.count("7p") == 0      # called-away, not in any meld here
    assert set(tiles) == {"5p", "9p"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_count_dora_glow OK")
