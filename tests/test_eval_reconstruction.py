# Unit tests for scripts/eval/eval_reconstruction.py helpers (QA tool).
# Plain script (no pytest dependency; also pytest-compatible).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval.eval_reconstruction import diff_zones, reject_categories
from majsoul_eye.state.observe import ObservedMeld, ObservedState


def test_reject_categories_maps_known_vocab():
    # One frame's violations -> distinct category set (frame counted once per
    # category, however many messages it produced). Vocabulary must track
    # assemble.py / observe.py message text.
    cats = reject_categories([
        "stray detection 5s (96px off-zone)",
        "stray detection 1m (260px off-zone)",
        "seat2 meld strip unparsable (3 cells)",
        "seat1 meld strip ambiguous (6 cells)",
        "seat0 river det off-grid (row residual 31px)",
        "seat3 river row1 hole (gap 92px)",
        "tile 5s seen 5>4 times across visible zones",
        "hero hand 10 + 3*0 melds != 13",
        "no dora marker visible",
        "seat 2 concealed 9 != 13(+1) for 0 melds",
    ])
    assert cats == {"stray", "meld_parse", "river_geometry", "tile_gt4",
                    "hand_size", "dora", "concealed"}


def test_reject_categories_unknown_bucket():
    assert reject_categories(["something novel"]) == {"other"}


def _obs_with_meld(**kw):
    o = ObservedState()
    o.hero_hand = ["1m"] * 10
    o.melds[1] = [ObservedMeld(type="pon", tiles=["P", "P", "P"],
                               called_pai="P", from_rel=2, **kw)]
    return o


def test_diff_zones_sees_called_pai_and_added_pai():
    # Final-review small item: obs_key used to compare melds only by
    # (type, from_rel, sorted tiles) — two kakans differing in which tile was
    # originally called (or a wrong added_pai) compared equal.
    a = ObservedState(); b = ObservedState()
    a.melds[1] = [ObservedMeld(type="kakan", tiles=["5p", "5p", "5p", "5pr"],
                               called_pai="5p", added_pai="5pr", from_rel=2)]
    b.melds[1] = [ObservedMeld(type="kakan", tiles=["5p", "5p", "5p", "5pr"],
                               called_pai="5pr", added_pai="5p", from_rel=2)]
    assert "melds" in diff_zones(a, b)
    b.melds[1] = [ObservedMeld(type="kakan", tiles=["5p", "5p", "5p", "5pr"],
                               called_pai="5p", added_pai="5pr", from_rel=2)]
    assert "melds" not in diff_zones(a, b)


def test_reject_categories_ordering_scores_before_kyotaku():
    # "scores sum" message text contains "kyotaku", so it must be checked FIRST.
    assert reject_categories(["scores sum 101000 + 1000*1 kyotaku != 100000"]) == {"hud_scores"}
    assert reject_categories(["kyotaku 0 < visible riichi count 1"]) == {"hud_kyotaku"}
    assert reject_categories(["wall count 60 vs predicted 64 (>1 off)"]) == {"hud_wall"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_eval_reconstruction OK")
