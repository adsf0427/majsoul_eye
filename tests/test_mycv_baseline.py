"""Tests for the mycv baseline scorer + engine class/seat maps.

Plain script (also pytest-compatible). Run: PYTHONPATH=. $PY tests/test_mycv_baseline.py
The engine-asset tests are skipped automatically if ../auto/mycv is absent.
"""

from __future__ import annotations

import os

from majsoul_eye.baselines.score import bag_tally, ZoneTally
from majsoul_eye.tiles import TILE_NAMES


def test_bag_perfect():
    t = bag_tally(["1m", "2m", "E"], ["1m", "2m", "E"])
    assert t.n_gt == 3 and t.n_pred == 3 and t.correct == 3
    assert t.end_to_end == 1.0 and t.precision == 1.0


def test_bag_misclassification():
    # same count, one wrong class -> class_acc 2/3, end2end 2/3
    t = bag_tally(["1m", "2m", "9p"], ["1m", "2m", "E"])
    assert t.correct == 2
    assert abs(t.end_to_end - 2 / 3) < 1e-9
    assert abs(t.precision - 2 / 3) < 1e-9


def test_bag_under_detection():
    # mycv found only 2 of 3 -> end2end 2/3 but class_acc 2/2 = 1.0
    t = bag_tally(["1m", "2m"], ["1m", "2m", "E"])
    assert t.correct == 2 and t.n_pred == 2 and t.n_gt == 3
    assert abs(t.end_to_end - 2 / 3) < 1e-9
    assert t.precision == 1.0


def test_bag_over_detection():
    # spurious extra tile -> end2end stays 3/3, class_acc 3/3 (min caps), n_pred>n_gt flags it
    t = bag_tally(["1m", "2m", "E", "E"], ["1m", "2m", "E"])
    assert t.n_pred == 4 and t.n_gt == 3 and t.correct == 3
    assert t.end_to_end == 1.0


def test_bag_duplicates_capped():
    t = bag_tally(["5m", "5m", "5m"], ["5m"])
    assert t.correct == 1  # only one real 5m


def test_red_five_lenient_vs_strict():
    # predicted plain 5m where GT is red 5mr: strict wrong, lenient right
    t = bag_tally(["5m"], ["5mr"])
    assert t.correct == 0
    assert t.correct_lenient == 1


def test_tally_aggregation():
    a = bag_tally(["1m", "2m"], ["1m", "9p"])  # correct 1 of 2
    b = bag_tally(["E", "S"], ["E", "S"])      # correct 2 of 2
    agg = ZoneTally()
    agg.add(a); agg.add(b)
    assert agg.n_gt == 4 and agg.correct == 3
    assert agg.end_to_end == 0.75


def test_screen_to_seat_consistency():
    # the engine's opponent-mask convention must match annotate.seatgt._screen_to_seat
    from majsoul_eye.annotate.seatgt import _screen_to_seat
    from majsoul_eye.baselines.mycv_engine import OPPONENT_MASKS
    hero = 1
    # raw mask k -> (hero+k)%4 ; k=1 right, k=2 across, k=3 left
    pos_for_k = {1: "right", 2: "across", 3: "left"}
    for k in OPPONENT_MASKS:
        assert _screen_to_seat(hero, pos_for_k[k]) == (hero + k) % 4


def test_engine_class_maps_are_identity():
    """mycv's class indices must align with our TILE_NAMES (no remap needed)."""
    mycv_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "auto", "mycv"))
    if not os.path.isdir(mycv_dir):
        print("SKIP engine class-map test (../auto/mycv absent)")
        return
    from majsoul_eye.baselines.mycv_engine import MycvEngine
    eng = MycvEngine(mycv_dir)
    # river/meld ResNet: 37 classes, indices 0..36 must equal TILE_NAMES[0..36]
    for idx, name in eng.river_meld_classes.items():
        assert TILE_NAMES[idx] == name, f"river/meld idx {idx}: {name} != {TILE_NAMES[idx]}"
    assert len(eng.river_meld_classes) == 37


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
