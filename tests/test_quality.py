"""Tests for the crop-quality gate (label/quality.py).

Plain script (also pytest-compatible). Run: PYTHONPATH=. $PY tests/test_quality.py
"""

from __future__ import annotations

import numpy as np

from majsoul_eye.label.quality import tile_face_fraction, tile_back_fraction, is_tile_present


def _felt(h=80, w=64):
    # blue table felt: B high, R/G low (BGR order)
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = 110   # B
    img[:, :, 1] = 70    # G
    img[:, :, 2] = 40    # R
    return img


def _tile(h=80, w=64):
    # white/cream tile face with some dark ink
    img = np.full((h, w, 3), 200, np.uint8)
    img[20:60, 24:40] = 30   # ink strokes (a chunk dark)
    return img


def test_felt_scores_near_zero():
    assert tile_face_fraction(_felt()) < 0.05
    assert not is_tile_present(_felt())


def test_tile_scores_high():
    f = tile_face_fraction(_tile())
    assert f > 0.5
    assert is_tile_present(_tile())


def test_empty_crop_is_zero():
    assert tile_face_fraction(np.zeros((0, 0, 3), np.uint8)) == 0.0
    assert not is_tile_present(np.zeros((0, 0, 3), np.uint8))


def test_half_slid_tile_between():
    # top half tile, bottom half felt -> fraction ~0.5, above the 0.35 default
    img = _felt()
    img[:40, :] = 200
    f = tile_face_fraction(img)
    assert 0.35 < f < 0.65
    assert is_tile_present(img)


def test_threshold_boundary():
    # a cell only ~20% covered by tile -> below 0.35 default -> dropped
    img = _felt()
    img[:16, :] = 200   # 16/80 = 0.20
    assert not is_tile_present(img)
    assert is_tile_present(img, min_face_frac=0.15)


def _back(h=80, w=64):
    # face-down tile back: warm orange/tan (BGR), R appreciably > B (measured ~(52,138,209))
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = 52    # B
    img[:, :, 1] = 138   # G
    img[:, :, 2] = 209   # R
    return img


def test_tile_back_scores_high_on_back_fraction_not_face_fraction():
    # a face-down tile is neither felt nor a white face -- must still count as "present"
    b = _back()
    assert tile_face_fraction(b) < 0.05          # not white/cream
    assert tile_back_fraction(b) > 0.9            # but is warm/orange
    assert is_tile_present(b)                     # so is_tile_present recognizes it


def test_felt_scores_near_zero_on_back_fraction_too():
    # felt (B > R) must not be mistaken for a tile back (R > B)
    assert tile_back_fraction(_felt()) < 0.05
    assert not is_tile_present(_felt())


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
