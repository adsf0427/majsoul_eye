"""Tests for coords.NormBox geometry (erode).

Plain script (also pytest-compatible). Run: PYTHONPATH=. $PY tests/test_coords.py
"""

from __future__ import annotations

from majsoul_eye.coords import NormBox


def test_erode_per_side_fractions():
    b = NormBox(0.2, 0.4, 0.6, 0.8)  # w=0.4 h=0.4
    e = b.erode(bottom=0.18, left=0.08, right=0.08)
    assert abs(e.x0 - (0.2 + 0.08 * 0.4)) < 1e-9
    assert abs(e.x1 - (0.6 - 0.08 * 0.4)) < 1e-9
    assert abs(e.y0 - 0.4) < 1e-9                       # top untouched
    assert abs(e.y1 - (0.8 - 0.18 * 0.4)) < 1e-9        # bottom trimmed


def test_erode_zero_is_identity():
    b = NormBox(0.1, 0.2, 0.3, 0.5)
    e = b.erode()
    assert (e.x0, e.y0, e.x1, e.y1) == (b.x0, b.y0, b.x1, b.y1)


def test_erode_shrinks_area():
    b = NormBox(0.0, 0.0, 1.0, 1.0)
    e = b.erode(top=0.1, bottom=0.1, left=0.1, right=0.1)
    assert e.w < b.w and e.h < b.h
    assert abs(e.w - 0.8) < 1e-9 and abs(e.h - 0.8) < 1e-9


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
