# Locator dispatch for arbitrary external screenshots (wide phones etc.).
# Plain script (no pytest dependency; also pytest-compatible).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from majsoul_eye.normalize import (BoardRegion, locate_auto, locate_fullscreen,
                                   locate_letterbox, locate_wide)


def _frame(w, h, fill=128):
    return np.full((h, w, 3), fill, dtype=np.uint8)


def test_locate_wide_centers_16x9():
    # 2.17:1 phone screenshot: the 3D table renders as a centered 16:9 rect
    # (verified on real samples); extra width is HUD-only side space.
    r = locate_wide(_frame(2868, 1320))
    bw = round(1320 * 16 / 9)
    assert (r.bw, r.bh) == (bw, 1320)
    assert r.ox == (2868 - bw) // 2 and r.oy == 0
    assert (r.fw, r.fh) == (2868, 1320)


def test_locate_fullscreen_carries_frame_dims():
    r = locate_fullscreen(_frame(1920, 1080))
    assert (r.ox, r.oy, r.bw, r.bh, r.fw, r.fh) == (0, 0, 1920, 1080, 1920, 1080)


def test_locate_auto_dispatch():
    assert locate_auto(_frame(1920, 1080)).ox == 0                    # 16:9 -> fullscreen
    assert locate_auto(_frame(2302, 1288)).ox == 0                    # ~16:9 (1.787) -> fullscreen
    wide = locate_auto(_frame(2868, 1320))                            # 2.17:1 -> wide
    assert wide.ox > 0 and wide.bw == round(1320 * 16 / 9)
    boxed = _frame(1920, 1200, fill=0)                                # letterboxed 16:9
    boxed[60:1140, :] = 128
    lb = locate_auto(boxed)
    assert (lb.oy, lb.bh) == (60, 1080)


def test_board_region_default_frame_dims_backcompat():
    # Old 4-arg construction (tests, callers) must keep working; frame dims
    # default to "unknown" and read back as the board rect itself.
    r = BoardRegion(0, 0, 1920, 1080)
    assert (r.fw or r.ox + r.bw, r.fh or r.oy + r.bh) == (1920, 1080)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_normalize OK")
