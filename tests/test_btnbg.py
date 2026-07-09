"""annotate/btnbg.py: per-game BTN_ZONE background model.

The model is the median of the zone over frames GT says carry NO buttons. The
median (not the mean) is what makes it survive the animated tablecloths, rain FX
and breathing 立绘 that live in that zone -- and it must never be built from a
frame that has a button on it, or the plate would be baked into the background.
"""
import os
import tempfile

import cv2
import numpy as np

from majsoul_eye.annotate import btnbg
from majsoul_eye.coords import BTN_ZONE
from majsoul_eye.normalize import locate_fullscreen
from majsoul_eye.state.replay import BoardState

W, H = 1920, 1080
region = locate_fullscreen(np.zeros((H, W, 3), np.uint8))
zx0, zy0, zx1, zy1 = region.norm_to_px(BTN_ZONE)
cy = (zy0 + zy1) // 2

tmp = tempfile.mkdtemp()


def _write(seq, plate=False, transient_x=None):
    img = np.zeros((H, W, 3), np.uint8)
    img[zy0:zy1, zx0:zx1] = 90                      # static table
    if transient_x is not None:                     # animation: bright FX, moves each frame
        img[cy - 30:cy + 30, transient_x:transient_x + 120] = 250
    if plate:
        img[cy - 48:cy + 48, zx0 + 100:zx0 + 350] = 150
    p = os.path.join(tmp, f"{seq:06d}.png")
    cv2.imwrite(p, img)
    return p


NO_BTN = BoardState(hero_seat=0, pending_ops=[1])            # dapai only -> no buttons
BTN = BoardState(hero_seat=0, pending_ops=[1, 3])            # pon offer -> buttons

frames, states = {}, {}
for i in range(12):                                          # 12 clean frames, FX sweeps across
    frames[i] = _write(i, transient_x=zx0 + 40 * i)
    states[i] = NO_BTN
for i in (12, 13):                                           # 2 button frames -> must be excluded
    frames[i] = _write(i, plate=True)
    states[i] = BTN

bg = btnbg.game_btn_background(states, frames)
assert bg is not None
assert bg.shape == (zy1 - zy0, zx1 - zx0), f"zone-shaped, got {bg.shape}"
assert bg.dtype == np.float32

# The median killed the sweeping FX: background is the static table everywhere.
assert abs(float(np.median(bg)) - 90) <= 1, f"median background {np.median(bg)}"
assert float(bg.max()) <= 120, f"transient FX leaked into the model (max {bg.max()})"

# The plate from the button frames must NOT be in the background -- if it were,
# the very buttons we want to find would cancel out against it.
plate_patch = bg[cy - 48 - zy0:cy + 48 - zy0, 100:350]
assert abs(float(np.median(plate_patch)) - 90) <= 1, "a button plate leaked into the background"

# Too few clean frames -> no model (caller must fall back / skip the game).
few = {i: frames[i] for i in range(4)}
assert btnbg.game_btn_background({i: NO_BTN for i in few}, few) is None

# A game where every frame offers buttons -> no clean frames -> None.
allbtn = {i: frames[i] for i in range(12)}
assert btnbg.game_btn_background({i: BTN for i in allbtn}, allbtn) is None

print("test_btnbg OK")
