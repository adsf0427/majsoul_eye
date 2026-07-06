"""reach_stick_boxes (Task 17a, spec §10): one box per seat currently in
riichi, mapped hero-relative like the score_* fields; label-only render check
(bright-bar fill) flags a box unreliable when nothing has rendered yet."""
import numpy as np

from majsoul_eye.annotate.hud import REACH_FILL_OK, reach_stick_boxes
from majsoul_eye.coords import REACH_STICK_SEEDS
from majsoul_eye.normalize import locate_fullscreen
from majsoul_eye.state.replay import BoardState

W, H = 1920, 1080
region = locate_fullscreen(np.zeros((H, W, 3), np.uint8))

# --- reach[0] True, hero=0 -> reach_stick_self, bar rendered -> reliable -----
img = np.zeros((H, W, 3), np.uint8)
x0, y0, x1, y1 = (int(v) for v in region.norm_to_px(REACH_STICK_SEEDS["reach_stick_self"]))
img[y0:y1, x0:x1] = 220                       # bright bar fills the whole slot

s = BoardState(hero_seat=0, reach=[True, False, False, False])
boxes = reach_stick_boxes(img, s, region)
assert len(boxes) == 1
b = boxes[0]
assert b["name"] == "reach_stick_self"
assert list(b["px_box"]) == [x0, y0, x1, y1]
assert b.get("reliable", True) is True
assert b["fill"] > REACH_FILL_OK

# --- reach all False -> no boxes ---------------------------------------------
s2 = BoardState(hero_seat=0, reach=[False, False, False, False])
assert reach_stick_boxes(img, s2, region) == []

# --- seat mapping: hero=2, seat 0 in reach -> reach_stick_across -------------
# (hero+i)%4==0 for i=2 -> REACH_STICK_NAMES[2] == "reach_stick_across"
black = np.zeros((H, W, 3), np.uint8)
s3 = BoardState(hero_seat=2, reach=[True, False, False, False])
boxes3 = reach_stick_boxes(black, s3, region)
assert len(boxes3) == 1 and boxes3[0]["name"] == "reach_stick_across"

# --- no bar rendered (black frame) -> reliable False -------------------------
s4 = BoardState(hero_seat=0, reach=[True, False, False, False])
boxes4 = reach_stick_boxes(black, s4, region)
assert len(boxes4) == 1
assert boxes4[0]["name"] == "reach_stick_self"
assert boxes4[0]["reliable"] is False
assert boxes4[0]["fill"] < REACH_FILL_OK

# --- hero_seat unknown (-1) guard --------------------------------------------
s5 = BoardState(hero_seat=-1, reach=[True, False, False, False])
assert reach_stick_boxes(img, s5, region) == []

print("test_reach_stick OK")
