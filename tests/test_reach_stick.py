"""reach_stick_boxes (Task 17a/17c, spec §10): one box per seat currently in
riichi, single detector class 'reach_stick' (symmetric object -> per-seat
classes are appearance-degenerate) + a 'slot' debug/QA field mapped
hero-relative like the score_* fields; label-only render check (bright-bar
fill) flags a box unreliable when nothing has rendered yet."""
import numpy as np

from majsoul_eye.annotate.hud import REACH_FILL_OK, reach_stick_boxes
from majsoul_eye.coords import REACH_STICK_SEEDS
from majsoul_eye.normalize import locate_fullscreen
from majsoul_eye.state.replay import BoardState

W, H = 1920, 1080
region = locate_fullscreen(np.zeros((H, W, 3), np.uint8))

# --- reach[0] True, hero=0 -> slot 'self', bar rendered -> reliable -----
img = np.zeros((H, W, 3), np.uint8)
x0, y0, x1, y1 = (int(v) for v in region.norm_to_px(REACH_STICK_SEEDS["self"]))
img[y0:y1, x0:x1] = 220                       # bright bar fills the whole slot

s = BoardState(hero_seat=0, reach=[True, False, False, False])
boxes = reach_stick_boxes(img, s, region)
assert len(boxes) == 1
b = boxes[0]
assert b["name"] == "reach_stick"          # single class -- see spec §10
assert b["slot"] == "self"                 # attribution/debug metadata, not the class
assert list(b["px_box"]) == [x0, y0, x1, y1]
assert b.get("reliable", True) is True
assert b["fill"] > REACH_FILL_OK

# --- reach all False -> no boxes ---------------------------------------------
s2 = BoardState(hero_seat=0, reach=[False, False, False, False])
assert reach_stick_boxes(img, s2, region) == []

# --- seat mapping: hero=2, seat 0 in reach -> slot 'across' -------------------
# (hero+i)%4==0 for i=2 -> REACH_STICK_SLOTS[2] == "across"
black = np.zeros((H, W, 3), np.uint8)
s3 = BoardState(hero_seat=2, reach=[True, False, False, False])
boxes3 = reach_stick_boxes(black, s3, region)
assert len(boxes3) == 1
assert boxes3[0]["name"] == "reach_stick"
assert boxes3[0]["slot"] == "across"

# --- fill gate is scoped to the reach window (2026-07-07 label fix) ----------
# The stick can only be missing/occluded on frames whose record is still in the
# reach window (is_score_anim_window); once settled it stays rendered to the end
# of the kyoku. A dark skinned stick (sword/syringe) fails the gray>=150 fill
# even fully rendered, so off-window frames must trust GT (the old
# unconditional gate silently dropped ~20% of across/left sticks -> trained the
# detector to treat skinned sticks as background).
s4 = BoardState(hero_seat=0, reach=[True, False, False, False])   # settled
boxes4 = reach_stick_boxes(black, s4, region)
assert len(boxes4) == 1
assert boxes4[0]["slot"] == "self"
assert boxes4[0]["fill"] < REACH_FILL_OK              # dark slot...
assert boxes4[0].get("reliable", True) is True        # ...but GT wins off-window

# in-window (reach_accepted bundled with next actor's tsumo) + unrendered slot
# -> the per-box safety net still fires
s5 = BoardState(hero_seat=0, reach=[True, False, False, False],
                last_event_types=frozenset({"reach_accepted", "tsumo"}))
boxes5 = reach_stick_boxes(black, s5, region)
assert boxes5[0]["reliable"] is False
# in-window but the bar HAS rendered -> fill passes, box stays reliable
boxes6 = reach_stick_boxes(img, s5, region)
assert boxes6[0].get("reliable", True) is True

# --- hero_seat unknown (-1) guard --------------------------------------------
s5 = BoardState(hero_seat=-1, reach=[True, False, False, False])
assert reach_stick_boxes(img, s5, region) == []

print("test_reach_stick OK")
