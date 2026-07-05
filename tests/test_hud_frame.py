"""annotate_frame emits hud_boxes; iter_hud_boxes flattens them; score-anim
window predicate flags reach frames."""
import numpy as np

from majsoul_eye.annotate import build_homographies, annotate_frame
from majsoul_eye.annotate.frame import iter_hud_boxes
from majsoul_eye.state.replay import BoardState, is_score_anim_window

s = BoardState(hero_seat=0, bakaze="E", kyoku=1, oya=0, in_round=True,
               scores=[25000] * 4, left_tile_count=64)
img = np.zeros((1080, 1920, 3), np.uint8)
hom = build_homographies(1920, 1080)
rec = annotate_frame(img, s, hom)
assert "hud_boxes" in rec
hb = list(iter_hud_boxes(rec))
names = {b.name for b in hb}
assert "round_label" in names and "seat_wind_self" in names
# black frame -> numeric fields have no ink -> unreliable, never wrong-text
for b in hb:
    if b.name == "score_self":
        assert b.reliable is False
    if b.name == "round_label":
        assert b.text == "E1"

assert is_score_anim_window(BoardState(last_event="reach_accepted"))
assert is_score_anim_window(BoardState(last_event="reach"))
assert not is_score_anim_window(BoardState(last_event="dahai"))
print("test_hud_frame OK")
