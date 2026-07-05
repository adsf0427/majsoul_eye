"""Button locator: bright banners on dark strip -> x-sorted candidates; GT op
set assigns classes; candidate/expected count mismatch -> all unreliable."""
import numpy as np

from majsoul_eye.annotate.hud import locate_button_candidates, button_boxes
from majsoul_eye.coords import BTN_ZONE
from majsoul_eye.normalize import locate_fullscreen
from majsoul_eye.state.replay import BoardState

W, H = 1920, 1080
img = np.zeros((H, W, 3), np.uint8)
# BTN_ZONE 内画两个高亮"按钮"（左：碰位，右：跳过位）
zx0, zy0, zx1, zy1 = (int(v) for v in (BTN_ZONE.x0 * W, BTN_ZONE.y0 * H,
                                       BTN_ZONE.x1 * W, BTN_ZONE.y1 * H))
cy = (zy0 + zy1) // 2
img[cy - 25:cy + 25, zx0 + 100:zx0 + 260] = 220
img[cy - 25:cy + 25, zx0 + 400:zx0 + 560] = 220
region = locate_fullscreen(img)

cands = locate_button_candidates(img, region)
assert len(cands) == 2 and cands[0][0] < cands[1][0]          # x-sorted

s = BoardState(hero_seat=0, pending_ops=[1, 3])               # pon offer
bb = button_boxes(img, s, region)
assert [b["name"] for b in bb] == ["btn_pon", "btn_skip"]     # left->right order rule
assert all(b.get("reliable", True) for b in bb)

s2 = BoardState(hero_seat=0, pending_ops=[1, 3, 9])           # expects 3, sees 2
bb2 = button_boxes(img, s2, region)
assert bb2 and all(b["reliable"] is False for b in bb2)
assert all(b.get("flag") == "count_mismatch" for b in bb2)

assert button_boxes(img, BoardState(pending_ops=[1]), region) == []   # dapai only
print("test_hud_buttons OK")
