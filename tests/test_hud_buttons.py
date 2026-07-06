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

# --- boxes are the BANNER (click area), not the text glyph (2026-07-07) ------
# The banner is a fixed-size plate (calibrated 250x96, centered 10px below the
# text-blob center across all 7 classes / 189 measured frames); the glyph box
# varies with display language, the banner does not.
from majsoul_eye.annotate.hud import BTN_BANNER_W, BTN_BANNER_H, BTN_BANNER_DY
tcx, tcy = zx0 + 180, cy                       # first blob's center (100..260)
exp = [tcx - BTN_BANNER_W // 2, tcy + BTN_BANNER_DY - BTN_BANNER_H // 2,
       tcx + BTN_BANNER_W // 2, tcy + BTN_BANNER_DY + BTN_BANNER_H // 2]
assert list(bb[0]["px_box"]) == exp, f"want banner box {exp}, got {bb[0]['px_box']}"
assert bb[0]["px_box"][2] - bb[0]["px_box"][0] == BTN_BANNER_W
assert bb[0]["px_box"][3] - bb[0]["px_box"][1] == BTN_BANNER_H

# --- oversized text blobs (merged banners/FX) are rejected as candidates -----
# (7.2% of v3 button labels were >300px-wide merged blobs -> garbage labels;
# rejecting them degrades the frame to count_mismatch = no labels, 宁缺毋滥)
img_big = np.zeros((H, W, 3), np.uint8)
img_big[cy - 25:cy + 25, zx0 + 100:zx0 + 260] = 220    # normal blob
img_big[cy - 25:cy + 25, zx0 + 300:zx0 + 700] = 220    # 400px merged blob
cands_big = locate_button_candidates(img_big, locate_fullscreen(img_big))
assert len(cands_big) == 1, f"oversized blob must be rejected, got {cands_big}"

s2 = BoardState(hero_seat=0, pending_ops=[1, 3, 9])           # expects 3, sees 2
bb2 = button_boxes(img, s2, region)
assert len(bb2) == 3, f"Expected 3 entries, got {len(bb2)}"
assert all(b["reliable"] is False for b in bb2)
assert all(b.get("flag") == "count_mismatch" for b in bb2)
# Exactly one entry (last) has px_box=None; others have real 4-int boxes
none_count = sum(1 for b in bb2 if b["px_box"] is None)
assert none_count == 1 and bb2[-1]["px_box"] is None, \
    f"Expected last entry to have px_box=None, got {bb2[-1]['px_box']}"
for b in bb2[:-1]:
    assert isinstance(b["px_box"], (list, tuple)) and len(b["px_box"]) == 4, \
        f"Expected 4-int box, got {b['px_box']}"

# Reverse mismatch: 3 candidate blobs but only 2 expected buttons
img3 = np.zeros((H, W, 3), np.uint8)
cy = (zy0 + zy1) // 2
img3[cy - 25:cy + 25, zx0 + 50:zx0 + 210] = 220      # left blob
img3[cy - 25:cy + 25, zx0 + 300:zx0 + 460] = 220     # center blob
img3[cy - 25:cy + 25, zx0 + 550:zx0 + 710] = 220     # right blob
region3 = locate_fullscreen(img3)
cands3 = locate_button_candidates(img3, region3)
assert len(cands3) == 3, f"Expected 3 candidates, got {len(cands3)}"

s3 = BoardState(hero_seat=0, pending_ops=[1, 3])     # expects 2 (btn_pon, btn_skip)
bb3 = button_boxes(img3, s3, region3)
assert len(bb3) == 2, f"Expected exactly 2 entries (len(expected)), got {len(bb3)}"
assert all(b["reliable"] is False for b in bb3), "All entries must be unreliable"
assert all(b.get("flag") == "count_mismatch" for b in bb3), "All must have count_mismatch flag"
# No entry should carry the third candidate's box coordinates
third_cand_x0 = cands3[2][0]
for b in bb3:
    if b["px_box"]:
        assert b["px_box"][0] != third_cand_x0, \
            f"Entry {b['name']} carries third candidate's x0={third_cand_x0}"

assert button_boxes(img, BoardState(pending_ops=[1]), region) == []   # dapai only
print("test_hud_buttons OK")
