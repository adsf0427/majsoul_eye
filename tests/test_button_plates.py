"""Button PLATE segmentation: buttons are an overlay on an otherwise static
BTN_ZONE, so `|frame - per-game background median|` isolates the banner plate
itself -- no brightness threshold, hence no dependence on tablecloth/skin
(which flooded the old gate) or on the display language (which biased the old
glyph-anchored box by ~16px).

Measured on 30 games of datasets/v5 (STATUS §1.55):
  count-matched button recall 94.8% (old brightness gate: 53.9%)
  false positives on no-button frames 0.06%
  plate center cx by language: ja 0.6784 / zh-Hans 0.6784 / zh-Hant 0.6786
  (old glyph-anchored box: ja 0.6812 vs zh-Hans 0.6755 -- an 11px language bias)
"""
import numpy as np

from majsoul_eye.annotate.hud import (locate_button_plates, button_boxes,
                                      plate_banner_box, BTN_BANNER_W, BTN_BANNER_H)
from majsoul_eye.coords import BTN_ZONE
from majsoul_eye.normalize import locate_fullscreen
from majsoul_eye.state.replay import BoardState

W, H = 1920, 1080
region = locate_fullscreen(np.zeros((H, W, 3), np.uint8))
zx0, zy0, zx1, zy1 = region.norm_to_px(BTN_ZONE)
ZH, ZW = zy1 - zy0, zx1 - zx0
cy = (zy0 + zy1) // 2


def _bg(level=90):
    """Per-game background model of the zone: gray float32, zone-shaped."""
    return np.full((ZH, ZW), float(level), np.float32)


def _frame(plates, bg_level=90, glyphs=()):
    """Board with `plates` = list of (x0,y0,x1,y1) ORIGINAL-px banner rects."""
    img = np.zeros((H, W, 3), np.uint8)
    img[zy0:zy1, zx0:zx1] = bg_level          # static background
    for x0, y0, x1, y1 in plates:
        img[y0:y1, x0:x1] = bg_level + 60     # the translucent plate overlay
    for x0, y0, x1, y1 in glyphs:
        img[y0:y1, x0:x1] = 230               # bright calligraphy inside it
    return img


# --- plates are found by overlay-difference, regardless of brightness --------
# Two 250x96 banners, gap 40px (measured minimum adjacent-plate gap: 39px).
p1 = (zx0 + 100, cy - 48, zx0 + 350, cy + 48)
p2 = (zx0 + 390, cy - 48, zx0 + 640, cy + 48)
img = _frame([p1, p2])
got = locate_button_plates(img, region, _bg())
assert len(got) == 2, f"want 2 plates, got {len(got)}: {got}"
assert got[0][0] < got[1][0], "plates must be x-sorted"
for want, g in zip((p1, p2), got):
    assert all(abs(a - b) <= 6 for a, b in zip(want, g)), f"want ~{want}, got {g}"

# A DARK plate on a BRIGHT background is found just the same -- the old
# `gray >= 140` gate is exactly what could not do this (IMG_1964's dark 吃).
dark = _frame([p1, p2], bg_level=170)
for x0, y0, x1, y1 in (p1, p2):
    dark[y0:y1, x0:x1] = 110
assert len(locate_button_plates(dark, region, _bg(170))) == 2, \
    "dark plates on a bright table must still be found"

# Nothing rendered yet (frame == background) -> no plates. This is the real
# capture-timing case (7.2% of GT button frames) and must stay detectable.
assert locate_button_plates(_frame([]), region, _bg()) == []

# A merged/oversized FX blob is not a plate.
big = _frame([(zx0 + 100, cy - 48, zx0 + 560, cy + 48)])
assert locate_button_plates(big, region, _bg()) == [], "460px blob must be rejected"


# --- the emitted box is centered on the PLATE, not on the glyph -------------
# This is the 16px bias regression: the old banner_box anchored a fixed-size box
# on the bright glyph's centroid, so a language whose glyph sits off-center (or
# is narrower) shifted the label off the click area.
off_center_glyph = (p1[0] + 150, cy - 20, p1[0] + 240, cy + 20)   # glyph hugs the right edge
img_g = _frame([p1], glyphs=[off_center_glyph])
(plate,) = locate_button_plates(img_g, region, _bg())
box = plate_banner_box(plate)
assert box[2] - box[0] == BTN_BANNER_W and box[3] - box[1] == BTN_BANNER_H
plate_cx = (p1[0] + p1[2]) // 2
box_cx = (box[0] + box[2]) // 2
glyph_cx = (off_center_glyph[0] + off_center_glyph[2]) // 2
assert abs(box_cx - plate_cx) <= 6, f"box must center on the plate ({plate_cx}), got {box_cx}"
assert abs(box_cx - glyph_cx) > 30, "box must NOT follow the glyph centroid"


# --- button_boxes(btn_bg=...) uses plates; GT assigns classes L->R ----------
s = BoardState(hero_seat=0, pending_ops=[1, 3])            # pon offer -> pon, skip
bb = button_boxes(_frame([p1, p2]), s, region, btn_bg=_bg())
assert [b["name"] for b in bb] == ["btn_pon", "btn_skip"]
assert all(b.get("reliable", True) for b in bb)
assert all(b["px_box"][2] - b["px_box"][0] == BTN_BANNER_W for b in bb)

# Buttons not rendered yet -> count_mismatch (frame later dropped from the
# detector set by build_dataset.has_unlabeled_buttons).
bb0 = button_boxes(_frame([]), s, region, btn_bg=_bg())
assert len(bb0) == 2 and all(b["reliable"] is False for b in bb0)
assert all(b.get("flag") == "count_mismatch" for b in bb0)

# No ops offered -> no button boxes at all, whatever the pixels show.
assert button_boxes(_frame([p1, p2]), BoardState(pending_ops=[1]), region,
                    btn_bg=_bg()) == []

# --- legacy brightness gate still reachable when no background model exists --
legacy = np.zeros((H, W, 3), np.uint8)
legacy[cy - 25:cy + 25, zx0 + 100:zx0 + 260] = 220
legacy[cy - 25:cy + 25, zx0 + 400:zx0 + 560] = 220
lb = button_boxes(legacy, s, locate_fullscreen(legacy))     # btn_bg omitted
assert [b["name"] for b in lb] == ["btn_pon", "btn_skip"]


# --- boxes must be JSON-serializable plain ints ------------------------------
# connectedComponentsWithStats hands back np.int32; annotate records are written
# as JSONL, so a numpy scalar leaking into px_box kills the whole game's
# annotation with "Object of type int32 is not JSON serializable".
import json

for b in locate_button_plates(_frame([p1, p2]), region, _bg()):
    assert all(type(v) is int for v in b), f"plate box must be plain ints: {[type(v) for v in b]}"
json.dumps([list(b) for b in locate_button_plates(_frame([p1, p2]), region, _bg())])
json.dumps(button_boxes(_frame([p1, p2]), s, region, btn_bg=_bg()))


# --- annotate_frame threads btn_bg down to button_boxes ---------------------
from majsoul_eye.annotate import build_homographies, annotate_frame

hom = build_homographies(W, H)
st = BoardState(hero_seat=0, bakaze="E", kyoku=1, oya=0, in_round=True,
                scores=[25000] * 4, left_tile_count=64, pending_ops=[1, 3])
rec = annotate_frame(_frame([p1, p2]), st, hom, btn_bg=_bg())
btn = [b for b in rec["hud_boxes"] if b["name"].startswith("btn_")]
assert [b["name"] for b in btn] == ["btn_pon", "btn_skip"], btn
assert all(b.get("reliable", True) for b in btn), "plates were found; boxes must be reliable"
assert not any(f.startswith("hud:error") for f in rec["flags"]), rec["flags"]

print("test_button_plates OK")
