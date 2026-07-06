"""ink_snap tightens to bright glyphs and clamps to the seed box; field_texts
maps BoardState -> reader-target strings (incl. seat rotation of scores)."""
import numpy as np

from majsoul_eye.annotate.hud import ink_snap, field_texts
from majsoul_eye.state.replay import BoardState

# --- ink_snap: bright "digits" on dark panel ---------------------------------
img = np.zeros((100, 200, 3), np.uint8)
img[40:52, 60:120] = 230                      # glyph band
snapped = ink_snap(img, (30, 20, 180, 80), pad=3)
x0, y0, x1, y1 = snapped
assert 55 <= x0 <= 58 and 117 <= x1 <= 125    # hugs 60..120 (+pad)
assert 35 <= y0 <= 38 and 49 <= y1 <= 56
assert ink_snap(np.zeros((100, 200, 3), np.uint8), (30, 20, 180, 80)) is None  # no ink
# clamp: glyph touching the seed edge must not escape it
img2 = np.zeros((100, 200, 3), np.uint8)
img2[20:80, 30:180] = 230
sx0, sy0, sx1, sy1 = ink_snap(img2, (30, 20, 180, 80), pad=5)
assert sx0 >= 30 and sy0 >= 20 and sx1 <= 180 and sy1 <= 80

# --- field_texts --------------------------------------------------------------
s = BoardState(hero_seat=2, bakaze="E", kyoku=3, honba=1, kyotaku=2, oya=1,
               in_round=True,   # riichi/honba fields are gated on in_round
               scores=[25000, 24000, 26000, 25000], left_tile_count=64)
t = field_texts(s)
assert t["score_self"] == "26000"             # scores[hero=2]
assert t["score_right"] == "25000"            # scores[3] (下家)
assert t["score_across"] == "25000"           # scores[0] (对家)
assert t["score_left"] == "24000"             # scores[1] (上家)
assert t["round_label"] == "E3"
assert t["wall_count"] == "余64"
assert t["riichi_stick_count"] == "x2" and t["honba_count"] == "x1"
assert t["seat_wind_self"] == "S"             # (2-1)%4=1 -> S
# missing GT -> field omitted
t2 = field_texts(BoardState())
assert "wall_count" not in t2 and "score_self" not in t2 and "seat_wind_self" not in t2

# --- wall_count: zero-padded text + fixed box (2026-07-07 label fix) ----------
# The client renders the count zero-padded to 2 digits (余09, not 余9), so the
# reader GT must match the pixels, and the string is constant-width -> the box
# is fixed (no extent snap; the old 42px seed clipped the digits off ALL labels).
from majsoul_eye.annotate.hud import hud_field_boxes
from majsoul_eye.coords import HUD_SEEDS
from majsoul_eye.normalize import BoardRegion

assert field_texts(BoardState(left_tile_count=4))["wall_count"] == "余04"
assert field_texts(BoardState(left_tile_count=0))["wall_count"] == "余00"

region = BoardRegion(0, 0, 1920, 1080)
frame = np.zeros((1080, 1920, 3), np.uint8)
frame[433:445, 923:997] = 230                 # 余NN ink band at its measured position
sw = BoardState(left_tile_count=43)
wc = next(d for d in hud_field_boxes(frame, sw, region) if d["name"] == "wall_count")
assert wc["px_box"] == list(region.norm_to_px(HUD_SEEDS["wall_count"]))  # fixed seed, not snapped
assert wc["px_box"][2] - wc["px_box"][0] >= 80    # covers 余 + both digits (~74px ink)
assert wc.get("reliable", True)
# not rendered (no ink in the 余 probe region) -> unreliable, same policy as before
wc2 = next(d for d in hud_field_boxes(np.zeros((1080, 1920, 3), np.uint8), sw, region)
           if d["name"] == "wall_count")
assert wc2.get("reliable", True) is False
print("test_hud_fields OK")
