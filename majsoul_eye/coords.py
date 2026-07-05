"""Normalized coordinate model for Majsoul easy-zone ROIs.

All boxes are normalized to 0–1 against a canonical 16:9 board, so they apply at
any resolution once a frame has been mapped to the canonical frame (see
``normalize.py``). Seed values are ported from ``auto/mycv`` (the proven 1920×1080
*web-client* layout) and Akagi's normalized hand slots.

⚠️ CALIBRATE: these seeds are from the web client at 1920×1080. Verify/adjust them
against a real ``session2`` capture (and again per target client — the Win client
may differ slightly). Edit the pixel tables below; everything else derives.
"""

from __future__ import annotations

from dataclasses import dataclass

MYCV_REF = (1920, 1080)  # reference resolution the mycv pixel ROIs were measured at


@dataclass(frozen=True)
class NormBox:
    """Axis-aligned box in normalized [0,1] coords (x0,y0 top-left, x1,y1 bot-right)."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float: return self.x1 - self.x0
    @property
    def h(self) -> float: return self.y1 - self.y0
    @property
    def cx(self) -> float: return (self.x0 + self.x1) / 2
    @property
    def cy(self) -> float: return (self.y0 + self.y1) / 2

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Map to integer pixel box (x0,y0,x1,y1) on a width×height canonical frame."""
        return (round(self.x0 * width), round(self.y0 * height),
                round(self.x1 * width), round(self.y1 * height))

    def yolo(self) -> tuple[float, float, float, float]:
        """YOLO box (cx,cy,w,h), all normalized."""
        return (self.cx, self.cy, self.w, self.h)

    def erode(self, top: float = 0.0, bottom: float = 0.0,
              left: float = 0.0, right: float = 0.0) -> "NormBox":
        """Shrink the box by per-side fractions of its own width/height. Used to
        trim next-tile bleed off packed river cells (the next discard bleeds into
        the bottom of a cell — see scripts/train/build_dataset.py --river-erode-*)."""
        w, h = self.w, self.h
        return NormBox(self.x0 + left * w, self.y0 + top * h,
                       self.x1 - right * w, self.y1 - bottom * h)


def px_box(x0: int, y0: int, x1: int, y1: int, ref: tuple[int, int] = MYCV_REF) -> NormBox:
    """Build a NormBox from a pixel ROI measured at `ref` resolution."""
    W, H = ref
    return NormBox(x0 / W, y0 / H, x1 / W, y1 / H)


# --- easy-zone ROIs (ported from mycv main2.py get_* methods, 1920×1080) -----
# Format: name -> pixel ROI (x0, y0, x1, y1).  # CALIBRATE on session2.
_MYCV_ROIS_PX: dict[str, tuple[int, int, int, int]] = {
    "riichi_sticks": (110, 140, 140, 180),   # get_bangzi
    "honba":         (250, 140, 280, 180),   # get_benchang
    "self_wind":     (800, 500, 850, 530),   # get_zifeng
    "round_wind":    (920, 390, 950, 420),   # get_changfeng
    "round_number":  (950, 390, 970, 420),   # get_jushu
    # per-seat score readouts (union of mycv's per-digit ROIs)
    "score_self":    (915, 472, 1000, 492),  # get_f1  (bottom / self)
    "score_right":   (863, 396, 888, 428),   # get_f2  (right, stacked digits)
    "score_across":  (951, 355, 995, 373),   # get_f3  (top / across)
    # "score_left":  get_f4 is computed differently — CALIBRATE/derive on session2
}

REGIONS: dict[str, NormBox] = {name: px_box(*roi) for name, roi in _MYCV_ROIS_PX.items()}

# Dora indicators are NOT on the dead wall — they're in the TOP-LEFT HUD panel,
# growing rightward as kan-dora reveal. NOTE: this is 2D HUD, so it's
# RESOLUTION-DEPENDENT (like scores) — fine at this 16:9 res; needs
# anchor-normalization for other resolutions.
# x-extent data-calibrated: brightness-scan of the 5-slot strip on session6 puts
# tile seams at px 82/138/194/250/306 → pitch ~56 px (0.0292), first edge ~28 px.
# The earlier single-tile 0.0395 width was ~33% too wide (boxes bled into the next
# slot), so x1 pulled 0.2125 → 0.1608 (5 × 56 px).
DORA_STRIP: NormBox = NormBox(0.015, 0.036, 0.1608, 0.111)
MAX_DORA = 5


def dora_slot(i: int, n: int = MAX_DORA) -> NormBox:
    """i-th dora indicator slot within DORA_STRIP (left→right)."""
    s = DORA_STRIP
    w = s.w / n
    return NormBox(s.x0 + i * w, s.y0, s.x0 + (i + 1) * w, s.y1)


# --- hero hand-tile slots ----------------------------------------------------
# Parametric model. Seeds: mycv flood-fill tiles are ~95×152 px at 1920×1080
# starting x≈235, y≈1002 (seed point inside tile); Akagi's normalized hand centers
# run x 0.139→0.788 at y 0.929. We model 13 concealed slots + a separated tsumo slot.
# x0 data-calibrated: brightness-scan of the first-tile left edge over 132 rendered
# session6 hands gives 223 px (mycv's 235 seed sat +12 px right); slot width 94.5≈95.
@dataclass(frozen=True)
class HandModel:
    x0: float = 223 / 1920          # left edge of first tile (measured)
    slot_w: float = 95 / 1920       # tile width (also slot pitch)  # measured 94.5
    y0: float = (1002 - 76) / 1080  # tile top (seed 1002 is mid; half-height ~76)
    tile_h: float = 152 / 1080      # tile height                   # CALIBRATE
    tsumo_gap: float = 0.015        # extra gap before the drawn tile (Akagi TSUMO_OFFSET_X)

    def slot_box(self, i: int, is_tsumo: bool = False) -> NormBox:
        """Box for the i-th hand tile (0-based). `is_tsumo` adds the drawn-tile gap."""
        x = self.x0 + i * self.slot_w + (self.tsumo_gap if is_tsumo else 0.0)
        return NormBox(x, self.y0, x + self.slot_w, self.y0 + self.tile_h)


HAND = HandModel()


# --- perspective river zones (screen-quadrant bounds, mycv lizhipai, 1080p) ---
# Coarse bounding boxes per opponent quadrant — used by P4 to route contour-detected
# tiles to a seat. Hero's own 河 (bottom) handled separately.  # CALIBRATE.
RIVER_ZONES_PX: dict[str, tuple[int, int, int, int]] = {
    "self":   (760, 600, 1180, 840),   # 自家 (bottom-center) — CALIBRATE (mycv read elsewhere)
    "across": (770, 130, 1140, 290),   # 对家 (top)
    "left":   (450, 265, 790, 650),    # 上家
    "right":  (1120, 265, 1430, 650),  # 下家
}
RIVER_ZONES: dict[str, NormBox] = {k: px_box(*v) for k, v in RIVER_ZONES_PX.items()}

# NOTE: the per-seat 河/副露 geometry (``RIVER_QUADS`` / ``MELD_STRIPS``, the
# equal-subdivision RiverGrid model) was removed here — it is superseded by the
# precise fullwarp annotator in ``majsoul_eye.annotate`` (data-calibrated
# DISCARD_GRID / composition-aware melds). See docs/STATUS.md §1.13.


# --- HUD field seed ROIs (px @ 1920x1080 web client) --------------------------
# CALIBRATED (Task 6 of the HUD plan) against real run_3/run_4/run_5/run_8
# ai_session frames (scripts/inspect/overlay_hud.py) — every box below was found
# by scanning per-row/per-column ink-brightness (`gray>=INK_THRESH`) profiles
# inside the center diamond + top-left panel, not eyeballed. Two seeds from the
# Task-5 seed guess were WRONG (not just loose) and had to move to a different
# glyph entirely:
#   - "round_label"/"wall_count" were swapped one row too high: the old
#     round_label box (y350-385) actually framed score_across's upside-down
#     digits, and the old wall_count box (y385-415) framed round_label's own
#     東N局 text. wall_count (余NN) is a further row down (y422-455).
#   - "seat_wind_self" pointed at empty panel chrome; the real corner wind tag
#     is BELOW the diamond (y488-540, near score_self), not beside score_left.
#     Verified via an E3 frame where hero is oya (seat_wind_self='E'): this box
#     turns solid red exactly like the other seats' oya-highlighted corner tags,
#     while the neighboring top-left corner tag (some other seat, out of scope —
#     HUD_NAMES has no seat_wind_left/across/right) does NOT.
# A SECOND calibration pass (still Task 6, after the first landed) found every
# numeric seed still touching a bright non-glyph neighbor on >=1 side, verified
# by re-running ink_snap and checking whether the *raw* (pre-pad) ink bbox sits
# within 1-2px of the seed edge across dozens of frames/games (a coincidental
# pad-only touch is fine; a raw-ink touch means real contamination bleeding the
# box open on that side):
#   - score_self/left/right/across each sit at one vertex of the center diamond,
#     which has a decorative glint/highlight (a plain glow normally, an ornate
#     flame/crown motif when the seat is oya) right at that vertex, OUTSIDE the
#     digits but INSIDE the old seed — score_self's seed reached down into the
#     glow below the digits (y0/y1 trimmed 460-500 -> 467-497), score_across's
#     reached up into the glow/crown above them (y0 pushed 325 -> 353),
#     score_left's reached into the glow at its own vertex (x0 pushed
#     850 -> 858), score_right's reached into the glow at its vertex (x1 pulled
#     in 1068 -> 1063).
#   - riichi_stick_count/honba_count reached down into the panel's own bottom
#     trim (a bright ~3px border, full seed width, right above y=180) — trimmed
#     y1 185 -> 175 for both. honba_count *also* reached right into the panel's
#     slanted bottom-right corner cut (measured creep-start as far left as
#     x=302 at the bottom sampled row) — trimmed x1 318 -> 300. Both also
#     reached left into their own icon glyph (the riichi-stick/honba-dice icon
#     to the left of "x N" is itself a bright white glyph, not glow) — x0 pulled
#     in a further 3px (88->85, 225->222) to land in the clean gap between icon
#     and text instead of flush against the text's own left edge.
#   - wall_count reached up into round_label's own descender bleed (y0 pushed
#     422 -> 427) and right into the panel's corner bezel highlight, a smaller
#     analog of honba_count's corner-cut problem (x1 pulled in 1010 -> 952).
# score_left/right also had 300%+ contamination risk from the Task-5 pass: their
# seed height reached up into an unrelated corner wind-tag badge, and
# score_right's width reached into a same-row bright blob (a per-seat decorative
# hand/tile graphic that can render beside the score) — both trimmed to the
# actual digit column measured via cv2.connectedComponentsWithStats.
#
# RESIDUAL / KNOWN LIMITATION (documented, not fixed here — see module docstring
# "2D HUD does not scale... resolution-dependent"): all of the above is verified
# clean (no raw-ink touch within 1-2px of any seed edge, sampled ~30 frames each
# across run_3 x4 games / run_4 / run_5 game1 / run_7 / run_8 x6 games / run_13 /
# run_14 — all native 1920x1080) plus 3 harmless one-off outliers (a mid-animation
# hand graphic transiently covering score_across; one frame that GT-paired to a
# client loading screen, a sync artifact unrelated to seeds; one 1px-tight but
# uncontaminated score_across touch). run_5/game2+game3 capture at 1923x1142
# (aspect 1.684, NOT 16:9 like every other capture) and show real contamination
# on riichi_stick_count/honba_count/score_across specifically because that
# resolution's top-left panel does not scale the icon<->text gap the same way
# the 1920x1080-derived normalized seed assumes — a non-uniform-scaling artifact
# of a non-16:9 source, not a seed placement error. Proper fix is the
# `AnchorLocator` TODO in normalize.py; out of scope for Task 6's fixed-slot seeds.
_HUD_SEEDS_PX: dict[str, tuple[int, int, int, int]] = {
    "score_self":         (900, 467, 1020, 497),   # y0/y1 trimmed clear of the diamond-vertex glow (see note)
    "score_right":        (1028, 385, 1063, 462),  # vertical digits; x1 trimmed short of the vertex glow (~x1067)
    "score_across":       (895, 353, 1030, 383),   # upside-down digits; y0 trimmed below the vertex glow/crown, y1 widened (to just short of round_label's seed) for bottom margin
    "score_left":         (858, 385, 900, 462),    # vertical digits; y0 trimmed below the corner wind-tag badge; x0 trimmed clear of the vertex glow (~x854)
    "round_label":        (912, 384, 1008, 414),   # 東N局 (below score_across, above wall_count)
    "wall_count":         (910, 427, 952, 455),    # 余NN (below round_label); y0 trimmed clear of round_label's descender bleed, x1 trimmed clear of the panel's corner bezel highlight (~x956)
    "seat_wind_self":     (793, 488, 852, 540),    # corner wind tag beside score_self (see note above)
    "riichi_stick_count": (85, 135, 178, 175),      # mycv get_bangzi; x0 clears the icon (ends ~x67) w/ 3px margin before the "x" glyph (starts ~x88), y1 trimmed clear of the panel's bottom trim (~y180)
    "honba_count":        (222, 135, 300, 175),     # mycv get_benchang; x0 clears the icon (ends ~x205) w/ 3px margin before the "x" glyph (starts ~x225), x1/y1 trimmed clear of the panel's bottom-right corner cut
}
HUD_SEEDS: dict[str, NormBox] = {k: px_box(*v) for k, v in _HUD_SEEDS_PX.items()}
