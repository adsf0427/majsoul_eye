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
# inside the center diamond + top-left panel, not eyeballed. All 9 fields moved
# from their initial Task-5 seed guess; two major wrong-glyph relocations:
#   - "round_label" was one row TOO HIGH (y350-385 framed score_across's
#     upside-down digits, not 東N局). RELOCATED to y384-414 (below
#     score_across, above wall_count).
#   - "seat_wind_self" pointed at EMPTY PANEL CHROME beside score_left. RELOCATED
#     to y488-540 (below the score-diamond, beside score_self, where the
#     corner wind tag actually is). Verified via an E3 frame where hero is
#     oya (seat_wind_self='E'): this box turns solid red exactly like the
#     other seats' oya-highlighted corner tags.
# All 9 seeds were then calibrated pixel-tight against the center diamond's
# decorative glint/highlight and the top-left panel's borders by scanning ink
# with cv2.connectedComponentsWithStats, stripping non-glyph neighbors (verified
# across ~30 frames per session, run_3 x4 games / run_4 / run_5 game1 / run_7 /
# run_8 x6 games / run_13 / run_14):
#   - score_self (900,460,1020,500) → (900,467,1020,497): y0/y1 trimmed clear
#     of the diamond-vertex glow below the digits.
#   - score_right (1040,330,1085,460) → (1028,385,1063,462): major relocation:
#     old y-range (330-460) spanned two unrelated HUD elements; new y-range
#     (385-462) matches the actual 下家 digit column's vertical span; x0/x1
#     trimmed clear of glow.
#   - score_across (900,295,1020,335) → (895,353,1030,383): relocated down
#     (y0 295→353) into the actual upside-down digit column; x0 pushed inward
#     (900→895), x1 widened (1020→1030) to include all digits w/ margin.
#   - score_left (835,330,880,460) → (858,385,900,462): major relocation to
#     actual 上家 digit column (y0 330→385, aligning with score_right); x0/x1
#     trimmed clear of the glow at the diamond's left vertex.
#   - wall_count (925,385,995,415) → (910,427,952,455): RELOCATED one row down
#     (y0 385→427, y1 415→455) into round_label's old row; x0 pushed inward
#     (925→910), x1 pulled in tight (995→952) clear of the panel's corner
#     bezel highlight.
#   - riichi_stick_count (95,135,175,185) → (85,135,178,175): x0 pulled in
#     tight (95→85) from the icon's edge; y1 trimmed (185→175) clear of the
#     panel's bottom trim.
#   - honba_count (235,135,315,185) → (222,135,300,175): x0 pulled in tight
#     (235→222) from the icon's edge; x1/y1 trimmed (315→300, 185→175) clear
#     of the panel's bottom-right corner cut.
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
# of a non-16:9 source, not a seed placement error.
#   RESOLVED for the RECOGNIZE path (2026-07-12): these seeds are training-time
#   only. recognize/hudstate.py crops each HUD field from its OWN detection box
#   (`frame_bgr[y0:y1, x0:x1]`), never from a seed, so a reflowing top-left panel
#   cannot contaminate a read no matter the resolution. The seeds below still
#   generate labels, and there every frame is a clean 1920x1080 render.
_HUD_SEEDS_PX: dict[str, tuple[int, int, int, int]] = {
    "score_self":         (900, 467, 1020, 497),   # y0/y1 trimmed clear of the diamond-vertex glow (see note)
    "score_right":        (1028, 385, 1063, 462),  # vertical digits; x1 trimmed short of the vertex glow (~x1067)
    "score_across":       (895, 353, 1030, 383),   # upside-down digits; y0 trimmed below the vertex glow/crown, y1 widened (to just short of round_label's seed) for bottom margin
    "score_left":         (858, 385, 900, 462),    # vertical digits; y0 trimmed below the corner wind-tag badge; x0 trimmed clear of the vertex glow (~x854)
    "round_label":        (912, 384, 1008, 414),   # 東N局 (below score_across, above wall_count)
    "wall_count":         (918, 428, 1002, 452),   # 余NN (below round_label). FIXED box, no extent snap: the count renders zero-padded to 2 digits, so the string is constant-width — glyph ink measured x923-997 y432-448, identical across ai_session/2/3 incl. the cloud table (2026-07-07 resurvey). The old 42px seed (x1=952, trimmed for a bezel highlight ~x956) sat BEFORE the digits (~x955+) and clipped them out of every label; with no snap, bezel/score-glow bleed can't stretch the box — render presence is probed on the 余 subregion only (WALL_COUNT_INK_PROBE)
    "seat_wind_self":     (793, 488, 852, 540),    # corner wind tag beside score_self (see note above)
    "riichi_stick_count": (85, 135, 178, 175),      # mycv get_bangzi; x0 clears the icon (ends ~x67) w/ 3px margin before the "x" glyph (starts ~x88), y1 trimmed clear of the panel's bottom trim (~y180)
    "honba_count":        (222, 135, 300, 175),     # mycv get_benchang; x0 clears the icon (ends ~x205) w/ 3px margin before the "x" glyph (starts ~x225), x1/y1 trimmed clear of the panel's bottom-right corner cut
}
HUD_SEEDS: dict[str, NormBox] = {k: px_box(*v) for k, v in _HUD_SEEDS_PX.items()}

# Render-presence probe for the fixed wall_count box: just the 余 glyph's cell
# (ink x923-948 — always present when the field renders, and safely inside the
# seed away from the bezel/score glow that contaminates a full-width ink scan).
WALL_COUNT_INK_PROBE = px_box(920, 430, 950, 450)


# Action-button strip (bottom, above the hand).
# CALIBRATED (Task 7 Step 5) against 22 real button frames in
# captures/raw/ai_session3/run_1/game1 (chi x12, pon x8, kan x3, ron x2,
# riichi x1). The Task-5 seed guess (0.30, 0.66, 0.98, 0.82 = px 576,713,1882,886
# @1920x1080) contained every real banner but ALSO two fixed HUD elements that
# sit in the same row and are bright enough to pass BTN_THRESH:
#   - a "<seat count> +20" turn/bonus indicator at px x~1502-1711 (present on
#     EVERY frame regardless of buttons) — caused every one of the 22 frames to
#     over-count by +1 candidate (100% count_mismatch before this fix).
#   - a small sakura-petal "callable tile" badge that floats above the discard
#     row at px y~713-747 (2 of 22 frames) — a separate, disconnected blob from
#     the button banner's own text glyph.
# Real button banners (chi/pon/kan/riichi/ron/skip) all sit within px
# x 663-1389, y 779-850 (skip is widest, up to x1=1389). Zone tightened to
# x1=0.74 (1420 px, well short of the 1502 px turn-indicator) and y0=0.705
# (761 px, below the 747 px badge, above the 779 px banner tops) — comfortable
# margin on both sides, verified 0 count_mismatch across all 22 real frames.
# ⚠️ KNOWN LIMIT (unverified ≥4-button rows): observed banner pitch is ~270-300 px,
# so a 4th action banner would land at px ~1500-1530 — INSIDE the excluded
# turn-indicator's territory. Such frames were absent from the Task-7 harvest; no
# static zone can hold both. If one appears it degrades to count_mismatch (frame
# contributes no button labels) — revisit with a smarter confounder filter then.
BTN_ZONE = NormBox(0.30, 0.705, 0.74, 0.82)


# --- reach-stick (立直棒) slots, one per seat (Task 17a/17c/17b; spec §10) ---
# CALIBRATED (T17b) against real reach-accepted frames (captures/raw/ai_session/
# run_3/{game1,game3,game4} + ai_session3/run_2/game1 [different "cloud" table
# skin — robustness check] + ai_session3/run_3/game1; scripts/inspect/overlay_hud.py).
# The Task-17a guess this replaces (across: px 905,278,1010,299) was WRONG: it sat
# on the discard row's own bright bottom bevel/trim (present regardless of reach —
# measured fill ~0.68-0.93 even with NO reach anywhere in the frame), not the stick.
# Found the real per-slot object by connected-components on gray>=200 in a window
# around each guess, across >=10 present frames each (self/right/left: pixel-identical
# box on every sample; across: object position confirmed via 3 independently-skinned
# cosmetics, see below), then verified visually (scripts/inspect/overlay_hud.py PNGs):
# self/across are horizontal bars just below/above the panel; left/right are vertical
# bars just left/right of it — same footprint the Task-17a guess described, just
# mis-sized/placed. ⚠️ COSMETIC DIVERSITY: the 立直棒 asset is a per-player equipped
# skin, not a fixed sprite — observed a plain white bar+red dot (self, right, left,
# and one across game), an ornate syringe-with-heart (across, run_3/game3), and a
# glowing purple arrow (across, ai_session3/run_2, the alt table skin) — all three
# occupy roughly this SAME screen real estate (confirming the box placement
# generalizes), but their brightness profile differs a lot (see REACH_FILL_OK note
# in annotate/hud.py). Boxes below = the plain-bar footprint (the common case) + a
# few px margin.
#
# Keyed by SLOT (self/right/across/left, hud.REACH_STICK_SLOTS), not by
# detector class — the stick is a single symmetric `reach_stick` class (spec
# §10 revision); these four boxes are WHERE each slot is on screen, used both
# to render/ink-check the annotator's per-slot box (annotate/hud.py) and, at
# calibration time, as the geometric reference the detection-relative seat
# attribution (recognize/hudstate.py) is expected to reproduce.
# Direction convention VERIFIED (T17b): feeding these four box centers + the
# round_label seed center (960, 399, from _HUD_SEEDS_PX above) through
# recognize.hudstate._attribute_slot reproduces self/across/left/right exactly:
# self dx=-7 dy=+122 (dominant-|dy|, dy>0) -> self; across dx=+5 dy=-72
# (dominant-|dy|, dy<0) -> across; left dx=-140 dy=+14 (dominant-|dx|, dx<0)
# -> left; right dx=+137 dy=+27 (dominant-|dx|, dx>0) -> right.
_REACH_STICK_SEEDS_PX: dict[str, tuple[int, int, int, int]] = {
    "self":   (870, 510, 1036, 531),   # horizontal bar, just below the panel
    "across": (888, 316, 1042, 338),   # horizontal bar, just above the panel
    "left":   (805, 356, 835, 470),    # vertical bar, just left of the panel
    "right":  (1078, 383, 1115, 468),  # vertical bar, just right of the panel
}
REACH_STICK_SEEDS: dict[str, NormBox] = {k: px_box(*v) for k, v in _REACH_STICK_SEEDS_PX.items()}
