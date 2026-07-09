"""GT-driven HUD field annotation: seed ROI (WHERE) + BoardState (WHAT).

Numeric fields are ink-snapped per frame (glyph width varies with the value),
EXCEPT constant-width ones (FIXED_BOX_NUMERIC, e.g. zero-padded wall_count)
which keep the seed box and only probe ink for render-presence; fixed-glyph
fields (round_label / seat_wind_self) keep the seed box. Buttons are
handled separately (button_boxes, Task 7). Output dict style matches
annotate.frame's hand_boxes: `reliable` is only ever SET False.
"""
from __future__ import annotations

import cv2
import numpy as np

from majsoul_eye.coords import (HUD_SEEDS, BTN_ZONE, REACH_STICK_SEEDS,
                                WALL_COUNT_INK_PROBE)
from majsoul_eye.hud import NUMERIC_FIELDS, REACH_STICK_SLOTS, buttons_for_ops
from majsoul_eye.state.replay import is_score_anim_window

# Numeric fields whose rendered string is constant-width, so the seed IS the
# label box (no extent snap; ink is probed for render-presence only). wall_count
# qualifies because the client zero-pads it to 2 digits; the probe covers just
# the 余 glyph, clear of the panel bezel / score glow that pollutes a full-width
# ink scan (the pollution that forced the old — digit-clipping — 42px seed).
FIXED_BOX_NUMERIC = {"wall_count": WALL_COUNT_INK_PROBE}

# CALIBRATED (Task 6): 150 only caught the brightest anti-aliased crest of the
# round_label/wall_count glyphs (cyan text tops out at gray~171 under BGR2GRAY,
# vs ~52 panel background) — measured histogram showed the bulk of their ink
# sitting in the 100-160 band. 120 gives full-body coverage for cyan text while
# staying far above the dark-panel background (~50-90) and orange/white digit
# fields (max gray 200+), which were already comfortably captured at 150.
INK_THRESH = 120   # gray level splitting glyph from dark panel
INK_MIN_PX = 12    # fewer bright px than this = field not rendered
INK_PAD = 3        # px of context kept around the glyph extent


def ink_snap(img: np.ndarray, px_box, thresh: int = INK_THRESH,
             pad: int = INK_PAD, min_px: int = INK_MIN_PX):
    """Tighten px_box to the bright-glyph extent inside it (clamped to px_box).
    Returns (x0,y0,x1,y1) or None when the field shows no ink (not rendered)."""
    x0, y0, x1, y1 = (int(v) for v in px_box)
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(g >= thresh)
    if len(xs) < min_px:
        return None
    return (max(x0, x0 + int(xs.min()) - pad), max(y0, y0 + int(ys.min()) - pad),
            min(x1, x0 + int(xs.max()) + 1 + pad), min(y1, y0 + int(ys.max()) + 1 + pad))


def field_texts(state) -> dict[str, str]:
    """BoardState -> {field name: exact string the reader must output}.
    Fields whose GT is missing are OMITTED (never guessed)."""
    t: dict[str, str] = {}
    hero = state.hero_seat
    if hero >= 0 and state.scores:
        for i, name in enumerate(("score_self", "score_right",
                                  "score_across", "score_left")):
            t[name] = str(state.scores[(hero + i) % 4])
    if state.bakaze and state.kyoku:
        t["round_label"] = f"{state.bakaze}{state.kyoku}"
    if state.left_tile_count is not None:
        # The client zero-pads to 2 digits (余09, never 余9) — verified on real
        # frames 2026-07-07; the reader GT must match the rendered glyphs.
        t["wall_count"] = f"余{state.left_tile_count:02d}"
    if state.in_round:
        t["riichi_stick_count"] = f"x{state.kyotaku}"
        t["honba_count"] = f"x{state.honba}"
    if hero >= 0 and state.oya >= 0:
        t["seat_wind_self"] = "ESWN"[(hero - state.oya) % 4]
    return t


def hud_field_boxes(img: np.ndarray, state, region) -> list[dict]:
    """Annotate every GT-known HUD field on one frame. Numeric fields are
    ink-snapped; a field with no ink is emitted unreliable (GT leads render /
    occluded), same policy as tile zones."""
    out: list[dict] = []
    for name, text in field_texts(state).items():
        seed = region.norm_to_px(HUD_SEEDS[name])
        box, fill = seed, 1.0
        if name in NUMERIC_FIELDS:
            probe = FIXED_BOX_NUMERIC.get(name)
            snapped = ink_snap(img, region.norm_to_px(probe) if probe else seed)
            if snapped is None:
                out.append({"name": name, "px_box": list(seed), "text": text,
                            "fill": 0.0, "reliable": False})
                continue
            if probe is None:                  # variable-width value: snap to ink
                box = snapped                  # (fixed-box fields keep the seed)
        d = {"name": name, "px_box": [int(v) for v in box], "text": text,
             "fill": fill}
        out.append(d)
    return out


# CALIBRATED (T17b) on real reach-accepted frames (same captures as coords.py's
# _REACH_STICK_SEEDS_PX; see that comment for the box-placement fix this fill
# measurement depends on). Measured fraction of gray>=150 px inside the
# CALIBRATED slot box, "settled" (>=2 frames after reach_accepted) vs the exact
# reach_accepted seq (the frame most likely to still be mid-animation):
#   self:   settled min 0.735 (n=116) | confirmed-lag 0.334 (seq1437,
#           run_3/game1 — hero's own hand-slam sprite covers the slot AND
#           riichi_stick_count still reads x0 in that same frame)
#   right:  settled min 0.610 (n=105) | no confirmed-lag sample caught mid-render
#   left:   settled min 0.428 (n=107) | confirmed-lag 0.0 (seq687, run_3/game1 —
#           riichi_stick_count already x2 but the slot is visibly still empty)
#   across: settled min 0.101 (n=123, wide spread — see cosmetic-diversity note
#           above: the ornate syringe skin is dimmer than the plain bar even
#           fully rendered) | confirmed-lag 0.0 (seq222/717, run_3/game3 — a
#           full-screen hand-slam FX covers the slot, riichi_stick_count still x0)
# 0.35 sits clear of self's only confirmed-lag value (0.334) and left's settled
# floor (0.428), so it reliably separates "mid-animation" from "rendered" for
# those two slots (right has no counter-example to violate it either).
# ⚠️ SCOPED TO THE REACH WINDOW (2026-07-07): the gate only applies when
# `is_score_anim_window(state)` says the frame's record is still in the
# reach/reach_accepted window — every confirmed lag/occlusion sample above sat
# on exactly that record (hand-slam FX at declaration); once settled the stick
# stays rendered to the end of the kyoku. Applied unconditionally, the
# luminance-only fill conflated "not yet rendered" with "rendered but dark
# skin" and silently dropped 22.3%/19.7% of across/left sticks in datasets/v3
# (192 resp. 201 of them with fill>=0.1, i.e. rendered dim skins — measured on
# a sword-skin frame: fill 0.264) — worse than dropped: those frames still
# trained the detector with the stick as BACKGROUND. Off-window frames now
# trust GT regardless of fill. In-window frames are already excluded from HUD
# label emission wholesale by build_dataset's frame-level is_score_anim_window
# gate (working since Task 18's last_event_types fix); this per-box check
# remains as the finer, per-seat safety net for other consumers of the
# annotations. NOTE the stale-fallback residual (see is_call_window docstring):
# a zero-event record inherits the previous record's last_event_types, which
# can only over-gate (conservative) here.
REACH_FILL_OK = 0.35


def reach_stick_boxes(img: np.ndarray, state, region) -> list[dict]:
    """One box per seat currently in riichi (spec §10). Single detector class
    `reach_stick` — the object is center-symmetric so per-seat classes would be
    appearance-degenerate inside the detected box (see hud.py's
    REACH_STICK_SLOTS docstring); `slot` here is QA/debug metadata only (which
    hero-relative slot lit up), NOT part of the YOLO label — build_dataset.py's
    hud_emit keys off `name` alone, and seat attribution at inference time is
    recovered from detection-relative geometry (recognize/hudstate.py), not
    from this annotation-time slot. Label-only like buttons — no text, the
    class itself is the label — so this does not go through
    ink_snap/NUMERIC_FIELDS at all; `fill` is a coarse lit/unlit render check
    (fraction of gray>=150 pixels in the seed slot), applied ONLY while the
    frame's record is still in the reach window (GT (`state.reach`) flips at
    `reach_accepted` a beat before the client draws the stick — same
    GT-leads-render race as every other zone in this module). Off-window the
    stick is guaranteed rendered, and a luminance fill would misread dark
    skinned sticks as absent (see REACH_FILL_OK note above)."""
    hero = state.hero_seat
    if hero < 0 or not state.reach:
        return []
    in_reach_window = is_score_anim_window(state)
    out: list[dict] = []
    for i, slot in enumerate(REACH_STICK_SLOTS):
        if not state.reach[(hero + i) % 4]:
            continue
        x0, y0, x1, y1 = (int(v) for v in region.norm_to_px(REACH_STICK_SEEDS[slot]))
        fill = 0.0
        roi = img[y0:y1, x0:x1]
        if roi.size:
            fill = float((cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) >= 150).mean())
        d = {"name": "reach_stick", "slot": slot, "px_box": [x0, y0, x1, y1], "fill": round(fill, 3)}
        if in_reach_window and fill < REACH_FILL_OK:
            d["reliable"] = False
        out.append(d)
    return out


# CALIBRATED (Task 7 Step 5) on 22 real button frames (chi x12, pon x8, kan x3,
# ron x2, riichi x1; captures/raw/ai_session3/run_1/game1). Majsoul's PC button
# banners are colored translucent parallelograms (green/blue/magenta/orange/
# red/gray per action) whose FILL is only mid-bright (gray ~40-95, barely above
# the ~40-60 table) — thresholding on brightness alone does NOT hug the whole
# banner shape, it isolates the bright white/gold calligraphy TEXT inside it
# (gray 150-230). That text-glyph box turned out to be a perfectly usable,
# consistent proxy for the button's location (measured area 5170-21008 px²,
# always w>h), so BTN_THRESH/BTN_MIN_AREA needed NO change from the Task-7
# seed guess — the only real bug was BTN_ZONE being too wide (see coords.py).
BTN_MIN_AREA = 2500    # px² @1080p; real banners measured 5170-21008 px²
BTN_THRESH = 140       # banner glyph glow vs table; verified across all 22 frames
# Glyph blobs beyond these dims are merged banners / stray FX, never a single
# button's text (clean glyphs measured w<=~220, h<=~85; 7.2% of v3 button labels
# were such merged blobs, up to 845px wide). Rejecting the candidate degrades
# the frame to count_mismatch -> it contributes no button labels (宁缺毋滥).
BTN_MAX_W = 300        # px @1080p, glyph-blob upper bound
BTN_MAX_H = 90
# CALIBRATED (2026-07-07) banner (= click area) geometry: the label box is the
# BANNER, not the text glyph — the glyph box varies with display language while
# the banner plate is a constant-size UI element. Measured by color-distance
# segmentation against 189 clean button frames across all 7 classes, default +
# skinned UI themes: banner bbox is 244-251 x 82-102 px, centered 8-13px below
# the glyph-blob center with |dcx| <= ~10. One fixed size keeps labels
# consistent for the detector; clicking anywhere in it hits the button.
BTN_BANNER_W = 250
BTN_BANNER_H = 96
BTN_BANNER_DY = 10     # banner center sits this far BELOW the glyph center


def banner_box(text_box) -> tuple[int, int, int, int]:
    """Fixed-size banner (click-area) box anchored on a glyph blob's center."""
    x0, y0, x1, y1 = text_box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2 + BTN_BANNER_DY
    return (cx - BTN_BANNER_W // 2, cy - BTN_BANNER_H // 2,
            cx + BTN_BANNER_W // 2, cy + BTN_BANNER_H // 2)
BTN_ORDER_LTR = True   # display order left->right == buttons_for_ops order —
                       # VERIFIED on all 22 real frames (incl. 3-button pon+kan
                       # and chi+ron frames): on-screen L->R order exactly
                       # matches buttons_for_ops's HUD_NAMES-order output.


# --- plate segmentation (2026-07-10, STATUS §1.55) --------------------------
# The brightness gate below is SKIN-DEPENDENT and was silently dropping 46.1% of
# all GT button frames (92.8% of which had the banner plainly rendered). Those
# frames kept their image but lost their button labels, so the detector trained
# them as background: val recall 0/92 on rendered-but-dropped buttons vs 99.3%
# on labeled ones. Two failure modes, both intrinsic to thresholding gray:
#   * bright/busy tablecloth or 立绘 -> the mask floods, blobs merge, the merged
#     blob exceeds BTN_MAX_W and is rejected -> zero candidates;
#   * a skin whose glyph is DARK (IMG_1964's 吃) -> the glyph never reaches 140.
#
# A button is an OVERLAY on an otherwise static zone, so the skin-agnostic signal
# is |frame - background|, where the background is that game's own median of the
# zone over frames GT says have no buttons (annotate/btnbg.py). That segments the
# PLATE (not the glyph), which is what we want to label anyway.
#
# CALIBRATED on 30 games of datasets/v5 (mixed skins, ja/zh-Hans/zh-Hant):
#   plate components measured w 151-248 (median 205), h 66-105 (median 89),
#   area 6379-18373 (median 12757); minimum gap between adjacent plates 39px, so
#   a 21-wide closing kernel cannot bridge two buttons. Bounds are widened past
#   the measured range because a partially-blended plate edge shrinks the
#   component -- a too-tight bound would silently recreate the very bug this
#   replaces. Recall 94.8% (count-matched), false positives on no-button frames
#   0.06%; the ~5% miss is dominated by the genuine not-yet-rendered frames.
PLATE_DIFF = 30        # gray delta vs background that counts as "overlay here"
PLATE_MIN_AREA = 5000  # px² @1080p; smallest measured plate component 5647
PLATE_MIN_W, PLATE_MAX_W = 120, 300
PLATE_MIN_H, PLATE_MAX_H = 55, 130


def locate_button_plates(img, region, bg) -> list[tuple[int, int, int, int]]:
    """Banner PLATES inside BTN_ZONE by overlay-difference against `bg` (that
    game's zone background median, gray float32, zone-shaped). x-sorted
    original-px boxes. Empty list = nothing rendered over the background."""
    x0, y0, x1, y1 = region.norm_to_px(BTN_ZONE)
    roi = img[y0:y1, x0:x1]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if g.shape != bg.shape:
        raise ValueError(f"btn background {bg.shape} != zone {g.shape}; "
                         "it must be built for this frame geometry")
    m = (np.abs(g - bg) > PLATE_DIFF).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 21), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        # int(): connectedComponentsWithStats yields np.int32, which json.dumps
        # rejects — these boxes are written straight into the annotation JSONL.
        x, y, w, h, area = (int(v) for v in stats[i][:5])
        if (area >= PLATE_MIN_AREA and w > h
                and PLATE_MIN_W <= w <= PLATE_MAX_W
                and PLATE_MIN_H <= h <= PLATE_MAX_H):
            out.append((x0 + x, y0 + y, x0 + x + w, y0 + y + h))
    return sorted(out)


def plate_banner_box(plate_box) -> tuple[int, int, int, int]:
    """Fixed-size banner (click-area) box centered on a PLATE component.

    Unlike banner_box(), which anchors on the bright-glyph centroid and so
    inherits its language dependence (measured: the skip box sat 11px further
    right in ja than in zh-Hans, and 16px right of the true plate center), the
    plate center is language- and skin-invariant (measured cx: ja 0.6784,
    zh-Hans 0.6784, zh-Hant 0.6786)."""
    x0, y0, x1, y1 = plate_box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    return (cx - BTN_BANNER_W // 2, cy - BTN_BANNER_H // 2,
            cx + BTN_BANNER_W // 2, cy + BTN_BANNER_H // 2)


def locate_button_candidates(img, region) -> list[tuple[int, int, int, int]]:
    """LEGACY brightness gate -- bright GLYPH blobs inside BTN_ZONE, x-sorted, as
    original-px boxes. Superseded by locate_button_plates(); kept only for callers
    with no per-game background model (overlay/inspect tools). Do not use it to
    build training labels: see the calibration note above."""
    x0, y0, x1, y1 = region.norm_to_px(BTN_ZONE)
    roi = img[y0:y1, x0:x1]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    m = (g >= BTN_THRESH).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 25), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if (w * h >= BTN_MIN_AREA and w > h              # wide banner shape
                and w <= BTN_MAX_W and h <= BTN_MAX_H):  # not a merged blob/FX
            out.append((x0 + x, y0 + y, x0 + x + w, y0 + y + h))
    return sorted(out)


def button_boxes(img, state, region, btn_bg=None) -> list[dict]:
    """GT-expected buttons matched to located candidates by order; emitted
    px_box is the fixed-size BANNER (click area). Count mismatch -> every box
    unreliable + flagged (frame contributes no button labels; 宁缺毋滥, and
    build_dataset then drops the frame from the detector set entirely).

    With `btn_bg` (this game's zone background median, see annotate/btnbg.py)
    the plate segmentation is used and the box centers on the plate. Without it
    the legacy brightness gate runs, which is skin- and language-biased -- the
    pipeline always passes btn_bg."""
    expected = buttons_for_ops(state.pending_ops or [])
    if not expected:
        return []
    if btn_bg is not None:
        cands = [plate_banner_box(p) for p in locate_button_plates(img, region, btn_bg)]
    else:
        cands = [banner_box(c) for c in locate_button_candidates(img, region)]
    ordered = expected if BTN_ORDER_LTR else expected[::-1]
    if len(cands) != len(expected):
        return [{"name": n,
                 "px_box": list(cands[i]) if i < len(cands) else None,
                 "reliable": False, "flag": "count_mismatch"}
                for i, n in enumerate(ordered)]
    return [{"name": n, "px_box": list(c)} for n, c in zip(ordered, cands)]
