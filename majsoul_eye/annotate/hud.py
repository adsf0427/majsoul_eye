"""GT-driven HUD field annotation: seed ROI (WHERE) + BoardState (WHAT).

Numeric fields are ink-snapped per frame (glyph width varies with the value);
fixed-glyph fields (round_label / seat_wind_self) keep the seed box. Buttons are
handled separately (button_boxes, Task 7). Output dict style matches
annotate.frame's hand_boxes: `reliable` is only ever SET False.
"""
from __future__ import annotations

import cv2
import numpy as np

from majsoul_eye.coords import HUD_SEEDS, BTN_ZONE, REACH_STICK_SEEDS
from majsoul_eye.hud import NUMERIC_FIELDS, REACH_STICK_SLOTS, buttons_for_ops

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
        t["wall_count"] = f"余{state.left_tile_count}"
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
            snapped = ink_snap(img, seed)
            if snapped is None:
                out.append({"name": name, "px_box": list(seed), "text": text,
                            "fill": 0.0, "reliable": False})
                continue
            box = snapped
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
# those two slots (right has no counter-example to violate it either). It DOES
# cost recall on `across`: ~half its settled samples (the dim syringe-skin
# games) fall below 0.35 and get flagged unreliable despite being fully
# rendered — accepted per this module's "reliable only ever SET False" policy
# (dropping a good frame is safe; the alternative, a low threshold, would miss
# the confirmed self/across hand-slam lag frames above, i.e. mislabel a
# not-yet-rendered frame as reliable, which this policy exists to prevent).
# ⚠️ Also NOTE: `state.replay.is_score_anim_window` (checked as a second,
# frame-level gate in annotate/frame.py) turns out to NEVER fire for these
# exact reach_accepted frames in this capture set — Majsoul bundles
# reach_accepted with the next actor's immediate tsumo in one record, so
# `state.last_event` is always overwritten to "tsumo" by the time the frame is
# snapshotted (verified across all 6 reach_accepted seqs sampled). This
# per-box fill check is therefore not a redundant safety net here — it is the
# only mechanism that actually catches these frames.
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
    (fraction of gray>=150 pixels in the seed slot), used only to flag
    `reliable=False` when the stick hasn't rendered yet (GT (`state.reach`)
    flips at `reach_accepted` a beat before the client draws the stick — same
    GT-leads-render race as every other zone in this module)."""
    hero = state.hero_seat
    if hero < 0 or not state.reach:
        return []
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
        if fill < REACH_FILL_OK:
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
BTN_ORDER_LTR = True   # display order left->right == buttons_for_ops order —
                       # VERIFIED on all 22 real frames (incl. 3-button pon+kan
                       # and chi+ron frames): on-screen L->R order exactly
                       # matches buttons_for_ops's HUD_NAMES-order output.


def locate_button_candidates(img, region) -> list[tuple[int, int, int, int]]:
    """Bright banner blobs inside BTN_ZONE, x-sorted, as original-px boxes."""
    x0, y0, x1, y1 = region.norm_to_px(BTN_ZONE)
    roi = img[y0:y1, x0:x1]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    m = (g >= BTN_THRESH).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 25), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= BTN_MIN_AREA and w > h:            # wide banner shape
            out.append((x0 + x, y0 + y, x0 + x + w, y0 + y + h))
    return sorted(out)


def button_boxes(img, state, region) -> list[dict]:
    """GT-expected buttons matched to located candidates by order.
    Count mismatch -> every box unreliable + flagged (frame contributes no
    button labels; 宁缺毋滥)."""
    expected = buttons_for_ops(state.pending_ops or [])
    if not expected:
        return []
    cands = locate_button_candidates(img, region)
    ordered = expected if BTN_ORDER_LTR else expected[::-1]
    if len(cands) != len(expected):
        return [{"name": n,
                 "px_box": list(cands[i]) if i < len(cands) else None,
                 "reliable": False, "flag": "count_mismatch"}
                for i, n in enumerate(ordered)]
    return [{"name": n, "px_box": list(c)} for n, c in zip(ordered, cands)]
