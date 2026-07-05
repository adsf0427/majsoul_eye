"""GT-driven HUD field annotation: seed ROI (WHERE) + BoardState (WHAT).

Numeric fields are ink-snapped per frame (glyph width varies with the value);
fixed-glyph fields (round_label / seat_wind_self) keep the seed box. Buttons are
handled separately (button_boxes, Task 7). Output dict style matches
annotate.frame's hand_boxes: `reliable` is only ever SET False.
"""
from __future__ import annotations

import cv2
import numpy as np

from majsoul_eye.coords import HUD_SEEDS
from majsoul_eye.hud import NUMERIC_FIELDS

INK_THRESH = 150   # gray level splitting glyph from dark panel  # CALIBRATE
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
