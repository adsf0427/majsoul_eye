"""Assemble detector HUD boxes + micro-reader outputs into one structured dict
(the HUD half of the recognized 场况; tile half comes from detector/classifier).
Takes runtime Detection objects from the detector. Crop->rotate-upright->read happens
here so runtime matches the training crops."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from majsoul_eye.hud import FIELD_ROT, HUD_NAMES, NUMERIC_FIELDS

_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE}
_SCORE_KEY = {"score_self": "self", "score_right": "right",
              "score_across": "across", "score_left": "left"}


@dataclass
class HudFieldEvidence:
    field_key: str
    value: object
    detections: list


@dataclass
class HudAssembly:
    values: dict
    fields: list[HudFieldEvidence]


def _to_int(s: str, strip: str = "") -> Optional[int]:
    s = s.lstrip(strip)
    try:
        return int(s)
    except ValueError:
        return None


def _attribute_slot(dx: float, dy: float) -> str:
    """Detection-relative geometry -> hero-relative slot (spec §10). The stick
    is a single symmetric class, so WHICH seat it belongs to is recovered from
    its position relative to the center-panel anchor, not from its class: the
    dominant axis (vertical vs horizontal offset) picks self/across vs
    left/right, and the sign picks the specific side. y grows downward."""
    if abs(dy) >= abs(dx):
        return "self" if dy > 0 else "across"
    return "right" if dx > 0 else "left"


def assemble_hud(dets, reader, frame_bgr: np.ndarray) -> dict:
    out = {"scores": {"self": None, "right": None, "across": None, "left": None},
           "round": None, "wall": None, "kyotaku": None, "honba": None,
           "seat_wind": None, "buttons": [],
           "riichi": {"self": False, "right": False, "across": False, "left": False}}
    anchor = None            # center of round_label detection (preferred)
    anchor_fallback = None   # center of wall_count detection (fallback)
    stick_centers = []       # (cx, cy) of every `reach_stick` detection
    for det in dets:
        cls = det.name
        x0, y0, x1, y1 = (int(round(v)) for v in det.xyxy)
        if cls not in HUD_NAMES:
            continue
        if cls.startswith("btn_"):
            out["buttons"].append(cls)
            continue
        if cls == "reach_stick":
            # label-only, like buttons -- no reader call. Seat attribution is
            # deferred until the anchor (round_label/wall_count) is known --
            # see the loop below.
            stick_centers.append(((x0 + x1) / 2, (y0 + y1) / 2))
            continue
        crop = frame_bgr[max(0, y0):y1, max(0, x0):x1]
        if crop.size == 0:
            continue
        rot = FIELD_ROT.get(cls, 0)
        if rot in _ROT:
            crop = cv2.rotate(crop, _ROT[rot])
        text = reader.read(crop, cls)
        if cls in _SCORE_KEY:
            out["scores"][_SCORE_KEY[cls]] = _to_int(text)
        elif cls == "wall_count":
            out["wall"] = _to_int(text, strip="余")
            if anchor_fallback is None:
                anchor_fallback = ((x0 + x1) / 2, (y0 + y1) / 2)
        elif cls == "riichi_stick_count":
            out["kyotaku"] = _to_int(text, strip="x")
        elif cls == "honba_count":
            out["honba"] = _to_int(text, strip="x")
        elif cls == "round_label":
            out["round"] = text
            anchor = ((x0 + x1) / 2, (y0 + y1) / 2)
        elif cls == "seat_wind_self":
            out["seat_wind"] = text
    out["buttons"].sort(key=HUD_NAMES.index)

    if stick_centers:
        a = anchor if anchor is not None else anchor_fallback
        if a is not None:
            ax, ay = a
            for cx, cy in stick_centers:
                out["riichi"][_attribute_slot(cx - ax, cy - ay)] = True
        # else: no anchor detection present in this frame -> leave riichi all False.
    return out


def assemble_hud_with_evidence(dets, reader, frame_bgr) -> HudAssembly:
    values = assemble_hud(dets, reader, frame_bgr)
    by_name = {}
    for det in dets:
        by_name.setdefault(det.name, []).append(det)
    fields = []
    score_names = ((0, "score_self"), (1, "score_right"),
                   (2, "score_across"), (3, "score_left"))
    score_values = values["scores"]
    relative_keys = ("self", "right", "across", "left")
    for seat, name in score_names:
        if by_name.get(name):
            value = (score_values[seat] if isinstance(score_values, list)
                     else score_values[relative_keys[seat]])
            fields.append(HudFieldEvidence(f"round.scores.{seat}",
                                           value, by_name[name]))
    for key, name in (("round.bakazeKyoku", "round_label"),
                      ("round.leftTileCount", "wall_count"),
                      ("round.kyotaku", "riichi_stick_count"),
                      ("round.honba", "honba_count"),
                      ("round.seatWindSelf", "seat_wind_self")):
        if by_name.get(name):
            lookup = {"round.bakazeKyoku": "round", "round.leftTileCount": "wall",
                      "round.kyotaku": "kyotaku", "round.honba": "honba",
                      "round.seatWindSelf": "seat_wind"}[key]
            fields.append(HudFieldEvidence(key, values[lookup], by_name[name]))
    return HudAssembly(values, fields)
