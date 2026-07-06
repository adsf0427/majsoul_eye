"""Assemble detector HUD boxes + micro-reader outputs into one structured dict
(the HUD half of the recognized 场况; tile half comes from detector/classifier).
Crop->rotate-upright->read happens here so runtime matches the training crops."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from majsoul_eye.hud import FIELD_ROT, HUD_NAMES, NUMERIC_FIELDS

_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE}
_SCORE_KEY = {"score_self": "self", "score_right": "right",
              "score_across": "across", "score_left": "left"}


def _to_int(s: str, strip: str = "") -> Optional[int]:
    s = s.lstrip(strip)
    try:
        return int(s)
    except ValueError:
        return None


def assemble_hud(dets, reader, frame_bgr: np.ndarray) -> dict:
    out = {"scores": {"self": None, "right": None, "across": None, "left": None},
           "round": None, "wall": None, "kyotaku": None, "honba": None,
           "seat_wind": None, "buttons": [],
           "riichi": {"self": False, "right": False, "across": False, "left": False}}
    for cls, (x0, y0, x1, y1) in dets:
        if cls not in HUD_NAMES:
            continue
        if cls.startswith("btn_"):
            out["buttons"].append(cls)
            continue
        if cls.startswith("reach_stick_"):
            # label-only, like buttons -- no reader call; the detected class IS
            # the seat's riichi state (bucketed under "riichi", not "buttons").
            out["riichi"][cls[len("reach_stick_"):]] = True
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
        elif cls == "riichi_stick_count":
            out["kyotaku"] = _to_int(text, strip="x")
        elif cls == "honba_count":
            out["honba"] = _to_int(text, strip="x")
        elif cls == "round_label":
            out["round"] = text
        elif cls == "seat_wind_self":
            out["seat_wind"] = text
    out["buttons"].sort(key=HUD_NAMES.index)
    return out
