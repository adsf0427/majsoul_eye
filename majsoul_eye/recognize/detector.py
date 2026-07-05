"""Tile (+ HUD) DETECTOR inference wrapper (Ultralytics YOLO).

The second shipped runtime model, alongside the classifier: it localizes tiles in
the hard/perspective zones (四家河/副露) and on external / mobile / layout-drifted
screenshots where the deterministic fixed-ROI path can't be trusted. The v2 head
adds 17 HUD-element classes after the frozen 38 tile classes (``majsoul_eye.hud``
= 55-class ``DET_NAMES``); v1 (tile-only, 38-class) weights still load fine since
their ids are a strict prefix of ``DET_NAMES``.

Parallels ``TileClassifier``: construct with a weights path, call ``predict(bgr)``.
``ultralytics`` is imported LAZILY inside ``__init__`` so ``import
majsoul_eye.recognize`` (and classifier-only users) never require it. Akagi-free —
this module never imports the dev-only ``capture/`` package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..hud import DET_NAMES
from ..tiles import TILE_NAMES


@dataclass
class Detection:
    """One detected tile or HUD element, in ORIGINAL image px."""
    xyxy: tuple                 # (x0, y0, x1, y1) axis-aligned bbox (always present)
    name: str                   # hud.DET_NAMES member (valid for all 55 classes)
    tile: Optional[str]         # tiles.TILE_NAMES member; None for HUD-class detections (see majsoul_eye.hud)
    cls: int
    score: float
    poly: tuple = None          # 4 oriented (x, y) corners for OBB models; None for HBB


def _name_and_tile(cls: int) -> tuple:
    """cls id -> (name, tile). ``name`` is always valid (hud.DET_NAMES); ``tile``
    is the same string for tile ids (< len(TILE_NAMES)) and None for HUD ids."""
    name = DET_NAMES[cls]
    tile = name if cls < len(TILE_NAMES) else None
    return name, tile


def _parse_result(res) -> list:
    """Flatten one ultralytics result into Detections, handling BOTH detector types.

    An OBB model (yolo*-obb) returns oriented boxes in ``res.obb`` (``res.boxes`` is
    None); a standard model returns axis-aligned boxes in ``res.boxes``. For OBB we
    keep ``xyxy`` (the enclosing bbox) for drop-in compatibility AND ``poly`` (the 4
    rotated corners). Pulled out of ``predict`` so it is unit-testable without a GPU.

    A cls id >= len(DET_NAMES) (unknown/future class the running code doesn't know
    about yet) is defensively SKIPPED rather than crashing or inventing a name.
    """
    obb = getattr(res, "obb", None)
    if obb is not None:                      # OBB model — even if empty, do NOT touch res.boxes
        out = []
        for o in obb:
            cls = int(o.cls[0])
            if cls >= len(DET_NAMES):
                continue
            name, tile = _name_and_tile(cls)
            out.append(Detection(
                xyxy=tuple(float(v) for v in o.xyxy[0].tolist()),
                name=name, tile=tile, cls=cls, score=float(o.conf[0]),
                poly=tuple((float(x), float(y)) for x, y in o.xyxyxyxy[0].tolist()),
            ))
        return out
    out = []
    for b in res.boxes:
        cls = int(b.cls[0])
        if cls >= len(DET_NAMES):
            continue
        name, tile = _name_and_tile(cls)
        out.append(Detection(
            xyxy=tuple(float(v) for v in b.xyxy[0].tolist()),
            name=name, tile=tile, cls=cls, score=float(b.conf[0]),
        ))
    return out


class TileDetector:
    """Inference wrapper: one BGR frame -> list[Detection]. Accepts standard (HBB)
    and oriented (OBB) tile-detector weights; OBB detections carry a 4-corner
    ``poly`` (see ``weights/detector/tile_detector_obb.pt``)."""

    def __init__(self, weights: str, device: str = "cpu", conf: float = 0.25,
                 imgsz: int = 1280):
        from ultralytics import YOLO          # lazy: keep recognize/ import light
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, bgr: np.ndarray) -> list:
        res = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf,
                                 device=self.device, verbose=False)[0]
        return _parse_result(res)
