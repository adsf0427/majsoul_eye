"""38-class tile DETECTOR inference wrapper (Ultralytics YOLO).

The second shipped runtime model, alongside the classifier: it localizes tiles in
the hard/perspective zones (四家河/副露) and on external / mobile / layout-drifted
screenshots where the deterministic fixed-ROI path can't be trusted.

Parallels ``TileClassifier``: construct with a weights path, call ``predict(bgr)``.
``ultralytics`` is imported LAZILY inside ``__init__`` so ``import
majsoul_eye.recognize`` (and classifier-only users) never require it. Akagi-free —
this module never imports the dev-only ``capture/`` package.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..tiles import TILE_NAMES


@dataclass
class Detection:
    """One detected tile, in ORIGINAL image px."""
    xyxy: tuple       # (x0, y0, x1, y1)
    tile: str         # tiles.TILE_NAMES member
    cls: int
    score: float


class TileDetector:
    """Inference wrapper: one BGR frame -> list[Detection]."""

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
        out = []
        for b in res.boxes:
            cls = int(b.cls[0])
            out.append(Detection(
                xyxy=tuple(float(v) for v in b.xyxy[0].tolist()),
                tile=TILE_NAMES[cls], cls=cls, score=float(b.conf[0]),
            ))
        return out
