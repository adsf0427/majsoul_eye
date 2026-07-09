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


def _dedup_overlaps(dets: list, iou_thresh: float = 0.8) -> list:
    """Class-agnostic overlap pass over the per-class NMS output. ultralytics
    suppresses duplicates only WITHIN a class, so one physical tile can carry
    a second lower-score box of another class (seen in the wild: a 4p double-
    detected as N on the same box — the phantom tile inflated a river by one
    and made an otherwise valid frame turn-infeasible). Keep the higher-score
    box of any pair with axis-aligned IoU >= iou_thresh; legitimate neighbours
    (adjacent river slots, kakan stacks) overlap far below it."""
    kept: list = []
    for d in sorted(dets, key=lambda d: d.score, reverse=True):
        x0, y0, x1, y1 = d.xyxy
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        for k in kept:
            kx0, ky0, kx1, ky1 = k.xyxy
            iw = min(x1, kx1) - max(x0, kx0)
            ih = min(y1, ky1) - max(y0, ky0)
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            karea = max(0.0, kx1 - kx0) * max(0.0, ky1 - ky0)
            if inter / (area + karea - inter) >= iou_thresh:
                break
        else:
            kept.append(d)
    return kept


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
        return _dedup_overlaps(_parse_result(res))
