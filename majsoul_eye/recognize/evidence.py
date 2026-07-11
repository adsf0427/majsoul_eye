from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from majsoul_eye.recognize.detector import Detection
from majsoul_eye.tiles import TILE_NAMES


@dataclass(frozen=True)
class CandidatePolicy:
    calibration_version: str | None
    top_k: int = 3

    @property
    def enabled(self) -> bool:
        return bool(self.calibration_version) and self.top_k > 0


@dataclass(frozen=True)
class CandidateScore:
    value: str
    confidence: float


@dataclass
class FieldObservation:
    field_key: str
    value: Any
    confidence: float | None
    detections: list[Detection] = field(default_factory=list)
    candidates: list[CandidateScore] = field(default_factory=list)


@dataclass
class AssemblyResult:
    observed: Any
    fields: list[FieldObservation]
    issues: list[str]


def crop_detection(frame_bgr: np.ndarray, det: Detection, size: int = 64) -> np.ndarray:
    if det.poly is not None:
        src = np.float32(det.poly)
        dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
        transform = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame_bgr, transform, (size, size))
    x0, y0, x1, y1 = (int(round(v)) for v in det.xyxy)
    h, w = frame_bgr.shape[:2]
    crop = frame_bgr[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(crop, (size, size))


def attach_tile_candidates(fields: list[FieldObservation], frame_bgr: np.ndarray,
                           classifier, policy: CandidatePolicy) -> None:
    if classifier is None or not policy.enabled:
        return
    indexed = [(i, field.detections[0]) for i, field in enumerate(fields)
               if field.detections and field.detections[0].tile is not None]
    if not indexed:
        return
    probabilities = classifier.predict_proba(
        [crop_detection(frame_bgr, det) for _, det in indexed])
    for row, (field_index, _) in zip(probabilities, indexed):
        order = np.argsort(-row)[:policy.top_k]
        fields[field_index].candidates = [
            CandidateScore(TILE_NAMES[int(class_id)], float(row[int(class_id)]))
            for class_id in order
        ]
