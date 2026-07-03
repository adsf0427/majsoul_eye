"""Per-box GT-consistency gate: crop -> classifier -> compare to GT class.

Catches boxes whose pixels don't match their GT label — chiefly discard-animation
occlusion (tile caught mid-flight, box lands on empty felt/arm), but also any
mislabel/occlusion. A frame-level smart-drop rule (see frame_decision) removes bad
boxes surgically, or the whole frame when too many are bad. Not a state predicate:
occlusion is intermittent (capture-timing-dependent), so we judge pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..tiles import NAME_TO_ID, TILE_NAMES
from ..label.quality import is_tile_present

TAU: float = 0.5        # min P(gt_cls) for a top1-mismatch box to still pass
MAX_BAD: int = 2        # per-frame bad-box budget before dropping the whole frame


@dataclass
class BoxVerdict:
    ok: bool
    gt: str
    pred: str
    conf: float          # P(gt_cls)
    reason: str          # "" | "mismatch" | "empty_felt"


def verdict_from_probs(prob_row: np.ndarray, gt: str, tau: float = TAU) -> BoxVerdict:
    """Pure verdict from one softmax row. Bad iff top1 != gt AND P(gt) < tau."""
    top = int(np.argmax(prob_row))
    pred = TILE_NAMES[top]
    conf = float(prob_row[NAME_TO_ID[gt]]) if gt in NAME_TO_ID else 0.0
    if pred == gt or conf >= tau:
        return BoxVerdict(True, gt, pred, conf, "")
    return BoxVerdict(False, gt, pred, conf, "mismatch")


def is_empty_felt(crop: np.ndarray, min_face_frac: float = 0.12) -> bool:
    """True when the crop is (almost) all table felt — no tile face present."""
    return not is_tile_present(crop, min_face_frac=min_face_frac)
