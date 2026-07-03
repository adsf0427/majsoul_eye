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


def score_frame(crops: list, gts: list, clf, *, tau: float = TAU, min_face_frac: float = 0.12) -> list[BoxVerdict]:
    """Verdict per (crop, gt). Empty-felt crops are bad up-front; the rest go through
    the classifier in one batch."""
    assert len(crops) == len(gts), (len(crops), len(gts))
    verdicts: list[BoxVerdict] = [None] * len(crops)  # type: ignore
    live_idx, live_crops = [], []
    for i, (crop, gt) in enumerate(zip(crops, gts)):
        if is_empty_felt(crop, min_face_frac=min_face_frac):
            verdicts[i] = BoxVerdict(False, gt, "", 0.0, "empty_felt")
        else:
            live_idx.append(i)
            live_crops.append(crop)
    if live_crops:
        probs = clf.predict_proba(live_crops)
        for k, i in enumerate(live_idx):
            verdicts[i] = verdict_from_probs(probs[k], gts[i], tau=tau)
    return verdicts


def frame_decision(verdicts: list[BoxVerdict], max_bad: int = MAX_BAD) -> tuple[str, list[int]]:
    """Decide whether to keep, drop boxes, or drop the whole frame.

    Returns ("keep", []) if no bad boxes.
    Returns ("drop_boxes", bad_indices) if 1 <= n_bad <= max_bad.
    Returns ("drop_frame", bad_indices) if n_bad > max_bad.
    """
    bad = [i for i, v in enumerate(verdicts) if not v.ok]
    if not bad:
        return "keep", []
    if len(bad) <= max_bad:
        return "drop_boxes", bad
    return "drop_frame", bad
