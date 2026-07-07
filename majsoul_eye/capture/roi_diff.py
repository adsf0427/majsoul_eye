"""Stability-ROI frame diff: the animation-motion signal for capture stability.

Restricting the diff to known play regions excludes the always-animating cloth
border, background art and the 2D HUD, so the threshold can be tight enough to
catch a moving tile/arm without being tripped by decoration. Shared by
autoplay_ai and FrameSyncer.

Multi-rect since 2026-07-07: the original single TABLE_ROI covered only the
central play surface (rivers/melds), so captures could pass stability while an
opponent hand row was still mid-理牌-compaction after a tedashi (GT settled,
pixels not — the backs experiment's misaligned-template frames, ~3.5-10%
measured; see STATUS §1.45 and annotate/backs.sorting_suspect, which stays as
the annotate-side gate for frames captured before this change). The added rects
are TIGHT hulls of the calibrated hand rows (annotate/_backs_manual.py manual
quads + drawn slots, ±margin) plus the hero hand row (its draw slide-in caused
the empty-slot hand mislabels too). Deliberately excluded: avatars where
possible (left/toimen), the dora panel (kan-dora glow pulses), HUD corners.
The right seat's rect unavoidably overlaps its avatar portrait (the row renders
in front of it) — portraits are static; a rare emote animation just delays that
capture (settle_cap bounds the wait).
"""
from __future__ import annotations

import numpy as np

# normalized (x0, y0, x1, y1) on the canonical 16:9 frame.
TABLE_ROI = (0.18, 0.16, 0.82, 0.92)            # central play surface (historical)
TOIMEN_ROW_ROI = (0.339, 0.0, 0.703, 0.088)     # across hand row + drawn slot
LEFT_ROW_ROI = (0.091, 0.088, 0.208, 0.625)     # left hand column + drawn slot
RIGHT_ROW_ROI = (0.806, 0.163, 0.936, 0.745)    # right hand column + drawn slot
HERO_ROW_ROI = (0.109, 0.855, 0.820, 1.0)       # hero hand (draw slide-in)

STABILITY_ROIS = (TABLE_ROI, TOIMEN_ROW_ROI, LEFT_ROW_ROI, RIGHT_ROW_ROI, HERO_ROW_ROI)


def _rect_diff(a: np.ndarray, b: np.ndarray, roi):
    """Mean abs diff over one rect; None if the rect quantizes to nothing (tiny
    frames — e.g. unit-test stubs — where a narrow rect covers <1px)."""
    h, w = a.shape[:2]
    x0, y0, x1, y1 = roi
    xa, ya, xb, yb = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    ra = a[ya:yb, xa:xb].astype(np.int16)
    rb = b[ya:yb, xa:xb].astype(np.int16)
    if ra.size == 0:
        return None
    return float(np.mean(np.abs(ra - rb)))


def roi_diff(a: np.ndarray, b: np.ndarray, roi=None) -> float:
    """Max mean-abs-diff over the stability rects (any moving region blocks).

    `roi` may be a single (x0,y0,x1,y1) rect (legacy callers/tests) or an
    iterable of rects; default = STABILITY_ROIS."""
    if a is None or b is None or a.shape != b.shape:
        return 1e9
    rois = STABILITY_ROIS if roi is None else (
        [roi] if isinstance(roi[0], (int, float)) else list(roi))
    diffs = [d for d in (_rect_diff(a, b, r) for r in rois) if d is not None]
    return max(diffs) if diffs else 1e9
