"""Table-ROI frame diff: the discard-animation motion signal for capture stability.

Restricting the diff to the play surface excludes the always-animating cloth border
and the 2D HUD, so the threshold can be tight enough to catch a moving tile/arm
without being tripped by decoration. Shared by autoplay_ai and FrameSyncer.
"""
from __future__ import annotations

import numpy as np

# normalized (x0, y0, x1, y1) of the play surface (canonical 16:9), HUD/border excluded.
TABLE_ROI = (0.18, 0.16, 0.82, 0.92)


def roi_diff(a: np.ndarray, b: np.ndarray, roi=TABLE_ROI) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 1e9
    h, w = a.shape[:2]
    x0, y0, x1, y1 = roi
    xa, ya, xb, yb = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    ra = a[ya:yb, xa:xb].astype(np.int16)
    rb = b[ya:yb, xa:xb].astype(np.int16)
    if ra.size == 0:
        return 1e9
    return float(np.mean(np.abs(ra - rb)))
