"""Crop quality gates for the auto-labeler.

The river/meld grids place a cell at every GT-expected position. But GT leads the
rendered image by ~1 board action (capture timing — see docs/STATUS.md §1.5), so
the *freshest* discard is often still mid-animation: its cell shows empty felt or a
half-slid tile. Saving such a crop under the GT label actively MISLABELS training
data ("empty felt = 8s").

A mean-brightness gate does NOT catch this: the blue table felt has mean brightness
~100-119, above any sane brightness floor. The discriminator that works is the
**tile-face fraction**: a mahjong tile face is white/cream (high in ALL channels),
while the felt is blue (B high, R/G low). Empirically (session6, 6k cells): real
tiles have face-fraction ~0.58-0.79; empty/partial felt ~0.0-0.11. A 0.35 cut
keeps ~99.7% of real tiles and drops ~75% of empty cells.
"""

from __future__ import annotations

import numpy as np

WHITE_THRESH = 140       # a pixel is "tile-face" if min(B,G,R) >= this (white/cream)
MIN_FACE_FRAC = 0.35     # cell is "empty/in-flight" below this fraction of tile-face pixels


def tile_face_fraction(crop_bgr: np.ndarray, white_thresh: int = WHITE_THRESH) -> float:
    """Fraction of pixels that look like a tile face (bright in every channel).

    Felt (blue) scores ~0; a rendered tile scores high regardless of the ink on it,
    because the tile background is white/cream. Returns 0.0 for an empty crop."""
    a = np.asarray(crop_bgr)
    if a.size == 0:
        return 0.0
    if a.ndim == 2:                       # grayscale fallback
        return float((a >= white_thresh).mean())
    return float((a.min(axis=2) >= white_thresh).mean())


def is_tile_present(crop_bgr: np.ndarray, min_face_frac: float = MIN_FACE_FRAC,
                    white_thresh: int = WHITE_THRESH) -> bool:
    """True if a tile is actually rendered in this cell (vs empty felt / in-flight)."""
    return tile_face_fraction(crop_bgr, white_thresh) >= min_face_frac
