"""Map an arbitrary screenshot onto the canonical 16:9 board frame.

This is what lets the fixed-slot ROIs in ``coords.py`` survive different
resolutions / windows / external screenshots (docs/DESIGN.md §3.5). A locator
returns a :class:`BoardRegion` (the board's pixel rect within the frame); ROIs
are then placed relative to that region.

- :func:`locate_fullscreen` — assume the whole frame IS the 16:9 board (clean
  fullscreen capture). The simplest, correct case for our own captures.
- :func:`locate_letterbox` — trim black bars (browser/letterboxed captures).
- ``AnchorLocator`` — TODO: detect UI landmarks and fit a transform for arbitrary
  external screenshots (needs real frames to build).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .coords import NormBox


@dataclass(frozen=True)
class BoardRegion:
    """The board's pixel rect within a frame: offset (ox,oy) + size (bw,bh)."""
    ox: int
    oy: int
    bw: int
    bh: int

    def norm_to_px(self, box: NormBox) -> tuple[int, int, int, int]:
        return (self.ox + round(box.x0 * self.bw), self.oy + round(box.y0 * self.bh),
                self.ox + round(box.x1 * self.bw), self.oy + round(box.y1 * self.bh))

    def px_to_norm_box(self, x0: int, y0: int, x1: int, y1: int) -> NormBox:
        """Map a full-frame pixel box back to a normalized canonical box."""
        return NormBox((x0 - self.ox) / self.bw, (y0 - self.oy) / self.bh,
                       (x1 - self.ox) / self.bw, (y1 - self.oy) / self.bh)

    def crop(self, frame: np.ndarray, box: NormBox) -> np.ndarray:
        x0, y0, x1, y1 = self.norm_to_px(box)
        h, w = frame.shape[:2]
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        return frame[y0:y1, x0:x1]

    @property
    def aspect(self) -> float:
        return self.bw / self.bh if self.bh else 0.0


def locate_fullscreen(frame: np.ndarray) -> BoardRegion:
    """Treat the entire frame as the 16:9 board."""
    h, w = frame.shape[:2]
    return BoardRegion(0, 0, w, h)


def locate_letterbox(frame: np.ndarray, black_thresh: int = 16) -> BoardRegion:
    """Trim near-black borders and return the content rect.

    Handles browser chrome only partially — use fullscreen capture when possible.
    """
    if frame.ndim == 3:
        gray = frame.max(axis=2)
    else:
        gray = frame
    cols = np.where(gray.max(axis=0) > black_thresh)[0]
    rows = np.where(gray.max(axis=1) > black_thresh)[0]
    if len(cols) == 0 or len(rows) == 0:
        return locate_fullscreen(frame)
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    return BoardRegion(x0, y0, x1 - x0, y1 - y0)


class AnchorLocator:
    """TODO (needs real frames): detect stable UI landmarks (corner badges, the
    score panel, the center round indicator), fit a similarity/perspective
    transform, and return the board rect — for arbitrary external screenshots
    where neither fullscreen nor simple letterbox holds. Falls back to letterbox.
    """

    def locate(self, frame: np.ndarray) -> BoardRegion:
        return locate_letterbox(frame)
