"""Hard-zone (河) auto-labeling via a per-seat perspective GRID + GT order.

The packed discard tiles touch, so a brightness mask merges each seat's whole
河 into ONE blob — per-tile contour detection can't separate them. So we DON'T
detect individual tiles. Instead:

    WHERE = a calibrated per-seat quad (the FULL 6×3 discard grid), subdivided by
            homography into cells;
    WHAT  = state.visible_river(seat) in discard order.

For a seat with K visible discards we place cells 0..K-1 (reading order) and read
each class from GT. Zero per-tile detection, zero hand-drawing. The quad is
calibrated once per seat on a full-river frame (see scripts + coords.RIVER_QUADS).
``detect_river_blocks`` is a calibration helper (finds the 4 blocks on a full
frame); runtime labeling uses the stored quad, not per-frame detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..coords import NormBox
from ..normalize import BoardRegion
from ..tiles import NAME_TO_ID
from .autolabel import LabelSample

RIVER_COLS = 6
RIVER_ROWS = 3


@dataclass(frozen=True)
class RiverGrid:
    """A seat's full discard grid as a quad, subdivided into cols×rows cells.

    Corners are normalized (0–1) canonical-board coords, ordered so cell index 0
    (first discard) is at ``g00`` and reading proceeds g00→g10 (columns) then
    g00→g01 (rows). Per-seat orientation is encoded by how the corners are
    assigned at calibration time, so the model itself is orientation-agnostic.
    """
    g00: tuple[float, float]   # col 0, row 0  (first discard corner)
    g10: tuple[float, float]   # col max, row 0
    g11: tuple[float, float]   # col max, row max
    g01: tuple[float, float]   # col 0, row max
    cols: int = RIVER_COLS
    rows: int = RIVER_ROWS

    @property
    def capacity(self) -> int:
        return self.cols * self.rows

    def _homography(self):
        import cv2
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        dst = np.array([self.g00, self.g10, self.g11, self.g01], np.float32)
        return cv2.getPerspectiveTransform(src, dst)

    def cell_corners(self, i: int) -> np.ndarray:
        """4 (x,y) normalized corners of cell i (reading order)."""
        import cv2
        col, row = i % self.cols, i // self.cols
        u0, u1 = col / self.cols, (col + 1) / self.cols
        v0, v1 = row / self.rows, (row + 1) / self.rows
        pts = np.array([[[u0, v0], [u1, v0], [u1, v1], [u0, v1]]], np.float32)
        return cv2.perspectiveTransform(pts, self._homography())[0]

    def cell_box(self, i: int, riichi: bool = False) -> NormBox:
        """Axis-aligned normalized box of cell i. Riichi tiles render sideways
        (wider than tall) — widen the cell to cover the rotated tile."""
        pts = self.cell_corners(i)
        x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
        if riichi:
            cx, w = (x0 + x1) / 2, (x1 - x0)
            x0, x1 = cx - w * 0.75, cx + w * 0.75   # ~1.5× width for the sideways tile
        return NormBox(x0, y0, x1, y1)


def label_river(region: BoardRegion, state, seat_pos: str, grid: RiverGrid) -> tuple[list[LabelSample], bool]:
    """Auto-label one seat's 河 from its calibrated grid + GT discard order.

    Returns (samples, ok); ok==False flags more discards than grid capacity
    (overflow / miscalibration) for review."""
    seat = _screen_to_seat(state.hero_seat, seat_pos)
    if seat is None:
        return [], False
    gt = state.visible_river(seat)              # list[RiverTile], called tiles excluded
    n = min(len(gt), grid.capacity)
    samples: list[LabelSample] = []
    for i in range(n):
        rt = gt[i]
        nb = grid.cell_box(i, riichi=getattr(rt, "riichi", False))
        samples.append(LabelSample("river", rt.pai, "tile", nb, region.norm_to_px(nb), NAME_TO_ID.get(rt.pai)))
    return samples, (len(gt) <= grid.capacity)


# --- calibration helper (used by T2; NOT in the runtime path) ----------------

def _bright_mask(gray: np.ndarray, thresh: int = 150, ksize: int = 9):
    import cv2
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    if ksize > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((ksize, ksize), np.uint8))
    return mask


def detect_river_blocks(frame: np.ndarray, thresh: int = 150) -> dict[str, dict]:
    """Find the 4 river blocks on a (near-full) frame and assign them to screen
    seats by center position. Returns {seat_pos: {bbox, center, quad}} in
    normalized coords. Calibration aid — seeds RIVER_QUADS; not used at runtime.
    """
    import cv2
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    mask = _bright_mask(gray, thresh)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        a = w * h
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        if a > 0.01 * H * W and 0.20 < cx < 0.80 and 0.08 < cy < 0.80:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True).reshape(-1, 2)
            quad = (approx / [W, H]).tolist() if len(approx) == 4 else None
            blobs.append({"area": a / (H * W),
                          "bbox": (x / W, y / H, (x + w) / W, (y + h) / H),
                          "center": (cx, cy), "quad": quad})
    blobs.sort(key=lambda b: -b["area"])
    blobs = blobs[:4]
    if not blobs:
        return {}
    out: dict[str, dict] = {}
    out["across"] = min(blobs, key=lambda b: b["center"][1])   # smallest cy (top)
    out["self"] = max(blobs, key=lambda b: b["center"][1])     # largest cy (bottom)
    rest = [b for b in blobs if b not in (out["across"], out["self"])]
    if rest:
        out["left"] = min(rest, key=lambda b: b["center"][0])
        out["right"] = max(rest, key=lambda b: b["center"][0])
    return out


def assign_from_gt(boxes_sorted: list, gt_tiles: list[str]):
    """Zip ordered boxes with the GT discard list; ok==True iff counts match."""
    n = min(len(boxes_sorted), len(gt_tiles))
    return list(zip(boxes_sorted[:n], gt_tiles[:n])), (len(boxes_sorted) == len(gt_tiles))


def _screen_to_seat(hero: int, seat_pos: str) -> Optional[int]:
    if hero < 0:
        return None
    return {"self": hero, "right": (hero + 1) % 4,
            "across": (hero + 2) % 4, "left": (hero + 3) % 4}.get(seat_pos)
