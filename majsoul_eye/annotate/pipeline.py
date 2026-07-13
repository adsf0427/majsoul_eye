#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mahjong Soul / digital riichi mahjong annotation utilities.

This version uses relative player positions instead of absolute wind/screen seats:

    seat = 0: 自家 / self      / bottom side in the current screen layout
    seat = 1: 下家 / shimocha  / right side in the current screen layout
    seat = 2: 对家 / toimen    / top side in the current screen layout
    seat = 3: 上家 / kamicha   / left side in the current screen layout

Coordinate spaces:
    original:   the raw 1920x1080 screenshot
    square:     1280x1280 rectified table square
    fullwarp:   full original image warped by the same homography, with expanded canvas

Important reliability notes:
    discard_slots are the relatively stable part of the current pipeline.
    meld/fuuro boxes are intentionally marked unreliable.  The code can normalize them
    as axis-aligned rectangles in fullwarp space, but they are not ground truth.

Dependencies:
    opencv-python, numpy
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple

import cv2
import numpy as np

# =============================================================================
# 1. Homography calibration constants
# =============================================================================
# Calibration is for the 1920x1080 screenshot geometry used in this conversation.
# If the screenshot is resized, either resize these points or recalibrate.
SRC_TABLE_CORNERS = np.float32([
    [522.05, 120.00],
    [1398.17, 120.00],
    [1567.30, 848.00],
    [353.94, 848.00],
])

TABLE_SIDE = 1280
DST_TABLE_CORNERS = np.float32([
    [0, 0],
    [TABLE_SIDE - 1, 0],
    [TABLE_SIDE - 1, TABLE_SIDE - 1],
    [0, TABLE_SIDE - 1],
])

DEFAULT_PAD = 40

# =============================================================================
# 2. Relative seat mapping
# =============================================================================
# Older data used screen-side labels produced by the first experiments.
# Current desired convention is relative-position IDs.
SCREEN_SIDE_TO_RELATIVE_SEAT = {
    "S": 0,  # bottom -> self
    "E": 1,  # right  -> shimocha
    "N": 2,  # top    -> toimen
    "W": 3,  # left   -> kamicha
}
RELATIVE_SEAT_TO_SCREEN_SIDE = {v: k for k, v in SCREEN_SIDE_TO_RELATIVE_SEAT.items()}
RELATIVE_SEAT_NAME = {
    0: "self",
    1: "shimocha",
    2: "toimen",
    3: "kamicha",
}
RELATIVE_SEAT_CN = {
    0: "自家",
    1: "下家",
    2: "对家",
    3: "上家",
}

# BGR colors for OpenCV.
RELATIVE_SEAT_COLORS = {
    0: (220, 80, 220),   # self: magenta
    1: (70, 180, 70),    # shimocha/right: green
    2: (70, 70, 255),    # toimen/top: red
    3: (80, 170, 255),   # kamicha/left: orange-ish
}
MELD_UNRELIABLE_COLOR = (0, 255, 255)  # yellow/cyan in BGR

# =============================================================================
# 3. Fixed discard tile dimensions in fullwarp space
# =============================================================================
# These are template dimensions in the current fullwarp coordinate system.
# They represent the visual footprint used by the current discard grid.
# Because the table homography maps the table plane, these values should be
# stable as long as the game camera, screenshot size, and calibration stay fixed.
#
# If you only want to annotate the top white face, use the FACE_INSET constants
# to shrink the footprint and remove most of the orange side/thickness.
DISCARD_FOOTPRINT_SIZE_FULLWARP = {
    0: (64.0, 76.0),  # self/bottom discard tile footprint
    1: (88.0, 68.0),  # shimocha/right footprint
    2: (64.0, 76.0),  # toimen/top footprint
    3: (88.0, 68.0),  # kamicha/left footprint
}

# Insets are (left, top, right, bottom) in fullwarp pixels.
# They are deliberately configurable.  The bottom inset removes the visible
# orange thickness from most face-up discard tiles.
DISCARD_FACE_INSET_FULLWARP = {
    0: (2.0, 2.0, 2.0, 10.0),
    1: (2.0, 2.0, 2.0, 9.0),
    2: (2.0, 2.0, 2.0, 10.0),
    3: (2.0, 2.0, 2.0, 9.0),
}

# For riichi sideways discards, the footprint can be represented as a rotated
# horizontal tile in the local discard grid.  The current data did not contain a
# reliable riichi example, so this is a template value, not learned from examples.
RIICHI_FOOTPRINT_SIZE_FULLWARP = {
    0: (100.0, 58.0),
    1: (58.0, 100.0),
    2: (100.0, 58.0),
    3: (58.0, 100.0),
}

# =============================================================================
# 4. Fuuro / meld convention
# =============================================================================
# Working assumption for future refinement:
#   In fullwarp space, fuuro tile bboxes may be treated as rectangles whose edges
#   are parallel/perpendicular to the table border.
#
# This is a useful annotation convention, because the same rectangle inverse-
# projects to a quadrilateral in the original image.  However the current fuuro
# coordinates are NOT reliable and should not be used as training labels.
FURO_BBOX_MODEL = "axis_aligned_rectangle_in_fullwarp"
FURO_RELIABLE = False
FURO_NOTE = (
    "UNRELIABLE: fuuro/meld bbox is only a rough candidate. "
    "Use axis-aligned fullwarp rectangles as an annotation convention, "
    "but recalibrate/detect fuuro separately before using it as ground truth."
)

# =============================================================================
# 5. Homography utilities
# =============================================================================
def build_homographies(image_width: int = 1920, image_height: int = 1080, pad: int = DEFAULT_PAD) -> Dict[str, Any]:
    """Build square and fullwarp homographies."""
    H_square = cv2.getPerspectiveTransform(SRC_TABLE_CORNERS, DST_TABLE_CORNERS)

    original_corners = np.float32([
        [0, 0],
        [image_width - 1, 0],
        [image_width - 1, image_height - 1],
        [0, image_height - 1],
    ]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(original_corners, H_square).reshape(-1, 2)
    min_xy = projected.min(axis=0)
    max_xy = projected.max(axis=0)

    translate = np.array([
        [1, 0, -min_xy[0] + pad],
        [0, 1, -min_xy[1] + pad],
        [0, 0, 1],
    ], dtype=np.float64)

    H_full = translate @ H_square
    full_size = (
        int(np.ceil(max_xy[0] - min_xy[0] + 2 * pad)),
        int(np.ceil(max_xy[1] - min_xy[1] + 2 * pad)),
    )
    full_offset = np.float32([-min_xy[0] + pad, -min_xy[1] + pad])

    return {
        "H_square": H_square,
        "H_square_inv": np.linalg.inv(H_square),
        "H_full": H_full,
        "H_full_inv": np.linalg.inv(H_full),
        "full_size": full_size,
        "full_offset": full_offset,
    }


def transform_points(points: Iterable[Iterable[float]], H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def original_to_square(points: Iterable[Iterable[float]], H_square: np.ndarray) -> np.ndarray:
    return transform_points(points, H_square)


def square_to_original(points: Iterable[Iterable[float]], H_square_inv: np.ndarray) -> np.ndarray:
    return transform_points(points, H_square_inv)


def original_to_fullwarp(points: Iterable[Iterable[float]], H_full: np.ndarray) -> np.ndarray:
    return transform_points(points, H_full)


def fullwarp_to_original(points: Iterable[Iterable[float]], H_full_inv: np.ndarray) -> np.ndarray:
    return transform_points(points, H_full_inv)


def square_to_fullwarp(points: Iterable[Iterable[float]], offset: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32) + offset


def fullwarp_to_square(points: Iterable[Iterable[float]], offset: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32) - offset


def warp_to_square(image: np.ndarray, H_square: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(image, H_square, (TABLE_SIDE, TABLE_SIDE), flags=cv2.INTER_CUBIC)


def warp_to_full(image: np.ndarray, H_full: np.ndarray, full_size: Tuple[int, int]) -> np.ndarray:
    # OpenCV dsize is (width, height). INTER_CUBIC is LOAD-BEARING despite being
    # ~60% of annotate_frame time: an INTER_LINEAR AB (2026-07-07, 976 frames)
    # left river fills unchanged (0 flips / 27.5k slots) but broke the meld snap
    # on the far seat — softer edges push crevice/edge contrast under the
    # MIN_CREVICE_CONTRAST / MIN_EDGE_GRAD thresholds (calibrated on cubic-sharp
    # profiles), flipping the candidate lock by 30px and dropping meld QA
    # agreement 1.0 -> 0.63. Don't downgrade without recalibrating the snap.
    return cv2.warpPerspective(image, H_full, full_size, flags=cv2.INTER_CUBIC)

# =============================================================================
# 6. Geometry helpers
# =============================================================================
def axis_aligned_bbox_poly(poly: Iterable[Iterable[float]]) -> np.ndarray:
    pts = np.asarray(poly, dtype=np.float32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def center_of_poly(poly: Iterable[Iterable[float]]) -> np.ndarray:
    return np.asarray(poly, dtype=np.float32).mean(axis=0)


def make_axis_aligned_rect(center: Iterable[float], w: float, h: float) -> np.ndarray:
    cx, cy = np.asarray(center, dtype=np.float32)
    return np.float32([
        [cx - w / 2, cy - h / 2],
        [cx + w / 2, cy - h / 2],
        [cx + w / 2, cy + h / 2],
        [cx - w / 2, cy + h / 2],
    ])


def shrink_rect(poly: Iterable[Iterable[float]], inset: Tuple[float, float, float, float]) -> np.ndarray:
    """Shrink an axis-aligned rectangle by (left, top, right, bottom)."""
    pts = axis_aligned_bbox_poly(poly)
    x1, y1 = pts[0]
    x2, y2 = pts[2]
    l, t, r, b = inset
    return np.float32([[x1 + l, y1 + t], [x2 - r, y1 + t], [x2 - r, y2 - b], [x1 + l, y2 - b]])


def enforce_discard_size(poly_fullwarp: Iterable[Iterable[float]], seat: int, riichi: bool = False) -> np.ndarray:
    """Return a fixed-size axis-aligned footprint centered at the given polygon."""
    w, h = RIICHI_FOOTPRINT_SIZE_FULLWARP[seat] if riichi else DISCARD_FOOTPRINT_SIZE_FULLWARP[seat]
    return make_axis_aligned_rect(center_of_poly(poly_fullwarp), w, h)


def discard_face_poly(poly_fullwarp: Iterable[Iterable[float]], seat: int) -> np.ndarray:
    """Return a smaller top-face polygon by removing configured side/thickness insets."""
    return shrink_rect(poly_fullwarp, DISCARD_FACE_INSET_FULLWARP[seat])

# =============================================================================
# 9b. GT-driven annotation GENERATION (calibrated 2026-07-01)
# =============================================================================
# The functions above only *resize* pre-existing polygons. The model below
# GENERATES discard + meld boxes for a whole board from ground truth (river tile
# lists, riichi index, meld list) using a fixed-camera calibration in fullwarp
# space, so every case_frame can be auto-annotated with GT labels (WHAT) placed at
# geometry (WHERE). Calibrated on the AB case_frames (1920x1080, ai_session_run_3_game1/ai_session_run_3_game3).
#
# Relative seat = screen position: 0=self(bottom) 1=shimocha(right)
# 2=toimen(top) 3=kamicha(left). Each seat's discard grid is one 90deg rotation.
#
# DISCARD grid: origin O = centre of the (row1,col1) tile; DCOL = one column step;
# DROW = one row step (outward, toward the player). Exact linear grid in fullwarp.
# Refit 2026-07-02 from 16 AI games x 40 frames (scripts/annotate/calibrate_annotation_model.py,
# ~5k crevice + ~800 edge pairs per seat, rmse 0.8-1.6px). Notable vs the first
# hand calibration: row pitches are ~108-110 (the tile-FACE plane keeps a small
# projective residual under the felt-fit homography), and the vertical rivers'
# origins move ~10px against the old blob-based (skirt-biased) measurement.
DISCARD_GRID = {
    0: {"o": (1346.9, 1240.4), "dcol": (72.46, 0.0), "drow": (0.0, 108.63)},  # self  down
    1: {"o": (1818.2, 767.1),  "dcol": (0.0, 74.88), "drow": (97.43, 0.0)},   # right rightward
    2: {"o": (1353.9, 683.0),  "dcol": (72.94, 0.0), "drow": (0.0, -109.95)}, # across up
    3: {"o": (1244.3, 764.3),  "dcol": (0.0, 74.74), "drow": (-97.61, 0.0)},  # left  leftward
}
DISCARD_COLS = 6
# Measured white FACE box (w,h) per seat (crevice-to-edge, skirt-free); the tile
# face is ~70.5x92.5 fullwarp px at every seat. Sideways (riichi) = rotated.
DISCARD_FOOT = {0: (70.2, 92.1), 1: (93.4, 70.5), 2: (70.4, 92.2), 3: (93.0, 71.0)}
RIICHI_FOOT  = {0: (92.1, 70.2), 1: (70.5, 93.4), 2: (92.2, 70.4), 3: (71.0, 93.0)}
# Sideways-tile cross alignment: offset of the sideways cell's centre vs the row
# centre. Re-measured 2026-07-02 AFTER the per-row offsets below (the earlier
# "top-aligned" reading was the row-position bias in disguise): ~centered.
SIDEWAYS_CROSS_SHIFT = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0), 3: (0.0, 0.0)}
# Per-row cross offsets (px along the row-advance direction, relative to row 1).
# The rows are NOT equally spaced and the pitch is NOT the same for self vs
# across (the fullwarp y-scale varies with distance to the camera): measured
# self r1->r2 ~96.4 but across r1->r2 ~111.4, r2->r3 ~97-98 everywhere.
# Calibrated per row via the edge/crevice chain (calibrate_annotation_model).
DISCARD_ROW_OFFSETS: Dict[int, list] = {
    0: [0.0, 96.4, 193.5],
    1: [0.0, 97.6, 194.0],
    2: [0.0, 104.9, 201.9],   # row1 sits 6.5px nearer the viewer than the old fit
    3: [0.0, 97.2, 194.8],
}
# Per-seat reading order = the 4-fold table rotation. disc0 is discard #0's corner.
#   disc0_at_col5: discard #0 is at the c=6 end; colsign/rowsign flip DCOL/DROW.
# Verified against partially-filled rows (self row2->left, across row2->right,
# right row2->bottom, left row2->down) via GT tile-id matching.
DISCARD_READ = {
    0: {"disc0_at_col5": False, "colsign": +1, "rowsign": +1},  # self
    1: {"disc0_at_col5": True,  "colsign": -1, "rowsign": +1},  # right
    2: {"disc0_at_col5": True,  "colsign": -1, "rowsign": +1},  # across
    3: {"disc0_at_col5": False, "colsign": +1, "rowsign": +1},  # left
}
# MELD strip: anchor = centre of the OUTERMOST meld tile (player's far corner);
# step = advance per tile toward table centre; foot=(w,h). Melds are corner-
# anchored (meld[0] at the corner, growing inward). Meld positions carry inherent
# per-round variation (~half a tile) since Majsoul lays them relative to the hand;
# reliable but approximate. self has no case here (set by 180deg symmetry).
MELD_STRIP = {
    0: {"anchor": (2360.0, 1850.0), "step": (-76.0, 0.0),  "foot": (66.0, 90.0)},  # self  BR grow-left
    1: {"anchor": (2406.0, 190.0),  "step": (0.0, 76.0),   "foot": (86.0, 78.0)},  # right TR grow-down
    2: {"anchor": (712.0, 178.0),   "step": (76.0, -1.0),  "foot": (66.0, 90.0)},  # across TL grow-right
    3: {"anchor": (675.0, 1762.0),  "step": (0.0, -80.0),  "foot": (92.0, 84.0)},  # left  BL grow-up
}
BOARD_CENTRE_FULLWARP = (1536.0, 1013.0)  # for orienting the kakan stack inward
DISCARD_FACE_INSET = 3.0                   # small safety inset inside the measured face

RELATIVE_SEAT_SIDE = {0: "S", 1: "E", 2: "N", 3: "W"}


def _rect_poly(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    return np.float32([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                       [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])


def river_sideways_index(full_river: list) -> int | None:
    """Display index (within the VISIBLE river) of the sideways-rendered tile.

    `full_river` = ordered discards incl. called-away ones, each a mapping with
    ``riichi`` and ``called`` bools. Normally the riichi-declaring discard is the
    sideways one; if that tile was claimed by another player, Majsoul renders the
    player's NEXT discard sideways instead.
    """
    ri = next((i for i, t in enumerate(full_river) if t.get("riichi")), None)
    if ri is None:
        return None
    j = ri
    while j < len(full_river) and full_river[j].get("called"):
        j += 1
    if j >= len(full_river):
        return None                     # riichi tile claimed, no later discard yet
    return sum(1 for t in full_river[:j] if not t.get("called"))


def generate_discard_slots(seat: int, river: list, H_full_inv: np.ndarray,
                           sideways_idx: int | None = None) -> List[Dict[str, Any]]:
    """Generate discard boxes for one seat from its visible river.

    `river` is the ordered list of discards, each a mapping with keys
    ``pai`` (str), ``riichi`` (bool), ``tsumogiri`` (bool). Returns relative-format
    slot dicts (fullwarp + original polys, GT label, riichi/face info).
    `sideways_idx` overrides which visible tile is rendered sideways (see
    river_sideways_index for the riichi-tile-claimed edge case).
    """
    g = DISCARD_GRID[seat]; rd = DISCARD_READ[seat]
    o = np.array(g["o"], float); dcol = np.array(g["dcol"], float); drow = np.array(g["drow"], float)
    disc0 = o + (DISCARD_COLS - 1) * dcol if rd["disc0_at_col5"] else o.copy()
    colvec = rd["colsign"] * dcol
    rowvec = rd["rowsign"] * drow
    row_pitch = float(np.linalg.norm(drow))
    row_unit = rowvec / (row_pitch + 1e-9)
    row_offs = DISCARD_ROW_OFFSETS.get(seat)

    def _row_shift(row: int) -> np.ndarray:
        if row_offs and row < len(row_offs):
            return row_offs[row] * row_unit
        return row * row_pitch * row_unit
    col_dir = colvec / (np.linalg.norm(colvec) + 1e-9)
    fw, fh = DISCARD_FOOT[seat]; rw, rh = RIICHI_FOOT[seat]
    side_along = rw if abs(col_dir[0]) > abs(col_dir[1]) else rh
    extra = max(0.0, side_along - np.linalg.norm(colvec))
    riichi_idx = sideways_idx
    if riichi_idx is None:
        riichi_idx = next((i for i, t in enumerate(river) if t.get("riichi")), None)
    slots: List[Dict[str, Any]] = []
    for i, rt in enumerate(river):
        row, col = divmod(i, DISCARD_COLS)
        eff_col = col
        if row >= 3:                       # 4th-row overflow (>18 discards, rare): wedge left of top row
            row, eff_col = 2, DISCARD_COLS + (i - 18)
        center = disc0 + eff_col * colvec + _row_shift(row)
        is_side = (i == riichi_idx)
        if riichi_idx is not None and (riichi_idx // DISCARD_COLS) == (i // DISCARD_COLS) and (riichi_idx % DISCARD_COLS) < col:
            center = center + col_dir * extra
        if is_side:
            center = center + col_dir * (extra / 2.0) + np.array(SIDEWAYS_CROSS_SHIFT[seat])
            poly = _rect_poly(center[0], center[1], rw, rh)
        else:
            poly = _rect_poly(center[0], center[1], fw, fh)
        face = shrink_rect(poly, (DISCARD_FACE_INSET,) * 4)
        slots.append({
            "area": "discard", "seat": seat, "relative_position": RELATIVE_SEAT_NAME[seat],
            "relative_position_cn": RELATIVE_SEAT_CN[seat], "screen_side": RELATIVE_SEAT_SIDE[seat],
            "index": i, "row": (i // DISCARD_COLS) + 1, "col": (i % DISCARD_COLS) + 1,
            "occupied": True, "riichi": bool(is_side), "tile": rt.get("pai"),
            "tsumogiri": bool(rt.get("tsumogiri", False)),
            "bbox_model": "fixed_axis_aligned_rect_in_fullwarp", "reliable": True,
            "poly_fullwarp": np.round(poly, 1).tolist(),
            "poly_original": np.round(fullwarp_to_original(poly, H_full_inv), 1).tolist(),
            "face_poly_fullwarp": np.round(face, 1).tolist(),
            "face_poly_original": np.round(fullwarp_to_original(face, H_full_inv), 1).tolist(),
        })
    return slots


# =============================================================================
# 9c. Composition-aware meld model v2 (2026-07-02)
# =============================================================================
# The v1 MELD_STRIP laid every meld tile as an equal-width upright cell — wrong in
# two ways that together produce the observed "per-case" drift:
#   1. The CLAIMED tile is rendered SIDEWAYS (wider along the strip, shallower
#      across it), and its position inside the meld encodes the discarder
#      (kamicha=first, toimen=middle, shimocha=last; chi always first).
#   2. Strip length therefore depends on meld COMPOSITION (a kan with a sideways
#      tile is ~1/3 tile longer than 4 uprights), so a uniform step accumulates
#      error over multi-meld strips.
# v2 lays variable-width cells from a fixed OUTER CORNER inward: upright cell =
# w(along)×d(cross), sideways cell = d(along)×w(cross) baseline-aligned, kakan =
# a second sideways tile stacked on the claimed one toward the table centre,
# ankan = back/face/face/back. Seat frames are the 4-fold table rotation.
#
# corner: outer corner of the strip (first meld's outermost tile, baseline edge);
# along:  unit vector, strip growth direction (toward the hand);
# cross:  unit vector, from the baseline toward the table centre;
# w: upright tile's along-width;  d: tile depth = sideways tile's along-width;
# gap: spacing between melds. Values calibrated on data (see calibrate script).
# Corners refit 2026-07-02 (rigid strip snap over 182 rounds; within-round
# σ≈0.9px, across-round σ≈4px → the strip is fixed per round and floats a few px
# between rounds; the annotator's per-frame snap absorbs the residual).
# w/d = the measured tile face (~70.5×92.5, same plane as the rivers).
# Recalibrated 2026-07-08 (STATUS §1.50): pos3 corner was ~+46px (half a tile) off
# ALONG — the stale offset parked snap_meld_strip at the aliasing midpoint (self-
# similar equal-pitch tiles), so pos3 flipped one whole tile on ~26% of frames
# ("上家副露严重失位"). Corner moved +45.5 ALONG (inward, toward the hand: y 1797.6→1752.1
# with along=(0,-1)) -> full-tile flips ~26%→<1% (residual near-miss ~5% is per-frame
# snap noise on the far seat; Phase 2 round consensus targets it). pos2 (对家) is NOT
# fixed here — its corner is well-calibrated; its adjacent-frame flicker is a per-round
# snap issue for Phase 2. pos0 ALSO shows a ~+46px cross offset, but its snap locks
# that correction rock-solid every frame (0% mislock) — LEFT AS-IS: recalibrating
# pos0 only traded a benign consistent offset for per-frame scatter (0%→4.4%).
# Re-run calibrate_annotation_model.py + verify with scripts/annotate/meld_snap_qa.py
# after any warp/mask change (guard catches a corner regressing toward aliasing).
MELD_STRIP2 = {
    0: {"corner": (2388.2, 1889.5), "along": (-1.0, 0.0), "cross": (0.0, -1.0), "w": 70.2, "d": 92.1, "gap": 0.0},
    1: {"corner": (2454.0, 153.0),  "along": (0.0, 1.0),  "cross": (-1.0, 0.0), "w": 70.5, "d": 93.4, "gap": 0.0},
    2: {"corner": (685.0, 135.0),   "along": (1.0, 0.0),  "cross": (0.0, 1.0),  "w": 70.4, "d": 92.2, "gap": 0.0},
    3: {"corner": (624.5, 1752.1),  "along": (0.0, -1.0), "cross": (1.0, 0.0),  "w": 71.0, "d": 93.0, "gap": 0.0},
}
# Which meld sits at the corner. Majsoul anchors the FIRST (oldest) meld at the
# fixed near corner (verified §1.9); walking from the corner therefore reverses
# each meld's own display order.
MELD_OLDEST_AT_CORNER = True
# Whether the corner walk reverses each meld's internal display order (True when
# the strip grows opposite to the owner's left-to-right reading direction).
MELD_WITHIN_REVERSED = True
# Sideways position inside a 4-tile daiminkan claimed from toimen ("middle"):
# display index 1 (second from the player's left). Calibratable.
KAN_TOIMEN_SIDEWAYS_IDX = 1

# --------------------------------------------------------------------------- #
# 3-player (sanma) geometry variants + mode switch
# --------------------------------------------------------------------------- #
# Sanma reuses the 4P screen ring unchanged (actors 0-2 = E1 chair indices,
# chair 3 = the would-be north seat renders empty all game — STATUS §1.59), the
# same fullwarp homography, DISCARD_READ, foot sizes and the snap machinery.
# Only the render metrics differ — slightly but systematically (STATUS §1.60;
# measured by calibrate_annotation_model over 788 records / 5 games / all three
# hero seats / plain + 2 skinned cloths; JSON: scratchpad/calib3p_run1_run2.json):
#   * side-river column pitch -2.6..-2.9% (74.88/74.74 -> 72.72/72.77)
#   * self-river row pitch +3.1%, across row pitch -5.2% (+6.3px row1 fold)
#   * SELF meld corner sits 46px further inward (cross, σ=0.00) — the hero
#     nukidora lane occupies the 4P corner position.
DISCARD_GRID_3P = {
    0: {"o": (1347.9, 1241.0), "dcol": (72.13, 0.0), "drow": (0.0, 112.0)},   # self  down
    1: {"o": (1818.0, 772.1),  "dcol": (0.0, 72.72), "drow": (97.95, 0.0)},   # right rightward
    2: {"o": (1354.5, 675.9),  "dcol": (72.61, 0.0), "drow": (0.0, -104.25)}, # across up
    3: {"o": (1244.0, 768.3),  "dcol": (0.0, 72.77), "drow": (-98.87, 0.0)},  # left  leftward
}
# pos2/pos3 o carry the row1-chain fold of the 2nd calibration pass (the tool's
# printed o comes from the edge fit alone; the residual uniform row shift is
# folded along the row-advance direction — verify pass showed +3.3/+2.8px).
DISCARD_ROW_OFFSETS_3P: Dict[int, list] = {
    0: [0.0, 96.2, 194.0],
    1: [0.0, 95.6, 195.0],
    2: [0.0, 96.3, 189.3],
    3: [0.0, 96.6, 193.2],
}
MELD_STRIP2_3P = {
    0: {"corner": (2388.0, 1843.5), "along": (-1.0, 0.0), "cross": (0.0, -1.0), "w": 70.2, "d": 92.1, "gap": 0.0},
    1: {"corner": (2453.5, 153.0),  "along": (0.0, 1.0),  "cross": (-1.0, 0.0), "w": 70.5, "d": 93.4, "gap": 0.0},
    2: {"corner": (686.2, 134.5),   "along": (1.0, 0.0),  "cross": (0.0, 1.0),  "w": 70.4, "d": 92.2, "gap": 0.0},
    3: {"corner": (625.0, 1750.1),  "along": (0.0, -1.0), "cross": (1.0, 0.0),  "w": 71.0, "d": 93.0, "gap": 0.0},
}

# Nukidora pile (3P only): face-centre anchor + per-tile step + face foot, per
# SCREEN pos, measured by scripts/annotate/calibrate_nukidora.py over run_1+run_2
# (n=105-306 per seat; anchor σ≤1px except SELF's ±12px per-round float along the
# row — a fill-maximizing 1-D snap in frame.py absorbs it). The pile sits in its
# own lane between the meld strip and the river; raw component centres include
# the camera-side skirt, so these anchors are face-trimmed by (blob-face)/2 along
# the radial-to-nadir direction (validated: pos2's trim lands exactly on the
# skirt-free first-pass measurement, 349.0 vs 348.6).
NUKI_STRIP_3P = {
    0: {"anchor": (2034.9, 1569.5), "step": (-72.61, 0.07), "foot": (72.0, 92.0)},  # self  grow-left
    1: {"anchor": (2180.3, 588.6),  "step": (-0.15, 85.06), "foot": (93.0, 71.0)},  # right grow-down
    2: {"anchor": (945.6, 349.0),   "step": (72.46, 0.0),   "foot": (73.0, 92.0)},  # across grow-right
    3: {"anchor": (860.2, 1487.4),  "step": (0.12, -78.23), "foot": (86.0, 72.0)},  # left  grow-up
}


def generate_nukidora_boxes(seat: int, count: int, H_full_inv: np.ndarray,
                            along_offset: float = 0.0) -> List[Dict[str, Any]]:
    """Face boxes for one seat's nukidora pile (3P). ``seat`` is the SCREEN pos.

    Shaped like meld cells (tile + poly_fullwarp/poly_original, plus
    ``nuki: True``) so frame.py can append them to ``rec["meld_boxes"]`` and
    every downstream consumer (crops, YOLO, QA) treats them as regular face-up
    N tiles. ``along_offset`` slides the pile along its step axis (the SELF
    pile floats per round like melds do).
    """
    if count <= 0:
        return []
    cfg = NUKI_STRIP_3P[seat]
    anchor = np.array(cfg["anchor"], float)
    step = np.array(cfg["step"], float)
    w, h = cfg["foot"]
    unit = step / (np.linalg.norm(step) + 1e-9)
    out: List[Dict[str, Any]] = []
    for k in range(count):
        cx, cy = anchor + k * step + along_offset * unit
        poly = np.float32([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                           [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
        out.append({"tile": "N", "nuki": True, "sideways": False,
                    "poly_fullwarp": np.round(poly, 1).tolist(),
                    "poly_original": np.round(fullwarp_to_original(poly, H_full_inv), 1).tolist()})
    return out


# Pristine per-mode tables, captured at import (inner containers copied so
# mutations of the ACTIVE dicts can never corrupt them).
def _copy_tables(grid, offs, strip):
    return ({k: dict(v) for k, v in grid.items()},
            {k: list(v) for k, v in offs.items()},
            {k: dict(v) for k, v in strip.items()})

_MODE_TABLES = {
    False: _copy_tables(DISCARD_GRID, DISCARD_ROW_OFFSETS, MELD_STRIP2),
    True:  _copy_tables(DISCARD_GRID_3P, DISCARD_ROW_OFFSETS_3P, MELD_STRIP2_3P),
}
_SANMA_ACTIVE = False


@dataclass(frozen=True)
class BoardGeometry:
    """An IMMUTABLE per-mode geometry bundle, passed explicitly.

    ``set_sanma()`` below swaps the module-level tables IN PLACE, process-wide.
    That is fine for the offline annotator and the calibration tools (single
    frame, single thread), but the recognition worker is a long-lived server that
    serves both modes: a per-request ``set_sanma()`` would corrupt a CONCURRENT
    request of the other mode, and it would do it silently — the wrong geometry
    does not raise, it just quietly reads a different, plausible, WRONG board.
    So recognition takes its geometry as an argument and never touches the globals.
    """
    sanma: bool
    discard_grid: dict
    discard_row_offsets: dict
    meld_strip2: dict
    nuki_strip: dict | None          # None in 4P: there is no north pile


def _frozen(sanma: bool) -> BoardGeometry:
    # Built from the PRISTINE import-time snapshot, never from the live dicts.
    # Aliasing them would mean one set_sanma(True) anywhere in the process (a
    # test, a calibration script) permanently poisons the recogniser's 4P
    # geometry — the exact class of silent corruption this class exists to stop.
    grid, offs, strip = _copy_tables(*_MODE_TABLES[sanma])
    nuki = ({seat: dict(cfg) for seat, cfg in NUKI_STRIP_3P.items()}
            if sanma else None)
    return BoardGeometry(sanma, grid, offs, strip, nuki)


GEOMETRY_4P = _frozen(False)
GEOMETRY_3P = _frozen(True)


def geometry_for(sanma: bool) -> BoardGeometry:
    return GEOMETRY_3P if sanma else GEOMETRY_4P


def set_sanma(flag: bool) -> None:
    """Swap the ACTIVE geometry constants between 4P and sanma, in place.

    Process-global, idempotent, cheap — callers set it per frame from
    ``BoardState.sanma`` (annotate_frame and the calibration tool do). The
    clear+update keeps dict identity, so both ``P.DISCARD_GRID`` attribute
    readers and ``from pipeline import DISCARD_GRID`` holders follow the swap.
    """
    global _SANMA_ACTIVE
    flag = bool(flag)
    if flag == _SANMA_ACTIVE:
        return
    for dst, src in zip((DISCARD_GRID, DISCARD_ROW_OFFSETS, MELD_STRIP2),
                        _copy_tables(*_MODE_TABLES[flag])):
        dst.clear()
        dst.update(src)
    _SANMA_ACTIVE = flag


def _remove_one(lst: list, item: str) -> list:
    out = list(lst)
    if item in out:
        out.remove(item)
    return out


def meld_display_cells(meld: Mapping[str, Any], seat: int) -> List[Dict[str, Any]]:
    """Flatten one meld into display-ordered cells (owner's view, left-to-right).

    `meld` needs: type, tiles (sorted), from_seat, called_pai, added_pai.
    Cell dict: {label, sideways: bool, stacked: [label] for a kakan added tile}.
    """
    mtype = meld["type"]
    tiles = [str(t) for t in meld.get("tiles", [])]
    called = str(meld.get("called_pai") or "")
    added = str(meld.get("added_pai") or "")
    if mtype == "ankan":
        vis = tiles[1:3]
        reds = [t for t in tiles if t.endswith("r")]
        if reds and reds[0] not in vis:            # Majsoul always shows the red 5
            vis = [reds[0], vis[-1] if vis else reds[0]]
        return [{"label": "back", "sideways": False},
                {"label": vis[0] if vis else "?", "sideways": False},
                {"label": vis[1] if len(vis) > 1 else "?", "sideways": False},
                {"label": "back", "sideways": False}]
    if mtype == "nukidora":                        # 3P only; render as upright stack
        return [{"label": t, "sideways": False} for t in tiles]

    rest = list(tiles)
    if mtype == "kakan" and added:
        rest = _remove_one(rest, added)
    if not called:                                 # legacy data: assume first tile
        called = rest[0] if rest else "?"
    rest = _remove_one(rest, called)

    rel = (int(meld.get("from_seat", seat)) - seat) % 4   # 1=shimocha 2=toimen 3=kamicha
    if mtype == "chi" or rel == 3:
        pos = 0
    elif rel == 2:
        pos = KAN_TOIMEN_SIDEWAYS_IDX if len(rest) >= 3 else 1
    else:
        pos = len(rest)
    cells = [{"label": t, "sideways": False} for t in rest]
    claimed_cell: Dict[str, Any] = {"label": called, "sideways": True}
    if mtype == "kakan":
        claimed_cell["stacked"] = [added if added else called]
    cells.insert(pos, claimed_cell)
    return cells


def generate_meld_boxes_v2(seat: int, melds: list, H_full_inv: np.ndarray,
                           along_offset: float = 0.0) -> List[Dict[str, Any]]:
    """Composition-aware meld boxes for one seat (fullwarp + original polys).

    `melds` = list of {type, tiles, from_seat, called_pai, added_pai} in CHRONO
    order (oldest first). `along_offset` slides the whole strip along its axis
    (used by the per-frame image snap; + = inward/toward the hand).
    """
    if not melds:
        return []
    cfg = MELD_STRIP2[seat]
    corner = np.array(cfg["corner"], float)
    along = np.array(cfg["along"], float)
    cross = np.array(cfg["cross"], float)
    w, d, gap = cfg["w"], cfg["d"], cfg["gap"]

    seq = melds if MELD_OLDEST_AT_CORNER else list(reversed(melds))
    out: List[Dict[str, Any]] = []
    p = float(along_offset)

    def _emit(a0: float, aw: float, c0: float, cd: float, label: str,
              sideways: bool, added: bool, meld_idx: int, mtype: str) -> None:
        pts = [corner + along * a0 + cross * c0,
               corner + along * (a0 + aw) + cross * c0,
               corner + along * (a0 + aw) + cross * (c0 + cd),
               corner + along * a0 + cross * (c0 + cd)]
        poly = axis_aligned_bbox_poly(np.float32(pts))
        out.append({
            "area": "fuuro", "seat": seat, "relative_position": RELATIVE_SEAT_NAME[seat],
            "tile": label, "meld_index": meld_idx, "meld_type": mtype,
            "sideways": bool(sideways), "is_added_kan": bool(added),
            "is_kan_tile": mtype in ("daiminkan", "ankan", "kakan"),
            "bbox_model": "axis_aligned_rect_in_fullwarp", "reliable": True,
            "note": "composition-aware corner-anchored strip (v2)",
            "poly_fullwarp": np.round(poly, 1).tolist(),
            "poly_original": np.round(fullwarp_to_original(poly, H_full_inv), 1).tolist(),
        })

    for mi, meld in enumerate(seq):
        cells = meld_display_cells(meld, seat)
        if MELD_WITHIN_REVERSED:                   # corner walk reverses display order
            cells = list(reversed(cells))
        for cell in cells:
            aw = d if cell["sideways"] else w      # sideways = deeper along, shallower across
            cd = w if cell["sideways"] else d
            _emit(p, aw, 0.0, cd, cell["label"], cell["sideways"], False, mi, meld["type"])
            for st_label in cell.get("stacked", []):
                _emit(p, aw, cd, cd, st_label, True, True, mi, meld["type"])
            p += aw
        p += gap
    return out


def meld_strip_len(seat: int, melds: list) -> float:
    """Total along-strip length of the meld area (for snap search windows)."""
    cfg = MELD_STRIP2[seat]
    w, d, gap = cfg["w"], cfg["d"], cfg["gap"]
    total = 0.0
    for meld in melds:
        for cell in meld_display_cells(meld, seat):
            total += d if cell["sideways"] else w
        total += gap
    return total


# =============================================================================
# 9d. Mask-based edge / crevice detectors + rigid meld-strip snap (2026-07-02)
# =============================================================================
# The tile's white THICKNESS SKIRT (the side facing the camera) merges with the
# face in any brightness mask, so blob correlation is biased toward the skirt.
# The unbiased features are:
#   * the thin dark CREVICE between laterally adjacent faces (3D tile edges), and
#   * the face's far-side EDGE transitions (felt/crevice -> face), which are
#     skirt-free: the skirt always points toward the camera nadir (+y in
#     fullwarp, lateral sign(NADIR_X - x)); occluded between packed tiles.
NADIR_X = 1536.0


def tile_face_mask(fullwarp_bgr: np.ndarray | None = None, *,
                   hsv: np.ndarray | None = None) -> np.ndarray:
    """White tile-face (+skirt) mask. Pass ``hsv`` (a precomputed BGR2HSV of the
    same image) when several masks share one frame — the conversion dominates."""
    if hsv is None:
        hsv = cv2.cvtColor(fullwarp_bgr, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 1] < 70) & (hsv[..., 2] > 165)).astype(np.uint8)


def tile_back_mask(fullwarp_bgr: np.ndarray | None = None, *,
                   hsv: np.ndarray | None = None) -> np.ndarray:
    """Colored tile-back mask for snap face/back discrimination (any skin).

    Saturation-based so it captures orange (default) AND skinned colored backs while
    staying disjoint from the white face mask (S<70) — snap needs to tell back cells
    from face cells. A near-white/grey skin back is (correctly) indistinguishable from
    a face here; its labeling reliability comes from tile_live_mask, not this mask.
    """
    if hsv is None:
        hsv = cv2.cvtColor(fullwarp_bgr, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] > 70).astype(np.uint8)


def tile_live_mask(fullwarp_bgr: np.ndarray | None = None, *,
                   hsv: np.ndarray | None = None) -> np.ndarray:
    """Skin-agnostic 'a tile is rendered here' mask: colored OR bright pixels.

    Used ONLY to judge liveness of a slot/cell GT already labels 'back' (drop the
    rare frames where GT leads the client render and the slot is still empty/black).
    Not for face/back discrimination — it lights up faces too (that is tile_back_mask's
    job). Colored-or-bright hedges both desaturated (grey) and dark skin backs.
    """
    if hsv is None:
        hsv = cv2.cvtColor(fullwarp_bgr, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 1] > 60) | (hsv[..., 2] > 110)).astype(np.uint8)


def _profile(mask: np.ndarray, x1: float, y1: float, x2: float, y2: float,
             axis: str) -> np.ndarray | None:
    """Mean-mask 1D profile over a window; indexed by x (axis='x') or y."""
    H, W = mask.shape[:2]
    x1, x2 = max(0, int(x1)), min(W, int(x2))
    y1, y2 = max(0, int(y1)), min(H, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    sub = mask[y1:y2, x1:x2]
    return sub.mean(axis=0) if axis == "x" else sub.mean(axis=1)


def find_crevice(mask: np.ndarray, span: tuple, pred: float, axis: str,
                 r: int = 10) -> tuple[float, float]:
    """Dark-minimum position near `pred` between two adjacent faces.

    span = (lo, hi) window in the OTHER axis. Returns (pos, contrast in 0..1);
    contrast <= 0 means nothing usable found."""
    if axis == "x":
        p = _profile(mask, pred - r, span[0], pred + r + 1, span[1], "x")
    else:
        p = _profile(mask, span[0], pred - r, span[1], pred + r + 1, "y")
    if p is None or len(p) < 5:
        return pred, 0.0
    i = int(np.argmin(p))
    sides = max(p[0], p[-1])
    contrast = float(sides - p[i])
    # parabolic sub-pixel refine
    if 0 < i < len(p) - 1:
        d = (p[i - 1] - p[i + 1]) / (2.0 * (p[i - 1] - 2 * p[i] + p[i + 1]) + 1e-9)
        i = i + float(np.clip(d, -1, 1))
    return float(pred - r + i), contrast


def find_edge(mask: np.ndarray, span: tuple, pred: float, axis: str,
              inside: int, r: int = 14) -> tuple[float, float]:
    """Face boundary near `pred`: strongest LOW->HIGH plateau transition toward
    the face side. Uses a k-wide plateau difference rather than a 1px gradient,
    so a razor-sharp 2px crevice (white on BOTH sides) cannot outscore the real
    but blurred felt->face edge. inside=+1 -> face at increasing coordinate."""
    if axis == "x":
        p = _profile(mask, pred - r, span[0], pred + r + 2, span[1], "x")
    else:
        p = _profile(mask, span[0], pred - r, span[1], pred + r + 2, "y")
    if p is None or len(p) < 5:
        return pred, 0.0
    if inside < 0:
        p = p[::-1]
    # asymmetric plateaus: the BEFORE window is long (12px) so a wide inter-tile
    # crevice (<=8px of dark, white on both sides) cannot impersonate felt; the
    # AFTER window stays short so the edge localizes sharply.
    kb = min(12, max(3, len(p) // 3))
    ka = min(5, max(2, len(p) // 4))
    if len(p) >= kb + ka + 2:
        cs = np.concatenate([[0.0], np.cumsum(p)])
        n = len(p) - kb - ka + 1
        j = np.arange(n)
        before = (cs[j + kb] - cs[j]) / kb
        after = (cs[j + kb + ka] - cs[j + kb]) / ka
        g = after - before
        i = int(np.argmax(g))
        pos = i + kb - 0.5
    else:
        g = np.diff(p)
        i = int(np.argmax(g))
        pos = i + 0.5
    if inside < 0:
        pos = (len(p) - 1) - pos
    return float(pred - r + pos), float(g[i] if len(g) else 0.0)


def _box_fill(ii: np.ndarray, x1, y1, x2, y2) -> float:
    """Mean mask inside a box, via cv2.integral matrix."""
    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    x1, x2 = max(0, min(W, int(x1))), max(0, min(W, int(x2)))
    y1, y2 = max(0, min(H, int(y1))), max(0, min(H, int(y2)))
    a = (x2 - x1) * (y2 - y1)
    if a <= 0:
        return 0.0
    return float(ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]) / a


MIN_CREVICE_CONTRAST = 0.18
MIN_EDGE_GRAD = 0.10


def snap_meld_strip(mask_face: np.ndarray, mask_back: np.ndarray,
                    mboxes: List[Dict[str, Any]], seat: int,
                    r_coarse: int = 60, r_cross: int = 16) -> tuple[float, float, dict]:
    """Rigid 2-dof snap of a generated meld strip against the masks.

    Majsoul anchors the strip loosely (it floats up to ~2/5 tile inward per
    round), so this is TWO-STAGE:
      1. coarse — slide the whole cell template +-r_coarse px along the strip,
         maximizing (face cells on face mask + back cells on back mask)
         coverage. Coverage is skirt-biased but only needs +-8px accuracy.
      2. fine  — skirt-free refinement around the coarse lock: interior
         crevices + the clean outer end (along); per-cell clean cross edges.
    Returns (d_along, d_cross, diag). Boxes are NOT modified."""
    cfg = MELD_STRIP2[seat]
    along = np.array(cfg["along"]); cross = np.array(cfg["cross"])
    horiz = abs(along[0]) > abs(along[1])          # strip runs along x?
    comb = ((mask_face | mask_back) > 0).astype(np.uint8)

    cells = [b for b in mboxes if not b.get("is_added_kan")]
    rects = [np.float32(b["poly_fullwarp"]) for b in cells]
    rr = [(float(p[:, 0].min()), float(p[:, 1].min()),
           float(p[:, 0].max()), float(p[:, 1].max())) for p in rects]
    ax = 0 if horiz else 1                          # along axis index
    order = np.argsort([r[ax] for r in rr])
    rr = [rr[i] for i in order]
    cells = [cells[i] for i in order]

    # ---- stage 1: candidate along-locks, judged by fine-stage evidence ----
    # Two coarse locators, each fallible in its own way: the strip's SKIRT-FREE
    # end edge (exact, but can be beaten by interior features on long skewed
    # strips) and a whole-template coverage scan (robust, but skirt-biased when
    # the along axis parallels the skirt). Each candidate is refined by the
    # crevice/edge fine stage and scored by the summed feature contrast — the
    # right lock puts a high-contrast crevice at every cell boundary.
    H, W = comb.shape[:2]
    step_vec = (1, 0) if horiz else (0, 1)

    def _shift(rr_, s):
        dx, dy = step_vec[0] * s, step_vec[1] * s
        return [(r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy) for r in rr_]

    candidates = {0}
    if horiz:
        use_max = float(np.mean([r[0] for r in rr])) > NADIR_X
        end = rr[-1] if use_max else rr[0]
        pred_e = end[2] if use_max else end[0]
        pos_d, g = find_edge(comb, (end[1] + 6, end[3] - 6), pred_e, "x",
                             -1 if use_max else +1, r=r_coarse)
    else:
        end = rr[0]
        pred_e = end[1]
        pos_d, g = find_edge(comb, (end[0] + 6, end[2] - 6), pred_e, "y", +1, r=r_coarse)
    if g >= MIN_EDGE_GRAD:
        candidates.add(int(round(pos_d - pred_e)))
    # blob coverage scan over a local window
    x0 = max(0, int(min(r[0] for r in rr)) - r_coarse - 4)
    y0 = max(0, int(min(r[1] for r in rr)) - r_coarse - 4)
    x1_ = min(W, int(max(r[2] for r in rr)) + r_coarse + 4)
    y1_ = min(H, int(max(r[3] for r in rr)) + r_coarse + 4)
    ii_f = cv2.integral(mask_face[y0:y1_, x0:x1_])
    ii_b = cv2.integral(mask_back[y0:y1_, x0:x1_])
    lw, lh = x1_ - x0, y1_ - y0

    def _cov(bx, ii, dx, dy):
        if not len(bx):
            return 0.0
        xx1 = np.clip(bx[:, 0] + dx, 0, lw); yy1 = np.clip(bx[:, 1] + dy, 0, lh)
        xx2 = np.clip(bx[:, 2] + dx, 0, lw); yy2 = np.clip(bx[:, 3] + dy, 0, lh)
        return float((ii[yy2, xx2] - ii[yy1, xx2] - ii[yy2, xx1] + ii[yy1, xx1]).sum())

    off4 = np.array([x0, y0, x0, y0], np.int32)
    fb = np.int32([r for r, c in zip(rr, cells) if c["tile"] != "back"]).reshape(-1, 4) - off4
    bbx = np.int32([r for r, c in zip(rr, cells) if c["tile"] == "back"]).reshape(-1, 4) - off4
    best_cov, blob_off = -1.0, 0
    for s in range(-r_coarse, r_coarse + 1, 2):
        dx, dy = step_vec[0] * s, step_vec[1] * s
        c = _cov(fb, ii_f, dx, dy) + _cov(bbx, ii_b, dx, dy)
        if c > best_cov:
            best_cov, blob_off = c, s
    candidates.add(blob_off)

    def _cross_coarse(rr_):
        vals = []
        for b, r in zip(cells, rr_):
            m = mask_back if b["tile"] == "back" else mask_face
            if horiz:
                pos, g2 = find_edge(m, (r[0] + 6, r[2] - 6), r[1], "y", +1, r=60)
                if g2 >= MIN_EDGE_GRAD:
                    vals.append(pos - r[1])
            else:
                east = (r[0] + r[2]) / 2 > NADIR_X
                pred = r[2] if east else r[0]
                pos, g2 = find_edge(m, (r[1] + 6, r[3] - 6), pred, "x", -1 if east else +1, r=60)
                if g2 >= MIN_EDGE_GRAD:
                    vals.append(pos - pred)
        return float(np.median(vals)) if len(vals) >= 2 else 0.0

    def _fine(rr_):
        """(along deltas, cross deltas, evidence score) around a candidate lock."""
        alist, clist, score = [], [], 0.0
        for a, b in zip(rr_[:-1], rr_[1:]):
            pred = (a[ax + 2] + b[ax]) / 2.0
            lo = max(a[1 - ax], b[1 - ax]) + 6
            hi = min(a[3 - ax], b[3 - ax]) - 6
            if hi - lo < 20:
                continue
            pos, c = find_crevice(comb, (lo, hi), pred, "x" if horiz else "y")
            if c >= MIN_CREVICE_CONTRAST:
                alist.append(pos - pred)
                score += c
        if horiz:
            use_max2 = float(np.mean([r[0] for r in rr_])) > NADIR_X
            e = rr_[-1] if use_max2 else rr_[0]
            pred = e[2] if use_max2 else e[0]
            pos, g2 = find_edge(comb, (e[1] + 6, e[3] - 6), pred, "x", -1 if use_max2 else +1)
        else:
            e = rr_[0]
            pred = e[1]
            pos, g2 = find_edge(comb, (e[0] + 6, e[2] - 6), pred, "y", +1)
        if g2 >= MIN_EDGE_GRAD:
            alist.append(pos - pred)
            score += g2
        for b, r in zip(cells, rr_):
            m = mask_back if b["tile"] == "back" else mask_face
            if horiz:
                pos, g2 = find_edge(m, (r[0] + 6, r[2] - 6), r[1], "y", +1, r=r_cross)
                if g2 >= MIN_EDGE_GRAD:
                    clist.append(pos - r[1])
                    score += g2
            else:
                east = (r[0] + r[2]) / 2 > NADIR_X
                pred = r[2] if east else r[0]
                pos, g2 = find_edge(m, (r[1] + 6, r[3] - 6), pred, "x", -1 if east else +1, r=r_cross)
                if g2 >= MIN_EDGE_GRAD:
                    clist.append(pos - pred)
                    score += g2
        # penalize large fine residues: the wrong lock "finds" scattered features
        if alist and np.std(alist) > 6.0:
            score *= 0.5
        return alist, clist, score

    best = None
    for cand in sorted(candidates, key=abs):
        rc = _shift(rr, cand)
        bc = _cross_coarse(rc)
        if abs(bc) > 6.0:
            cvec = (0.0, bc) if horiz else (bc, 0.0)
            rc = [(r[0] + cvec[0], r[1] + cvec[1], r[2] + cvec[0], r[3] + cvec[1]) for r in rc]
        else:
            bc = 0.0
        alist, clist, score = _fine(rc)
        if best is None or score > best[0] + 1e-9:
            best = (score, cand, bc, alist, clist)

    score, base_off, base_cross, d_along_meas, d_cross_meas = best
    da = base_off + (float(np.median(d_along_meas)) if d_along_meas else 0.0)
    dc = base_cross + (float(np.median(d_cross_meas)) if d_cross_meas else 0.0)
    # measurements are along the +x/+y axes; convert to along/cross vectors
    axis_along = np.array([1.0, 0.0]) if horiz else np.array([0.0, 1.0])
    axis_cross = np.array([0.0, 1.0]) if horiz else np.array([1.0, 0.0])
    d_along = da * float(np.dot(axis_along, along))
    d_cross = dc * float(np.dot(axis_cross, cross))
    diag = {"n_along": len(d_along_meas), "n_cross": len(d_cross_meas),
            "coarse_px": base_off, "raw_along_px": da, "raw_cross_px": dc,
            "score": round(score, 2)}
    return d_along, d_cross, diag


def shift_boxes(mboxes: List[Dict[str, Any]], seat: int, d_along: float,
                d_cross: float, H_full_inv: np.ndarray) -> List[Dict[str, Any]]:
    """Return copies of meld boxes rigidly shifted by (d_along, d_cross)."""
    cfg = MELD_STRIP2[seat]
    vec = np.array(cfg["along"]) * d_along + np.array(cfg["cross"]) * d_cross
    out = []
    for b in mboxes:
        nb = dict(b)
        poly = np.float32(b["poly_fullwarp"]) + vec
        nb["poly_fullwarp"] = np.round(poly, 1).tolist()
        nb["poly_original"] = np.round(fullwarp_to_original(poly, H_full_inv), 1).tolist()
        out.append(nb)
    return out
