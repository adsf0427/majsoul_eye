"""Spike (1.9b): single top-down table homography ``H_table`` for 河/副露 geometry.

Replaces the per-seat-quad geometry (which drifts on far/angled seats and floats
the 加杠 box onto empty felt) with ONE homography that rectifies the whole 16:9
board into a bird's-eye view. All coplanar elements — 4 rivers, 4 melds, riichi,
the three kans — are then placed in a single rectified frame, and the 4 seats are
literal 90° rotations of one canonical seat.

This is a VISUALIZATION spike (per docs/STATUS.md §1.9 precedent: see the boxes
before integrating). It does NOT modify the package. Outputs to fails/topdown_demo/
(git-ignored, regenerable).

Status (docs/STATUS.md §1.13): ARCHIVED + de-load-beared. Its H_table geometry is
superseded by the precise fullwarp annotator (``majsoul_eye.annotate``); it is kept
only as a runnable debug visualizer. The shared GT plumbing it used to own now lives
in the package (``annotate.cases.CASES``, ``annotate.seatgt``, ``capture.gtframes.load_pair``),
which it imports; the superseded ``RiverGrid`` + self meld strip are inlined below so
this file stands alone.

Run from repo root with PYTHONPATH=. and the conda `auto` python:
    PYTHONPATH=. $PY scripts/annotate/spike_topdown.py --list-seqs --capture captures/intermediate/gt/ai_run_3_game1.jsonl
    PYTHONPATH=. $PY scripts/annotate/spike_topdown.py --all-cases            # Mode A (warp) + B (original)
    PYTHONPATH=. $PY scripts/annotate/spike_topdown.py --case C_kakan_single --mode both
    PYTHONPATH=. $PY scripts/annotate/spike_topdown.py --warp --case rivers_full   # debug: warp + symmetry

FINDINGS (2026-06-30 spike, ai_run_3_game1/ai_run_3_game3 1080p):
  * A single H_table (fit from the 4 play-square corners, PLAY_CORNERS_NORM)
    rectifies the whole board to top-down. Camera fixed → H is constant; the
    normalized form is resolution-independent (BoardRegion absorbs the scale).
  * SQUARE WARP — the first calibration was SKEWED because the 4 hand-read corners
    were not left-right symmetric (a rotation). The capture camera has no roll/yaw
    (only pitch), so the board is MIRROR-symmetric in image space: build the quad
    from the reliably-detected LEFT play edge (Hough) mirrored about CX. Symmetric
    trapezoid = true projected square → un-skewed warp. The table's PHYSICAL corners
    are off-screen (extrapolated); we calibrate the visible inner play boundary.
  * AutoMajsoul (_external/AutoMajsoul) detects the felt quad by HSV colour
    segmentation then warps it to a square — but that needs the felt to be an INSET
    shape on a dark surround (its Android / a zoomed-out view). On our FULLSCREEN
    16:9 the blue felt FILLS the frame (ran its real detect(): returns ~the whole
    frame, mask covers 60%+), and Hough/edge can't isolate the top/bottom play edges
    (tile rows + hand strip confuse them). So we use a fixed-camera calibration of
    the play boundary instead of per-frame auto-detection. detect_table_quad() is a
    faithful port kept for the inset-view case (--detect).
  * RIVERS: the felt homography over-stretched the far region + ignored tile-height
    parallax, so the 4 rivers warped to DIFFERENT sizes (self row ~0.06, across ~0.085,
    right ~0.045) → boxes couldn't be uniform. FIX = FACE-PLANE REFIT: re-fit H from the
    four rivers' near-edge points (measured in the felt warp) to a congruent symmetric
    target (H_table_norm via _RIVER_SRC_FELT → _RIVER_TGT) — a metric rectification of the
    tile-FACE plane, so all 4 rivers become identical uniform grids (verified: self/across
    cell 0.038×0.060, left/right 0.060×0.038 = the 90° rotation). Grid = one symmetric bbox
    per seat (RIVER_BBOX_RECT, 4 rotations); reading order from _UNIT_CORNERS (ROT_SIGN=-1,
    the +1 build silently swapped left↔right rivers).
  * KAKAN is CO-PLANAR, not elevated. The §1.9 z-lift floated the added-tile box
    onto empty felt; here the added kan tile is modeled as an in-plane extra cell
    and the warp confirms the 8m kakan is a flat block on the table (see
    _meld_labels_with_kakan). This is the fix for the user's complaint.
  * MELDS: one self strip + rotation + a radial MELD_OUTWARD shift aligns the
    in-line meld columns (daiminkan/chi) reasonably. KNOWN residuals → integration:
    (1) H from hand-read felt corners is mildly asymmetric, so the LEFT periphery
        (long meld columns / ankan) is offset — refit H with findHomography (felt
        corners + center wind-tile anchors) to tighten;
    (2) the 1-row uniform-cell strip does not model the rotated *called* tile
        (wider, sideways) nor the kakan added tile sitting BESIDE it (2-D offset);
    (3) recalibrate the canonical meld strip directly in rect space instead of the
        radial-shift hack.
  Conclusion: the top-down approach is the right direction and fixes rivers +
  kakan; melds need the rect-space calibration above before integration.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from majsoul_eye import paths
from majsoul_eye.annotate.cases import CASES
from majsoul_eye.annotate.seatgt import SEAT_POS, _screen_to_seat
from majsoul_eye.capture.gtframes import build_seq_state, load_frames, load_pair
from majsoul_eye.normalize import BoardRegion


# --------------------------------------------------------------------------- #
# self-contained (archived-spike) geometry: the equal-subdivision RiverGrid and
# the self meld strip were removed from the package (majsoul_eye.label.river /
# coords.MELD_STRIPS) as superseded — inlined here so this spike stands alone.
# --------------------------------------------------------------------------- #
RIVER_COLS = 6
RIVER_ROWS = 3


@dataclass(frozen=True)
class RiverGrid:
    """A seat's discard grid as a quad subdivided into cols×rows cells.

    Reading order: cell 0 at ``g00``, columns g00→g10 then rows g00→g01.
    """
    g00: tuple
    g10: tuple
    g11: tuple
    g01: tuple
    cols: int = RIVER_COLS
    rows: int = RIVER_ROWS

    @property
    def capacity(self) -> int:
        return self.cols * self.rows

    def _homography(self):
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        dst = np.array([self.g00, self.g10, self.g11, self.g01], np.float32)
        return cv2.getPerspectiveTransform(src, dst)

    def cell_corners(self, i: int) -> np.ndarray:
        """4 (x,y) normalized corners of cell i (reading order)."""
        col, row = i % self.cols, i // self.cols
        u0, u1 = col / self.cols, (col + 1) / self.cols
        v0, v1 = row / self.rows, (row + 1) / self.rows
        pts = np.array([[[u0, v0], [u1, v0], [u1, v1], [u0, v1]]], np.float32)
        return cv2.perspectiveTransform(pts, self._homography())[0]


# self meld strip (was coords.MELD_STRIPS["self"]): bottom-right, right-anchored.
_MELD_SELF = {"quad": ((0.702, 0.861), (0.911, 0.873), (0.911, 0.960), (0.702, 0.950)), "count": 6}

DEFAULT_CAPTURE = "captures/intermediate/gt/ai_run_3_game1.jsonl"
OUT_DIR = "fails/topdown_demo"
RECT = 1000  # square rectified canvas (4-fold rotation is exact on a square)

# Play-square corners on a canonical 1080p frame (ai_run_3_game1 seq1458), normalized 0–1.
# Order [FL, FR, BR, BL] = bottom-left, bottom-right, top-right, top-left, SELF at
# the bottom (near) edge.
#
# The capture camera has no roll/yaw (only pitch), so the board is MIRROR-symmetric
# left↔right in image space. We therefore build the quad from the reliably-detected
# LEFT play-boundary edge (Hough: top corner TL, bottom corner BL) mirrored about
# the board's vertical axis CX. This guarantees a symmetric trapezoid = a true
# projected square → a square (un-skewed) warp. Refine = tweak CX / the left edge.
_TL = (0.2688, 0.1352)   # top-left play corner   (left edge, top)
_BL = (0.1911, 0.7361)   # bottom-left play corner (left edge, bottom)
CX = 0.490               # board vertical-symmetry axis (image x), CALIBRATE


def _mirror(p, cx=None):
    cx = CX if cx is None else cx
    return (2 * cx - p[0], p[1])


PLAY_CORNERS_NORM = [
    _BL,            # FL  front-left  = bottom-left
    _mirror(_BL),   # FR  front-right = bottom-right (mirror of BL)
    _mirror(_TL),   # BR  back-right  = top-right    (mirror of TL)
    _TL,            # BL  back-left   = top-left
]
MARGIN = 0.16         # play square occupies the central (1-2*MARGIN) of the canvas
MELD_OUTWARD = 72     # rect-units: push meld strips radially out toward the wall (CALIBRATE)
# Rotation direction bottom→right→top→left (screen pos 0=self,1=right,2=across,3=left).
# -1 verified: pos1 'right'→screen right, pos3 'left'→screen left (+1 swapped them).
ROT_SIGN = -1


# --------------------------------------------------------------------------- #
# --list-seqs : print candidate seqs so cases can be (re)discovered
# (load_pair / build_seq_state / load_frames come from majsoul_eye.capture.gtframes)
# --------------------------------------------------------------------------- #

def _meld_brief(state) -> str:
    parts = []
    for seat in range(4):
        ms = state.melds[seat]
        if ms:
            parts.append(f"s{seat}:" + ",".join(m.type for m in ms))
    return " | ".join(parts) if parts else "-"


def list_seqs(capture: str, frames_dir: Optional[str] = None) -> None:
    frames_dir = frames_dir or paths.frames_dir_for(capture)
    seq_state = build_seq_state(capture)
    frames = load_frames(frames_dir)
    print(f"# {capture}  (board-changing seqs with a saved frame)")
    print(f"# {'seq':>6} {'round':>6} {'rivers':>16} {'hero':>4}  melds")
    for seq in sorted(seq_state):
        if seq not in frames:
            continue
        s = seq_state[seq]
        rivers = [len(s.visible_river(k)) for k in range(4)]
        print(f"  {seq:>6} {str(s.bakaze)+str(s.kyoku):>6} {str(rivers):>16} "
              f"{len(s.hero_hand):>4}  {_meld_brief(s)}")


# --------------------------------------------------------------------------- #
# table-quad detection (ported from _external/AutoMajsoul ingame_recognizer.py):
# segment the felt by colour → largest 4-pt contour → order corners. Robust &
# resolution-independent — replaces hand-read PLAY_CORNERS so the warp is square.
# --------------------------------------------------------------------------- #

DET_SAMPLE_FRAC = 0.10
DET_DELTA_H = 15
DET_DELTA_SV = 60
DET_MIN_AREA_RATIO = 0.22
DET_APPROX_EPS = 0.02


def _order_quad(quad: np.ndarray) -> np.ndarray:
    """Order 4 pts as [TL, TR, BL, BR] (AutoMajsoul convention)."""
    s = quad.sum(axis=1)
    d = np.diff(quad, axis=1).ravel()
    o = np.zeros((4, 2), np.float32)
    o[0] = quad[np.argmin(s)]   # TL
    o[1] = quad[np.argmin(d)]   # TR
    o[2] = quad[np.argmax(d)]   # BL
    o[3] = quad[np.argmax(s)]   # BR
    return o


def _quad_from_contour(mask: np.ndarray, img_area: int):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < DET_MIN_AREA_RATIO * img_area:
        return None
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, DET_APPROX_EPS * peri, True)
    if len(approx) > 4:
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
    if len(approx) != 4:
        return None
    return _order_quad(approx.reshape(4, 2).astype(np.float32))


def detect_table_quad(frame_bgr: np.ndarray):
    """Return the felt quad as [TL,TR,BL,BR] pixel pts, or None. Colour first,
    Canny-edge fallback (AutoMajsoul IngameRecognizer.detect)."""
    h, w = frame_bgr.shape[:2]
    area = h * w
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    cy, cx = h // 2, int(w // 2 - w * 0.25)        # sample left-of-centre felt
    dh, dw = int(h * DET_SAMPLE_FRAC / 2), int(w * DET_SAMPLE_FRAC / 2)
    sample = hsv[max(0, cy - dh):cy + dh, max(0, cx - dw):cx + dw].reshape(-1, 3)
    mh, ms, mv = np.mean(sample, axis=0)
    lower = np.array([max(0, mh - DET_DELTA_H), max(0, ms - DET_DELTA_SV), max(0, mv - DET_DELTA_SV)], np.uint8)
    upper = np.array([min(179, mh + DET_DELTA_H), min(255, ms + DET_DELTA_SV), min(255, mv + DET_DELTA_SV)], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=3)
    quad = _quad_from_contour(mask, area)
    if quad is None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.dilate(cv2.Canny(gray, 50, 150), None, iterations=2)
        quad = _quad_from_contour(edges, area)
    return quad


# --------------------------------------------------------------------------- #
# H_table geometry (normalized board 0–1  <->  rectified RECT×RECT top-down)
# --------------------------------------------------------------------------- #

def _rect_square(rect: int) -> np.ndarray:
    m = MARGIN * rect
    R = rect
    # match PLAY_CORNERS order [FL, FR, BR, BL]
    return np.array([[m, R - m], [R - m, R - m], [R - m, m], [m, m]], np.float32)


def _felt_H(rect: int = RECT) -> np.ndarray:
    """Felt-plane homography from the play-boundary corners (norm 0–1 -> rect)."""
    src = np.array(PLAY_CORNERS_NORM, np.float32)
    dst = _rect_square(rect)
    return cv2.getPerspectiveTransform(src, dst)


# Face-plane refit: the felt homography over-stretches the far region and ignores
# tile-thickness parallax, so the 4 rivers come out different sizes. We re-fit H
# from the four rivers' near-edge end-points (measured in the FELT warp) to a
# symmetric, congruent target — a metric rectification of the tile-FACE plane, so
# all 4 rivers become identical uniform grids. (Measured on ai_run_3_game1 seq1458.)
#   src = tile near-edge ends in the felt warp (rect fractions); ordered L,R / T,B.
_RIVER_SRC_FELT = [
    (0.395, 0.615), (0.625, 0.615),   # self   near (top) : left, right
    (0.388, 0.395), (0.618, 0.395),   # across near (bot) : left, right
    (0.408, 0.360), (0.408, 0.630),   # left   near (right): top, bottom
    (0.640, 0.360), (0.640, 0.630),   # right  near (left) : top, bottom
]
_RIVER_TGT = [
    (0.385, 0.615), (0.615, 0.615),
    (0.385, 0.385), (0.615, 0.385),
    (0.385, 0.385), (0.385, 0.615),
    (0.615, 0.385), (0.615, 0.615),
]
_H_CACHE: dict = {}


def H_table_norm(rect: int = RECT) -> np.ndarray:
    """3x3 homography: normalized board coords (0–1) -> rectified top-down (0..rect).
    Re-fit from the river tiles (face plane) so all 4 rivers rectify congruently."""
    if rect in _H_CACHE:
        return _H_CACHE[rect]
    Hf = _felt_H(rect)
    src_felt = np.array([(x * rect, y * rect) for x, y in _RIVER_SRC_FELT], np.float32)
    norm_src = rect_to_norm(src_felt, Hf)           # norm positions of the measured tile corners
    tgt = np.array([(x * rect, y * rect) for x, y in _RIVER_TGT], np.float32)
    H, _ = cv2.findHomography(norm_src, tgt)
    _H_CACHE[rect] = H.astype(np.float32)
    return _H_CACHE[rect]


def norm_to_rect(pts, H: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def rect_to_norm(pts, H: np.ndarray) -> np.ndarray:
    Hinv = np.linalg.inv(H)
    p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, Hinv).reshape(-1, 2)


def rotate_rect(pts, k: int, rect: int = RECT) -> np.ndarray:
    """Rotate rect-space pts by k*90° about the canvas center (4-fold symmetry)."""
    c = rect / 2.0
    out = np.asarray(pts, np.float32).reshape(-1, 2).copy()
    for _ in range(k % 4):
        x, y = out[:, 0] - c, out[:, 1] - c
        # one 90° step; ROT_SIGN selects visual direction
        if ROT_SIGN > 0:
            out[:, 0], out[:, 1] = c - y, c + x
        else:
            out[:, 0], out[:, 1] = c + y, c - x
    return out


def _sym_cost(frame, region, corners, rect, mask):
    """4-fold symmetry cost: MSE between the warp and its 90/180/270° rotations
    over an annulus mask (the river band — symmetric; excludes HUD/avatars)."""
    src = np.array(corners, np.float32)
    m = MARGIN * rect
    dst = np.array([[m, rect - m], [rect - m, rect - m], [rect - m, m], [m, m]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    S = np.array([[1.0 / region.bw, 0, -region.ox / region.bw],
                  [0, 1.0 / region.bh, -region.oy / region.bh], [0, 0, 1.0]], np.float64)
    g = cv2.warpPerspective(frame, (H.astype(np.float64) @ S), (rect, rect))
    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cost = 0.0
    for code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        r = cv2.rotate(g, code)
        cost += float(np.mean(((g - r) ** 2)[mask]))
    return cost


def optimize_corners(frame, region, rect=RECT, iters=400):
    """Coordinate-descent on the 4 play corners to maximize warp 4-fold symmetry."""
    yy, xx = np.mgrid[0:rect, 0:rect]
    rr = np.sqrt((xx - rect / 2) ** 2 + (yy - rect / 2) ** 2) / rect
    mask = (rr > 0.10) & (rr < 0.24)            # river-band annulus
    corners = [list(c) for c in PLAY_CORNERS_NORM]
    best = _sym_cost(frame, region, corners, rect, mask)
    step = 0.02
    for _ in range(iters):
        improved = False
        for ci in range(4):
            for axis in (0, 1):
                for d in (step, -step):
                    trial = [list(c) for c in corners]
                    trial[ci][axis] += d
                    c = _sym_cost(frame, region, trial, rect, mask)
                    if c < best - 1e-6:
                        best, corners, improved = c, trial, True
        if not improved:
            step *= 0.5
            if step < 0.0006:
                break
    return [tuple(round(v, 4) for v in c) for c in corners], best


def pixel_to_rect_H(region: BoardRegion, H: np.ndarray) -> np.ndarray:
    """Compose frame-pixel -> normalized -> rect into one 3x3 for warpPerspective."""
    S = np.array([[1.0 / region.bw, 0, -region.ox / region.bw],
                  [0, 1.0 / region.bh, -region.oy / region.bh],
                  [0, 0, 1.0]], np.float64)
    return (H.astype(np.float64) @ S)


# --------------------------------------------------------------------------- #
# rect-space seat geometry: define ONE seat, rotate for the other three
# --------------------------------------------------------------------------- #

# Each seat's 6×3 river block as an axis-aligned bbox in rect (top-down) space
# (fractions of RECT), measured DIRECTLY on the clean warp. H isn't perfectly
# 4-fold symmetric (vertical rivers sit a touch further out than horizontal), so
# we snap each seat to its own measured bbox rather than rotating one grid.
#   pos_idx: 0=self(bottom) 1=right 2=across(top) 3=left
# After the face-plane refit the 4 rivers are congruent & symmetric, so the bboxes
# are 4 rotations of one: near edge at 0.385/0.615 (6-tile span), extending 3 rows
# outward (row block depth 0.18). _UNIT_CORNERS supplies each seat's reading order.
RIVER_BBOX_RECT = {
    0: (0.385, 0.615, 0.615, 0.795),   # self   (extends down)
    1: (0.615, 0.385, 0.795, 0.615),   # right  (extends right)
    2: (0.385, 0.205, 0.615, 0.385),   # across (extends up)
    3: (0.205, 0.385, 0.385, 0.615),   # left   (extends left)
}
# Unit-square corner order [g00,g10,g11,g01] for each seat = self order rotated
# k·90° (sense matches rotate_rect / ROT_SIGN=-1), so cell 0 is the first discard
# and reading orientation is correct per seat.
_UNIT_CORNERS = {
    0: [(0, 0), (1, 0), (1, 1), (0, 1)],
    1: [(0, 1), (0, 0), (1, 0), (1, 1)],
    2: [(1, 1), (0, 1), (0, 0), (1, 0)],
    3: [(1, 0), (1, 1), (0, 1), (0, 0)],
}


def seat_river_grid(pos_idx: int, H: np.ndarray) -> RiverGrid:
    """River grid for screen position pos_idx: map the per-seat rect bbox + rotated
    corner order back to normalized board coords (precise per-seat, not rotated H)."""
    x0, y0, x1, y1 = RIVER_BBOX_RECT[pos_idx]
    corners_rect = np.array([(x0 + u * (x1 - x0), y0 + v * (y1 - y0)) for u, v in _UNIT_CORNERS[pos_idx]],
                            np.float32) * RECT
    cn = rect_to_norm(corners_rect, H)
    return RiverGrid(tuple(cn[0]), tuple(cn[1]), tuple(cn[2]), tuple(cn[3]))


def _quad_norm_to_pts(region: BoardRegion, corners) -> np.ndarray:
    return np.array([region.norm_to_px(_box(x, y, x, y))[:2] for x, y in corners], np.int32)


def _strip_grid_norm(quad_norm, count: int, k: int) -> RiverGrid:
    """1-row grid of k cells from a strip quad (norm), anchored at the g10 end
    (first meld fixed at the near corner; column grows back toward g00). Mirrors
    meld._strip_grid with anchor='end' for all seats (the §1.9 unified rule)."""
    g = [np.array(p, float) for p in quad_norm]
    step = (g[1] - g[0]) / max(1, count)
    h = g[3] - g[0]
    g10 = g[1]
    g00 = g[1] - step * k
    return RiverGrid(tuple(g00), tuple(g10), tuple(g10 + h), tuple(g00 + h), cols=max(1, k), rows=1)


def seat_meld_strip_quad(pos_idx: int, H: np.ndarray, outward: float = MELD_OUTWARD):
    """Meld strip quad (norm) for screen position pos_idx, from the self strip
    mapped to rect, rotated, and pushed radially out toward the wall by `outward`
    rect-units (the self strip sits a touch inward of the real melds)."""
    base_rect = norm_to_rect(_MELD_SELF["quad"], H)
    rot_rect = rotate_rect(base_rect, pos_idx)
    c = RECT / 2.0
    d = rot_rect.mean(axis=0) - c
    n = d / (np.linalg.norm(d) + 1e-6)
    rot_rect = rot_rect + n * outward
    return rect_to_norm(rot_rect, H)


def _meld_labels_with_kakan(melds):
    """Flatten melds (reverse order) to (label, sideways?) cells. Kakan is modeled
    IN-PLANE: the added tile is a 2nd sideways cell appended next to the called
    tile — NOT lifted off the table (the §1.9 z-lift floated it onto empty felt)."""
    cells = []  # list of (label, is_added_kan)
    for m in reversed(list(melds)):
        if m.type == "ankan":
            t = m.tiles  # 4 identical; shown back, face, face, back
            cells += [("back", False), (t[1], False), (t[2], False), ("back", False)]
        elif m.type == "kakan":
            # 3 base (the pon) + 1 added tile, all coplanar
            base = m.tiles[:3]
            added = m.tiles[3] if len(m.tiles) > 3 else m.tiles[-1]
            cells += [(t, False) for t in base] + [(added, True)]
        else:
            cells += [(t, False) for t in m.tiles]
    return cells


def draw_meld_boxes(frame, region, state, H, thickness=2):
    """Draw all 4 seats' meld cells with GT labels. Kakan added tile in magenta."""
    for pos_idx, name in enumerate(SEAT_POS):
        seat = _screen_to_seat(state.hero_seat, name)
        if seat is None or not state.melds[seat]:
            continue
        cells = _meld_labels_with_kakan(state.melds[seat])
        quad = seat_meld_strip_quad(pos_idx, H)
        count = _MELD_SELF["count"]
        grid = _strip_grid_norm(quad, count, len(cells))
        for i, (lab, is_kan) in enumerate(cells):
            pts = grid.cell_corners(i)
            poly = _quad_norm_to_pts(region, [tuple(p) for p in pts])
            color = (255, 0, 255) if is_kan else ((0, 165, 255) if lab == "back" else (255, 200, 0))
            cv2.polylines(frame, [poly], True, color, thickness)
            c = poly.mean(axis=0).astype(int)
            cv2.putText(frame, lab, (c[0] - 14, c[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def draw_grids_on_warp(canvas, state, H, rect=RECT):
    """Overlay the rect-space river + meld grids on the top-down warp (for calibration)."""
    def poly_rect(norm_pts, color, t=2):
        rp = norm_to_rect([tuple(p) for p in norm_pts], H).astype(np.int32)
        cv2.polylines(canvas, [rp], True, color, t)
    for pos_idx, name in enumerate(SEAT_POS):
        seat = _screen_to_seat(state.hero_seat, name)
        if seat is None:
            continue
        gt = state.visible_river(seat)
        grid = seat_river_grid(pos_idx, H)
        for i in range(min(len(gt), grid.capacity)):
            poly_rect(grid.cell_corners(i), (0, 255, 255), 1)
        if state.melds[seat]:
            cells = _meld_labels_with_kakan(state.melds[seat])
            mg = _strip_grid_norm(seat_meld_strip_quad(pos_idx, H), _MELD_SELF["count"], len(cells))
            for i, (lab, is_kan) in enumerate(cells):
                poly_rect(mg.cell_corners(i), (255, 0, 255) if is_kan else (255, 200, 0), 2)
    return canvas


def draw_river_boxes(frame, region, state, H, thickness=2):
    """Draw all 4 seats' river cells (perspective quads) with GT labels on the frame."""
    for pos_idx, name in enumerate(SEAT_POS):
        seat = _screen_to_seat(state.hero_seat, name)
        if seat is None:
            continue
        gt = state.visible_river(seat)
        if not gt:
            continue
        grid = seat_river_grid(pos_idx, H)
        n = min(len(gt), grid.capacity)
        for i in range(n):
            rt = gt[i]
            pts = grid.cell_corners(i)  # 4 normalized corners
            poly = _quad_norm_to_pts(region, [tuple(p) for p in pts])
            color = (0, 255, 255)  # river = yellow
            cv2.polylines(frame, [poly], True, color, thickness)
            c = poly.mean(axis=0).astype(int)
            cv2.putText(frame, rt.pai, (c[0] - 14, c[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------- #
# --grid : dump a frame with a normalized-coordinate grid for eyeballing coords
# --------------------------------------------------------------------------- #

def draw_norm_grid(frame: np.ndarray, region: BoardRegion, step: float = 0.05) -> np.ndarray:
    img = frame.copy()
    n = int(round(1.0 / step))
    for i in range(n + 1):
        u = i * step
        x0, y0, x1, y1 = region.norm_to_px(_box(u, 0, u, 1))
        major = abs(u - round(u, 1)) < 1e-6 and (round(u * 10) % 1 == 0)
        c = (0, 200, 255) if (round(u, 2) in (0.5,)) else ((0, 160, 0) if major else (60, 60, 60))
        t = 2 if (round(u, 2) == 0.5) else (1 if major else 1)
        cv2.line(img, (x0, y0), (x1, y1), c, t)
        x0, y0, x1, y1 = region.norm_to_px(_box(0, u, 1, u))
        cv2.line(img, (x0, y0), (x1, y1), c, t)
        if major:
            px = region.norm_to_px(_box(u, 0, u, 0))
            cv2.putText(img, f"{u:.1f}", (px[0] + 2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            py = region.norm_to_px(_box(0, u, 0, u))
            cv2.putText(img, f"{u:.1f}", (4, py[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # overlay the current PLAY_CORNERS quad (magenta) to compare against the felt line
    poly = np.array([[region.norm_to_px(_box(x, y, x, y))[:2]] for x, y in PLAY_CORNERS_NORM], np.int32)
    cv2.polylines(img, [poly], True, (255, 0, 255), 2)
    for (x, y), name in zip(PLAY_CORNERS_NORM, ["FL", "FR", "BR", "BL"]):
        p = region.norm_to_px(_box(x, y, x, y))
        cv2.circle(img, (p[0], p[1]), 6, (255, 0, 255), -1)
        cv2.putText(img, name, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    return img


def _box(x0, y0, x1, y1):
    from majsoul_eye.coords import NormBox
    return NormBox(x0, y0, x1, y1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=DEFAULT_CAPTURE)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--case", default=None, help="named preset from CASES")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--mode", choices=["A", "B", "both"], default="both")
    ap.add_argument("--rect", type=int, default=RECT)
    ap.add_argument("--all-cases", action="store_true")
    ap.add_argument("--list-seqs", action="store_true")
    ap.add_argument("--check-symmetry", action="store_true")
    ap.add_argument("--grid", action="store_true", help="dump frame with a normalized-coord grid")
    ap.add_argument("--warp", action="store_true", help="dump top-down warp with play-square overlay")
    ap.add_argument("--detect", action="store_true", help="detect felt quad (AutoMajsoul method) + warp")
    ap.add_argument("--extract-frames", action="store_true", help="save each CASE's raw original frame to <out>/case_frames/")
    args = ap.parse_args()

    if args.extract_frames:
        out = os.path.join(args.out_dir, "case_frames")
        os.makedirs(out, exist_ok=True)
        for tag, cfg in CASES.items():
            frame, state, _ = load_pair(cfg["capture"], cfg["seq"], None)
            p = os.path.join(out, f"{tag}.png")
            cv2.imwrite(p, frame)
            print(f"{tag:16} {os.path.basename(cfg['capture']):14} seq{cfg['seq']:<5} hero={state.hero_seat}  {cfg.get('note','')}")
        print(f"\nwrote {len(CASES)} raw frames to {out}/")
        return

    if args.list_seqs:
        list_seqs(args.capture, args.frames_dir)
        return

    os.makedirs(args.out_dir, exist_ok=True)

    capture, seq = args.capture, args.seq
    if args.case:
        capture, seq = CASES[args.case]["capture"], CASES[args.case]["seq"]

    if args.grid:
        if seq is None:
            raise SystemExit("--grid needs --seq or --case")
        frame, state, region = load_pair(capture, seq, args.frames_dir)
        out = os.path.join(args.out_dir, f"grid_{os.path.splitext(os.path.basename(capture))[0]}_seq{seq}.png")
        cv2.imwrite(out, draw_norm_grid(frame, region))
        print(f"wrote {out}  (frame {frame.shape[1]}x{frame.shape[0]} hero_seat={state.hero_seat})")
        return

    if args.detect:
        if seq is None:
            raise SystemExit("--detect needs --seq or --case")
        frame, state, region = load_pair(capture, seq, args.frames_dir)
        quad = detect_table_quad(frame)
        stem = os.path.splitext(os.path.basename(capture))[0]
        if quad is None:
            print("detection FAILED (no felt quad)")
            return
        print("detected quad [TL,TR,BL,BR] (px):", quad.tolist())
        print("  normalized:", [(round((x - region.ox) / region.bw, 4), round((y - region.oy) / region.bh, 4)) for x, y in quad])
        dbg = frame.copy()
        cv2.polylines(dbg, [quad.astype(np.int32)], True, (0, 255, 0), 3)
        for (x, y), lab in zip(quad.astype(int), ["TL", "TR", "BL", "BR"]):
            cv2.circle(dbg, (x, y), 9, (255, 0, 0), -1)
            cv2.putText(dbg, lab, (x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(args.out_dir, f"detect_{stem}_seq{seq}.png"), dbg)
        S = args.rect
        dst = np.array([[0, 0], [S, 0], [0, S], [S, S]], np.float32)
        mat = cv2.getPerspectiveTransform(quad, dst)
        warped = cv2.warpPerspective(frame, mat, (S, S))
        cv2.imwrite(os.path.join(args.out_dir, f"detectwarp_{stem}_seq{seq}.png"), warped)
        print(f"wrote detect_{stem}_seq{seq}.png + detectwarp_{stem}_seq{seq}.png")
        return

    if args.warp:
        if seq is None:
            raise SystemExit("--warp needs --seq or --case")
        frame, state, region = load_pair(capture, seq, args.frames_dir)
        H = H_table_norm(args.rect)
        canvas = cv2.warpPerspective(frame, pixel_to_rect_H(region, H), (args.rect, args.rect))
        R, m = args.rect, int(MARGIN * args.rect)
        draw_grids_on_warp(canvas, state, H, args.rect)
        # play square (should land exactly on the warped felt boundary)
        cv2.rectangle(canvas, (m, m), (R - m, R - m), (0, 0, 255), 2)
        cv2.line(canvas, (R // 2, 0), (R // 2, R), (0, 180, 0), 1)
        cv2.line(canvas, (0, R // 2), (R, R // 2), (0, 180, 0), 1)
        for f in (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9):
            cv2.line(canvas, (int(f * R), 0), (int(f * R), R), (70, 70, 70), 1)
            cv2.line(canvas, (0, int(f * R)), (R, int(f * R)), (70, 70, 70), 1)
        out = os.path.join(args.out_dir, f"warp_{os.path.splitext(os.path.basename(capture))[0]}_seq{seq}.png")
        cv2.imwrite(out, canvas)
        # symmetry overlay: blend the warp with its own 90° rotation. If H rectifies
        # correctly, the felt boundary + 4 river grids coincide with their rotation.
        plain = cv2.warpPerspective(frame, pixel_to_rect_H(region, H), (args.rect, args.rect))
        rot = cv2.rotate(plain, cv2.ROTATE_90_CLOCKWISE)
        blend = cv2.addWeighted(plain, 0.5, rot, 0.5, 0)
        cv2.rectangle(blend, (m, m), (R - m, R - m), (0, 0, 255), 1)
        symout = os.path.join(args.out_dir, f"sym_{os.path.splitext(os.path.basename(capture))[0]}_seq{seq}.png")
        cv2.imwrite(symout, blend)
        # quantitative: MSE between play-square ring of warp and its 90° rotation
        ring = plain[m:R - m, m:R - m].astype(np.float32)
        ringr = rot[m:R - m, m:R - m].astype(np.float32)
        mse = float(np.mean((ring - ringr) ** 2))
        print(f"wrote {out} and {symout}  symmetry MSE(play-square)={mse:.0f}")
        return

    def render_one(cap, sq, tag):
        frame, state, region = load_pair(cap, sq, args.frames_dir)
        H = H_table_norm(args.rect)
        if args.mode in ("B", "both"):
            img = frame.copy()
            draw_river_boxes(img, region, state, H)
            draw_meld_boxes(img, region, state, H)
            out = os.path.join(args.out_dir, f"{tag}_B.png")
            cv2.imwrite(out, img)
            print(f"wrote {out}  hero_seat={state.hero_seat}")
        if args.mode in ("A", "both"):
            canvas = cv2.warpPerspective(frame, pixel_to_rect_H(region, H), (args.rect, args.rect))
            draw_grids_on_warp(canvas, state, H, args.rect)
            outA = os.path.join(args.out_dir, f"{tag}_A.png")
            cv2.imwrite(outA, canvas)
            print(f"wrote {outA}")

    if args.all_cases:
        for tag, cfg in CASES.items():
            render_one(cfg["capture"], cfg["seq"], tag)
        return

    if seq is None:
        raise SystemExit("need --seq or --case (or use --list-seqs)")
    render_one(capture, seq, args.case or f"{os.path.splitext(os.path.basename(capture))[0]}_seq{seq}")
    return


if __name__ == "__main__":
    main()
