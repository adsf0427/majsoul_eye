"""Detections -> ObservedState (spec 2026-07-05 §3.2).

Runs the calibrated annotate/pipeline geometry BACKWARD: detection centers are
mapped original->canonical(1920x1080)->fullwarp, then matched to the discard
grid / meld strip. Akagi-free (annotate.pipeline is pure geometry; capture/ is
never imported)."""
from __future__ import annotations

import numpy as np

from majsoul_eye.annotate import pipeline as P
from majsoul_eye.normalize import BoardRegion
from majsoul_eye.state.observe import ObservedRiverTile

CANON_W, CANON_H = 1920, 1080


def _fw_points(det, region: BoardRegion, H_full) -> np.ndarray:
    """Detection corners (poly if OBB else xyxy box) -> fullwarp, via canonical px."""
    if det.poly:
        pts = np.float32(det.poly)
    else:
        x0, y0, x1, y1 = det.xyxy
        pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    nb = [region.px_to_norm_box(float(x), float(y), float(x), float(y)) for x, y in pts]
    canon = np.float32([[b.x0 * CANON_W, b.y0 * CANON_H] for b in nb])
    return P.original_to_fullwarp(canon, H_full)


def _river_frame(seat: int):
    g = P.DISCARD_GRID[seat]
    rd = P.DISCARD_READ[seat]
    o = np.array(g["o"], float)
    dcol = np.array(g["dcol"], float)
    drow = np.array(g["drow"], float)
    disc0 = o + (P.DISCARD_COLS - 1) * dcol if rd["disc0_at_col5"] else o.copy()
    colv = rd["colsign"] * dcol
    colu = colv / np.linalg.norm(colv)
    rowu = rd["rowsign"] * drow / np.linalg.norm(drow)
    return disc0, colu, rowu, float(np.linalg.norm(dcol))


def _assign_river(seat: int, items):
    """items = [(det, corners_fw)] -> (ordered ObservedRiverTiles, violations).

    Row = nearest DISCARD_ROW_OFFSETS entry; order within a row = along-column
    projection (handles the riichi extra-shift and the >18 overflow, since only
    ORDER matters). Sideways = footprint longer along the column axis."""
    disc0, colu, rowu, col_pitch = _river_frame(seat)
    offs = P.DISCARD_ROW_OFFSETS[seat]
    row_pitch = offs[1] - offs[0]
    rows: dict[int, list] = {0: [], 1: [], 2: []}
    viol: list[str] = []
    for det, pts in items:
        c = pts.mean(axis=0)
        v = float(np.dot(c - disc0, rowu))
        r = int(np.argmin([abs(v - x) for x in offs]))
        if abs(v - offs[r]) > 0.5 * row_pitch:
            viol.append(f"seat{seat} river det off-grid (row residual {v - offs[r]:.0f}px)")
            continue
        u = float(np.dot(c - disc0, colu))
        ext_col = float(np.ptp(pts @ colu))
        ext_row = float(np.ptp(pts @ rowu))
        rows[r].append((u, ObservedRiverTile(det.tile, sideways=ext_col > ext_row)))
    out: list[ObservedRiverTile] = []
    for r in (0, 1, 2):
        rows[r].sort(key=lambda t: t[0])
        if rows[r] and r > 0 and len(rows[r - 1]) != P.DISCARD_COLS:
            viol.append(f"seat{seat} river row{r} occupied but row{r-1} not full")
        if r < 2 and len(rows[r]) > P.DISCARD_COLS:
            viol.append(f"seat{seat} river row{r} has {len(rows[r])}>6 tiles")
        out.extend(t for _, t in rows[r])
    return out, viol
