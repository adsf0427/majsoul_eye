"""Detections -> ObservedState (spec 2026-07-05 §3.2).

Runs the calibrated annotate/pipeline geometry BACKWARD: detection centers are
mapped original->canonical(1920x1080)->fullwarp, then matched to the discard
grid / meld strip. Akagi-free (annotate.pipeline is pure geometry; capture/ is
never imported)."""
from __future__ import annotations

import numpy as np

from majsoul_eye.annotate import pipeline as P
from majsoul_eye.normalize import BoardRegion
from majsoul_eye.state.observe import ObservedMeld, ObservedRiverTile
from majsoul_eye.tiles import red_to_normal

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


def _strip_cells(seat: int, items):
    """Sort meld-zone detections into display cells walking from the corner.
    Returns cells = [{label, sideways, stacked: [label]}] in CORNER order."""
    cfg = P.MELD_STRIP2[seat]
    corner = np.array(cfg["corner"], float)
    along = np.array(cfg["along"], float)
    cross = np.array(cfg["cross"], float)
    d = cfg["d"]
    raw = []
    for det, pts in items:
        c = pts.mean(axis=0)
        a = float(np.dot(c - corner, along))
        cr = float(np.dot(c - corner, cross))
        ext_a = float(np.ptp(pts @ along))
        ext_c = float(np.ptp(pts @ cross))
        raw.append({"a": a, "c": cr, "label": det.tile,
                    "sideways": ext_a > ext_c, "stacked_on": None})
    raw.sort(key=lambda x: x["a"])
    cells, i = [], 0
    while i < len(raw):
        cell = {"label": raw[i]["label"], "sideways": raw[i]["sideways"], "stacked": []}
        j = i + 1
        # a kakan's added tile shares the same along-slot, offset across (c ~ d..2d)
        while j < len(raw) and abs(raw[j]["a"] - raw[i]["a"]) < 0.4 * cfg["w"]:
            top = raw[j] if raw[j]["c"] > raw[i]["c"] else raw[i]
            base = raw[i] if top is raw[j] else raw[j]
            cell = {"label": base["label"], "sideways": True, "stacked": [top["label"]]}
            j += 1
        cells.append(cell)
        i = j
    return cells


def _hypotheses(group):
    """Candidate (type, tiles, called, added, from_rel) for a cell group."""
    labels = [c["label"] for c in group]
    out = []
    if len(group) == 4 and labels.count("back") == 2:
        face = next(l for l in labels if l != "back")
        tiles = [face] * 4
        base = red_to_normal(face)
        if base[0] == "5":                     # 4 copies of a five must include the red
            tiles = [base + "r"] + [base] * 3
        out.append(("ankan", tiles, "", "", 0))
        return out
    stacked = next((c for c in group if c["stacked"]), None)
    if stacked is not None:
        called, added = stacked["label"], stacked["stacked"][0]
        tiles = [c["label"] for c in group] + [added]
        for rel in (1, 2, 3):
            out.append(("kakan", tiles, called, added, rel))
        return out
    side = [c for c in group if c["sideways"]]
    if len(side) != 1:
        return []
    called = side[0]["label"]
    norm = [red_to_normal(l) for l in labels]
    if len(group) == 4:
        if len(set(norm)) == 1:
            for rel in (1, 2, 3):
                out.append(("daiminkan", labels, called, "", rel))
    elif len(group) == 3:
        if len(set(norm)) == 1:
            for rel in (1, 2, 3):
                out.append(("pon", labels, called, "", rel))
        elif all(len(x) == 2 and x[0].isdigit() for x in norm) \
                and len({x[1] for x in norm}) == 1:        # one suit only
            ranks = sorted(int(x[0]) for x in norm)
            if ranks[1] - ranks[0] == 1 and ranks[2] - ranks[1] == 1:
                out.append(("chi", labels, called, "", 3))
    return out


def _match_group(seat, group):
    """Find the hypothesis whose FORWARD rendering equals the observed cells."""
    obs_cells = [(c["label"], c["sideways"], tuple(c["stacked"])) for c in group]
    for type_, tiles, called, added, rel in _hypotheses(group):
        m = {"type": type_, "tiles": sorted(tiles), "from_seat": (seat + rel) % 4,
             "called_pai": called, "added_pai": added}
        cells = P.meld_display_cells(m, seat)
        if P.MELD_WITHIN_REVERSED:
            cells = list(reversed(cells))
        want = [(c["label"], bool(c["sideways"]), tuple(c.get("stacked", [])))
                for c in cells]
        if want == obs_cells:
            return ObservedMeld(type_, sorted(tiles), called, added, rel)
    return None


def _parse_melds(seat: int, items):
    """Meld-zone detections -> (melds in SCREEN order oldest-first, violations).

    Whole-strip recursive parse: a group size that matches locally but leaves an
    unparsable tail is backtracked (e.g. pon KKK followed by a meld starting
    with K would otherwise be swallowed as a fake daiminkan)."""
    cells = _strip_cells(seat, items)

    def parse(i: int):
        if i == len(cells):
            return []
        for size in (4, 3):
            if i + size <= len(cells):
                got = _match_group(seat, cells[i:i + size])
                if got is not None:
                    rest = parse(i + size)
                    if rest is not None:
                        return [got] + rest
        return None

    melds = parse(0)
    if melds is None:
        return [], [f"seat{seat} meld strip unparsable ({len(cells)} cells)"]
    return melds, []
