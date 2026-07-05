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

    Whole-strip recursive parse trying size 4 then 3 at each position. A strip
    with more than one forward-consistent decomposition is REJECTED (violation)
    rather than guessed at: with legal tile counts such ambiguity cannot arise
    (a same-kind 3+4 collision needs >4 copies, which check_observed also
    rejects), so it only fires on detector noise — prefer no parse to a wrong
    one."""
    cells = _strip_cells(seat, items)

    def parse(i: int):
        """Complete parses from cell i, capped at 2."""
        if i == len(cells):
            return [[]]
        out = []
        for size in (4, 3):
            if i + size <= len(cells):
                got = _match_group(seat, cells[i:i + size])
                if got is not None:
                    for rest in parse(i + size):
                        out.append([got] + rest)
                        if len(out) >= 2:
                            return out
        return out

    parses = parse(0)
    if not parses:
        return [], [f"seat{seat} meld strip unparsable ({len(cells)} cells)"]
    if len(parses) > 1:
        return [], [f"seat{seat} meld strip ambiguous ({len(cells)} cells)"]
    return parses[0], []


from majsoul_eye.coords import DORA_STRIP, HAND
from majsoul_eye.state.observe import ObservedState, check_observed

HAND_MIN_H = 0.11            # hand tiles are ~0.141 canon-high; hero meld tiles ~0.083


def assemble(dets, region: BoardRegion) -> ObservedState:
    """One frame's detections -> ObservedState. HUD fields stay None (their
    readers are the 2026-07-04 spec). 'back' detections only ever route to MELD
    zones (ankan renders back/face/face/back); opponents' concealed rows sit off
    the felt plane, land outside every calibrated zone after the homography and
    are dropped silently (concealed_counts stays None — cross-check only)."""
    o = ObservedState()
    Hs = P.build_homographies(CANON_W, CANON_H)
    hand_cand, dora_cand, table = [], [], []
    conf: dict[str, list] = {}

    def note(zone, det):
        conf.setdefault(zone, []).append(det.score)

    for det in dets:
        x0, y0, x1, y1 = det.xyxy
        nb = region.px_to_norm_box(x0, y0, x1, y1)
        if DORA_STRIP.x0 <= nb.cx <= DORA_STRIP.x1 and \
                DORA_STRIP.y0 <= nb.cy <= DORA_STRIP.y1:
            if det.tile != "back":
                dora_cand.append((nb.x0, det))
                note("dora", det)
            continue
        if det.tile != "back" and nb.h >= HAND_MIN_H and nb.cy >= HAND.y0 - 0.02:
            hand_cand.append((nb.x0, nb, det))
            note("hand", det)
            continue
        table.append((det, _fw_points(det, region, Hs["H_full"])))

    # hand + drawn (gap of >= ~half a slot before the last tile)
    hand_cand.sort(key=lambda t: t[0])
    o.hero_hand = [d.tile for _, _, d in hand_cand]
    if len(hand_cand) >= 2:
        gap = hand_cand[-1][0] - hand_cand[-2][0]
        if gap > HAND.slot_w + 0.5 * HAND.tsumo_gap:
            o.drawn_tile = o.hero_hand.pop()
    o.dora_markers = [d.tile for _, d in sorted(dora_cand, key=lambda t: t[0])]

    # route table detections to the nearest seat zone (river vs meld)
    per_river: list[list] = [[] for _ in range(4)]
    per_meld: list[list] = [[] for _ in range(4)]
    for det, pts in table:
        c = pts.mean(axis=0)
        best = None                                    # (dist, kind, seat)
        for seat in range(4):
            disc0, colu, rowu, col_pitch = _river_frame(seat)
            offs = P.DISCARD_ROW_OFFSETS[seat]
            u = float(np.dot(c - disc0, colu))
            v = float(np.dot(c - disc0, rowu))
            du = max(0.0, -u, u - 10 * col_pitch)
            dv = min(abs(v - x) for x in offs)
            d_river = float(np.hypot(du, dv))
            cfg = P.MELD_STRIP2[seat]
            a = float(np.dot(c - np.array(cfg["corner"]), np.array(cfg["along"])))
            cr = float(np.dot(c - np.array(cfg["corner"]), np.array(cfg["cross"])))
            da = max(0.0, -a, a - 16 * cfg["w"])
            dc = max(0.0, -cr, cr - 2.2 * cfg["d"])
            d_meld = float(np.hypot(da, dc))
            kinds = (("meld", d_meld),) if det.tile == "back" else \
                    (("river", d_river), ("meld", d_meld))
            for kind, dist in kinds:
                if best is None or dist < best[0]:
                    best = (dist, kind, seat)
        if best[0] > 60.0:
            if det.tile != "back":     # opponents' concealed rows are expected strays
                o.violations.append(
                    f"stray detection {det.tile} ({best[0]:.0f}px off-zone)")
            continue
        (per_river if best[1] == "river" else per_meld)[best[2]].append((det, pts))
        note(f"{best[1]}{best[2]}", det)

    for seat in range(4):
        o.rivers[seat], v1 = _assign_river(seat, per_river[seat])
        melds, v2 = _parse_melds(seat, per_meld[seat])
        o.melds[seat] = melds
        o.violations.extend(v1 + v2)
        o.reach[seat] = any(t.sideways for t in o.rivers[seat])
    o.zone_confidence = {z: min(s) for z, s in conf.items()}
    o.violations.extend(check_observed(o))
    return o
