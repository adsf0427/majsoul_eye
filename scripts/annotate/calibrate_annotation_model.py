"""Measure & refit the fullwarp annotation geometry against many real frames.

The generator in ``mahjong_relative_annotation_pipeline`` places boxes from
static constants (DISCARD_GRID / MELD_STRIP2) calibrated on a handful of AB
case frames. This script measures those constants against MANY frames across
MANY games using the skirt-free image features (see pipeline §9d):

  * lateral CREVICES between adjacent tile faces  -> column pitch + origin,
  * far-side face EDGE transitions                -> row origin + pitch,
  * end-cell widths / sideways-cell widths        -> true face w / d,
  * rigid meld-strip snap offsets                 -> corner + per-round float.

Everything is recorded as (predicted, detected) coordinate pairs, pooled per
seat/axis across all frames (the camera is fixed), then fit with a robust
linear map det = a*pred + b. --refit prints suggested new constants.

The newest discard of the last actor is EXCLUDED (GT leads the render by ~1
action; that slot may still be empty felt).

Run (conda `auto` env, repo root):
  PYTHONPATH=. $PY scripts/annotate/calibrate_annotation_model.py --per-game 40 --out scratchpad/calib.json
  PYTHONPATH=. $PY scripts/annotate/calibrate_annotation_model.py --refit scratchpad/calib.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from majsoul_eye.annotate import pipeline as P
from majsoul_eye.annotate.seatgt import seat_gt, SEAT_POS
from majsoul_eye import paths
from majsoul_eye.capture.gtframes import build_seq_state, load_frames

HORIZ = {0: True, 1: False, 2: True, 3: False}   # river rows run along x?


def sample_seqs(seq_state: dict, frames: dict, k: int) -> list[int]:
    seqs = [s for s in sorted(seq_state) if s in frames]
    if len(seqs) <= k:
        return seqs
    idx = np.linspace(0, len(seqs) - 1, k).astype(int)
    return [seqs[i] for i in sorted(set(idx))]


# --------------------------------------------------------------------------- #
# river structured measurement
# --------------------------------------------------------------------------- #

def _rects(slots) -> list[tuple]:
    out = []
    for s in slots:
        p = np.float32(s["poly_fullwarp"])
        out.append((float(p[:, 0].min()), float(p[:, 1].min()),
                    float(p[:, 0].max()), float(p[:, 1].max())))
    return out


def measure_river(mask, slots, pos: int, skip_last: bool) -> dict:
    """Structured (pred, det) pairs for one seat's river in one frame."""
    res = {"lat": [], "vert": [], "w": [], "d": [], "side_dtop": []}
    if not slots:
        return res
    horiz = HORIZ[pos]
    rects = _rects(slots)
    suspect = len(slots) - 1 if skip_last else -1

    # group into display rows by the cross coordinate
    cross_of = (lambda r: (r[1] + r[3]) / 2) if horiz else (lambda r: (r[0] + r[2]) / 2)
    along_lo = (lambda r: r[0]) if horiz else (lambda r: r[1])
    rows: dict[int, list[int]] = defaultdict(list)
    pitch = abs(P.DISCARD_GRID[pos]["drow"][1 if horiz else 0])
    base = min(cross_of(r) for r in rects)
    for i, r in enumerate(rects):
        rows[int(round((cross_of(r) - base) / max(1.0, pitch)))].append(i)

    for _, idxs in sorted(rows.items()):
        idxs = sorted(idxs, key=lambda i: along_lo(rects[i]))
        # lateral crevices between adjacent cells
        for a, b in zip(idxs[:-1], idxs[1:]):
            if a == suspect or b == suspect:
                continue
            ra, rb = rects[a], rects[b]
            if horiz:
                pred = (ra[2] + rb[0]) / 2
                span = (max(ra[1], rb[1]) + 4, min(ra[3], rb[3]) - 4)
                pos_d, c = P.find_crevice(mask, span, pred, "x")
            else:
                pred = (ra[3] + rb[1]) / 2
                span = (max(ra[0], rb[0]) + 4, min(ra[2], rb[2]) - 4)
                pos_d, c = P.find_crevice(mask, span, pred, "y")
            if c >= P.MIN_CREVICE_CONTRAST:
                res["lat"].append((pred, pos_d))
        # outer lateral edges of the row. For vertical rivers the south end is the
        # skirt side (columns run along the camera axis) — measure the north end only.
        ends = ((idxs[0], +1), (idxs[-1], -1)) if horiz else ((idxs[0], +1),)
        for endi, inside in ends:
            if endi == suspect:
                continue
            r = rects[endi]
            if horiz:
                pred = r[0] if inside > 0 else r[2]
                pos_d, g = P.find_edge(mask, (r[1] + 4, r[3] - 4), pred, "x", inside)
            else:
                pred = r[1] if inside > 0 else r[3]
                pos_d, g = P.find_edge(mask, (r[0] + 4, r[2] - 4), pred, "y", inside)
            if g >= P.MIN_EDGE_GRAD:
                res["lat"].append((pred, pos_d))
                # end-cell width: outer edge + inner crevice -> face size along row
                if len(idxs) >= 2:
                    j = idxs[1] if inside > 0 else idxs[-2]
                    ra, rb = (rects[endi], rects[j]) if inside > 0 else (rects[j], rects[endi])
                    predc = (ra[2] + rb[0]) / 2 if horiz else (ra[3] + rb[1]) / 2
                    spanc = ((max(ra[1], rb[1]) + 4, min(ra[3], rb[3]) - 4) if horiz
                             else (max(ra[0], rb[0]) + 4, min(ra[2], rb[2]) - 4))
                    posc, cc = P.find_crevice(mask, spanc, predc, "x" if horiz else "y")
                    if cc >= P.MIN_CREVICE_CONTRAST:
                        wmeas = (posc - pos_d) if inside > 0 else (pos_d - posc)
                        slot = slots[endi]
                        (res["d"] if slot.get("riichi") else res["w"]).append(wmeas)
        # row far-side edge (top in fullwarp for horizontal; clean x-side for vertical)
        clean = [i for i in idxs if i != suspect and not slots[i].get("riichi")]
        if clean:
            r0 = [rects[i] for i in clean]
            if horiz:
                pred = float(np.median([r[1] for r in r0]))
                span = (min(r[0] for r in r0) + 6, max(r[2] for r in r0) - 6)
                pos_d, g = P.find_edge(mask, span, pred, "y", +1)
            else:
                cx = float(np.mean([(r[0] + r[2]) / 2 for r in r0]))
                east = cx > P.NADIR_X
                pred = float(np.median([r[2] if east else r[0] for r in r0]))
                span = (min(r[1] for r in r0) + 6, max(r[3] for r in r0) - 6)
                pos_d, g = P.find_edge(mask, span, pred, "x", -1 if east else +1)
            if g >= P.MIN_EDGE_GRAD:
                res["vert"].append((pred, pos_d))
        # sideways tile far-side edge relative to the row's (cross alignment check)
        for i in idxs:
            if slots[i].get("riichi") and i != suspect:
                r = rects[i]
                if horiz:
                    pos_d, g = P.find_edge(mask, (r[0] + 4, r[2] - 4), r[1], "y", +1)
                    if g >= P.MIN_EDGE_GRAD:
                        res["side_dtop"].append(pos_d - r[1])
                else:
                    cx = (r[0] + r[2]) / 2
                    east = cx > P.NADIR_X
                    pred = r[2] if east else r[0]
                    pos_d, g = P.find_edge(mask, (r[1] + 4, r[3] - 4), pred, "x", -1 if east else +1)
                    if g >= P.MIN_EDGE_GRAD:
                        res["side_dtop"].append((pos_d - pred) * (-1 if east else 1))
    return res


# --------------------------------------------------------------------------- #
# per-row chain measurement (rows are NOT exactly equally spaced)
# --------------------------------------------------------------------------- #

def measure_rows(mask, slots, pos: int, skip_last: bool) -> list[tuple]:
    """Measure each present row's anti-skirt anchor edge via an edge/crevice
    chain. Returns [(row_index_0based, delta_along_rowdir_px), ...]: delta>0 =
    the real row lies further along the row-advance direction than predicted."""
    if not slots:
        return []
    horiz = HORIZ[pos]
    rects = _rects(slots)
    suspect = len(slots) - 1 if skip_last else -1
    rows: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(slots):
        if i == suspect or s.get("riichi"):
            continue
        rows[min(int(s["row"]) - 1, 2)].append(i)
    if not rows:
        return []
    drow = np.array(P.DISCARD_GRID[pos]["drow"], float)
    rowdir = drow / (np.linalg.norm(drow) + 1e-9)
    if horiz:
        # anchor = north (min-y) edge; the northmost row has the exterior edge
        anchor = lambda r: r[1]
        span_of = lambda rr_: (min(x[0] for x in rr_) + 6, max(x[2] for x in rr_) - 6)
        axis, ext_sign, adv = "y", +1, rowdir[1]
        extreme_max = False
    else:
        east = float(np.mean([r[0] for r in rects])) > P.NADIR_X
        anchor = (lambda r: r[2]) if east else (lambda r: r[0])
        span_of = lambda rr_: (min(x[1] for x in rr_) + 6, max(x[3] for x in rr_) - 6)
        axis, ext_sign, adv = "x", (-1 if east else +1), rowdir[0]
        extreme_max = east
    keyed = sorted(rows.items(),
                   key=lambda kv: float(np.median([anchor(rects[i]) for i in kv[1]])),
                   reverse=extreme_max)
    out = []
    for k, (ridx, idxs) in enumerate(keyed):
        rr_ = [rects[i] for i in idxs]
        pred = float(np.median([anchor(r) for r in rr_]))
        span = span_of(rr_)
        if span[1] - span[0] < 30:
            continue
        if k == 0:                                  # exterior felt->face edge
            det, q = P.find_edge(mask, span, pred, axis, ext_sign, r=18)
            if q < P.MIN_EDGE_GRAD:
                continue
        else:                                       # crevice vs the previous row's skirt
            det, q = P.find_crevice(mask, span, pred - ext_sign * 2.0, axis, r=14)
            if q < P.MIN_CREVICE_CONTRAST:
                continue
            det = det + ext_sign * 1.5
        out.append((ridx, float((det - pred) * adv)))
    return out


# --------------------------------------------------------------------------- #
# per-frame measurement
# --------------------------------------------------------------------------- #

def measure_frame(img: np.ndarray, state, hom: dict) -> list[dict]:
    Hinv = hom["H_full_inv"]
    full = P.warp_to_full(img, hom["H_full"], hom["full_size"])
    mw = P.tile_face_mask(full)
    mb = P.tile_back_mask(full)
    out = []
    kyoku = f"{state.bakaze}{state.kyoku}.{getattr(state, 'honba', 0)}"

    for pos in range(4):
        river, sideways_idx, melds, seat = seat_gt(state, pos)
        if len(river) >= 4:
            slots = P.generate_discard_slots(pos, river, Hinv, sideways_idx=sideways_idx)
            r = measure_river(mw, slots, pos, skip_last=(state.last_actor == seat))
            r["rows"] = measure_rows(mw, slots, pos, skip_last=(state.last_actor == seat))
            if r["lat"] or r["vert"] or r["rows"]:
                out.append({"kind": "river", "pos": pos, "kyoku": kyoku, **r})
        if melds:
            mboxes = P.generate_meld_boxes_v2(pos, melds, Hinv)
            da, dc, diag = P.snap_meld_strip(mw, mb, mboxes, pos)
            out.append({"kind": "meld", "pos": pos, "kyoku": kyoku,
                        "n_melds": len(melds), "types": ",".join(m["type"] for m in melds),
                        "d_along": da, "d_cross": dc, **diag})
    return out


# --------------------------------------------------------------------------- #
# aggregate / refit
# --------------------------------------------------------------------------- #

def _robust_linfit(pairs: list) -> tuple[float, float, float]:
    """det = a*pred + b with one outlier-rejection pass. Returns (a, b, rmse)."""
    a = np.asarray(pairs, float)
    if len(a) < 8:
        return 1.0, float(np.median(a[:, 1] - a[:, 0])) if len(a) else 0.0, 0.0
    x, y = a[:, 0], a[:, 1]
    for _ in range(2):
        A = np.polyfit(x, y, 1)
        resid = y - np.polyval(A, x)
        keep = np.abs(resid - np.median(resid)) < 4.0
        if keep.sum() < 8 or keep.all():
            break
        x, y = x[keep], y[keep]
    A = np.polyfit(x, y, 1)
    rmse = float(np.sqrt(np.mean((y - np.polyval(A, x)) ** 2)))
    return float(A[0]), float(A[1]), rmse


def refit(measure_path: str) -> None:
    with open(measure_path, encoding="utf-8") as f:
        data = json.load(f)
    riv = defaultdict(lambda: {"lat": [], "vert": [], "w": [], "d": [], "side_dtop": []})
    rowm = defaultdict(lambda: defaultdict(list))
    melds = defaultdict(list)
    meld_rounds = defaultdict(list)
    for rec in data["records"]:
        if rec["kind"] == "river":
            for k in ("lat", "vert", "w", "d", "side_dtop"):
                riv[rec["pos"]][k] += rec.get(k, [])
            for ridx, delta in rec.get("rows", []):
                rowm[rec["pos"]][ridx].append(delta)
        else:
            melds[rec["pos"]].append((rec["d_along"], rec["d_cross"]))
            meld_rounds[(rec["capture"], rec["kyoku"], rec["pos"])].append(
                (rec["d_along"], rec["d_cross"]))

    print("=== RIVER structured fit (det = a*pred + b, per seat/axis) ===")
    new_grid = {}
    for pos in sorted(riv):
        m = riv[pos]
        al, bl, el = _robust_linfit(m["lat"])
        av, bv, ev = _robust_linfit(m["vert"])
        w = float(np.median(m["w"])) if m["w"] else None
        d = float(np.median(m["d"])) if m["d"] else None
        sdt = float(np.median(m["side_dtop"])) if m["side_dtop"] else None
        print(f"  pos{pos} {SEAT_POS[pos]:7} lat n={len(m['lat']):5} a={al:.4f} b={bl:+7.2f} rmse={el:.2f} | "
              f"vert n={len(m['vert']):4} a={av:.4f} b={bv:+7.2f} rmse={ev:.2f}")
        print(f"        face w={w and round(w,1)} (n={len(m['w'])})  d={d and round(d,1)} (n={len(m['d'])})  "
              f"sideways far-edge Δ={sdt and round(sdt,1)} (n={len(m['side_dtop'])})")
        g = P.DISCARD_GRID[pos]
        o = np.array(g["o"], float); dcol = np.array(g["dcol"], float); drow = np.array(g["drow"], float)
        horiz = HORIZ[pos]
        fw, fh = P.DISCARD_FOOT[pos]
        # vert pairs are far-side face EDGES, not centers: reconstruct the center
        # from the fitted true edge + the newly measured face depth.
        if horiz:
            d_use = d or fh
            top_true = av * (o[1] - fh / 2) + bv
            o2 = (al * o[0] + bl, top_true + d_use / 2)
            dcol2 = (al * dcol[0], dcol[1]); drow2 = (drow[0], av * drow[1])
        else:
            d_use = d or fw
            east = o[0] > P.NADIR_X                 # right river: clean edge = east
            edge_pred = o[0] + fw / 2 if east else o[0] - fw / 2
            edge_true = av * edge_pred + bv
            o2 = (edge_true - d_use / 2 if east else edge_true + d_use / 2,
                  al * o[1] + bl)
            dcol2 = (dcol[0], al * dcol[1]); drow2 = (av * drow[0], drow[1])
        new_grid[pos] = {"o": tuple(round(v, 1) for v in o2),
                         "dcol": tuple(round(v, 2) for v in dcol2),
                         "drow": tuple(round(v, 2) for v in drow2),
                         "w": w and round(w, 1), "d": d and round(d, 1)}
    print("--- suggested DISCARD_GRID ---")
    for pos, g in new_grid.items():
        print(f'  {pos}: {{"o": {g["o"]}, "dcol": {g["dcol"]}, "drow": {g["drow"]}}},  # face w={g["w"]} d={g["d"]}')

    if rowm:
        print("=== per-row chain deltas (px along row-advance; suggested DISCARD_ROW_OFFSETS) ===")
        for pos in sorted(rowm):
            meds = {}
            for ridx, vals in sorted(rowm[pos].items()):
                a = np.array(vals, float)
                meds[ridx] = float(np.median(a))
                print(f"  pos{pos} row{ridx + 1}: n={len(a):4} med {np.median(a):+6.1f} ±{a.std():5.2f}")
            base = meds.get(0, 0.0)
            pitch = float(np.linalg.norm(P.DISCARD_GRID[pos]["drow"]))
            cur = P.DISCARD_ROW_OFFSETS.get(pos) or [k * pitch for k in range(3)]
            sug = [round(cur[k] + meds.get(k, 0.0) - base, 1) if (k in meds or k == 0) else None
                   for k in range(3)]
            print(f"    -> DISCARD_ROW_OFFSETS[{pos}] = {sug}   (fold row1 med {base:+.1f} into o)")

    print("=== MELD strip rigid-snap offsets (along + = inward) ===")
    for pos in sorted(melds):
        a = np.array(melds[pos], float)
        print(f"  pos{pos} {SEAT_POS[pos]:7} n={len(a):4}  along med {np.median(a[:,0]):+6.1f} ±{a[:,0].std():5.2f}  "
              f"cross med {np.median(a[:,1]):+6.1f} ±{a[:,1].std():5.2f}")
        cfg = P.MELD_STRIP2[pos]
        c = np.array(cfg["corner"], float)
        nc = c + np.array(cfg["along"]) * np.median(a[:, 0]) + np.array(cfg["cross"]) * np.median(a[:, 1])
        print(f"        suggested corner: ({nc[0]:.1f}, {nc[1]:.1f})")
    within, across = [], []
    for _, vals in meld_rounds.items():
        v = np.array(vals, float)
        if len(v) >= 2:
            within.append(v[:, 0].std())
        across.append(v[:, 0].mean())
    if across:
        print(f"=== MELD along float: within-round σ≈{np.mean(within) if within else 0:.2f}px, "
              f"across-round σ≈{np.std(across):.2f}px (n_rounds={len(across)}) ===")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", nargs="*", default=None,
                    help="capture jsonl files (default: all converted GT under captures/intermediate/gt/)")
    ap.add_argument("--per-game", type=int, default=40)
    ap.add_argument("--out", default="scratchpad/calib_measure.json")
    ap.add_argument("--refit", default=None, help="aggregate an existing measurement JSON")
    args = ap.parse_args()

    if args.refit:
        refit(args.refit)
        return

    captures = args.captures or paths.converted_gt_captures()
    hom = P.build_homographies(1920, 1080)
    records = []
    for cap in captures:
        try:
            seq_state = build_seq_state(cap)
            frames = load_frames(paths.frames_dir_for(cap))
        except Exception as e:
            print(f"  {cap}: SKIP ({e})")
            continue
        seqs = sample_seqs(seq_state, frames, args.per_game)
        n = 0
        for seq in seqs:
            img = cv2.imread(frames[seq])
            if img is None:
                continue
            if img.shape[1] != 1920:
                img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
            for rec in measure_frame(img, seq_state[seq], hom):
                rec["capture"] = os.path.basename(cap)
                rec["seq"] = seq
                records.append(rec)
            n += 1
        print(f"  {cap}: measured {n} frames ({len(records)} records total)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"records": records}, f, ensure_ascii=False)
    print(f"wrote {args.out} ({len(records)} records)")


if __name__ == "__main__":
    main()
