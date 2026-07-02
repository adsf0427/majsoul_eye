"""Batch-annotate every captured frame of the MahjongCopilot AI games.

For each (frame, GT state) pair this emits precise tile boxes for:
  * all 4 discard rivers  — calibrated fullwarp face grid (pipeline §9b/§9d),
    riichi-sideways + riichi-claimed edge case + 4th-row overflow;
  * all 4 meld areas      — composition-aware strip (v2) + a rigid per-frame
    image snap (the strip floats a few px per round);
  * the hero hand         — existing calibrated HandModel (majsoul_eye.coords).

Every box carries an image-evidence confidence (`fill` = mask coverage of the
face box). The newest discard is often not rendered yet (GT leads the client by
~1 action): such slots are flagged ``unrendered`` instead of being emitted as
trustworthy boxes. Low-fill boxes elsewhere are flagged ``low_conf``.

Output (default out/ai_session_annotations/):
  <game>.jsonl      one JSON record per frame (all boxes, fullwarp + original px)
  summary.json      per-game + global QA stats
  overlays/         every Nth frame rendered with its boxes (spot checking)

Optional --qa-classifier runs the 97.6% tile classifier over sampled face crops
and reports agreement with the GT labels per zone (end-to-end precision proxy).

Run (conda `auto` env, repo root):
  PYTHONPATH=. $PY scripts/annotate_ai_session.py                    # all captures/intermediate/gt/*
  PYTHONPATH=. $PY scripts/annotate_ai_session.py \
      --captures captures/intermediate/gt/ai_run_3_game1.jsonl --overlay-every 40 --qa-classifier

By default the frames dir is derived from the capture name (X.jsonl -> X/, via
majsoul_eye.paths.frames_dir_for). Pass --frames-dir to point one capture at a
different frames folder (e.g. de-letterboxed frames from deletterbox_frames.py)
while keeping the GT + output name of the original capture:
  PYTHONPATH=. $PY scripts/annotate_ai_session.py --captures captures/intermediate/gt/ai_run_5_game2.jsonl \
      --frames-dir captures/intermediate/derived/ai_run_5_game2_fixed --qa-classifier
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np

import mahjong_relative_annotation_pipeline as P
from majsoul_eye import paths
from majsoul_eye.label.autolabel import label_frame
from majsoul_eye.normalize import locate_fullscreen
from scripts.calibrate_annotation_model import seat_gt
from scripts.spike_topdown import build_seq_state, load_frames, _frames_dir_for

FILL_OK = 0.25          # face-mask coverage below this = not rendered / occluded
SNAP_MAX_ALONG = 70.0   # clamp for the rigid meld snap (the strip floats per round)
SNAP_MAX_CROSS = 70.0   # the SELF strip also floats vertically (up to ~1/2 tile)


def _fill(ii: np.ndarray, poly) -> float:
    p = np.float32(poly)
    return P._box_fill(ii, p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())


def annotate_frame(img: np.ndarray, state, hom: dict, hand_suspect: bool = False) -> dict:
    """Full annotation record for one frame. `hand_suspect` marks frames right
    after a kyoku start, where the deal/sort animation may not match GT order."""
    Hinv = hom["H_full_inv"]
    full = P.warp_to_full(img, hom["H_full"], hom["full_size"])
    mw = P.tile_face_mask(full)
    mb = P.tile_back_mask(full)
    ii_w = cv2.integral(mw)
    ii_b = cv2.integral(mb)

    rec = {"hero_seat": state.hero_seat,
           "kyoku": f"{state.bakaze}{state.kyoku}",
           "discard_slots": {}, "meld_boxes": {}, "hand_boxes": [], "flags": []}

    for pos in range(4):
        river, sideways_idx, melds, seat = seat_gt(state, pos)
        slots = P.generate_discard_slots(pos, river, Hinv, sideways_idx=sideways_idx)
        newest = len(slots) - 1 if state.last_actor == seat else -1
        for i, s in enumerate(slots):
            f = _fill(ii_w, s["face_poly_fullwarp"])
            s["fill"] = round(f, 3)
            if f < FILL_OK:
                s["reliable"] = False
                if i == newest:
                    s["unrendered"] = True      # GT leads the render by ~1 action
                    rec["flags"].append(f"pos{pos}:river[{i}]:unrendered")
                else:
                    s["low_conf"] = True
                    rec["flags"].append(f"pos{pos}:river[{i}]:low_fill={f:.2f}")
        rec["discard_slots"][str(pos)] = slots

        boxes = P.generate_meld_boxes_v2(pos, melds, Hinv)
        if boxes:
            da, dc, diag = P.snap_meld_strip(mw, mb, boxes, pos)
            if diag["n_along"] + diag["n_cross"] >= 2:
                da = float(np.clip(da, -SNAP_MAX_ALONG, SNAP_MAX_ALONG))
                dc = float(np.clip(dc, -SNAP_MAX_CROSS, SNAP_MAX_CROSS))
                boxes = P.shift_boxes(boxes, pos, da, dc, Hinv)
            for b in boxes:
                ii = ii_b if b["tile"] == "back" else ii_w
                f = _fill(ii, b["poly_fullwarp"])
                b["fill"] = round(f, 3)
                b["snap"] = (round(da, 1), round(dc, 1))
                if f < FILL_OK:
                    b["reliable"] = False
                    b["low_conf"] = True
                    rec["flags"].append(f"pos{pos}:meld[{b['tile']}]:low_fill={f:.2f}")
        rec["meld_boxes"][str(pos)] = boxes

    # hero hand via the calibrated HandModel (settled 13-tile states only)
    try:
        region = locate_fullscreen(img)
        hb = []
        for s in label_frame(img, state, region, zones=frozenset({"hand"})):
            x1, y1, x2, y2 = s.px_box
            f = 0.0
            if x2 > x1 and y2 > y1:
                hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
                f = float(((hsv[..., 1] < 70) & (hsv[..., 2] > 165)).mean())
            hb.append({"tile": s.label, "px_box": list(s.px_box), "fill": round(f, 3)})
        if hb and float(np.median([h["fill"] for h in hb])) < 0.30:
            # deal/draw animation still playing — GT leads the render
            for h in hb:
                h["reliable"] = False
            rec["flags"].append("hand:unrendered")
        elif hb and hand_suspect:
            # kyoku just started: tiles may be rendered but not yet GT-sorted
            for h in hb:
                h["reliable"] = False
            rec["flags"].append("hand:deal_unsorted")
        rec["hand_boxes"] = hb
    except Exception as e:                       # hand layout is best-effort
        rec["flags"].append(f"hand:error:{e}")
    return rec


def render_overlay(img: np.ndarray, rec: dict, path: str) -> None:
    vis = img.copy()
    for pos_k, slots in rec["discard_slots"].items():
        for s in slots:
            col = (0, 255, 0) if s.get("reliable", True) else (
                (0, 165, 255) if s.get("unrendered") else (0, 0, 255))
            cv2.polylines(vis, [np.int32(s["face_poly_original"])], True, col, 2, cv2.LINE_AA)
    for pos_k, boxes in rec["meld_boxes"].items():
        for b in boxes:
            col = (255, 0, 255) if b.get("is_added_kan") else (
                (255, 200, 0) if b.get("reliable", True) else (0, 0, 255))
            cv2.polylines(vis, [np.int32(b["poly_original"])], True, col, 2, cv2.LINE_AA)
    for h in rec["hand_boxes"]:
        x1, y1, x2, y2 = h["px_box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.imwrite(path, vis)


def _crop_quad(img: np.ndarray, poly, size: int = 64) -> np.ndarray:
    src = np.float32(poly)
    dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (size, size))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", nargs="*", default=None,
                    help="capture jsonl files (default: all converted GT under captures/intermediate/gt/)")
    ap.add_argument("--frames-dir", default=None,
                    help="override the frames dir (must contain frames.jsonl); "
                         "only valid with exactly one --captures")
    ap.add_argument("--out", default="out/ai_session_annotations")
    ap.add_argument("--overlay-every", type=int, default=60)
    ap.add_argument("--qa-classifier", action="store_true",
                    help="classify sampled face crops and report GT agreement")
    ap.add_argument("--qa-per-game", type=int, default=400)
    args = ap.parse_args()

    captures = args.captures or paths.converted_gt_captures()
    if args.frames_dir and len(captures) != 1:
        ap.error("--frames-dir requires exactly one --captures")
    os.makedirs(os.path.join(args.out, "overlays"), exist_ok=True)
    hom = P.build_homographies(1920, 1080)

    clf = None
    if args.qa_classifier:
        from majsoul_eye.recognize.classifier import TileClassifier
        clf = TileClassifier("majsoul_eye/recognize/tile_classifier.pt")

    summary = {}
    for cap in captures:
        name = os.path.splitext(os.path.basename(cap))[0]
        try:
            seq_state = build_seq_state(cap)
            frames = load_frames(args.frames_dir or _frames_dir_for(cap))
        except Exception as e:
            print(f"{name}: SKIP ({e})")
            continue
        seqs = [s for s in sorted(seq_state) if s in frames]
        # first 2 board-changing seqs of each kyoku = deal animation window
        suspect_seqs = set()
        last_kyoku, fresh = None, 0
        for s in sorted(seq_state):
            st = seq_state[s]
            kyo = f"{st.bakaze}{st.kyoku}.{getattr(st, 'honba', 0)}"
            if kyo != last_kyoku:
                last_kyoku, fresh = kyo, 2
            if fresh > 0:
                suspect_seqs.add(s)
                fresh -= 1
        stats = defaultdict(float)
        qa = defaultdict(lambda: [0, 0])         # zone -> [n, correct]
        qa_bad = []
        out_path = os.path.join(args.out, f"{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for k, seq in enumerate(seqs):
                img = cv2.imread(frames[seq])
                if img is None:
                    continue
                if img.shape[1] != 1920:
                    img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
                rec = annotate_frame(img, seq_state[seq], hom,
                                     hand_suspect=(seq in suspect_seqs))
                rec["capture"] = name
                rec["seq"] = seq
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                stats["frames"] += 1
                for slots in rec["discard_slots"].values():
                    stats["river_boxes"] += len(slots)
                    stats["river_ok"] += sum(1 for s in slots if s.get("reliable", True))
                    stats["river_unrendered"] += sum(1 for s in slots if s.get("unrendered"))
                for boxes in rec["meld_boxes"].values():
                    stats["meld_boxes"] += len(boxes)
                    stats["meld_ok"] += sum(1 for b in boxes if b.get("reliable", True))
                stats["hand_boxes"] += len(rec["hand_boxes"])

                if args.overlay_every and k % args.overlay_every == 0:
                    render_overlay(img, rec, os.path.join(args.out, "overlays", f"{name}_seq{seq}.png"))

                if clf is not None and k % 3 == 0 and qa["river"][0] + qa["meld"][0] < args.qa_per_game:
                    # sideways cells are 90°-rotated vs the training crops: classify
                    # both rotations and accept either (QA-only leniency).
                    crops, keys = [], []
                    for slots in rec["discard_slots"].values():
                        for s in slots:
                            if s.get("reliable", True):
                                c = _crop_quad(img, s["face_poly_original"])
                                if s.get("riichi"):
                                    crops += [np.rot90(c).copy(), np.rot90(c, 3).copy()]
                                    keys.append(("river", s["tile"], 2))
                                else:
                                    crops.append(c)
                                    keys.append(("river", s["tile"], 1))
                    for boxes in rec["meld_boxes"].values():
                        for b in boxes:
                            if b.get("reliable", True):
                                c = _crop_quad(img, b["poly_original"])
                                if b.get("sideways"):
                                    crops += [np.rot90(c).copy(), np.rot90(c, 3).copy()]
                                    keys.append(("meld", b["tile"], 2))
                                else:
                                    crops.append(c)
                                    keys.append(("meld", b["tile"], 1))
                    for h in rec["hand_boxes"]:
                        x1, y1, x2, y2 = h["px_box"]
                        if h.get("reliable", True) and x2 > x1 and y2 > y1:
                            crops.append(cv2.resize(img[y1:y2, x1:x2], (64, 64)))
                            keys.append(("hand", h["tile"], 1))
                    if crops:
                        preds = clf.predict(crops)
                        pi = 0
                        for zone, gtl, ncand in keys:
                            cand = preds[pi:pi + ncand]
                            pi += ncand
                            qa[zone][0] += 1
                            qa[zone][1] += int(gtl in cand)
                            if gtl not in cand and len(qa_bad) < 60:
                                qa_bad.append(f"seq{seq} {zone} gt={gtl} pred={'/'.join(cand)}")

        s = {k: int(v) for k, v in stats.items()}
        if clf is not None:
            s["qa"] = {z: {"n": n, "agree": round(c / n, 4) if n else None}
                       for z, (n, c) in qa.items()}
            s["qa_mismatch_examples"] = qa_bad[:20]
        summary[name] = s
        riv_pct = 100 * s.get("river_ok", 0) / max(1, s.get("river_boxes", 1))
        meld_pct = 100 * s.get("meld_ok", 0) / max(1, s.get("meld_boxes", 1))
        qa_str = ""
        if clf is not None:
            qa_str = "  QA " + " ".join(f"{z}:{v['agree']}" for z, v in s["qa"].items() if v["n"])
        print(f"{name}: {s['frames']} frames  river {s.get('river_boxes',0)} boxes ({riv_pct:.1f}% ok, "
              f"{s.get('river_unrendered',0)} unrendered)  meld {s.get('meld_boxes',0)} ({meld_pct:.1f}% ok)"
              f"  hand {s.get('hand_boxes',0)}{qa_str}", flush=True)

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}/summary.json")


if __name__ == "__main__":
    main()
