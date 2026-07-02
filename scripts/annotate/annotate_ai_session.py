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
  PYTHONPATH=. $PY scripts/annotate/annotate_ai_session.py                    # all captures/intermediate/gt/*
  PYTHONPATH=. $PY scripts/annotate/annotate_ai_session.py \
      --captures captures/intermediate/gt/ai_run_3_game1.jsonl --overlay-every 40 --qa-classifier

By default the frames dir is derived from the capture name (X.jsonl -> X/, via
majsoul_eye.paths.frames_dir_for). Pass --frames-dir to point one capture at a
different frames folder (e.g. de-letterboxed frames from deletterbox_frames.py)
while keeping the GT + output name of the original capture:
  PYTHONPATH=. $PY scripts/annotate/annotate_ai_session.py --captures captures/intermediate/gt/ai_run_5_game2.jsonl \
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

from majsoul_eye.annotate import pipeline as P
from majsoul_eye import paths
from majsoul_eye.annotate.frame import annotate_frame, crop_quad
from majsoul_eye.capture.gtframes import build_seq_state, load_frames


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
    for d in rec.get("dora_boxes", []):
        x1, y1, x2, y2 = d["px_box"]
        if not d.get("reliable", True):
            col = (0, 0, 255)                        # red   = not rendered / low fill
        elif d.get("back"):
            col = (0, 140, 255)                      # orange = face-down back slot
        else:
            col = (0, 255, 255)                      # yellow = revealed indicator
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.putText(vis, d["tile"], (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.imwrite(path, vis)


def _process_capture(cap, cfg):
    """Annotate one capture end-to-end -> (name, summary_entry | None). Standalone
    so it can run in a worker process (see --workers); keeps cv2/torch single-
    threaded so N workers use N cores without oversubscription."""
    cv2.setNumThreads(1)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    name = os.path.splitext(os.path.basename(cap))[0]
    try:
        seq_state = build_seq_state(cap)
        frames = load_frames(cfg["frames_dir"] or paths.frames_dir_for(cap))
    except Exception as e:
        print(f"{name}: SKIP ({e})", flush=True)
        return name, None

    clf = None
    if cfg["qa_classifier"]:
        from majsoul_eye.recognize.classifier import TileClassifier
        clf = TileClassifier("majsoul_eye/recognize/tile_classifier.pt")
    hom = P.build_homographies(1920, 1080)

    try:
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
        out_path = os.path.join(cfg["out"], f"{name}.jsonl")
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
                stats["dora_boxes"] += len(rec["dora_boxes"])
                stats["dora_ok"] += sum(1 for d in rec["dora_boxes"] if d.get("reliable", True))

                if cfg["overlay_every"] and k % cfg["overlay_every"] == 0:
                    render_overlay(img, rec, os.path.join(cfg["out"], "overlays", f"{name}_seq{seq}.png"))

                if clf is not None and k % 3 == 0 and qa["river"][0] + qa["meld"][0] < cfg["qa_per_game"]:
                    # sideways cells are 90°-rotated vs the training crops: classify
                    # both rotations and accept either (QA-only leniency).
                    crops, keys = [], []
                    for slots in rec["discard_slots"].values():
                        for s in slots:
                            if s.get("reliable", True):
                                c = crop_quad(img, s["face_poly_original"])
                                if s.get("riichi"):
                                    crops += [np.rot90(c).copy(), np.rot90(c, 3).copy()]
                                    keys.append(("river", s["tile"], 2))
                                else:
                                    crops.append(c)
                                    keys.append(("river", s["tile"], 1))
                    for boxes in rec["meld_boxes"].values():
                        for b in boxes:
                            if b.get("reliable", True):
                                c = crop_quad(img, b["poly_original"])
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
    except Exception as e:                       # isolate a bad game from the pool
        print(f"{name}: FAILED ({type(e).__name__}: {e})", flush=True)
        return name, None

    s = {k: int(v) for k, v in stats.items()}
    if clf is not None:
        s["qa"] = {z: {"n": n, "agree": round(c / n, 4) if n else None}
                   for z, (n, c) in qa.items()}
        s["qa_mismatch_examples"] = qa_bad[:20]
    riv_pct = 100 * s.get("river_ok", 0) / max(1, s.get("river_boxes", 1))
    meld_pct = 100 * s.get("meld_ok", 0) / max(1, s.get("meld_boxes", 1))
    qa_str = ""
    if clf is not None:
        qa_str = "  QA " + " ".join(f"{z}:{v['agree']}" for z, v in s["qa"].items() if v["n"])
    dora_pct = 100 * s.get("dora_ok", 0) / max(1, s.get("dora_boxes", 1))
    print(f"{name}: {s['frames']} frames  river {s.get('river_boxes',0)} boxes ({riv_pct:.1f}% ok, "
          f"{s.get('river_unrendered',0)} unrendered)  meld {s.get('meld_boxes',0)} ({meld_pct:.1f}% ok)"
          f"  hand {s.get('hand_boxes',0)}  dora {s.get('dora_boxes',0)} ({dora_pct:.1f}% ok){qa_str}", flush=True)
    return name, s


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
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4) - 2)),
                    help="parallel worker processes, one game each "
                         "(default ~cpu_count-2; pass 1 for sequential)")
    args = ap.parse_args()

    captures = args.captures or paths.converted_gt_captures()
    if args.frames_dir and len(captures) != 1:
        ap.error("--frames-dir requires exactly one --captures")
    os.makedirs(os.path.join(args.out, "overlays"), exist_ok=True)

    cfg = {"out": args.out, "frames_dir": args.frames_dir,
           "overlay_every": args.overlay_every,
           "qa_classifier": args.qa_classifier, "qa_per_game": args.qa_per_game}

    # merge into an existing summary so multiple runs into the same --out (e.g.
    # session6 then session5, which needs its own --frames-dir) accumulate rather
    # than clobber. A clean regen starts from a freshly-deleted out/ dir anyway.
    summary_path = os.path.join(args.out, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        try:
            summary = json.load(open(summary_path, encoding="utf-8"))
        except Exception:
            summary = {}

    workers = min(max(1, args.workers), len(captures))
    if workers > 1:
        import functools
        from concurrent.futures import ProcessPoolExecutor
        worker = functools.partial(_process_capture, cfg=cfg)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(worker, captures))
    else:
        results = [_process_capture(cap, cfg) for cap in captures]
    for name, entry in results:
        if entry is not None:
            summary[name] = entry

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
