"""Visualize classifier failure cases: run a model over labeled crops, collect the
misclassifications, and render montages grouped by confusion pair (gt -> pred),
most frequent first. Use it to SEE what the classifier confuses (neighbor bleed,
perspective, red fives, ...) rather than just reading a number.

Examples (PowerShell):
  $PY = "C:/Users/zsx/miniforge3/envs/auto/python.exe"; $env:PYTHONPATH = "."
  # all crops of a game vs the shipped model:
  & $PY scripts/visualize_failures.py --crops datasets/ai_g1/crops --out fails/ai_g1
  # only a held-out kyoku (true generalization failures):
  & $PY scripts/visualize_failures.py --crops datasets/session6_erode/crops `
        --val-capture captures/session6.jsonl --val-kyoku E3.0,S2.0 --out fails/s6val

Outputs: <out>/confusions.png (montage), <out>/summary.txt (overall acc + top confusions
+ worst classes). Runs on GPU if available.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter, defaultdict

import numpy as np


def seq_to_kyoku(capture: str) -> dict:
    from majsoul_eye.state.replay import Replayer
    from majsoul_eye.capture.schema import read_records
    from majsoul_eye.capture.sync import RELEVANT_EVENTS
    rp = Replayer(); m = {}
    for r in read_records(capture):
        rp.apply_record(r)
        if r.mjai and any(e.get("type") in RELEVANT_EVENTS for e in r.mjai):
            m[r.seq] = f"{rp.state.bakaze}{rp.state.kyoku}.{rp.state.honba}"
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", action="append", required=True, help="dataset crops dir (repeatable)")
    ap.add_argument("--model", default="majsoul_eye/recognize/tile_classifier.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-capture", default="", help="restrict to a held-out split's capture")
    ap.add_argument("--val-kyoku", default="", help="comma kyoku list, e.g. E3.0,S2.0 (with --val-capture)")
    ap.add_argument("--top", type=int, default=20, help="top-N confusion pairs to show")
    ap.add_argument("--per", type=int, default=12, help="example crops per confusion pair")
    ap.add_argument("--cell", type=int, default=56)
    args = ap.parse_args()

    import cv2, torch
    from majsoul_eye.tiles import NAME_TO_ID, TILE_NAMES
    from majsoul_eye.recognize.classifier import TileNet, preprocess

    val_set = set(args.val_kyoku.split(",")) if args.val_kyoku else None
    sk = seq_to_kyoku(args.val_capture) if (args.val_capture and val_set) else None

    items = []  # (path, gt_class)
    for cdir in args.crops:
        for cls in os.listdir(cdir):
            if cls not in NAME_TO_ID:
                continue
            for p in glob.glob(os.path.join(cdir, cls, "*.png")):
                if sk is not None:
                    seq = int(os.path.basename(p).split("_")[0])
                    if sk.get(seq) not in val_set:
                        continue
                items.append((p, cls))
    if not items:
        raise SystemExit("no crops matched (check --crops / --val-kyoku)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = TileNet().to(dev)
    model.load_state_dict(torch.load(args.model, map_location=dev))
    model.eval()

    fails = defaultdict(list)         # (gt, pred) -> [paths]
    per_cls = defaultdict(lambda: [0, 0])  # cls -> [correct, total]
    correct = 0
    for i in range(0, len(items), 512):
        batch = items[i:i + 512]
        x = torch.stack([preprocess(cv2.imread(p)) for p, _ in batch]).to(dev)
        with torch.no_grad():
            pred = model(x).argmax(1).cpu().numpy()
        for (p, gt), k in zip(batch, pred):
            pr = TILE_NAMES[k]
            per_cls[gt][1] += 1
            if pr == gt:
                per_cls[gt][0] += 1; correct += 1
            else:
                fails[(gt, pr)].append(p)

    acc = correct / len(items)
    os.makedirs(args.out, exist_ok=True)

    # --- text summary ---
    pairs = sorted(fails.items(), key=lambda kv: -len(kv[1]))
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"model={args.model}\ncrops={args.crops} val={args.val_kyoku or '(all)'}\n")
        f.write(f"n={len(items)}  acc={acc:.4f}  errors={len(items)-correct}\n\n")
        f.write("top confusions (gt -> pred  count):\n")
        for (gt, pr), ps in pairs[:40]:
            f.write(f"  {gt:4s} -> {pr:4s}  {len(ps)}\n")
        f.write("\nworst classes (acc, n):\n")
        for cls in sorted(per_cls, key=lambda c: per_cls[c][0] / per_cls[c][1]):
            c, n = per_cls[cls]
            if c < n:
                f.write(f"  {cls:4s} {c/n:.3f} ({c}/{n})\n")
    print(f"n={len(items)} acc={acc:.4f} errors={len(items)-correct}  top confusions:")
    for (gt, pr), ps in pairs[:12]:
        print(f"  {gt:4s} -> {pr:4s}  x{len(ps)}")

    # --- montage: one row per top confusion pair ---
    cw = ch = args.cell
    label_w = 120
    pad = 4
    rows = []
    import random
    random.seed(0)
    for (gt, pr), ps in pairs[:args.top]:
        ex = ps[:]; random.shuffle(ex); ex = ex[:args.per]
        cells = []
        for p in ex:
            img = cv2.imread(p)
            cells.append(cv2.resize(img, (cw, ch)) if img is not None else np.zeros((ch, cw, 3), np.uint8))
        while len(cells) < args.per:
            cells.append(np.full((ch, cw, 3), 30, np.uint8))
        strip = np.hstack(cells)
        lab = np.full((ch, label_w, 3), 30, np.uint8)
        cv2.putText(lab, f"{gt}->{pr}", (4, ch // 2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(lab, f"x{len(ps)}", (4, ch // 2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        rows.append(np.hstack([lab, strip]))
    if rows:
        width = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.full((r.shape[0], width - r.shape[1], 3), 30, np.uint8)]) for r in rows]
        sep = np.full((pad, width, 3), 60, np.uint8)
        stacked = [rows[0]]
        for r in rows[1:]:
            stacked.append(sep); stacked.append(r)
        canvas = np.vstack(stacked)
        out = os.path.join(args.out, "confusions.png")
        cv2.imwrite(out, canvas)
        print(f"\nsaved {out}  ({len(rows)} confusion rows)  + summary.txt")
    else:
        print("\nno misclassifications — nothing to montage")


if __name__ == "__main__":
    main()
