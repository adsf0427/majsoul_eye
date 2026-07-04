"""Train the 38-class tile classifier on auto-labeled crops from one or more games.

Splitting matters: the same physical discarded tile appears in ~10 frames, so a
frame/random split leaks it across train/val and inflates accuracy. Hold out
whole KYOKU (or a whole SESSION) so val contains DIFFERENT physical tiles —
the honest generalization test. With ≥2 games, a cross-session val is strongest.

    # train on both games, validate on held-out session6 kyoku:
    python scripts/train/train_classifier.py \
        --data session5=datasets/session5/crops:captures/raw/manual/session5.jsonl \
        --data session6=datasets/session6_hr/crops:captures/raw/manual/session6.jsonl \
        --val session6:E3.0,S2.0 --epochs 20

    # pure cross-game: train session5, validate ALL of session6:
        --val session6:*
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from majsoul_eye.tiles import TILE_NAMES, NAME_TO_ID
from majsoul_eye.recognize.classifier import TileNet, preprocess
from majsoul_eye.capture.schema import read_records
from majsoul_eye.capture.sync import RELEVANT_EVENTS
from majsoul_eye.state.replay import Replayer


def seq_to_kyoku(capture: str) -> dict[int, str]:
    rp = Replayer(); m = {}
    for r in read_records(capture):
        rp.apply_record(r)
        if r.mjai and any(e.get("type") in RELEVANT_EVENTS for e in r.mjai):
            m[r.seq] = f"{rp.state.bakaze}{rp.state.kyoku}.{rp.state.honba}"
    return m


def _augment(img):
    h, w = img.shape[:2]
    ang = random.uniform(-7, 7)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[0, 2] += random.uniform(-0.06, 0.06) * w
    M[1, 2] += random.uniform(-0.06, 0.06) * h
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    img = np.clip(img.astype(np.float32) * random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
    return img


class CropDS(Dataset):
    def __init__(self, items, train=False):
        self.items = items; self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = cv2.imread(p)
        if self.train:
            img = _augment(img)
        return preprocess(img), y


def dataset_data_specs(ds_dir: str, sub: str) -> list:
    """Expand a versioned dataset's ``games.json`` (scripts/data/build_datasets.py)
    into ``(name, <ds_dir>/<game dir>/<sub>, capture)`` tuples (sub = 'crops'/'yolo').
    Tuples, not ``NAME=DIR:CAPTURE`` strings — a Windows drive colon would break the
    string form's ``:`` split."""
    manifest = os.path.join(ds_dir, "games.json")
    if not os.path.exists(manifest):
        raise SystemExit(f"{manifest} not found — is {ds_dir!r} a dataset version built by "
                         f"scripts/data/build_datasets.py?")
    m = json.load(open(manifest, encoding="utf-8"))
    return [(g["name"], os.path.join(ds_dir, g["dir"], sub).replace(os.sep, "/"), g["capture"])
            for g in m["games"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", default=None, help="NAME=CROPSDIR:CAPTURE (repeatable)")
    ap.add_argument("--dataset", action="append", default=None,
                    help="versioned dataset dir with games.json (scripts/data/build_datasets.py); "
                         "repeatable — expands to one --data entry per game (crops). Duplicate "
                         "NAMEs: the later spec (and any explicit --data) wins.")
    ap.add_argument("--val", default="", help="VAL spec 'NAME:k1,k2' or 'NAME:*' (whole session)")
    ap.add_argument("--out", default="majsoul_eye/recognize/tile_classifier.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers (try 6 on GPU so cv2.imread doesn't starve it).")
    args = ap.parse_args()

    # parse sources + val spec (--dataset manifests expand first, --data can override)
    entries = []
    for dsd in (args.dataset or []):
        entries += dataset_data_specs(dsd, "crops")
    for d in (args.data or []):
        name, rest = d.split("=", 1)
        crops, capture = rest.split(":", 1)
        entries.append((name, crops, capture))
    if not entries:
        ap.error("need --data and/or --dataset")
    sources = {}
    for name, crops, capture in entries:
        if name in sources:
            print(f"note: duplicate game {name!r} — keeping the later spec ({crops})")
        sources[name] = (crops, capture)
    val_name, val_kyoku = (args.val.split(":", 1) + [""])[:2] if args.val else ("", "")
    val_set = "*" if val_kyoku == "*" else set(val_kyoku.split(",")) if val_kyoku else set()

    train, val = [], []
    for name, (crops, capture) in sources.items():
        sk = seq_to_kyoku(capture)
        for cls in os.listdir(crops):
            if cls not in NAME_TO_ID:
                continue
            y = NAME_TO_ID[cls]
            for p in glob.glob(os.path.join(crops, cls, "*.png")):
                seq = int(os.path.basename(p).split("_")[0])
                is_val = name == val_name and (val_set == "*" or sk.get(seq) in val_set)
                (val if is_val else train).append((p, y))
    random.seed(0); random.shuffle(train)
    print(f"sources={list(sources)}  val={args.val or '(none)'}")
    print(f"train={len(train)} val={len(val)}  classes train={len(set(y for _,y in train))} val={len(set(y for _,y in val))}")
    if not val:
        print("WARNING: empty val set");

    cls_count = Counter(y for _, y in train)
    weights = torch.tensor([1.0 / cls_count.get(y, 1) for _, y in train])
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    pin = torch.cuda.is_available()
    tl = DataLoader(CropDS(train, train=True), batch_size=args.batch, sampler=sampler,
                    num_workers=args.workers, pin_memory=pin,
                    persistent_workers=args.workers > 0)
    vl = (DataLoader(CropDS(val), batch_size=args.batch, num_workers=args.workers,
                     pin_memory=pin, persistent_workers=args.workers > 0) if val else None)

    import time
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    print(f"device={dev} ({gpu})  train={len(train)} val={len(val)} batches/epoch={len(tl)}", flush=True)
    model = TileNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best = 0.0
    best_per, best_tot = Counter(), Counter()
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run_loss = 0.0; nb = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step()
            run_loss += loss.item(); nb += 1
        train_loss = run_loss / max(nb, 1); dt = time.time() - t0
        if not vl:
            print(f"epoch {ep+1:2d}/{args.epochs}  train_loss={train_loss:.4f}  ({dt:.1f}s)", flush=True)
            continue
        model.eval(); correct = 0; per, per_tot = Counter(), Counter()
        with torch.no_grad():
            for x, y in vl:
                pred = model(x.to(dev)).argmax(1).cpu()
                correct += (pred == y).sum().item()
                for p_, y_ in zip(pred.numpy(), y.numpy()):
                    per_tot[y_] += 1; per[y_] += int(p_ == y_)
        acc = correct / len(val)
        print(f"epoch {ep+1:2d}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={acc:.4f}  ({dt:.1f}s)", flush=True)
        if acc >= best:
            best = acc
            best_per, best_tot = per, per_tot     # snapshot per-class of the SAVED weight
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(model.state_dict(), args.out)
    if not vl:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save(model.state_dict(), args.out)
    print(f"\nbest val_acc={best:.4f}  saved -> {args.out}")
    if best_tot:
        worst = sorted(((best_per[y] / best_tot[y], TILE_NAMES[y], best_tot[y]) for y in best_tot), key=lambda t: t[0])
        print("worst classes of the SAVED (best-epoch) model (acc, class, n_val):")
        for a, c, n in worst[:12]:
            print(f"  {a:.3f}  {c:4s}  n={n}")


if __name__ == "__main__":
    main()
