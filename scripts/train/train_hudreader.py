"""Train the HUD micro-readers: DigitCTC over NUMERIC_FIELDS crops (segmentation-free
CRNN-CTC) + round_label / seat_wind_self CE heads (TileNet), all three saved into
ONE checkpoint consumed by ``majsoul_eye.recognize.hudreader.HudReader``.

Consumes per-game ``hud/<field>/<seq>.png`` crops + ``hud/labels.jsonl`` emitted by
``scripts/train/build_dataset.py`` (Task 9: padded 15%, already rotated upright),
addressed through the SAME versioned-dataset manifests as ``train_classifier.py``'s
``--dataset`` convention (``datasets/<v>/games.json`` — see
``scripts/data/build_datasets.py``): ``{"val": <held-out game name>, "games": [{"name",
"dir", ...}, ...]}``.

Unlike the tile classifier, the held-out split here is a WHOLE GAME (the manifest's
``val`` name), not a per-kyoku split: HUD field text (scores/wall count/round/wind)
repeats across a game's frames the same way discards do, so per-kyoku leakage is the
same risk — but a HUD field has no kyoku-scoped GT the way ``Replayer`` gives tiles,
and a whole-game hold-out is already how ``build_datasets.py`` splits the detector
(``--val {game}:*``), so this reuses that same convention rather than inventing one.

    python scripts/train/train_hudreader.py --dataset datasets/v2 \\
        --out majsoul_eye/recognize/hud_reader.pt

Smoke (CPU-cheap, no real dataset needed — see tests/test_hudreader.py + task report
for how a 2-pseudo-game synthetic manifest is assembled under scratchpad/):
    python scripts/train/train_hudreader.py --dataset scratchpad/t11_ds --epochs 1 \\
        --out scratchpad/t11_ds/hud_reader_smoke.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from majsoul_eye.hud import CTC_CHARSET, NUMERIC_FIELDS, ROUND_CLASSES, WIND_CLASSES
from majsoul_eye.recognize.classifier import TileNet, preprocess
from majsoul_eye.recognize.hudreader import DigitCTC, _strip, ctc_decode, encode_text


# ---- dataset manifest expansion (mirrors train_classifier.py's --dataset) -----

def dataset_hud_specs(ds_dir: str) -> tuple[list[str] | None, list[tuple[str, str]]]:
    """Expand a versioned dataset's ``games.json`` into ``(name, hud_dir)`` tuples
    + the manifest's held-out ``val`` game names (whole-game hold-out, see module
    docstring). ``write_manifest`` stores ``val`` as a LIST (multi-val convention);
    a legacy scalar is normalized to a one-element list. ``hud_dir`` may not exist
    for datasets built before Task 9 (no HUD crops emitted yet) — callers must
    tolerate a missing ``labels.jsonl``."""
    manifest = os.path.join(ds_dir, "games.json")
    if not os.path.exists(manifest):
        raise SystemExit(f"{manifest} not found — is {ds_dir!r} a dataset version built by "
                         f"scripts/data/build_datasets.py?")
    m = json.load(open(manifest, encoding="utf-8"))
    specs = [(g["name"], os.path.join(ds_dir, g["dir"], "hud").replace(os.sep, "/"))
             for g in m["games"]]
    val = m.get("val")
    return ([val] if isinstance(val, str) else val), specs


def load_rows(games: list[tuple[str, str]], val_names: list[str]) -> tuple[list[dict], list[dict]]:
    """Read every game's hud/labels.jsonl -> rows {"path","text","pad","field"},
    split train/val by WHOLE GAME (name in val_names -> val)."""
    train, val = [], []
    for name, hud_dir in games:
        lp = os.path.join(hud_dir, "labels.jsonl")
        if not os.path.exists(lp):
            print(f"note: no {lp} — skipping game {name!r} (built before HUD crops existed?)")
            continue
        bucket = val if name in val_names else train
        for line in open(lp, encoding="utf-8"):
            r = json.loads(line)
            bucket.append({
                "path": os.path.join(hud_dir, r["file"]).replace(os.sep, "/"),
                "text": r["text"], "pad": float(r.get("pad", 0.15)), "field": r["name"],
            })
    return train, val


# ---- CTC sub-training -----------------------------------------------------

def _augment_crop(bgr: np.ndarray, pad: float, jitter: float = 0.08) -> np.ndarray:
    """Random re-crop within the stored pad (simulates an imperfect detector box)
    + brightness jitter. Re-crop is bounded by min(jitter, pad) per side so it
    never eats into the true field content when pad < jitter. Shared by the CTC
    strips AND the round/wind CE heads: the dataset crops are fixed-ROI ink-snap
    (near-identical framing every frame), so a head trained without this jitter
    overfits the exact framing and collapses on the detector's tighter runtime
    boxes (measured: round_label 26.7% / seat_wind 72.7% end-to-end vs 100%
    crop-level before augmentation was applied to the CE heads)."""
    j = min(jitter, pad)
    h, w = bgr.shape[:2]
    x0 = int(round(w * random.uniform(0, j)))
    x1 = w - int(round(w * random.uniform(0, j)))
    y0 = int(round(h * random.uniform(0, j)))
    y1 = h - int(round(h * random.uniform(0, j)))
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    bgr = bgr[y0:y1, x0:x1]
    bgr = np.clip(bgr.astype(np.float32) * random.uniform(0.85, 1.15), 0, 255).astype(np.uint8)
    return bgr


class CtcDS(Dataset):
    """rows -> (1xHxW strip tensor, encoded target ids, raw text). Reuses
    hudreader._strip so train-time preprocessing is bit-identical to HudReader.read()."""

    def __init__(self, rows: list[dict], train: bool = False):
        self.rows = rows
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(r["path"])
        if self.train:
            img = _augment_crop(img, r["pad"])
        x = _strip(img)[0]                       # 1x32xW (drop _strip's batch dim)
        y = torch.tensor(encode_text(r["text"]), dtype=torch.long)
        return x, y, r["text"]


def ctc_collate(batch):
    xs, ys, texts = zip(*batch)
    maxw = max(x.shape[-1] for x in xs)
    padded = torch.zeros(len(xs), 1, 32, maxw)
    for i, x in enumerate(xs):
        padded[i, :, :, :x.shape[-1]] = x
    input_lengths = torch.tensor([max(1, x.shape[-1] // 4) for x in xs], dtype=torch.long)
    target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    targets = torch.cat(ys) if ys else torch.zeros(0, dtype=torch.long)
    return padded, input_lengths, targets, target_lengths, texts


def eval_ctc(model, rows_val, batch, workers, dev) -> float | None:
    if not rows_val:
        return None
    dl = DataLoader(CtcDS(rows_val), batch_size=batch, shuffle=False,
                    collate_fn=ctc_collate, num_workers=workers)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, input_lengths, _targets, _target_lengths, texts in dl:
            logits = model(x.to(dev)).cpu()      # B,Tmax,C
            for i, t_len in enumerate(input_lengths.tolist()):
                pred = ctc_decode(logits[i, :t_len])
                correct += int(pred == texts[i])
                total += 1
    return correct / total if total else None


def train_ctc(rows_train, rows_val, epochs, batch, workers, dev):
    model = DigitCTC().to(dev)
    if not rows_train:
        print("[ctc] 0 train samples — skipping (checkpoint keeps random-init weights)")
        return {k: v.cpu().clone() for k, v in model.state_dict().items()}, None

    # field-type-balanced sampling: wall_count is naturally digit-diverse, scores
    # skew 0/2/5-heavy — weight inversely by field so the CTC head sees every
    # NUMERIC_FIELDS type about equally often per epoch.
    field_count = Counter(r["field"] for r in rows_train)
    weights = torch.tensor([1.0 / field_count[r["field"]] for r in rows_train])
    sampler = WeightedRandomSampler(weights, len(weights))
    dl = DataLoader(CtcDS(rows_train, train=True), batch_size=batch, sampler=sampler,
                    collate_fn=ctc_collate, num_workers=workers)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CTCLoss(blank=0, zero_infinity=True)
    best_acc, best_sd = -1.0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    for ep in range(epochs):
        model.train(); t0 = time.time(); run_loss = 0.0; nb = 0
        for x, input_lengths, targets, target_lengths, _texts in dl:
            x = x.to(dev)
            logp = model(x).permute(1, 0, 2)      # CTCLoss wants T,B,C
            loss = lossf(logp, targets, input_lengths, target_lengths)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item(); nb += 1
        train_loss = run_loss / max(nb, 1); dt = time.time() - t0
        acc = eval_ctc(model, rows_val, batch, workers, dev)
        acc_s = f"val_exact={acc:.4f}" if acc is not None else "val_exact=n/a"
        print(f"[ctc]   epoch {ep+1:2d}/{epochs}  train_loss={train_loss:.4f}  {acc_s}  ({dt:.1f}s)",
              flush=True)
        if acc is not None and acc >= best_acc:
            best_acc = acc
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_acc < 0:      # val was empty every epoch -> keep the final weights
        best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_acc = None
    return best_sd, best_acc


# ---- round / wind CE sub-trainings (share TileNet + classifier.preprocess) ----

class ClsDS(Dataset):
    def __init__(self, rows: list[dict], classes: list[str], train: bool = False):
        self.rows = rows; self.classes = classes; self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(r["path"])
        if self.train:
            img = _augment_crop(img, r["pad"], jitter=0.15)
        return preprocess(img), self.classes.index(r["text"])


def train_cls(tag, rows_train, rows_val, classes, epochs, batch, workers, dev):
    model = TileNet(n_classes=len(classes)).to(dev)
    if not rows_train:
        print(f"[{tag}] 0 train samples — skipping (checkpoint keeps random-init weights)")
        return {k: v.cpu().clone() for k, v in model.state_dict().items()}, None

    tl = DataLoader(ClsDS(rows_train, classes, train=True), batch_size=batch, shuffle=True,
                    num_workers=workers)
    vl = (DataLoader(ClsDS(rows_val, classes), batch_size=batch, num_workers=workers)
          if rows_val else None)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best_acc, best_sd = -1.0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    for ep in range(epochs):
        model.train(); t0 = time.time(); run_loss = 0.0; nb = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step()
            run_loss += loss.item(); nb += 1
        train_loss = run_loss / max(nb, 1); dt = time.time() - t0
        if vl:
            model.eval(); correct = 0; tot = 0
            with torch.no_grad():
                for x, y in vl:
                    pred = model(x.to(dev)).argmax(1).cpu()
                    correct += (pred == y).sum().item(); tot += len(y)
            acc = correct / tot
            print(f"[{tag}] epoch {ep+1:2d}/{epochs}  train_loss={train_loss:.4f}  "
                 f"val_top1={acc:.4f}  ({dt:.1f}s)", flush=True)
            if acc >= best_acc:
                best_acc = acc
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            print(f"[{tag}] epoch {ep+1:2d}/{epochs}  train_loss={train_loss:.4f}  "
                 f"val_top1=n/a  ({dt:.1f}s)", flush=True)
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_sd, (best_acc if best_acc >= 0 else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", default=None,
                    help="versioned dataset dir with games.json (scripts/data/build_datasets.py); "
                         "repeatable — collects every game's hud/labels.jsonl.")
    ap.add_argument("--val", action="append", default=None,
                    help="held-out whole game name (repeatable; default: the 'val' list of "
                         "the FIRST --dataset manifest that defines one).")
    ap.add_argument("--out", default="majsoul_eye/recognize/hud_reader.pt")
    ap.add_argument("--epochs", type=int, default=20, help="epochs for EACH of the 3 sub-trainings")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers (try higher on GPU so cv2.imread doesn't starve it).")
    args = ap.parse_args()

    if not args.dataset:
        ap.error("need --dataset (repeatable)")

    games: list[tuple[str, str]] = []
    val_names = args.val
    for dsd in args.dataset:
        v, specs = dataset_hud_specs(dsd)
        games += specs
        if val_names is None:
            val_names = v
    names = [n for n, _ in games]
    if not val_names:
        ap.error("no --val given and no --dataset manifest defines one")
    missing = [v for v in val_names if v not in names]
    if missing:
        ap.error(f"--val {missing!r} not among discovered games {names}")

    train_rows, val_rows = load_rows(games, val_names)
    ctc_train = [r for r in train_rows if r["field"] in NUMERIC_FIELDS]
    ctc_val = [r for r in val_rows if r["field"] in NUMERIC_FIELDS]
    round_train = [r for r in train_rows if r["field"] == "round_label"]
    round_val = [r for r in val_rows if r["field"] == "round_label"]
    wind_train = [r for r in train_rows if r["field"] == "seat_wind_self"]
    wind_val = [r for r in val_rows if r["field"] == "seat_wind_self"]

    print(f"games={names}  val={val_names}")
    print(f"ctc:   train={len(ctc_train)}  val={len(ctc_val)}  "
         f"(fields: {dict(Counter(r['field'] for r in ctc_train))})")
    print(f"round: train={len(round_train)}  val={len(round_val)}")
    print(f"wind:  train={len(wind_train)}  val={len(wind_val)}")
    if not (ctc_train or round_train or wind_train):
        raise SystemExit("no training rows found at all — check --dataset/hud/labels.jsonl paths")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    print(f"device={dev} ({gpu})", flush=True)

    ctc_sd, ctc_acc = train_ctc(ctc_train, ctc_val, args.epochs, args.batch, args.workers, dev)
    round_sd, round_acc = train_cls("round", round_train, round_val, ROUND_CLASSES,
                                    args.epochs, args.batch, args.workers, dev)
    wind_sd, wind_acc = train_cls("wind", wind_train, wind_val, WIND_CLASSES,
                                  args.epochs, args.batch, args.workers, dev)

    ckpt = {
        "ctc": ctc_sd, "round": round_sd, "wind": wind_sd,
        "charset": CTC_CHARSET,
        "meta": {
            "games": names, "val": val_names, "epochs": args.epochs,
            "ctc_exact_match": ctc_acc, "round_top1": round_acc, "wind_top1": wind_acc,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(ckpt, args.out)

    def _fmt(a):
        return f"{a:.4f}" if a is not None else "n/a"
    print(f"\nsaved -> {args.out}")
    print(f"  ctc exact-match = {_fmt(ctc_acc)}")
    print(f"  round top1      = {_fmt(round_acc)}")
    print(f"  wind  top1      = {_fmt(wind_acc)}")


if __name__ == "__main__":
    main()
