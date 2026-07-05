"""Assemble an Ultralytics YOLO detection dataset from per-game ``build_dataset``
YOLO exports, splitting BY KYOKU/GAME (never by frame — the same physical discard
spans ~10 near-duplicate frames, so a frame split leaks it and inflates mAP).

Mirrors ``train_classifier.py``'s interface. Each ``--data`` entry is
``NAME=YOLODIR:CAPTURE`` where YOLODIR holds ``images/`` + ``labels/`` (e.g.
``datasets/precise_ai_run_3_game1/yolo``, produced by ``build_dataset.py``) and
CAPTURE is the GT jsonl, used only to map each frame's global ``seq`` -> kyoku for
a leakage-safe split. ``--val NAME:k1,k2`` holds those kyoku out; ``--val NAME:*``
holds the whole game out (the cross-game val — strongest).

Writes into ``--out`` (default ``datasets/detector``) WITHOUT copying images, using
REPO-ROOT-RELATIVE POSIX paths so the whole tree is portable: tar ``datasets/`` + the
repo to another machine (e.g. a GPU server) and train there with NO regeneration.
**Run from the repo root** (as all tooling does) — the paths are relative to it.
Ultralytics resolves the image paths against the CWD and recovers ``.../labels/<seq>.txt``
by swapping the ``images`` path segment; it normalizes ``/``→os.sep first, so forward
slashes work on Windows too.
    train.txt / val.txt   newline lists of repo-root-relative POSIX image paths
    data.yaml             relative path/train/val + 55 = frozen 38 tiles + 17 HUD
                          (majsoul_eye.hud.DET_NAMES); v1 (pre-HUD) label files only
                          ever used class ids 0-37, so old + new labels mix freely
                          under the 55-class head.

Milestone 1 (one game, hold out its last kyoku):
    python scripts/train/build_detector_dataset.py \
        --data g1=datasets/precise_ai_run_3_game1/yolo:captures/raw/ai_session/run_3/game1/game1.jsonl \
        --val g1:S4.0 --out datasets/detector_g1

Milestone 2 (many games, hold out one whole game):
    python scripts/train/build_detector_dataset.py \
        --data g1=datasets/precise_ai_run_3_game1/yolo:captures/raw/ai_session/run_3/game1/game1.jsonl \
        ...(one --data per game)... \
        --data v=datasets/precise_ai_run_8_game1/yolo:captures/raw/ai_session/run_8/game1/game1.jsonl \
        --val v:* --out datasets/detector
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from majsoul_eye.hud import DET_NAMES


def seq_to_kyoku(capture: str) -> dict:
    """Map each record ``seq`` -> ``"{bakaze}{kyoku}.{honba}"`` by replaying the GT.

    Re-implemented here (not imported from ``train_classifier``) to keep scripts
    free of inter-script imports; identical logic to the classifier trainer.
    """
    from majsoul_eye.capture.schema import read_records
    from majsoul_eye.capture.sync import RELEVANT_EVENTS
    from majsoul_eye.state.replay import Replayer

    rp = Replayer(); m = {}
    for r in read_records(capture):
        rp.apply_record(r)
        if r.mjai and any(e.get("type") in RELEVANT_EVENTS for e in r.mjai):
            m[r.seq] = f"{rp.state.bakaze}{rp.state.kyoku}.{rp.state.honba}"
    return m


def parse_data_arg(spec: str):
    """``NAME=YOLODIR:CAPTURE`` -> (name, yolodir, capture)."""
    name, rest = spec.split("=", 1)
    yolodir, capture = rest.split(":", 1)
    return name, yolodir, capture


def dataset_data_specs(ds_dir: str, sub: str = "yolo") -> list:
    """Expand a versioned dataset's ``games.json`` (scripts/data/build_datasets.py)
    into ``(name, <ds_dir>/<game dir>/<sub>, capture)`` tuples. Tuples, not
    ``NAME=DIR:CAPTURE`` strings — a Windows drive colon would break the ``:`` split.
    Re-implemented here (not imported from train_classifier) to keep scripts free of
    inter-script imports."""
    manifest = os.path.join(ds_dir, "games.json")
    if not os.path.exists(manifest):
        raise SystemExit(f"{manifest} not found — is {ds_dir!r} a dataset version built by "
                         f"scripts/data/build_datasets.py?")
    m = json.load(open(manifest, encoding="utf-8"))
    return [(g["name"], os.path.join(ds_dir, g["dir"], sub).replace(os.sep, "/"), g["capture"])
            for g in m["games"]]


def split_images(sources: dict, val_name: str, val_set, kyoku_fn=seq_to_kyoku):
    """sources: name -> (yolodir, capture). Returns (train, val) lists of image paths
    with the yolodir's relativity PRESERVED, POSIX-slashed (relative yolodir in →
    portable relative path out). ``val_set == "*"`` holds the whole game out; otherwise
    a kyoku set from ``kyoku_fn(capture)``. Non-val games go entirely to train."""
    train, val = [], []
    for name, (yolodir, capture) in sources.items():
        want_val = name == val_name
        sk = kyoku_fn(capture) if (want_val and val_set != "*") else {}
        for p in sorted(glob.glob(os.path.join(yolodir, "images", "*.png"))):
            ap = p.replace(os.sep, "/")           # POSIX sep; keep relativity → portable
            if not want_val:
                train.append(ap)
            elif val_set == "*":
                val.append(ap)
            else:
                seq = int(os.path.splitext(os.path.basename(p))[0])
                (val if sk.get(seq) in val_set else train).append(ap)
    return train, val


def _relposix(p: str) -> str:
    """Repo-root-relative POSIX path; falls back to absolute for cross-drive outs."""
    try:
        return os.path.relpath(p).replace(os.sep, "/")
    except ValueError:
        return os.path.abspath(p).replace(os.sep, "/")


def build_data_yaml_text(out_dir: str, train_rel: str = "train.txt",
                         val_rel: str = "val.txt") -> str:
    """data.yaml text with class names sourced from the 55-class hud.DET_NAMES
    (frozen 38 tiles + 17 HUD elements)."""
    lines = [
        "# generated by build_detector_dataset.py — do not edit by hand",
        "# paths are repo-root-relative — run training FROM THE REPO ROOT (portable / tar-and-go)",
        f"path: {_relposix(out_dir)}",
        f"train: {train_rel}",
        f"val: {val_rel}",
        f"nc: {len(DET_NAMES)}",
        "names:",
    ]
    lines += [f"  {i}: '{name}'" for i, name in enumerate(DET_NAMES)]
    return "\n".join(lines) + "\n"


def write_dataset(out_dir: str, train: list, val: list) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for fname, items in (("train.txt", train), ("val.txt", val)):
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(items) + ("\n" if items else ""))
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(build_data_yaml_text(out_dir))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", default=None,
                    help="NAME=YOLODIR:CAPTURE (repeatable). YOLODIR holds images/+labels/.")
    ap.add_argument("--dataset", action="append", default=None,
                    help="versioned dataset dir with games.json (scripts/data/build_datasets.py); "
                         "repeatable — expands to one --data entry per game (yolo). Use several to "
                         "assemble a COMBINED split across versions. Duplicate NAMEs: later wins.")
    ap.add_argument("--val", default="", help="VAL spec 'NAME:k1,k2' or 'NAME:*' (whole game)")
    ap.add_argument("--out", default="datasets/detector")
    args = ap.parse_args()

    entries = []
    for dsd in (args.dataset or []):
        entries += dataset_data_specs(dsd)
    for d in (args.data or []):
        entries.append(parse_data_arg(d))
    if not entries:
        ap.error("need --data and/or --dataset")
    sources = {}
    for name, yolodir, capture in entries:
        if name in sources:
            print(f"note: duplicate game {name!r} — keeping the later spec ({yolodir})")
        sources[name] = (yolodir, capture)
    val_name, val_kyoku = (args.val.split(":", 1) + [""])[:2] if args.val else ("", "")
    val_set = "*" if val_kyoku == "*" else set(val_kyoku.split(",")) if val_kyoku else set()

    train, val = split_images(sources, val_name, val_set)
    write_dataset(args.out, train, val)
    print(f"sources={list(sources)}  val={args.val or '(none)'}")
    print(f"train imgs={len(train)}  val imgs={len(val)}  classes={len(DET_NAMES)}  -> {args.out}")
    if not val:
        print("WARNING: empty val set")


if __name__ == "__main__":
    main()
