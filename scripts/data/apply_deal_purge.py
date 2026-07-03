"""Apply a deal-frame purge to a datasets/ tree from a precomputed manifest.

PORTABLE companion to purge_deal_frames.py, for a machine that has the transferred
``datasets/`` but NOT the ``majsoul_eye`` package or ``captures/`` GT (e.g. a GPU
training server). Pure stdlib — no imports from this repo.

On the SOURCE machine (where captures/ live), generate the manifest:
    PYTHONPATH=. python scripts/data/purge_deal_frames.py --write-manifest deal_manifest.json
Copy ``deal_manifest.json`` + this file to the server (next to ``datasets/``) and:
    python apply_deal_purge.py deal_manifest.json                 # dry-run (preview)
    python apply_deal_purge.py deal_manifest.json --apply

Manifest = ``{"precise_<name>": [seq, ...], ...}``. For each dataset it deletes the
deal-window artifacts ``crops/<tile>/<seq>_*.png`` + ``yolo/images/<seq>.png`` +
``yolo/labels/<seq>.txt``, then drops the matching image lines from every
``datasets/detector*/{train,val}.txt`` (matched by seq, so the preview is exact and
it does not depend on the files being deleted first). Idempotent; run from the dir
that holds ``datasets/`` (same relativity the detector lists use).
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def line_is_deal(line: str, manifest: dict) -> bool:
    """A detector-list image line -> True if it is a deal-window frame.

    Line looks like ``datasets/precise_<name>/yolo/images/<seq>.png`` (repo-root-
    relative POSIX). Match the ``precise_<name>`` dir + the 6-digit seq basename.
    """
    parts = line.replace("\\", "/").split("/")
    if "yolo" not in parts:
        return False
    ds = parts[parts.index("yolo") - 1]              # precise_<name>
    try:
        seq = int(os.path.splitext(parts[-1])[0])
    except ValueError:
        return False
    return seq in set(manifest.get(ds, []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="deal_manifest.json from purge_deal_frames.py --write-manifest")
    ap.add_argument("--datasets-dir", default="datasets")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    tot_crops = tot_imgs = tot_lbls = 0
    for ds_name, seqs in sorted(manifest.items()):
        ds = os.path.join(args.datasets_dir, ds_name)
        if not os.path.isdir(ds):
            continue                                 # this game not present on this box
        victims = []
        for seq in seqs:
            s6 = f"{int(seq):06d}"
            victims += glob.glob(os.path.join(ds, "crops", "*", f"{s6}_*.png"))
            for sub, ext in (("yolo/images", "png"), ("yolo/labels", "txt")):
                p = os.path.join(ds, sub, f"{s6}.{ext}")
                if os.path.exists(p):
                    victims.append(p)
        n_crop = sum(1 for v in victims if os.sep + "crops" + os.sep in v)
        n_img = sum(1 for v in victims if v.endswith(".png") and "images" in v)
        n_lbl = sum(1 for v in victims if v.endswith(".txt"))
        tot_crops += n_crop; tot_imgs += n_img; tot_lbls += n_lbl
        print(f"{ds_name}: deal seqs={len(seqs)}  crops={n_crop} yolo-img={n_img} yolo-lbl={n_lbl}")
        if args.apply:
            for v in victims:
                os.remove(v)

    # Drop deal-frame image lines from the assembled detector split(s), matched by seq.
    for lst in glob.glob(os.path.join(args.datasets_dir, "detector*", "*.txt")):
        if os.path.basename(lst) not in ("train.txt", "val.txt"):
            continue
        with open(lst, encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        kept = [ln for ln in lines if not line_is_deal(ln, manifest)]
        dropped = len(lines) - len(kept)
        if dropped:
            print(f"{lst}: drop {dropped} deal-image lines ({len(lines)} -> {len(kept)})")
            if args.apply:
                with open(lst, "w", encoding="utf-8") as f:
                    f.write("\n".join(kept) + ("\n" if kept else ""))

    mode = "DELETED" if args.apply else "would delete (dry-run; pass --apply)"
    print(f"\nTOTAL {mode}: crops={tot_crops}  yolo-images={tot_imgs}  yolo-labels={tot_lbls}")


if __name__ == "__main__":
    main()
