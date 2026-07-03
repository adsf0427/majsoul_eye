"""Delete deal-in animation (rivers-empty) frame artifacts from PRE-FIX datasets.

The build/annotate pipeline now drops these frames automatically at build time
(``majsoul_eye.state.replay.is_deal_window`` — a kyoku has started but no discard
has happened yet, so the hero hand is mid-deal/sort and GT boxes don't match the
pixels). This one-time tool cleans datasets that were built BEFORE that fix.

For each ``datasets/precise_<name>/`` it replays the matching GT capture, finds the
deal-window seqs, and removes their classifier crops (``crops/<tile>/<seq>_*.png``)
+ YOLO image/label (``yolo/images/<seq>.png`` / ``yolo/labels/<seq>.txt``). It then
rewrites every ``datasets/detector*/{train,val}.txt`` to drop image lines whose file
no longer exists (so the assembled detector split stays valid).

Capture resolution: ``precise_<name>`` -> ``captures/intermediate/gt/<name>.jsonl``
(AI games) or ``captures/raw/manual/<name>.jsonl`` (manual sessions).

Dry-run by DEFAULT (prints what it WOULD delete); pass ``--apply`` to delete.
Idempotent: a second run finds nothing to remove.

  PYTHONPATH=. $PY scripts/data/purge_deal_frames.py            # dry-run
  PYTHONPATH=. $PY scripts/data/purge_deal_frames.py --apply
"""
from __future__ import annotations

import argparse
import glob
import os

from majsoul_eye.capture.schema import read_records
from majsoul_eye.capture.sync import RELEVANT_EVENTS
from majsoul_eye.state.replay import Replayer, is_deal_window


def deal_window_seqs(capture: str) -> set[int]:
    """seqs whose reconstructed BoardState is in the deal-in window (rivers-empty)."""
    rp = Replayer()
    out: set[int] = set()
    for r in read_records(capture):
        rp.apply_record(r)
        if r.mjai and any(e.get("type") in RELEVANT_EVENTS for e in r.mjai):
            if is_deal_window(rp.state):
                out.add(r.seq)
    return out


def resolve_capture(name: str) -> str | None:
    """precise_<name> stem -> its GT capture jsonl (AI gt/ or manual/)."""
    for cand in (os.path.join("captures", "intermediate", "gt", f"{name}.jsonl"),
                 os.path.join("captures", "raw", "manual", f"{name}.jsonl")):
        if os.path.exists(cand):
            return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-dir", default="datasets")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    ap.add_argument("--write-manifest", metavar="PATH", default=None,
                    help="also write {precise_<name>: [deal seqs]} JSON for apply_deal_purge.py "
                         "(portable purge on a machine that has only datasets/, no captures/).")
    args = ap.parse_args()

    import json
    manifest: dict[str, list[int]] = {}
    tot_crops = tot_imgs = tot_lbls = 0
    for ds in sorted(glob.glob(os.path.join(args.datasets_dir, "precise_*"))):
        name = os.path.basename(ds)[len("precise_"):]
        cap = resolve_capture(name)
        if cap is None:
            print(f"{name}: SKIP (no capture jsonl found)")
            continue
        seqs = deal_window_seqs(cap)
        manifest[os.path.basename(ds)] = sorted(seqs)
        victims = []
        for seq in seqs:
            s6 = f"{seq:06d}"
            victims += glob.glob(os.path.join(ds, "crops", "*", f"{s6}_*.png"))
            for sub, ext in (("yolo/images", "png"), ("yolo/labels", "txt")):
                p = os.path.join(ds, sub, f"{s6}.{ext}")
                if os.path.exists(p):
                    victims.append(p)
        n_crop = sum(1 for v in victims if os.sep + "crops" + os.sep in v)
        n_img = sum(1 for v in victims if v.endswith(".png") and "images" in v)
        n_lbl = sum(1 for v in victims if v.endswith(".txt"))
        tot_crops += n_crop; tot_imgs += n_img; tot_lbls += n_lbl
        print(f"{name}: deal seqs={sorted(seqs)}  crops={n_crop} yolo-img={n_img} yolo-lbl={n_lbl}")
        if args.apply:
            for v in victims:
                os.remove(v)

    # Fix assembled detector splits: drop lines whose image no longer exists.
    for lst in glob.glob(os.path.join(args.datasets_dir, "detector*", "*.txt")):
        if os.path.basename(lst) not in ("train.txt", "val.txt"):
            continue
        with open(lst, encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        kept = [ln for ln in lines if os.path.exists(ln)]
        dropped = len(lines) - len(kept)
        if dropped:
            print(f"{lst}: drop {dropped} missing-image lines ({len(lines)} -> {len(kept)})")
            if args.apply:
                with open(lst, "w", encoding="utf-8") as f:
                    f.write("\n".join(kept) + ("\n" if kept else ""))

    if args.write_manifest:
        with open(args.write_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
        print(f"wrote manifest ({sum(len(v) for v in manifest.values())} deal seqs) -> {args.write_manifest}")

    mode = "DELETED" if args.apply else "would delete (dry-run; pass --apply)"
    print(f"\nTOTAL {mode}: crops={tot_crops}  yolo-images={tot_imgs}  yolo-labels={tot_lbls}")


if __name__ == "__main__":
    main()
