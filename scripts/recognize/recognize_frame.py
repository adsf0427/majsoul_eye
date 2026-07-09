"""Single-frame board recognition CLI — screenshot(s) in, JSON out.

Wraps the runtime chain (Akagi-free): ``TileDetector`` -> ``assemble`` ->
``ObservedState`` -> ``reconstruct`` -> legal hero-perspective mjai sequence.
🔁 recurring tool (PIPELINE.md §4) — NOT a pipeline stage; the library chain it
wraps is the shipped product, this file is just an entrypoint for external
callers (other processes / languages / quick inspection).

One JSON object per input image on stdout (JSON lines; --pretty to indent):

  {"file": ..., "ok": bool,            # ok = frame accepted AND reconstructed
   "violations": [...],                # non-empty => frame rejected, rest null
   "observed": {...} | null,           # ObservedState (hand/rivers/melds/dora/reach;
                                       #  HUD fields null until HudReader lands)
   "mjai": [...] | null,               # start_game..now; hero abs seat = start_game.id
   "fabricated": {...} | null,         # what reconstruct invented (haipai, defaults)
   "reason": str | null}               # why reconstruct failed (ok=false, no violations)

Usage:
  PYTHONPATH=. python scripts/recognize/recognize_frame.py shot.png
  PYTHONPATH=. python scripts/recognize/recognize_frame.py --device cuda \
      --weights weights/detector/tile_detector_obb_20260709_055509.pt frames/*.png
  # letterboxed (non-16:9) screenshots:
  PYTHONPATH=. python scripts/recognize/recognize_frame.py --letterbox shot.png
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys


def newest_obb_weights() -> str | None:
    cands = glob.glob("weights/detector/tile_detector_obb_*.pt")
    return max(cands, key=os.path.getmtime) if cands else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="screenshot path(s)")
    ap.add_argument("--weights", default=None,
                    help="detector weights (default: newest weights/detector/"
                         "tile_detector_obb_*.pt by mtime)")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--letterbox", action="store_true",
                    help="force letterbox trim (default auto-detects by aspect: "
                         "~16:9 fullscreen / wider phone / narrower letterbox)")
    ap.add_argument("--no-reconstruct", action="store_true",
                    help="stop after ObservedState (skip mjai synthesis)")
    ap.add_argument("--pretty", action="store_true", help="indent JSON output")
    args = ap.parse_args()

    weights = args.weights or newest_obb_weights()
    if not weights:
        sys.exit("no --weights given and no weights/detector/tile_detector_obb_*.pt found")
    print(f"[recognize_frame] weights: {weights}  device: {args.device}",
          file=sys.stderr)

    import cv2
    from majsoul_eye.normalize import locate_auto, locate_letterbox
    from majsoul_eye.recognize.assemble import assemble
    from majsoul_eye.recognize.detector import TileDetector
    from majsoul_eye.state.reconstruct import reconstruct

    det = TileDetector(weights, device=args.device)
    # auto: ~16:9 -> fullscreen, wider (phones) -> centered-16:9 board with
    # screen-corner dora rescue, narrower -> letterbox trim
    locate = locate_letterbox if args.letterbox else locate_auto

    for path in args.images:
        out = {"file": path, "ok": False, "violations": None,
               "observed": None, "mjai": None, "fabricated": None, "reason": None}
        img = cv2.imread(path)
        if img is None:
            out["reason"] = "cannot read image"
        else:
            obs = assemble(det.predict(img), locate(img))
            out["violations"] = obs.violations
            out["observed"] = dataclasses.asdict(obs)
            if obs.violations:
                out["reason"] = "frame rejected by consistency gate"
            elif args.no_reconstruct:
                out["ok"] = True
            else:
                r = reconstruct(obs)
                out["ok"] = r.ok
                if r.ok:
                    out["mjai"], out["fabricated"] = r.events, r.fabricated
                else:
                    out["reason"] = r.reason
        print(json.dumps(out, ensure_ascii=False,
                         indent=2 if args.pretty else None), flush=True)


if __name__ == "__main__":
    main()
