"""Single-frame board recognition CLI — screenshot(s) in, JSON out.

Wraps the manifest-bound runtime chain (Akagi-free): one
``RecognitionRuntime.from_manifest`` load (detector + classifier + HUD reader,
fixed SHA-verified assets), then per image ``recognize_bytes`` ->
override-aware ``reconstruct_draft`` -> legal hero-perspective mjai sequence.
🔁 recurring tool (PIPELINE.md §4) — NOT a pipeline stage; the library chain it
wraps is the shipped product, this file is just an entrypoint for external
callers (other processes / languages / quick inspection). The production worker
is ``scripts/recognize/serve_worker.py``; this CLI shares its exact runtime.

Model selection is manifest-first: ``--manifest`` names one immutable asset set
(no mtime-based weight guessing). Detector experiments go through an explicit
alternate manifest, never loose ``--weights`` paths.

One JSON object per input image on stdout (JSON lines; --pretty to indent):

  {"file": ..., "ok": bool,            # ok = draft reconstructed (or --no-reconstruct)
   "violations": [...],                # recognition issue codes (blocking => draft incomplete)
   "observed": {...} | null,           # ObservedState projection of the recognized draft
   "draft": {...},                     # WhatCutDraftV1 (schemaVersion 1, recognizer meta, evidence)
   "mjai": [...] | null,               # start_game..now; hero abs seat = start_game.id
   "fabricated": {...} | null,         # what reconstruct invented (hidden draws, defaults)
   "reason": [...] | null}             # reconstruct issues (ok=false)

Usage:
  PYTHONPATH=. python scripts/recognize/recognize_frame.py shot.png
  PYTHONPATH=. python scripts/recognize/recognize_frame.py --device cuda \
      --manifest majsoul_eye/recognize/model-manifest.internal-v1.json frames/*.png
  # experimental (not-yet-promoted) layout support:
  PYTHONPATH=. python scripts/recognize/recognize_frame.py --allow-experimental shot.png
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="screenshot path(s)")
    ap.add_argument("--manifest",
        default="majsoul_eye/recognize/model-manifest.internal-v1.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eye-revision", default=os.environ.get("EYE_REVISION", "local-cli"))
    ap.add_argument("--allow-experimental", action="store_true")
    ap.add_argument("--no-reconstruct", action="store_true",
                    help="stop after recognition (skip mjai synthesis)")
    ap.add_argument("--pretty", action="store_true", help="indent JSON output")
    args = ap.parse_args()

    from majsoul_eye.recognize.runtime import (
        RecognitionContext, RecognitionRuntime,
    )
    from majsoul_eye.what_cut.adapter import draft_to_observed

    runtime = RecognitionRuntime.from_manifest(
        args.manifest, device=args.device, eye_revision=args.eye_revision)
    runtime.warmup()
    for path in args.images:
        image_bytes = open(path, "rb").read()
        digest = hashlib.sha256(image_bytes).hexdigest()
        context = RecognitionContext(f"cli:{path}", f"cli:{digest[:16]}", digest,
                                     runtime.manifest.layout_id,
                                     args.allow_experimental, None)
        recognized = runtime.recognize_bytes(image_bytes, context)
        adapted = draft_to_observed(recognized["draft"])
        observed = dataclasses.asdict(adapted.observed) if adapted.observed else None
        violations = [issue["code"] for issue in recognized["issues"]]
        if args.no_reconstruct:
            out = {"file": path, "ok": True, "violations": violations,
                   "observed": observed, "draft": recognized["draft"],
                   "mjai": None, "fabricated": None, "reason": None}
        else:
            rebuilt = runtime.reconstruct_draft(recognized["draft"], 0)
            out = {"file": path, "ok": rebuilt["ok"], "violations": violations,
                   "observed": observed, "draft": recognized["draft"],
                   "mjai": rebuilt["mjai"], "fabricated": rebuilt["fabricated"],
                   "reason": None if rebuilt["ok"] else rebuilt["issues"]}
        print(json.dumps(out, ensure_ascii=False,
                         indent=2 if args.pretty else None), flush=True)


if __name__ == "__main__":
    main()
