"""Pick the BOARD_TOO_SMALL floor from evidence — a self-consistency scale sweep.

There is no held-out golden set yet (the layout is still ``experimental``), so
this measures scale robustness without GT: every screenshot that recognizes
CLEANLY at native size becomes its own reference. The frame is then rescaled so
the FITTED BOARD height hits each target rung, re-recognized with the size floor
disabled, and the semantic draft diff vs the native draft is counted (same
field-diff as the golden evaluator). The floor should sit at the lowest rung
whose diffs are still ~0 — below that, tiles genuinely stop being readable and
rejection is honest.

🔁 recurring tool (PIPELINE.md §4) — threshold-selection evidence, not a
pipeline stage. Rerun whenever the detector/classifier or the floor changes.

Usage:
  PYTHONPATH=. python scripts/eval/sweep_board_floor.py \
      --images 'datasets/v6/*/frames/*.png' --sample 40 \
      --manifest majsoul_eye/recognize/model-manifest.internal-v1.json \
      --device cuda --out out/sweep_board_floor
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics

import cv2
import numpy as np

from majsoul_eye.normalize import locate_anchor
from majsoul_eye.recognize.runtime import (
    RecognitionContext, RecognitionRuntime, RuntimeFailure,
)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from eval_what_cut_goldens import modification_count, semantic_fields  # noqa: E402


DEFAULT_HEIGHTS = (720, 708, 680, 650, 620, 560, 500)


def recognize(runtime, image, *, tag: str):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeFailure("INVALID_IMAGE", "re-encode failed")
    body = encoded.tobytes()
    digest = hashlib.sha256(body).hexdigest()
    context = RecognitionContext(f"sweep:{tag}", f"sweep:{digest[:16]}", digest,
                                 runtime.manifest.layout_id, True, None)
    recognized = runtime.recognize_bytes(body, context)
    rebuilt = runtime.reconstruct_draft(
        recognized["draft"], recognized["draft"]["revision"])
    blocking = [i for i in (*recognized["issues"], *rebuilt["issues"])
                if i["severity"] == "blocking"]
    entered = bool(rebuilt["ok"]) and not blocking
    return recognized["draft"], entered


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True,
                    help="glob for candidate screenshots (native-size references)")
    ap.add_argument("--sample", type=int, default=40,
                    help="max reference screenshots to use (evenly spaced)")
    ap.add_argument("--heights", default=",".join(str(h) for h in DEFAULT_HEIGHTS))
    ap.add_argument("--manifest",
        default="majsoul_eye/recognize/model-manifest.internal-v1.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="out/sweep_board_floor")
    ap.add_argument("--eye-revision",
                    default=os.environ.get("EYE_REVISION", "sweep-floor"))
    args = ap.parse_args()
    heights = [int(h) for h in args.heights.split(",")]

    runtime = RecognitionRuntime.from_manifest(
        args.manifest, device=args.device, eye_revision=args.eye_revision,
        evaluation_mode=True)
    runtime.warmup()
    # The sweep must be able to descend BELOW the current floor to find where
    # readability actually ends; the floor under test is the output, not a gate.
    layout = runtime.manifest.raw["layout"]
    layout["minBoardWidth"], layout["minBoardHeight"] = 2, 2

    paths = sorted(glob.glob(args.images, recursive=True))
    if not paths:
        raise SystemExit(f"no images match {args.images}")
    if args.sample and len(paths) > args.sample:
        step = len(paths) / args.sample
        paths = [paths[int(i * step)] for i in range(args.sample)]

    per_height: dict[int, dict[str, list]] = {
        h: {"edits": [], "entered": []} for h in heights}
    references = 0
    skipped = {"unlocatable": 0, "not_clean_native": 0, "board_too_short": 0}
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        dets = runtime.detector.predict(image)
        found = locate_anchor(image, dets)
        if found is None:
            skipped["unlocatable"] += 1
            continue
        native_bh = found.region.bh
        try:
            reference, entered = recognize(runtime, image, tag=path)
        except RuntimeFailure:
            entered = False
        if not entered:
            skipped["not_clean_native"] += 1
            continue
        references += 1
        for target in heights:
            if target >= native_bh:            # never upscale: that tests nothing
                skipped["board_too_short"] += 1
                continue
            factor = target / native_bh
            resized = cv2.resize(image, None, fx=factor, fy=factor,
                                 interpolation=cv2.INTER_AREA)
            try:
                draft, sub_entered = recognize(
                    runtime, resized, tag=f"{path}@{target}")
                edits = modification_count(draft, reference)
            except RuntimeFailure:
                sub_entered, edits = False, len(semantic_fields(reference))
            per_height[target]["entered"].append(sub_entered)
            per_height[target]["edits"].append(edits)

    report = {"references": references, "skipped": skipped, "rungs": {}}
    for target in heights:
        entered = per_height[target]["entered"]
        edits = per_height[target]["edits"]
        if not entered:
            continue
        report["rungs"][target] = {
            "n": len(entered),
            "enteredRate": round(sum(entered) / len(entered), 4),
            "medianEdits": statistics.median(edits),
            "p90Edits": sorted(edits)[max(0, -(-len(edits) * 9 // 10) - 1)],
            "meanEdits": round(statistics.fmean(edits), 3),
        }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
