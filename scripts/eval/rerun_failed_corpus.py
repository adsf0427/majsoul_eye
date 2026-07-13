"""Replay a failed-upload corpus through the CURRENT runtime — the acceptance
gate for recognition-gate changes.

Input is a corpus snapshot as archived from prod (see corpus/*/README.md):
``<corpus>/images/<draft_id>.<ext>`` plus ``<corpus>/metadata.jsonl`` rows with
at least ``draft_id`` and ``error_code`` (the code prod rejected it with).

For every image this reruns ``recognize_bytes`` (+ ``reconstruct_draft``) and
reports, per prior error code, what the current runtime does instead:

  outcome "draft+entered"  — draft produced, reconstruct ok, no blocking issue
          "draft+editable" — draft produced but blocked/incomplete (user can fix)
          "<FAILURE_CODE>" — still hard-rejected, possibly with a different code

Optionally renders an overlay per image (fitted board rect + required-region
boxes) for human spot-checks of the localization itself.

🔁 recurring tool (PIPELINE.md §4) — NOT a pipeline stage; paired with the
prod-side extended failed-draft retention this is the long-term failure-corpus
feedback loop.

Usage:
  PYTHONPATH=. python scripts/eval/rerun_failed_corpus.py \
      --corpus corpus/failed-uploads-20260713 \
      --manifest majsoul_eye/recognize/model-manifest.internal-v1.json \
      --device cuda --out out/rerun_failed_corpus --overlays
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import time

import cv2
import numpy as np

from majsoul_eye.normalize import PANEL_LANDMARKS, clipped_sides, locate_anchor
from majsoul_eye.recognize.runtime import (
    RecognitionContext, RecognitionRuntime, RuntimeFailure,
)


def load_metadata(corpus: str) -> list[dict]:
    rows = []
    for line in open(os.path.join(corpus, "metadata.jsonl"), encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def image_path_for(corpus: str, draft_id: str) -> str | None:
    matches = glob.glob(os.path.join(corpus, "images", f"{draft_id}.*"))
    return matches[0] if len(matches) == 1 else None


def _required_boxes():
    """Normalized canonical boxes for the hand row + center panel (as gated)."""
    from majsoul_eye.coords import HAND, HUD_SEEDS

    hand = [HAND.slot_box(0), HAND.slot_box(13, is_tsumo=True)]
    panel = [HUD_SEEDS[name] for name in PANEL_LANDMARKS]
    def union(boxes):
        return (min(b.x0 for b in boxes), min(b.y0 for b in boxes),
                max(b.x1 for b in boxes), max(b.y1 for b in boxes))
    return {"hand": union(hand), "panel": union(panel)}


def render_overlay(image, dets, out_path: str, label: str) -> None:
    """Fitted rect (orange) + required-region boxes (cyan) + outcome label."""
    canvas = image.copy()
    found = locate_anchor(image, dets)
    if found is not None:
        r = found.region
        cv2.rectangle(canvas, (r.ox, r.oy), (r.ox + r.bw, r.oy + r.bh),
                      (0, 165, 255), 3)
        for fx0, fy0, fx1, fy1 in _required_boxes().values():
            cv2.rectangle(canvas,
                          (int(r.ox + fx0 * r.bw), int(r.oy + fy0 * r.bh)),
                          (int(r.ox + fx1 * r.bw), int(r.oy + fy1 * r.bh)),
                          (255, 255, 0), 2)
        label += (f"  fit: res={found.residual:.1f} in={found.inliers}/{found.total}"
                  f" clip={','.join(clipped_sides(r)) or '-'}")
    else:
        label += "  fit: NONE"
    scale = min(1.0, 1280 / canvas.shape[1])
    if scale < 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--manifest",
        default="majsoul_eye/recognize/model-manifest.internal-v1.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="out/rerun_failed_corpus")
    ap.add_argument("--overlays", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eye-revision",
                    default=os.environ.get("EYE_REVISION", "rerun-corpus"))
    args = ap.parse_args()

    runtime = RecognitionRuntime.from_manifest(
        args.manifest, device=args.device, eye_revision=args.eye_revision,
        evaluation_mode=True)
    runtime.warmup()

    rows = load_metadata(args.corpus)
    if args.limit:
        rows = rows[: args.limit]
    os.makedirs(args.out, exist_ok=True)
    if args.overlays:
        os.makedirs(os.path.join(args.out, "overlays"), exist_ok=True)

    results = []
    matrix: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows:
        draft_id, prior = row["draft_id"], row["error_code"]
        path = image_path_for(args.corpus, draft_id)
        if path is None:
            matrix[prior]["MISSING_IMAGE"] += 1
            continue
        body = open(path, "rb").read()
        digest = hashlib.sha256(body).hexdigest()
        context = RecognitionContext(
            f"rerun:{draft_id}", draft_id, digest, runtime.manifest.layout_id,
            True, None)
        started = time.monotonic()
        issues: list = []
        try:
            recognized = runtime.recognize_bytes(body, context)
            issues = recognized["issues"]
            rebuilt = runtime.reconstruct_draft(
                recognized["draft"], recognized["draft"]["revision"])
            blocking = [i for i in (*issues, *rebuilt["issues"])
                        if i["severity"] == "blocking"]
            outcome = ("draft+entered" if rebuilt["ok"] and not blocking
                       else "draft+editable")
        except RuntimeFailure as exc:
            outcome = exc.code
        except Exception as exc:  # noqa: BLE001 — a crash is a result here, not a bug escape
            outcome = f"ERROR:{type(exc).__name__}"
        elapsed_ms = int((time.monotonic() - started) * 1000)
        matrix[prior][outcome] += 1
        results.append({"draftId": draft_id, "priorCode": prior,
                        "outcome": outcome,
                        "issueCodes": sorted({i["code"] for i in issues}),
                        "width": row.get("width"), "height": row.get("height"),
                        "ms": elapsed_ms})
        if args.overlays:
            image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                dets = runtime.detector.predict(image)
                render_overlay(image, dets,
                               os.path.join(args.out, "overlays", f"{draft_id}.jpg"),
                               f"{prior} -> {outcome}")

    with open(os.path.join(args.out, "results.jsonl"), "w", encoding="utf-8") as fh:
        for item in results:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    recovered = sum(count for outcomes in matrix.values()
                    for outcome, count in outcomes.items()
                    if outcome.startswith("draft"))
    total = sum(sum(outcomes.values()) for outcomes in matrix.values())
    summary = {"total": total, "recovered": recovered,
               "matrix": {prior: dict(outcomes)
                          for prior, outcomes in sorted(matrix.items())}}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
