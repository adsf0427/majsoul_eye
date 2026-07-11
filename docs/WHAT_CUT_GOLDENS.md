# What-Cut Desktop Golden Set

Each JSONL row contains exactly: `schemaVersion=1`, one shared `datasetVersion`,
`sampleId`, `gameId`, a dataset-relative `imagePath`, original-byte
`imageSha256`, dataset-relative `expectedDraftPath`, and
`independentOfTraining=true`. Expected drafts use `WhatCutDraftV1`; semantic
evaluation excludes IDs, revisions, evidence, annotations, recognizer metadata,
baseline fields and provenance, but includes current visible/ghost values.

The set is independent: none of its games or derived frames may appear in a
classifier, detector, HUD-reader, calibration, threshold-selection or training
dataset. The evaluator rejects the whole candidate set if any pair has 64-bit
dHash Hamming distance at most 4; remove one member of every such pair and run
again. The resulting checked set must contain at least 100 screenshots from at
least 20 games. `structuralEntryRate` counts only responses with no blocking
recognition/reconstruction issue; editable but incomplete drafts remain useful
for diagnosis, but do not satisfy this accuracy metric.

Run the fixed experimental baseline:

    PYTHONPATH=. python scripts/eval/eval_what_cut_goldens.py \
      --manifest majsoul_eye/recognize/model-manifest.internal-v1.json \
      --goldens goldens/what-cut/majsoul-desktop-16x9-v1/goldens.jsonl \
      --out majsoul_eye/recognize/model-manifest.internal-v1.accuracy.json \
      --device cuda --eye-revision "$EYE_REVISION"

Promotion sequence:

1. Collect and correct the held-out set until it meets count and independence rules.
2. Change only `supportStatus` from `experimental` to `supported`; keep model,
   inference, layout and candidate settings unchanged.
3. Run the evaluator against that final manifest so the report records its full SHA-256.
4. Keep the report and detached `.sha256` beside the manifest.
5. Run `serve_worker.py --check-only`; readiness must reject a failed, missing,
   stale, or differently configured report.

Passing thresholds are immutable: structural editor-entry rate >=0.95, median
modified fields =0, and p90 modified fields <=2. Lowering rejection or changing
inference after evaluation changes the manifest SHA and invalidates the report.
The report must also contain `comparisonVersion=what-cut-semantic-v1` and the
exact count/rate/edit/dHash threshold snapshot; readiness rejects either drift.
