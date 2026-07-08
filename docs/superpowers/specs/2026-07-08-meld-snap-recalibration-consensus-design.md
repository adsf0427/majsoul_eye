# Meld-snap stability: corner recalibration + per-round consensus

**Date:** 2026-07-08
**Status:** Design — awaiting user review
**Scope:** `majsoul_eye/annotate/` (meld/fuuro box placement) + the dataset builders

## 1. Problem

Reviewing the `backs_review` FiftyOne dataset, the user found the opponents' **副露
(furo/meld)** boxes are frequently mis-placed:

- **上家 (pos3 / kamicha): severe mis-placement** — the whole strip lands a full
  tile off in many rounds.
- **对家 (pos2 / toimen): adjacent frames disagree** — two consecutive frames with
  an *unchanged* meld area get very different boxes.

This contradicts the expectation that meld annotation is "fairly reliable."

## 2. Root cause (measured, not assumed)

Meld boxes are produced in two stages:

1. **Generate** (`generate_meld_boxes_v2`) — GT-driven, deterministic. Places a
   template strip from a calibrated outer corner (`MELD_STRIP2[seat]`). **Reliable.**
2. **Per-frame snap** (`snap_meld_strip`) — a discrete 2-DOF rigid snap that slides
   the template onto the current frame's face/back masks, to absorb Majsoul's
   per-round ~½-tile float. **This is the fragile step; every reported defect is here.**

Diagnostics over `ai_session` + `ai_session3` (~3.5k confident meld frames/seat,
snap measured directly) isolate **two layers**:

### Layer A — a systematic corner mis-calibration (the dominant defect)

Per-seat dominant snap offset (= how far each corner is off) and mislock rate
(fraction of confident frames *outside* the dominant cluster):

| seat | along offset | along mislock | cross offset | cross mislock | reading |
|------|-------------:|--------------:|-------------:|--------------:|---------|
| pos0 self    | ~0   | 0%      | **+46** | 0%   | benign systematic cross offset (snap always corrects) |
| pos1 下家    | ~0   | 3%      | ~0      | 0%   | clean |
| pos2 对家    | ~1   | 2%      | ~0      | 2%   | well-calibrated; rare per-round mislocks |
| pos3 上家    | **+46** | **26%** | ~0   | 0%   | **half-tile along offset → snap sits at aliasing midpoint** |

pos3's corner is off by **~+46px ≈ half a tile depth** along the strip. A half-tile
offset is the *worst case* for the snap: the strip of equal-pitch tiles is
self-similar, so a lock shifted one whole tile scores almost as high as the correct
lock. The snap therefore flips between the two ~26% of the time → the "上家严重失位."

### Layer B — residual per-frame/per-round snap flips

Even the well-calibrated seats mislock ~1–3% of frames (single-frame or
single-round). Example: `run_8_game4` pos2 held `dc=+24` (the correct per-round
cross float) for a stretch, then one frame flipped to `dc=−0.5` and back → the
"对家相邻帧差别." This is genuine per-round float plus occasional snap noise; it is
independent of Layer A.

### Verification of the fix direction

Pre-shifting pos3's strip by +46 before snapping (i.e. simulating a recalibrated
corner) collapses its behaviour to the good-seat level:

```
pos3, current corner:      mislock 25.9%   (dominant-cluster frac 0.69)
pos3, corner + 46 along:   mislock  0.9%   (dominant-cluster frac 0.94)
```

The 74% of pos3 frames that already snap correctly **do not move** (they currently
emit corner+snap(+46); after recalibration they emit corner(+46)+snap(≈0) — same
pixels). So recalibration is low-risk to already-correct data.

`calibrate_annotation_model.py` already derives corners as
`corner + along·median(d_along) + cross·median(d_cross)`; because the current pos3
median is +46, **re-running the existing calibrator reproduces exactly this
correction**. The corner is stale (never re-fit after a post-2026-07-02 warp/mask
change), not methodologically broken.

## 3. Design

Two phases, each independently shippable. Phase 1 removes the severe defect with a
minimal, verified change; Phase 2 removes the residual and the cross-frame flicker.

### Phase 1 — Recalibrate the meld corners (fixes 上家严重失位)

- Re-run `scripts/annotate/calibrate_annotation_model.py` over the current AI
  sessions; apply the suggested `MELD_STRIP2[seat].corner` for every seat.
  Expected deltas: **pos3 +46 along, pos0 +46 cross**, pos1/pos2 ≈0.
- Add a QA guard (reuse the `agg_offset` diagnostic from this investigation): after
  recalibration, each seat's snap mislock rate must be < 3% and dominant offset ≈ 0.
  Keep this as a periodic check — the stale corner is a maintenance gap; a warp/mask
  change can re-introduce an offset silently.
- No pipeline-shape change; only calibration constants move. Datasets must be
  rebuilt (see §6).

**Note:** Phase 1 does NOT help pos2 (already well-calibrated) — the 对家 flicker is
a Layer-B problem, addressed in Phase 2.

### Phase 2 — Per-round snap consensus (fixes 对家相邻帧差别 + pos3 residual)

The strip is physically fixed within a kyoku (STATUS: within-round σ≈0.9px). So the
correct `(d_along, d_cross)` is ~constant per `(game, bakaze, kyoku, honba, seat)`.
Per-frame independent snapping discards the strongest signal — agreement across the
round. Consensus is *safe only after Phase 1*: pre-recalibration, pos3 had large
*consistent* mislock blocks (32 frames) that could out-vote the truth; post-recal the
true offset dominates (frac 0.94), so a confidence-weighted vote is robust.

**New module `majsoul_eye/annotate/meldsnap.py`** (keeps `frame.py` lean):

- `measure_meld_snaps(img, state, hom) -> {seat: (d_along, d_cross, score, n)}`
  — the warp + meld masks + `snap_meld_strip` only (no river/hand/dora/hud), for the
  measurement pass.
- `round_meld_consensus(samples) -> {(kyoku_key, seat): (d_along, d_cross, conf)}`
  — per group, take the **confidence-weighted dominant cluster** (cluster width
  ≈ ⅓ tile) of the per-frame offsets; consensus = weighted median of that cluster;
  `conf` = cluster weight / total. Below a confidence floor → `None`.
- Constants: `CLUSTER_TOL`, `MIN_ROUND_CONF`, `MIN_ROUND_FRAMES`.

**`annotate_frame` refactor** — split the meld block into *measure* and *apply*, add
`meld_snap_override: dict[int, tuple] | None = None`:
- `None` (default) → current per-frame snap behaviour. Backward-compatible for
  `annotate_ai_session --qa`, `overlay_*`, and any single-frame caller.
- provided → skip the per-frame snap, shift by `override[seat]`, recompute fills and
  reliability against the shifted boxes.

**Build orchestration** (`build_datasets.py` and `build_backs_review.py`) becomes
per-game two-pass:
1. Pass 1: `measure_meld_snaps` for every usable frame → collect by round key.
2. `round_meld_consensus` → per-(kyoku,seat) offset.
3. Pass 2: `annotate_frame(..., meld_snap_override=offsets_for_this_frame)`.

Cost: ~one extra warp per frame in Pass 1 (measurement is lighter than full
annotate). A fused single-warp variant is possible later if rebuild time matters.

### Low-confidence rounds

When `round_meld_consensus` returns `None` for a `(kyoku, seat)` (no frame with
strong enough meld edges — heavy occlusion / contrast-killing skin):
**use the template (offset 0) but mark those meld boxes `reliable=False`** (dropped
from YOLO labels by the existing `if b.reliable` filter, exactly like the
deal/call-window policy) **and emit a `meld:low_round_conf` flag** so
`build_backs_review` / FiftyOne surfaces them for spot-check. *(Decision made on the
user's behalf while away — see §7; override candidate: keep-and-emit template boxes.)*

## 4. Data flow

```
frame + GT state
  └─ Pass 1: measure_meld_snaps ──► samples[(game,bakaze,kyoku,honba,seat)] += (da,dc,score,n)
                                        │
                              round_meld_consensus
                                        │
                                        ▼  {(kyoku,seat): (da,dc,conf)}
  └─ Pass 2: annotate_frame(meld_snap_override=…) ──► shifted meld boxes + fills + reliability
```

## 5. Testing / QA

- **Unit** (`tests/test_meldsnap.py`): `round_meld_consensus` on synthetic
  distributions — majority+outliers → majority; all-scattered / too-few → `None`;
  the pos3-style `{46:31, −19:32}` block → picks +46 only when weighted toward the
  denser/higher-score cluster (documents the post-recal safety assumption).
- **Regression metric**: per `(kyoku,seat)`, max intra-kyoku *emitted* offset jump
  → assert ≈0 on a sample after the fix (the `snap_diag` script from this
  investigation, gated to emitted boxes).
- **Recalibration QA**: `agg_offset` mislock rate < 3% / seat, offset ≈ 0.
- **Visual**: re-render the known-bad frames (`run_8_game6` seq1714/1715 pos3;
  `run_8_game4` seq219/220 pos2) and confirm boxes sit on tiles.
- Full existing suite stays green.

## 6. Pipeline impact (per CLAUDE.md discipline)

- **Stales all derived data**: `MELD_STRIP2` corners + the new consensus change every
  meld box → rebuild affected datasets (`build_datasets.py <v> --force`) and the
  `backs_review` QA set. Note in the rebuild that classifier/detector retrain follows.
- **Changes a build step**: builds become two-pass → update `docs/PIPELINE.md` build
  stage + add a `docs/STATUS.md` §1.5x entry. Classify `meldsnap.py` as pipeline lib.
- `recognize/` untouched (stays Akagi-free; this is annotation-side only).

## 7. Decisions made on the user's behalf (please confirm)

1. **Low-conf rounds** → mark unreliable + drop + QA flag (§3). Alternative:
   keep-template-and-emit (max data, small systematic-offset risk).
2. **Phase split** — recommend doing **Phase 1 now** (verified 26%→0.9%, minimal
   change) and gating **Phase 2** on a look at the post-Phase-1 review set; but both
   are needed to fully close *both* reported symptoms (Phase 1 → 上家, Phase 2 → 对家).

## 8. Open questions

- Confirm the +46 pos3 / +46 pos0 deltas on a non-1080p / 4K session (melds scale
  with the table, and builds resize to 1920 first, so expected to hold — verify).
- Should the `agg_offset` mislock guard become a committed test / CI check to prevent
  silent corner drift after future warp/mask changes? (Recommended.)
