# mycv pipeline reproduction & accuracy baseline — design

**Date:** 2026-06-27
**Status:** Approved (approach B)
**Goal:** Faithfully reproduce mycv's full tile-face recognition pipeline inside the
`majsoul_eye` framework and measure its accuracy against Akagi GT on our captured
frames — to get an honest mycv baseline to compare against our own classifier (93.5%).

## Why

Earlier mycv accuracy numbers (64% / 56% / 35%) were invalid: they came from feeding
mycv mismatched inputs (4K grid crops, wrong model, hand-made "isolation"). To measure
mycv's *real* native accuracy we must run its *complete native pipeline* on frames it can
handle, scored against ground truth. mycv is a working bot, so its true accuracy is
expected to be high; this experiment establishes the real number and the per-region
breakdown, and tells us whether mycv's contour-isolation is worth borrowing.

## Scope

**In:** mycv's three neural tile-face pipelines — **hand**, **river**, **meld**.
**Out:** dora / scores / round-info (mycv reads those by template matching, not the tile
nets); the "who called" direction inference in `recpai3` (that is *who*, not *what* — it
does not affect tile-face identity); `reveal()/reveal2()/bq` mask variants (kept default
for a clean baseline, noted as a fidelity caveat).

## mycv pipeline facts (from `../auto/mycv`)

Two classifiers, three segmentation paths, all calibrated to **1080p web-client default skin**:

| Region | Segmentation | Classifier | Preprocess |
|---|---|---|---|
| hand | `getHandTiles`: floodFill from `[235,1002]` along the hand row | `tile.model` (TileNet, 38-class incl. `back`) | Resize 32×32 + ToTensor + Normalize(0.5) |
| river+meld | seat-mask `copyTo` → `cutPic`: white-face `inRange` + `dealContour` geometric filter + per-tile contour isolation (mask neighbours to white) | `myweight.pth` (ResNet b1..b4 + Conv2d(256,8) + Flatten + Linear(512,37), 37-class no `back`) | Resize 60×60 + `cv.split(img)/255` (BGR, **no** mean/std) |

- **River vs meld routing:** `getType(center)` looks up a region-code in `m/mask.png`
  (`typeMask`, resized to 1080p); `dicttype={886:0,532:0,1127:0, 1249:1,537:2,1314:3, 1213:4}`
  → 0=river, 1/2/3=meld(3 directions), 4=hand.
- **Seat selection:** seat masks `m/m{0..3}.png` `copyTo`-fill everything outside one
  seat region with table colour `(95,58,37)`; `cutPic` then runs on that masked frame.
- **Class order:** both classifiers use the **same order as our `tiles.TILE_NAMES`** —
  identity index map (river/meld 0–36 direct; hand 0–37 direct incl. `back`). No remap.
- **Internal resolution:** `cutPic` does `cv.resize(img,(1920,1080))` first; `copyTo` uses
  a 1080p mask. So a fullscreen-16:9 frame at any resolution is downscaled to 1080p before
  segmentation.

## Deviation: approach B → B′ (real-engine adapter)

During context-gathering the original blocker for using mycv's real code (heavy
deps: pyautogui/matplotlib) turned out **not to exist** — they import fine in the
`auto` env, and `myCV()` instantiates. Re-porting mycv's algorithm by hand only
adds risk of a threshold/off-by-one divergence. So we run mycv's **actual code**
(`cutPic`/`model`/`getType`/`getHandTiles`) behind a thin in-framework adapter
(`baselines/mycv_engine.py`). Same spirit as B (faithful reproduction measured in
our framework), zero fidelity risk.

Probe findings that reshaped the measurement (validated on real session6 frames):
- **Self river is outside mycv's vision**: every seat mask fills the self-river
  region with table colour; mycv knows its own discards from its own play. So
  "river" = the **3 opponent seats**. Seat masks `m{1,2,3}` cleanly isolate the
  three opponent rivers; raw mask k → absolute seat `(hero+k)%4`. `m0` is a loose
  combined mask, unused.
- **Seat masks overlap in meld zones** (one pon appears under several masks), so
  meld→seat assignment can't trust "which mask caught it"; meld detections are a
  global de-duped pool, scored against opponent GT melds.
- **Scoring is multiset (bag) matching**, not coordinate matching — fair to mycv
  and free of any dependency on our own (possibly imperfect) coord calibration.

## Architecture (approach B′: real-engine adapter)

A dev-only benchmark, **not** under `recognize/` (preserve the shipped product's
Akagi-free boundary). Files:

1. **`majsoul_eye/baselines/mycv_port.py`** — faithful port of mycv's recognition
   front-end. No pyautogui/matplotlib deps. Loads real assets from `../auto/mycv` by
   absolute path. Public surface:
   - `MycvPort(mycv_dir)` — loads `myweight.pth`, `tile.model`, `m/mask.png`, `m/m{0..3}.png`.
   - `recognize_hand(frame_bgr) -> list[(name, (x,y))]`
   - `recognize_seat(frame_bgr, seat_mask_idx) -> {"river":[(name,(x,y))], "meld":[(name,(x,y))]}`
     (replicates the shared `cutPic → classify → getType` core of recpai2/recpai3).
   - Ported helpers: `makeMask`, `dealContour`, `cutPic`, `getType`, `pickShapeRGB`,
     `lizhipai`, floodFill hand reader, and both net definitions. Copied verbatim in
     behaviour (same thresholds/constants) — fidelity over cleanliness.
2. **`scripts/inspect/mycv_baseline.py`** — the scoring harness. Uses our `Replayer`,
   `coords`, `tiles`. Per session: replay GT, run the port per frame, align, score.

### Seat & resolution alignment

- `m{0..3}.png → screen position` resolved once from the mask PNG centroids
  (bottom=self, top=across, left=left, right=right); recorded as a constant with a check.
- `screen_to_seat(hero_seat, sp)` = `{self:hero, right:hero+1, across:hero+2, left:hero+3}`
  (already in `label/river.py`) maps screen → absolute seat for GT comparison.
- mycv operates in its own 1080p space (raw `resize(1920,1080)`). GT tile positions for
  matching come from our `coords` (`RIVER_QUADS`/`HAND`/`MELD_STRIPS`) evaluated on the
  canonical board × (1920,1080). Assumption: capture is fullscreen 16:9 so raw-resize ≈
  canonical board (true for session6 and session5_16x9). Stated, and validated by the
  visual gate below.

### Scoring (per region, per session)

Match mycv detections to GT, then report **two** metrics (as requested):

- **Matching:** spatial greedy nearest-neighbour between detected tile centers and GT tile
  centers (GT centers from our `coords` grid cells / hand slots / meld strips in 1080p
  space), within a distance threshold. Hand additionally uses left→right ordinal zip as a
  cross-check. Unmatched GT = miss (FN); unmatched detection = spurious (FP).
- **(a) End-to-end accuracy** = correct (matched **and** class-correct) / total GT;
  misses, spurious, and misclassifications all count against. (mycv's real usefulness.)
- **(b) Classification accuracy** = class-correct / matched. (Comparable to our 93.5%.)
- Also report detection **recall** (matched/GT) and **precision** (matched/detected), and
  a top-confusions list per region.

### Output

Per region × session table: end-to-end %, classification %, detection recall/precision,
n. Aggregate across sessions. Printed to terminal; full per-frame detail to scratchpad.

## Verification gate (critical — runs FIRST)

Skin/resolution mismatch is the top risk: if our captures use a different table skin than
mycv's calibration, mycv's white `inRange` thresholds and seat masks misfire, and the
numbers would reflect *mismatch*, not mycv's true accuracy. **Before trusting any
aggregate number**, overlay mycv's detected boxes + predicted classes on 2–3 frames and
eyeball alignment. Only after the overlay looks right do we run the full sweep and report.

## Testing

- Unit (no client): class-index→name identity map; `screen_to_seat`; the spatial matcher
  on synthetic point sets (known FN/FP/correct); `getType` region-code lookup on a few
  known pixels.
- Integration: the visual-overlay gate on real frames; a small-N run with sanity bounds
  (e.g. river detection recall > 0 and class accuracy in a plausible band) before full run.

## Risks / caveats

- **Skin/resolution mismatch** → mitigated by the visual gate.
- **floodFill hand reader** is hardcoded to 1080p start pixel `[235,1002]`; if our hand row
  sits elsewhere it returns nothing — caught by the gate.
- **Detection vs classification conflation** — separated by reporting both metrics.
- **Default masks only** (no reveal/bq variants) — fidelity caveat, noted in the report.
- mycv assets live in a sibling repo (`../auto/mycv`); path is a CLI arg, defaulting to the
  documented sibling location.
