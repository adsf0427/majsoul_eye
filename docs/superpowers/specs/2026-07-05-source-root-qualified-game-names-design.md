# Source-root-qualified game names (lift the cross-source run-uniqueness rule)

**Date:** 2026-07-05
**Status:** approved (design)
**Scope:** `majsoul_eye/paths.py`, `scripts/data/build_datasets.py`, tests, live docs, one memory.

## Problem

`build_datasets.py v2 --sources captures/raw/ai_session captures/raw/ai_session2`
aborts. `ai_session2` inside `captures/raw/raw.7z` uses the pristine layout
`run_1/run_2/run_3`, which collides with `ai_session/run_1` and `ai_session/run_3`.

The collision is systemic, not cosmetic. Three pipeline stages independently call
`paths.ai_game_name(cap)`, whose result is a function of only the last two path
segments (`run_N`, `gameM`) — the source root is ignored:

1. `build_datasets.discover_games` — dataset dir + `games.json` manifest key.
2. `scripts/annotate/annotate_ai_session.py` — *writes* `annotations/<name>.jsonl`.
3. `scripts/train/build_dataset.py --from-annotations DIR` — *reads*
   `annotations/<name>.jsonl`.

So `ai_session/run_1/game1` and `ai_session2/run_1/game1` don't merely clash in the
manifest — both resolve to `ai_run_1_game1.jsonl` and silently clobber each other's
annotation, and stage 3's lookup becomes ambiguous. `discover_games` currently
hard-fails on the manifest clash ("run numbering must be unique across sources"),
which is why `ai_session2` was hand-renamed on disk to `run_21/22/23`. We want the
raw.7z layout to build directly, with no rename.

## Decision (confirmed with user)

- **Naming: prefix *every* root** with its directory basename (symmetric — the
  canonical `ai_session` is renamed too), not only the non-canonical roots.
- **Contamination: build everything.** `ai_session2/run_1` (2 games) is flagged
  contaminated in STATUS §1.25, but the builder stays free of hardcoded data-quality
  carve-outs; excluding it is the user's job (don't extract / delete run_1).

### Why the canonical rename is leak-safe here

Renaming the canonical `ai_session` games (`ai_run_8_game1` → `ai_session_run_8_game1`)
would, in general, make the *same physical kyoku* carry different names in two dataset
versions and leak across a mixed-version held-out `--val`. That risk is **moot in this
repo**: `datasets/` and `captures/` are gitignored (local-only) and **no versioned
`games.json` manifest exists** — the local data is the flat `datasets/precise_ai_run_*`
per-game dirs. v2 is built fresh and self-contained, so one consistent scheme inside v2
is leak-free. (If a v1 manifest is ever built in the old scheme and mixed with v2, the
held-out game must be named identically in both — out of scope here.)

## Core change: `paths.ai_game_name`

Replace the literal `ai_` prefix with the source-root directory basename. The root is
the path segment immediately above `run_N`:

- game shape `…/<ROOT>/run_N/gameM[/gameM].jsonl` → `<ROOT>_run_N_gameM`
- single-game-run shape `…/<ROOT>/run_N[/run_N].jsonl` → `<ROOT>_run_N`
- anything else (manual `sessionN`, …) → basename stem, unchanged.

Derivation reuses the existing ancestor logic:

- game branch (parent matches `run_\d+`, stem matches `game\d+`): `root = anc[-2]`.
- run branch (stem matches `run_\d+`): `root = anc[-1]`.

Guard the index so a malformed/too-short path falls back to `"ai"` rather than
emitting a leading-underscore name; real captures are always
`captures/raw/<ROOT>/run_N/…` so the guard is defensive only.

| capture path | old | new |
|---|---|---|
| `…/ai_session/run_8/game1/game1.jsonl` | `ai_run_8_game1` | `ai_session_run_8_game1` |
| `…/ai_session/run_8/game6.jsonl` (sibling) | `ai_run_8_game6` | `ai_session_run_8_game6` |
| `…/ai_session/run_1/run_1.jsonl` | `ai_run_1` | `ai_session_run_1` |
| `…/ai_session/run_1.jsonl` (sibling) | `ai_run_1` | `ai_session_run_1` |
| `…/ai_session2/run_1/game1/game1.jsonl` | `ai_run_1_game1` (collided) | `ai_session2_run_1_game1` |
| `…/manual/session5.jsonl` | `session5` | `session5` |

Because all three stages call this one pure function on the same capture path, they
stay consistent for free — no name is threaded through CLI args.

## Cascade changes

`scripts/data/build_datasets.py`:
- `DEFAULT_VAL`: `"ai_run_8_game1"` → `"ai_session_run_8_game1"`.
- `FRAMES_OVERRIDE` keys: `ai_run_5_game2`, `ai_run_5_game3` →
  `ai_session_run_5_game2`, `ai_session_run_5_game3`.
- `discover_games`: keep the duplicate-name check, but reword — it is now a
  true-duplicate safety net (e.g. the same source listed twice), not a cross-source
  run-number guard. Module docstring's "run numbering must be unique across sources"
  paragraph rewritten to describe root-qualified names.

`majsoul_eye/paths.py`: `ai_game_name` docstring updated to the new mapping.

Live command docs (the held-out val name changes, so `--val` examples must too):
`README.md`, `majsoul_eye/CLAUDE.md`, `docs/PIPELINE.md` SOP lines — replace
`--val ai_run_8_game1:*` / `ai_run_8_game1` (as the held-out-game convention) with
`ai_session_run_8_game1`. Dataset *directory* examples that name existing on-disk dirs
(`precise_ai_run_1`, etc.) are NOT changed — those dirs are unchanged.

Historical `docs/STATUS.md` entries and `docs/superpowers/specs/*` stay as written
(history). Add one new STATUS entry describing this change.

Memory `skinned-capture-ai-session2.md`: remove the now-obsolete "Any future capture
root must use globally unique run numbers" rule; note that roots are now
root-qualified.

## Tests (TDD — adjust/add before the code)

- `tests/test_paths.py`: update the 5 `ai_game_name` assertions to the new names; add
  a non-canonical-root case (`ai_session2/run_1/game1` → `ai_session2_run_1_game1`).
- `tests/test_downstream_rewire.py`: line-14 assertion → `ai_session_run_3_game1`.
- `tests/test_build_datasets.py`:
  - `test_discover_games_shapes_and_kinds`: names → `ai_session_run_1`,
    `ai_session_run_2_game1`, `session9`; the `dir` default assertion follows.
  - `test_frames_override_applies`: `ai_session_run_5_game2`.
  - `test_discover_games_collision_and_empty`: **flip** — two roots with the same run
    number must now produce *distinct* names (no SystemExit); keep the empty-root
    abort. Add a real true-duplicate case if convenient (same root twice) to keep the
    safety net covered.
  - Assertions using `ai_run_*` as arbitrary manifest strings
    (`test_manifest_roundtrip…`, `test_apply_existing_dirs…`) are inputs, not derived —
    leave unless they read more clearly renamed.
- `tests/test_migrate_ai.py`: assertions `ai_run_3_game1`, `ai_run_1` → the
  `ai_session_`-prefixed forms (verify the temp paths' root basename first — if the
  test builds under a non-`ai_session` temp dir, the prefix follows that dir).

## Explicitly NOT touched

- On-disk `datasets/precise_ai_run_*` / `obb_precise_ai_run_*` dirs, and the tests
  referencing them by literal path (`test_consistency_golden.py`, `test_detector.py`):
  a function rename does not move directories.
- `ai_session2/run_1` contamination handling (user chose "build everything").
- The uncommitted `verify_game_yolo` work already in `build_datasets.py` — preserved.

## Verification

1. `for t in tests/test_*.py; do PYTHONPATH=. python "$t" || break; done` — all green.
2. `PYTHONPATH=. python scripts/data/build_datasets.py v2 --sources
   captures/raw/ai_session captures/raw/ai_session2 --dry-run` — discovers both roots'
   games (`ai_session_run_*` + `ai_session2_run_*`) with **no collision/duplicate
   abort**, default val `ai_session_run_8_game1` resolves, and the printed per-game
   commands carry the qualified names. (Run against whatever is on disk — the exact
   game count varies with disk state, e.g. `ai_session/run_1`'s GTRecord jsonl is
   missing per STATUS §1.27; the raw.7z original layout is the target once extracted.)
