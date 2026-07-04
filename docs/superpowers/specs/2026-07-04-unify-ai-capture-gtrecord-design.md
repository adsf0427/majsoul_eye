# Unify AI capture to the manual `GTRecord` format

**Date:** 2026-07-04
**Status:** Approved design, pending implementation plan
**Branch:** `feat/unify-ai-capture-gtrecord` (proposed)

## 1. Goal

Make the AI-autoplay capture path (`scripts/capture/autoplay_ai.py`) write, **at
capture time**, the same on-disk format the manual `record_gt.py` path writes: a
per-game `GTRecord` JSONL (parsed `raw_liqi` + derived `mjai` inline) plus a
screenshot index. This eliminates the separate `convert_mjcopilot.py` pass and the
`captures/intermediate/gt/` layer, so **both capture lines produce identical data**
and every downstream consumer treats AI and manual games the same way.

User's framing (verbatim): *"我想让这两条线的数据集统一，让 a 自动捕获的时候，直接
存下来和人工捕获一样的数据，方便后续处理。"*

## 2. Background — why the two lines differ today

| Stage | manual (`session5/6`) | AI (`ai_run_*`) |
|---|---|---|
| Raw file | `raw/manual/sessionN.jsonl` | `raw/ai_session/run_*/game*/frames.jsonl` |
| Raw record | full `GTRecord` (`raw_liqi` + `mjai` inline) | `{seq, ts, b64}` (liqi wire only) |
| Needs convert? | No — already `GTRecord` | Yes — `convert_mjcopilot` decodes wire + derives MJAI → `intermediate/gt/*.jsonl` |
| `build_dataset` reads | the raw jsonl directly | the converted jsonl |

Key realization that shapes this design: **downstream is already unified on
`GTRecord`.** `convert_mjcopilot.py` already emits a real `GTRecord` JSONL (via
`capture.schema.write_records`) plus a screenshot index. `build_dataset`,
`annotate_ai_session`, and `rebuild_datasets` all consume the AI data in that
`GTRecord` shape. The only difference is **where** the `GTRecord` is produced: in a
separate offline convert pass (AI) vs. at capture time (manual). This design moves
AI production to capture time, matching manual.

## 3. Constraints & context

- `autoplay_ai.py` already runs `game_state.input(msg)` live to get the bot's
  reaction, so the MJAI events are already being derived in-process. The converter's
  only extra trick is intercepting `GameState.mjai_pending_input_msgs` (a
  `TracedGameState` subclass whose `mjai_pending_input_msgs` is a `CapList` that
  `copy.deepcopy`s every appended event — needed because GameState **mutates the AI
  hand in place**, which would otherwise overwrite `start_kyoku.tehais`).
- The live loop is delicate (browser thread + serial action queue + real Mortal
  bot + clicking). The recording addition must be isolated: wrapped in try/except
  and fed to a **background writer**, exactly like the manual `akagi_tap.py` tap, so
  it can never stall or break the bot loop.
- `recognize/` must stay Akagi/MahjongCopilot-free. All new capture code is
  dev-only and lives under `majsoul_eye/capture/` or `scripts/`.
- The real Mortal bot only **reads** `mjai_pending_input_msgs`; GameState derives it
  independent of which bot consumes it. Therefore the live derivation equals the
  offline convert derivation — proven by test (§7), not merely asserted.
- **Abnormal exit is a hard requirement** (user): if the game window closes
  mid-play or the script is Ctrl-C'd / crashes, the in-progress game must still be a
  usable, self-consistent game on disk. This is what forces incremental writing (§4).

## 4. Mechanism — inline incremental `GTRecord` writing

Reject "convert at game-end": if the run dies mid-game the trigger never fires and
that game yields no `GTRecord`. Instead write the `GTRecord` **per liqi message,
during play**, the same way `akagi_tap.py` writes one record per `parse_liqi`. An
interrupted game keeps every record up to the last message.

Two proven pieces are reused:

- **`GTWriter`** (currently in `capture/akagi_tap.py`) — the background-thread JSONL
  writer manual already uses. It writes the `_schema` header and one `GTRecord` per
  line off the hot path. It will be **moved to a neutral module** so both the Akagi
  tap and `autoplay_ai` import it without pulling Akagi in (see §5).
- **The MJAI-extraction trick** — extracted from `convert_mjcopilot.py` into a shared
  dev-only helper `make_capturing_game_state(GameStateCls, bot) -> (gs, drain_mjai)`.
  `gs` is a normal (traced) `GameState` instance, so every existing
  `game_state.<attr>` access/assignment in `autoplay_ai` is unchanged; `drain_mjai()`
  returns the deep-copied mjai events derived since the previous call.

`autoplay_ai` hot-path delta (all recording wrapped in try/except):

```python
game_state, drain_mjai = make_capturing_game_state(GameState, bot)   # was GameState(bot)
...
reaction = game_state.input(msg)          # unchanged
mjai = drain_mjai()                        # NEW: events THIS message derived
if mjai:                                   # emit-on-new-mjai (see below)
    gt_writer.put(GTRecord(seq=seq, ts=ts, flow_id="", seat=game_state.seat,
                           last_op_step=0, syncing=False,
                           method=..., action_name=..., raw_liqi=msg, mjai=mjai))
```

**Emit policy:** emit a `GTRecord` only for messages that produced **new mjai**,
matching today's `convert_mjcopilot` output (the manual tap instead emits one record
per message, including empty-mjai ones). Both are tolerated downstream
(`gtframes.build_seq_state` filters on `mjai` ∩ `RELEVANT_EVENTS`); choosing
emit-on-new-mjai keeps the regenerated AI datasets **byte-identical to the current
convert output** (§7 regression).

`method`/`action_name` are derived from the parsed `msg` the same way the converter
does (`method = msg["method"]`; `action_name = msg["data"]["name"]` when method is
`.lq.ActionPrototype`).

## 5. On-disk layout — mirror manual exactly

The `frames.jsonl` name currently holds the b64 wire in the AI tree. Free that name
for the screenshot index (matching manual) and rename the wire to `liqi.jsonl`.
Per-game layout becomes the `X.jsonl ↔ X/` convention `paths.frames_dir_for`
already encodes:

```
captures/raw/ai_session/run_N/
  ai_settings.json                 (unchanged)
  gameM.jsonl            ← GTRecord JSONL (raw_liqi + mjai)          [NEW — the consumable]
  gameM/
    liqi.jsonl           ← raw liqi wire {seq, ts, b64}             [was gameM/frames.jsonl]
    frames.jsonl         ← screenshot index {seq, file, status, ts} [NEW, written incrementally]
    frames/NNNNNN.png    ← screenshots                              (unchanged)
    metadata.json        ← {language}                              (unchanged)
```

This is byte-for-byte the manual structure (`raw/manual/sessionN.jsonl` +
`sessionN/frames.jsonl` + `sessionN/frames/`), nested under `run_N/`.

- **Raw wire kept** (`liqi.jsonl`) — tiny, and the re-derivable source of truth if the
  derivation logic ever changes. Still written incrementally for crash-safety.
- **Screenshot index written incrementally** — `autoplay_ai` currently just dumps
  PNGs and lets the converter glob them; now it appends `{seq, file, status, ts}` to
  `gameM/frames.jsonl` as each PNG is saved (file field index-relative
  `"frames/NNNNNN.png"`, the same shape `FrameSyncer` writes). So an
  abnormally-ended game has a complete-up-to-interruption index with no game-end pass.

Result: an interrupted game leaves `gameM.jsonl`, `gameM/frames.jsonl`,
`gameM/frames/*.png`, `gameM/liqi.jsonl`, `gameM/metadata.json` mutually consistent
and directly consumable by `build_dataset` — no convert, no `intermediate/gt/`.

### Module boundary note (`GTWriter`)

`GTWriter` lives in `akagi_tap.py` today. `autoplay_ai` must not import `akagi_tap`
(it triggers the Akagi `MajsoulBridge` import). Move `GTWriter` to a neutral,
Akagi-free module (candidate: `capture/schema.py`, which already defines `GTRecord`
+ JSONL header and has no Akagi deps) and have `akagi_tap` import it from there.
This keeps both taps on one writer and preserves the `capture/` import boundaries.

## 6. Shared code + downstream rewire

**New shared helper** — `majsoul_eye/capture/mjcopilot_gt.py` (dev-only,
MahjongCopilot-coupled, never imported by `recognize/`): holds
`make_capturing_game_state(...)` and any `GTRecord`-assembly used by both paths.
`convert_mjcopilot.py` is refactored to call it, so the live path and the
legacy-migration path share **one** derivation and cannot drift.

**`paths.py`** — add:
- `ai_captures()` — glob `raw/ai_session/run_*/game*.jsonl` (the per-game GTRecord
  files; excludes `ai_settings.json` and the deeper `game*/frames.jsonl`/`liqi.jsonl`).
- `ai_game_name(path)` — derive the stable flattened name `ai_run_N_gameM` from
  `.../run_N/gameM.jsonl`, so dataset dirs (`precise_ai_run_3_game1`, held-out val
  `ai_run_8_game1`, etc.) stay identical and nothing churns.
- `converted_gt_captures()` is repointed to `ai_captures()` (or retired with callers
  updated).

**Consumers repointed** from `intermediate/gt/` to `raw/ai_session/`:
- `rebuild_datasets.py` — capture discovery + `gt_frames_dir(name)` (now
  `run_N/gameM/` instead of `intermediate/gt/name`) + `FRAMES_OVERRIDE` keys.
- `annotate_ai_session.py` — default capture glob.
- `ingest_run.py` — its convert step is dropped; it becomes build[/train] only (or is
  folded into `rebuild_datasets`, decided at plan time).
- `build_dataset.py` — **no change**: it already takes `(capture.jsonl, frames_dir)`
  and reads a `GTRecord` jsonl directly (exactly how manual already works).

## 7. Legacy migration (the 16 existing b64 games)

One-time `scripts/data/migrate_ai_to_gtrecord.py` — dry-run default, idempotent, same
discipline as the existing `migrate_captures_layout.py`. For each existing
`run_N/gameM/frames.jsonl` (b64):
1. Run the shared `convert_game` over the wire → write `run_N/gameM.jsonl` (GTRecord).
2. Rewrite `run_N/gameM/frames.jsonl` as the screenshot index (glob `frames/*.png`,
   index-relative `file`), OR write it fresh after step 3's rename.
3. Rename the old b64 `frames.jsonl` → `gameM/liqi.jsonl`.

After it runs, all 16 AI games are indistinguishable from freshly-captured ones and
`captures/intermediate/gt/` (all-derived, gitignored) can be deleted.

Letterboxed games (`ai_run_5_game2/3`, handled via `FRAMES_OVERRIDE` /
`intermediate/derived/*_fixed`) keep their existing de-letterbox override; only the
GT-source location changes, not the frame override.

## 8. Testing / verification

- **Equivalence test (correctness proof):** feed one real game's `liqi.jsonl`
  through the shared `make_capturing_game_state` (with a StubBot) and through the old
  `convert_game`; assert the resulting `GTRecord` lists are identical. Guarantees the
  refactor + the live derivation match today's validated output.
- **Round-trip test:** a tiny 1-game capture (or fixture) → assert the layout files
  exist, the frame index resolves via `paths.resolve_frame_path`, and
  `Replayer.check_invariants()` passes on the replayed states.
- **Regression:** after migration + rewire, run `rebuild_datasets.py --yes` and
  confirm crops + YOLO labels are byte-identical to the pre-change datasets (the whole
  point — nothing downstream should change).
- Keep all existing test suites green (`test_replay`, `test_label`, `test_classifier`,
  `test_annotate_frame`, etc.).

## 9. Out of scope / non-goals

- No change to the manual `record_gt.py` / `akagi_tap.py` capture behavior (only the
  `GTWriter` module move, which is behavior-preserving).
- No retraining of the classifier or detector — datasets are expected byte-identical,
  so weights are untouched. If regression surfaces a diff, that is a separate task.
- No change to `recognize/` (the shipped runtime) — it never touched either capture
  path and must stay Akagi/MahjongCopilot-free.
- The `--overlay`, `--auto-next`, `--autojoin`, language-metadata, and deal-window-skip
  behaviors of `autoplay_ai` are unchanged.

## 10. Risks

- **Live derivation ≠ offline derivation.** Mitigated by the equivalence test (§7)
  and by a cheap optional post-capture self-check (convert `liqi.jsonl` and diff
  against the inline `gameM.jsonl`). Low risk: bot only reads the mjai list.
- **Hot-path regression in `autoplay_ai`.** Mitigated by try/except isolation +
  background `GTWriter` (both proven in `akagi_tap`), and by the `gs` returned from
  the helper being a drop-in for `GameState` (no call-site rewrites beyond the two
  new lines).
- **Migration corrupts existing captures.** Mitigated by dry-run default,
  idempotency, and keeping the original wire (renamed, not deleted). The 16 games are
  also re-derivable from the wire at any time.
