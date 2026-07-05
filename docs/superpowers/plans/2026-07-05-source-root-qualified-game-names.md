# Source-Root-Qualified Game Names Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `build_datasets.py v2 --sources captures/raw/ai_session captures/raw/ai_session2` build the pristine `raw.7z` layout (where `ai_session2` uses `run_1/2/3`) without the cross-source run-number collision, by tagging every game name with its source-root basename.

**Architecture:** One pure function, `paths.ai_game_name`, is the single namer three pipeline stages call (`build_datasets.discover_games`, `annotate_ai_session`, `build_dataset --from-annotations`). Change its leading tag from the fixed `ai_` to the source-root directory basename (the path segment above `run_N`); all three stages stay consistent for free. A fourth caller (the deprecated `migrate_ai_to_gtrecord.py`) has a duplicate namer that we collapse onto `paths.ai_game_name`. Then rename the two constants that hard-code the old canonical names (`DEFAULT_VAL`, `FRAMES_OVERRIDE` keys) and update tests + live docs.

**Tech Stack:** Python 3 (conda `auto` env). Tests are plain scripts under `tests/` (no pytest dependency; also pytest-compatible). Run everything from the repo root with `PYTHONPATH=.`.

## Global Constraints

- Run all commands from the repo root `majsoul_eye/` with `PYTHONPATH=.`.
- Default-PATH python has NO numpy; in this harness's Bash use `C:/Users/zsx/miniforge3/envs/auto/python.exe` in place of `python`. (Linux training box: use the `auto` env's python — see memory; either way it is the `auto` interpreter.)
- 38-class order and runtime `recognize/` are untouched — this is a naming/plumbing change only.
- Naming rule (verbatim): tag = source-root directory basename (segment above `run_N`). `…/ai_session/run_8/game1` → `ai_session_run_8_game1`; `…/ai_session2/run_1/game1` → `ai_session2_run_1_game1`; single-game run `…/<root>/run_1` → `<root>_run_1`; manual `sessionN` → `sessionN` (fallback unchanged).
- `FRAMES_OVERRIDE` values are physical dirs on disk named `ai_run_5_game2_fixed` / `ai_run_5_game3_fixed` — only the KEYS (game names) change, the VALUES stay.
- `DEFAULT_VAL` becomes `"ai_session_run_8_game1"` (the val game `captures/raw/ai_session/run_8/game1` under the new scheme).
- Do NOT rename on-disk `datasets/precise_ai_run_*` / `obb_precise_ai_run_*` dirs or the tests that reference them by literal path (`test_consistency_golden.py`, `test_detector.py`) — a function rename does not move directories.
- Do NOT touch historical `docs/STATUS.md` entries or `docs/superpowers/specs/*` (history); only append one new STATUS entry.
- Git: no `Co-Authored-By` trailer (user global rule). Commit only the files each task lists (leave the pre-existing uncommitted `verify_game_yolo` / `docs/PIPELINE.md` / `regen_detector_dataset.sh` working-tree changes alone unless a task edits that same file).

---

### Task 1: Root-qualify the shared namer + consuming constants + all name tests (atomic)

This is a cross-cutting rename: changing `paths.ai_game_name` simultaneously breaks the tests that assert the old names in `test_paths.py`, `test_downstream_rewire.py`, `test_build_datasets.py`, and `test_migrate_ai.py`. There is no ordering that keeps the suite green mid-way, so the function, the two `build_datasets.py` constants, the `migrate_ai` duplicate namer, and all four test files move in ONE commit. TDD: update every affected assertion to the new expected names first (suite goes red), then flip the four source files to make them green.

**Files:**
- Modify: `majsoul_eye/paths.py` (function `ai_game_name`, ~lines 147-166)
- Modify: `scripts/data/build_datasets.py` (module docstring lines 17-22; `FRAMES_OVERRIDE` lines 56-59; `DEFAULT_VAL` line 61; `discover_games` docstring line 129 + collision branch lines 140-143)
- Modify: `scripts/data/migrate_ai_to_gtrecord.py` (`plan_targets`, lines 58-66)
- Test: `tests/test_paths.py`, `tests/test_downstream_rewire.py`, `tests/test_build_datasets.py`, `tests/test_migrate_ai.py`

**Interfaces:**
- Produces: `paths.ai_game_name(capture_path: str) -> str` — now returns `"<root>_run_N_gameM"` / `"<root>_run_N"` where `<root>` is the source-root basename; the fallback (non run/game paths, e.g. manual) still returns the basename stem. Consumed unchanged (same signature) by `build_datasets.discover_games`, `annotate_ai_session`, `build_dataset`, `migrate_ai_to_gtrecord.plan_targets`, `purge_deal_frames`.

- [ ] **Step 1: Update `tests/test_paths.py` assertions to the new names (RED)**

Replace the three assertion bodies (keep the comments) at lines 15-31:

```python
def test_ai_game_name_multi_game():
    # nested (new canon): jsonl inside its own frames dir
    assert paths.ai_game_name("captures/raw/ai_session/run_3/game1/game1.jsonl") == "ai_session_run_3_game1"
    # absolute + backslash variants resolve the same
    assert paths.ai_game_name(r"D:\x\captures\raw\ai_session\run_10\game2\game2.jsonl") == "ai_session_run_10_game2"
    # legacy sibling shape still resolves (un-migrated trees)
    assert paths.ai_game_name("captures/raw/ai_session/run_8/game6.jsonl") == "ai_session_run_8_game6"


def test_ai_game_name_single_game_run():
    assert paths.ai_game_name("captures/raw/ai_session/run_1/run_1.jsonl") == "ai_session_run_1"  # nested
    assert paths.ai_game_name("captures/raw/ai_session/run_1.jsonl") == "ai_session_run_1"        # legacy sibling


def test_ai_game_name_source_root_qualified():
    # a NON-canonical root: its basename tags the name, so the same run number under
    # two roots is distinct (raw.7z's ai_session2/run_1 vs ai_session/run_1 no longer clash).
    assert paths.ai_game_name("captures/raw/ai_session2/run_1/game1/game1.jsonl") == "ai_session2_run_1_game1"
    assert paths.ai_game_name("captures/raw/ai_session2/run_1/run_1.jsonl") == "ai_session2_run_1"


def test_ai_game_name_fallback_for_manual():
    # manual sessions (or anything not matching run/game) fall back to the stem
    assert paths.ai_game_name("captures/raw/manual/session5.jsonl") == "session5"
```

- [ ] **Step 2: Update `tests/test_downstream_rewire.py` line 14 (RED)**

```python
def test_annotate_and_build_agree_on_name():
    # Both derive the annotation filename the same way for an AI capture (nested layout).
    cap = "captures/raw/ai_session/run_3/game1/game1.jsonl"
    assert paths.ai_game_name(cap) == "ai_session_run_3_game1"
```

- [ ] **Step 3: Update `tests/test_migrate_ai.py` name assertions (RED)**

Line 22 (`test_plan_targets_multi_game`) and line 32 (`test_plan_targets_single_game_run`):

```python
    # in test_plan_targets_multi_game:
    assert t["name"] == "ai_session_run_3_game1"
    # in test_plan_targets_single_game_run:
    assert t["name"] == "ai_session_run_1"
```

(Leave every other assertion in that file — the path assertions are structure-derived, not name-derived.)

- [ ] **Step 4: Update `tests/test_build_datasets.py` — shapes, frames-override, and the flipped collision test (RED)**

In `test_discover_games_shapes_and_kinds` (root basename is `ai_session`), replace lines 32-40:

```python
        assert set(by_name) == {"ai_session_run_1", "ai_session_run_2_game1", "session9"}, by_name
        assert by_name["ai_session_run_1"]["kind"] == "ai"
        assert by_name["ai_session_run_2_game1"]["kind"] == "ai"
        assert by_name["session9"]["kind"] == "manual"
        # frames dir = capture with .jsonl stripped, POSIX-slashed
        assert by_name["ai_session_run_2_game1"]["frames_dir"].endswith("run_2/game1")
        assert "\\" not in by_name["ai_session_run_2_game1"]["frames_dir"]
        # dir defaults to the game name (no prefix)
        assert by_name["ai_session_run_1"]["dir"] == "ai_session_run_1"
```

Replace the whole `test_discover_games_collision_and_empty` (lines 43-59) with a renamed test that asserts the two roots are now DISTINCT, a same-root-twice duplicate still aborts, and empty still aborts:

```python
def test_discover_games_source_qualified_and_empty():
    """Same run number in two roots must NOT collide now (names are source-root
    qualified); the SAME root listed twice is a real duplicate and aborts; an empty
    root aborts."""
    with tempfile.TemporaryDirectory() as td:
        r1 = os.path.join(td, "ai_session")
        r2 = os.path.join(td, "ai_session2")
        _touch(os.path.join(r1, "run_1", "game1", "game1.jsonl"))
        _touch(os.path.join(r2, "run_1", "game1", "game1.jsonl"))
        names = {g["name"] for g in bds.discover_games([r1, r2])}
        assert names == {"ai_session_run_1_game1", "ai_session2_run_1_game1"}, names
        # same root passed twice -> genuine duplicate -> abort
        try:
            bds.discover_games([r1, r1])
            raise AssertionError("duplicate not detected")
        except SystemExit as e:
            assert "duplicate" in str(e)
        # empty root -> abort
        try:
            bds.discover_games([os.path.join(td, "empty")])
            raise AssertionError("empty root not detected")
        except SystemExit as e:
            assert "no captures" in str(e)
```

In `test_frames_override_applies` replace lines 68-69:

```python
        assert g["name"] == "ai_session_run_5_game2"
        assert g["frames_dir"] == bds.FRAMES_OVERRIDE["ai_session_run_5_game2"].replace(os.sep, "/")
```

(Leave `test_manifest_roundtrip_and_training_expansion` and `test_apply_existing_dirs_preserves_v1_style_prefixes` — their `ai_run_*` strings are explicit manifest inputs, not derived names.)

- [ ] **Step 5: Run the four tests and confirm they FAIL**

Run:
```bash
PYTHONPATH=. python tests/test_paths.py; PYTHONPATH=. python tests/test_downstream_rewire.py; PYTHONPATH=. python tests/test_build_datasets.py; PYTHONPATH=. python tests/test_migrate_ai.py
```
Expected: `AssertionError` from at least `test_paths` / `test_build_datasets` / `test_migrate_ai` / `test_downstream_rewire` (old function still returns `ai_run_*`; the new `ai_session_run_*` expectations mismatch). This proves the tests exercise the change.

- [ ] **Step 6: Change `paths.ai_game_name` to tag by source root**

Replace the function (lines 147-166) with:

```python
def ai_game_name(capture_path: str) -> str:
    """Stable flattened dataset name for an AI GTRecord capture, tagged by SOURCE ROOT.

    The leading tag is the source-root directory basename (the path segment above
    ``run_N``), so the same run number under two roots yields DISTINCT names — no
    cross-source run-renumbering needed (raw.7z's ``ai_session2/run_1`` no longer
    collides with ``ai_session/run_1``):

    ``.../ai_session/run_N/gameM/gameM.jsonl`` -> ``ai_session_run_N_gameM`` (nested);
    ``.../ai_session2/run_N/gameM.jsonl``      -> ``ai_session2_run_N_gameM`` (sibling);
    ``.../<root>/run_1/run_1.jsonl``           -> ``<root>_run_1`` (single-game run, either shape);
    anything else                              -> the basename stem  (manual sessions pass through).
    """
    p = os.path.abspath(capture_path).replace("\\", "/")
    parts = p.split("/")
    stem = os.path.splitext(parts[-1])[0]                 # gameM  or  run_N
    anc = parts[:-1]
    if anc and anc[-1] == stem:                           # nested: drop the frames dir itself
        anc = anc[:-1]
    parent = anc[-1] if anc else ""
    if re.fullmatch(r"run_\d+", parent) and re.fullmatch(r"game\d+", stem):
        root = anc[-2] if len(anc) >= 2 else "ai"         # source-root basename, above run_N
        return f"{root}_{parent}_{stem}"                  # <root>_run_N_gameM
    if re.fullmatch(r"run_\d+", stem):
        root = anc[-1] if anc else "ai"                   # run_N's parent is the source root
        return f"{root}_{stem}"                           # <root>_run_N
    return stem
```

- [ ] **Step 7: Rename the two hard-coded constants + reword the collision guard/docstrings in `scripts/data/build_datasets.py`**

`DEFAULT_VAL` (line 61):
```python
DEFAULT_VAL = "ai_session_run_8_game1"   # held-out whole game (classifier + detector convention)
```

`FRAMES_OVERRIDE` (lines 56-59) — keys change, VALUES stay (on-disk dir names unchanged):
```python
FRAMES_OVERRIDE = {
    "ai_session_run_5_game2": os.path.join(paths.DERIVED, "ai_run_5_game2_fixed"),
    "ai_session_run_5_game3": os.path.join(paths.DERIVED, "ai_run_5_game3_fixed"),
}
```

Module docstring (lines 17-22) — replace the "must be UNIQUE / keep run numbering global" paragraph with:
```
Sources: each root is scanned for the AI shapes (``run_*/game*.jsonl`` and
``run_*.jsonl``) plus the manual shape (``session*.jsonl``). Game names come from
``paths.ai_game_name``, which tags each name with its SOURCE-ROOT basename
(``captures/raw/ai_session2/run_1/game1`` -> ``ai_session2_run_1_game1``), so the same
run number under different roots never collides — no cross-source run-renumbering
needed. AI games go annotate -> build ``--from-annotations``; manual games build
direct (no annotate stage).
```

`discover_games` docstring (line 129):
```python
    Raises SystemExit on an empty root or a duplicate game name (e.g. the same source
    root listed twice; distinct roots are source-root-qualified and never collide).
```

`discover_games` collision branch (lines 140-143):
```python
            name = paths.ai_game_name(cap)
            if name in seen:
                raise SystemExit(f"duplicate game name {name!r} from both {seen[name]} and "
                                 f"{cap} — pass each source root once (names are already "
                                 f"source-root-qualified, so distinct roots never collide)")
```

- [ ] **Step 8: Collapse the duplicate namer in `scripts/data/migrate_ai_to_gtrecord.py` onto `paths.ai_game_name` (DRY)**

Replace `plan_targets` lines 58-66 (the `gd`/`game`/`parent`/`name` block through `gt_path`) with:

```python
    gd = os.path.abspath(game_dir).replace("\\", "/")
    game = gd.split("/")[-1]
    name = paths.ai_game_name(gd)                      # source-root-qualified (shared namer)
    gt_path = os.path.join(game_dir, f"{game}.jsonl")  # nested: GT INSIDE the game dir
```

(`paths` is already imported at line 27; the old `parts`/`parent` locals were only used to build `name`, so they go away. `game` is still needed for `gt_path`.)

- [ ] **Step 9: Run the four changed tests — confirm they now PASS**

Run:
```bash
PYTHONPATH=. python tests/test_paths.py && PYTHONPATH=. python tests/test_downstream_rewire.py && PYTHONPATH=. python tests/test_build_datasets.py && PYTHONPATH=. python tests/test_migrate_ai.py
```
Expected: each prints its `... OK` line (e.g. `test_paths OK`).

- [ ] **Step 10: Run the FULL suite — confirm nothing else regressed**

Run:
```bash
for t in tests/test_*.py; do echo "== $t =="; PYTHONPATH=. python "$t" || { echo "FAILED: $t"; break; }; done
```
Expected: every file prints its `OK` line, no `FAILED:`. (In particular `test_consistency_golden.py` and `test_detector.py`, which reference `datasets/precise_ai_run_*` by literal path, are unaffected — those dirs are not renamed. If either is skipped for missing local weights/data, that is its normal behavior, not a regression.)

- [ ] **Step 11: Commit**

```bash
git add majsoul_eye/paths.py scripts/data/build_datasets.py scripts/data/migrate_ai_to_gtrecord.py \
        tests/test_paths.py tests/test_downstream_rewire.py tests/test_build_datasets.py tests/test_migrate_ai.py
git commit -m "feat(data): source-root-qualified game names (lift cross-source run-uniqueness)"
```

---

### Task 2: Update live docs, code comments, STATUS history, and memory

Naming-convention prose and command examples that reference the held-out val game by its old name are now wrong for a freshly-built dataset. Update the LIVE ones (README, project CLAUDE.md, PIPELINE.md SOP, two stale code comments), append one STATUS entry, and drop the obsolete "globally unique run numbers" rule from memory. No code behavior changes here, so verification is by grep + read.

**Files:**
- Modify: `README.md`, `majsoul_eye/CLAUDE.md`, `docs/PIPELINE.md`
- Modify: `majsoul_eye/annotate/cases.py` (comments, lines 11-12, 17-18), `majsoul_eye/annotate/pipeline.py` (comment, line 277)
- Modify: `docs/STATUS.md` (append one new entry)
- Modify: `/hszhao-f1/h3011050/.claude/projects/-hszhao-f1-h3011050-workspace-phoenix-phoenix-server-majsoul-eye/memory/skinned-capture-ai-session2.md`

**Interfaces:** none (docs/comments only).

- [ ] **Step 1: Confirm no false-positive `precise_ai_run_8_game1` in the three doc files**

Run:
```bash
grep -rn "precise_ai_run_8_game1" README.md majsoul_eye/CLAUDE.md docs/PIPELINE.md
```
Expected: NO output. (Proves a `replace_all` of `ai_run_8_game1` → `ai_session_run_8_game1` in these files only touches the val-convention string, never a directory name.)

- [ ] **Step 2: Rename the val-convention string in the three doc files**

In each of `README.md`, `majsoul_eye/CLAUDE.md`, `docs/PIPELINE.md`, replace every occurrence of `ai_run_8_game1` with `ai_session_run_8_game1` (Edit with `replace_all: true` per file). Do NOT touch `precise_ai_run_1` (README line ~166) or any other `precise_*` dir example. Affected lines: README ~130 & ~135; CLAUDE.md ~63; PIPELINE.md ~105, ~109, ~116, ~145.

- [ ] **Step 3: Fix the two stale code comments referencing the AB-case game names**

`majsoul_eye/annotate/cases.py` lines 11-12 and 17-18 — update the `# was ai_run_3_game1` / `ai_run_3_game3` mentions to the qualified names:
```python
Seat mapping (screen pos from hero): ai_session_run_3_game1 hero=3 → self3/right0/across1/left2;
                                     ai_session_run_3_game3 hero=1 → self1/right2/across3/left0.
```
```python
_G1 = f"{_AI}/run_3/game1/game1.jsonl"      # ai_session_run_3_game1 (hero=3)
_G3 = f"{_AI}/run_3/game3/game3.jsonl"      # ai_session_run_3_game3 (hero=1)
```

`majsoul_eye/annotate/pipeline.py` line 277:
```python
# geometry (WHERE). Calibrated on the AB case_frames (1920x1080, ai_session_run_3_game1/ai_session_run_3_game3).
```

- [ ] **Step 4: Append a STATUS.md entry**

Add after the last §1.28 entry (do not edit older entries). Use the next section number (§1.29) and today's date:

```markdown
### 1.29 game 名按 source 根加前缀（拜托跨 source run 唯一限制）（2026-07-05）
- **动机**（用户）：`raw.7z` 的原始布局里 `ai_session2` 用 `run_1/2/3`，与 `ai_session/run_1`、
  `ai_session/run_3` 撞名——旧规则"run 编号必须跨 source 根全局唯一"逼得 `ai_session2` 落盘时被
  改名 `run_21/22/23`。目标：`build_datasets.py v2 --sources captures/raw/ai_session
  captures/raw/ai_session2` 直接吃原始布局。
- **根因**：撞名是系统性的——三个阶段（`discover_games` / `annotate_ai_session` /
  `build_dataset --from-annotations`）各自调 `paths.ai_game_name(cap)`，而它只看路径最后两段
  （`run_N/gameM`），忽略 source 根 → 两个 `run_1/game1` 都解析成 `ai_run_1_game1.jsonl`，标注互相
  覆盖。
- **改动**：`ai_game_name` 的前缀由固定 `ai_` 改为 **source 根目录 basename**（`run_N` 上一层）：
  `ai_session/run_8/game1 → ai_session_run_8_game1`、`ai_session2/run_1/game1 →
  ai_session2_run_1_game1`；manual `sessionN` 仍走 basename 兜底。三个阶段同调一个纯函数 → 自动一致。
  第 4 个调用方 `migrate_ai_to_gtrecord.plan_targets`（弃用工具，自带一份重复命名）改为同调
  `paths.ai_game_name`（DRY）。`build_datasets.py` 随迁 `DEFAULT_VAL`
  (`ai_run_8_game1→ai_session_run_8_game1`) 与 `FRAMES_OVERRIDE` 键（值即磁盘 `ai_run_5_game*_fixed`
  不变）；`discover_games` 撞名分支改为"真重复"兜底（同一根传两次才报错）。
- **前缀改的是规范 canonical 根也一起加前缀**（用户选"prefix every root"）；因 `datasets/`、`captures/`
  均 gitignore、无 `games.json` 版本清单，v2 全新自包含 → 无跨版本 val 泄漏。既有磁盘
  `datasets/precise_ai_run_*` 目录与按字面路径引用它们的测试不受影响（函数改名不搬目录）。
- **验证**：TDD——`test_paths`（含新 `ai_session2` 用例）、`test_downstream_rewire`、`test_build_datasets`
  （撞名测试翻转为断言"不同名 + 同根传两次仍报错"）、`test_migrate_ai` 先红后绿；全量测试通过；
  `build_datasets.py v2 --sources ai_session ai_session2 --dry-run` 两根共存无撞名、默认 val
  `ai_session_run_8_game1` 命中。README/CLAUDE.md/PIPELINE.md 命令示例同步。
```

- [ ] **Step 5: Update the memory file (drop the obsolete global-uniqueness rule)**

In `…/memory/skinned-capture-ai-session2.md`, replace the first bullet (the "Runs renamed … Any future capture root must use globally unique run numbers." block) with a note that names are now source-root-qualified:

```markdown
- **Originally renamed `run_1..3` → `run_21..23`** on disk to dodge a collision: back
  then `ai_game_name()` ignored the source root, so `ai_session2/run_3/game1` clashed
  with `ai_session/run_3/game1` and `discover_games` hard-failed. **As of 2026-07-05
  (STATUS §1.29) this rename is NO LONGER needed** — `ai_game_name` now tags every name
  with its source-root basename (`ai_session2/run_1/game1` → `ai_session2_run_1_game1`),
  so `raw.7z`'s pristine `ai_session2/run_1..3` builds directly alongside `ai_session`.
```

- [ ] **Step 6: Verify docs are internally consistent (no stray old val name in live docs)**

Run:
```bash
grep -rn "ai_run_8_game1" README.md majsoul_eye/CLAUDE.md docs/PIPELINE.md
```
Expected: NO output (all live occurrences renamed; historical STATUS entries and spec docs are intentionally untouched and not in this grep set).

- [ ] **Step 7: Commit**

```bash
git add README.md majsoul_eye/CLAUDE.md docs/PIPELINE.md majsoul_eye/annotate/cases.py \
        majsoul_eye/annotate/pipeline.py docs/STATUS.md
git commit -m "docs(data): update val convention + STATUS/comments for source-root-qualified names"
```
(The memory file lives outside the repo — it is saved by the Write in Step 5, not committed here.)

---

### Task 3: End-to-end acceptance — dry-run over both roots

Prove the user's actual command works: two source roots coexist with no collision, names are qualified, and the default val resolves. Run against whatever is on disk (currently `ai_session2` is the renamed `run_21..23`; the mechanism is identical, and the exact same-run-number collision is already covered precisely by the `test_discover_games_source_qualified_and_empty` unit test).

**Files:** none modified (verification only).

- [ ] **Step 1: Dry-run the versioned build over both roots**

Run:
```bash
PYTHONPATH=. python scripts/data/build_datasets.py v2 \
    --sources captures/raw/ai_session captures/raw/ai_session2 --dry-run
```

- [ ] **Step 2: Confirm the acceptance criteria in the output**

Expected:
- Prints `DRY RUN datasets/v2  (N game(s), val=ai_session_run_8_game1)` — i.e. it does NOT abort with a `duplicate game name` / `collision` SystemExit, and the default val resolved to `ai_session_run_8_game1`.
- The `games:` line lists both `ai_session_run_*` names AND `ai_session2_run_*` names (on the current disk the latter appear as `ai_session2_run_21_game1`, … — with `raw.7z`'s pristine layout extracted they would be `ai_session2_run_1_game1`, …).
- The printed per-game `$ … build_dataset.py …/ai_session2/run_*/… --out datasets/v2/ai_session2_run_*` commands carry the qualified `--out` dirs (no bare `ai_run_*` for the ai_session2 games).

If it instead aborts with "no captures under …" for a root, that root has no discoverable games on disk (e.g. `ai_session/run_1`'s GTRecord jsonl is missing per STATUS §1.27) — that is a disk-state issue, not a code regression; note it and, if needed, re-run pointing only at roots that have data.

- [ ] **Step 3: (No commit)** This task produces no file changes; it is the final acceptance gate. If it surfaces a defect, fix in the relevant Task-1 file and re-run its tests before re-running this dry-run.

---

## Self-Review

**Spec coverage:** core `ai_game_name` change → Task 1 Step 6. Three-stage consistency → inherent (all call the one function). `discover_games` collision reword → Task 1 Step 7. `migrate_ai` duplicate namer (found during planning; DRY'd) → Task 1 Step 8. `DEFAULT_VAL` + `FRAMES_OVERRIDE` → Task 1 Step 7. All four affected test files → Task 1 Steps 1-4. Collision-test flip → Task 1 Step 4. Live docs → Task 2 Steps 2-3. STATUS entry → Task 2 Step 4. Memory rule drop → Task 2 Step 5. "NOT touched" (precise_* dirs, contamination) → Global Constraints. Acceptance dry-run → Task 3. Covered.

**Placeholder scan:** every code/edit step carries verbatim content; no TBD/TODO/"handle edge cases".

**Type consistency:** `ai_game_name(str) -> str` signature unchanged across all steps; new names (`ai_session_run_8_game1`, `ai_session2_run_1_game1`, `ai_session_run_5_game2`) are spelled identically in the function, constants, tests, and docs; `DEFAULT_VAL` value matches the val name asserted in the dry-run acceptance.
