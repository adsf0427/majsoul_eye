"""Central definition of the ``captures/`` layout + frame-path resolution.

Single source of truth for where raw vs. intermediate capture data lives and how
a ``frames.jsonl`` ``file`` field resolves to an on-disk PNG. Import this instead
of hardcoding ``"captures/..."`` paths or re-deriving the frames-dir stem rule.

Layout::

    captures/
      raw/            ai_session/ (GTRecord, written inline by autoplay_ai.py — same
                      format as manual)  +  manual/ (record_gt sessions)
      intermediate/   derived/ (cropped / de-letterboxed) — ``gt/`` (converted GT +
                      hollow indexes) is RETIRED; AI captures now write ``GTRecord``
                      directly under ``raw/ai_session/`` instead of a separate
                      converted tree (the ``GT`` constant below survives only for the
                      vestigial standalone ``convert_mjcopilot`` CLI / one-time legacy
                      migration).
      legacy/         archived byte-identical duplicates

Output-role data (``datasets/``, ``out/``, ``fails/``) lives OUTSIDE ``captures/``.

``frames.jsonl`` ``file`` entries are stored RELATIVE (new layout): index-relative
(``frames/000009.png``) for self-contained frame dirs. (Captures-relative entries
like ``raw/ai_session/run_3/game1/frames/000009.png``, used by the old hollow
``gt/`` indexes to point back into the raw tree, are a legacy form now that ``gt/``
is retired.) ``resolve_frame_path`` also still accepts the legacy absolute paths,
so un-migrated indexes keep loading.
"""
from __future__ import annotations

import glob as _glob
import os
import re

# Repo root = parent of the majsoul_eye package dir. All tooling runs from here.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Layout constants are repo-root-relative strings (all scripts run with cwd = repo
# root and PYTHONPATH=.); use them directly in globs / argparse defaults.
CAPTURES = "captures"
RAW = os.path.join(CAPTURES, "raw")
RAW_AI_SESSION = os.path.join(RAW, "ai_session")
RAW_MANUAL = os.path.join(RAW, "manual")
INTERMEDIATE = os.path.join(CAPTURES, "intermediate")
GT = os.path.join(INTERMEDIATE, "gt")
DERIVED = os.path.join(INTERMEDIATE, "derived")
LEGACY = os.path.join(CAPTURES, "legacy")

# Absolute captures root, for resolving relative index entries regardless of cwd.
CAPTURES_ABS = os.path.join(REPO_ROOT, CAPTURES)


def frames_dir_for(capture: str) -> str:
    """``captures/.../ai_run_3_game1.jsonl`` -> ``captures/.../ai_run_3_game1``.

    The frames dir is the capture path with its ``.jsonl`` suffix stripped — a
    sibling with the same stem. The one place this ``X.jsonl <-> X/`` coupling
    is defined.
    """
    stem, _ = os.path.splitext(str(capture))
    return stem


def rel_to_captures(path: str) -> str:
    """POSIX path of ``path`` relative to the captures/ root (for writing indexes).

    e.g. ``.../captures/raw/ai_session/run_3/game1/frames/000009.png`` ->
    ``raw/ai_session/run_3/game1/frames/000009.png``.
    """
    rel = os.path.relpath(os.path.abspath(path), CAPTURES_ABS)
    return rel.replace(os.sep, "/")


def rel_frame(png_path: str, index_dir: str) -> str:
    """POSIX path of a PNG relative to its own index dir (self-contained indexes).

    e.g. ``.../session5_16x9/frames/000009.png`` with index_dir ``.../session5_16x9``
    -> ``frames/000009.png``. Portable: the frame dir can move anywhere.
    """
    rel = os.path.relpath(os.path.abspath(png_path), os.path.abspath(index_dir))
    return rel.replace(os.sep, "/")


def _reroot_at_captures(abs_path: str):
    """Rebuild a (possibly moved) absolute path under the current CAPTURES_ABS by
    re-anchoring at its last ``captures`` path segment — rescues indexes whose
    absolute paths point at an old captures location. Returns None if no segment."""
    parts = abs_path.replace("\\", "/").split("/")
    if "captures" in parts:
        i = len(parts) - 1 - parts[::-1].index("captures")
        return os.path.join(REPO_ROOT, *parts[i:])
    return None


def resolve_frame_path(file_field: str, index_dir=None) -> str:
    """Resolve a ``frames.jsonl`` ``file`` entry to a usable on-disk path.

    Handles relative (new) and absolute (legacy) entries. Order: absolute-and-exists
    first (legacy back-compat), then index-relative, then captures-relative, then a
    moved-absolute re-root, then basename fallbacks; else return the input unchanged
    so ``cv2.imread`` fails loudly with the real path.
    """
    f = file_field
    cands = []
    if os.path.isabs(f):
        cands.append(f)                                       # 1. legacy absolute
    if index_dir is not None:
        cands.append(os.path.join(index_dir, f))             # 2. index-relative
    cands.append(os.path.join(CAPTURES, f))                  # 3. captures-relative (cwd)
    cands.append(os.path.join(CAPTURES_ABS, f))              #    captures-relative (repo-anchored)
    if os.path.isabs(f):
        rr = _reroot_at_captures(f)                          # 4. moved-absolute rescue
        if rr:
            cands.append(rr)
    if index_dir is not None:
        base = os.path.basename(f)
        cands.append(os.path.join(index_dir, base))          # 5. basename fallbacks
        cands.append(os.path.join(index_dir, "frames", base))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return file_field                                        # 6. fail loud with the real path


def ai_game_name(capture_path: str) -> str:
    """Stable flattened dataset name for an AI GTRecord capture.

    ``.../run_N/gameM.jsonl`` -> ``ai_run_N_gameM``;
    ``.../run_1.jsonl``       -> ``ai_run_1`` (single-game legacy run);
    anything else             -> the basename stem (manual sessions pass through).
    """
    p = os.path.abspath(capture_path).replace("\\", "/")
    parts = p.split("/")
    stem = os.path.splitext(parts[-1])[0]                 # gameM  or  run_N
    parent = parts[-2] if len(parts) >= 2 else ""
    if re.fullmatch(r"run_\d+", parent) and re.fullmatch(r"game\d+", stem):
        return f"ai_{parent}_{stem}"
    if re.fullmatch(r"run_\d+", stem):
        return f"ai_{stem}"
    return stem


def _ai_captures_in(ai_session_dir: str) -> list:
    """AI GTRecord jsonls under a given ai_session root (test seam for ai_captures)."""
    multi = _glob.glob(os.path.join(ai_session_dir, "run_*", "game*.jsonl"))
    single = _glob.glob(os.path.join(ai_session_dir, "run_*.jsonl"))
    return sorted(multi + single)


def ai_captures() -> list:
    """Sorted AI GTRecord capture jsonls under raw/ai_session/ (both shapes)."""
    return _ai_captures_in(RAW_AI_SESSION)


def converted_gt_captures() -> list:
    """Back-compat alias: AI GT captures now live in raw/ai_session/ (no convert)."""
    return ai_captures()


def manual_captures() -> list:
    """Sorted manual F11 session capture jsonl files under raw/manual/."""
    return sorted(_glob.glob(os.path.join(RAW_MANUAL, "session*.jsonl")))
