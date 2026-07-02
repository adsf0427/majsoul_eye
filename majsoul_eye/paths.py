"""Central definition of the ``captures/`` layout + frame-path resolution.

Single source of truth for where raw vs. intermediate capture data lives and how
a ``frames.jsonl`` ``file`` field resolves to an on-disk PNG. Import this instead
of hardcoding ``"captures/..."`` paths or re-deriving the frames-dir stem rule.

Layout::

    captures/
      raw/            ai_session/ (MahjongCopilot)  +  manual/ (record_gt sessions)
      intermediate/   gt/ (converted GT + hollow indexes)  +  derived/ (cropped / de-letterboxed)
      legacy/         archived byte-identical duplicates

Output-role data (``datasets/``, ``out/``, ``fails/``) lives OUTSIDE ``captures/``.

``frames.jsonl`` ``file`` entries are stored RELATIVE (new layout): index-relative
(``frames/000009.png``) for self-contained frame dirs, or captures-relative
(``raw/ai_session/run_3/game1/frames/000009.png``) for the hollow ``gt/`` indexes
that point back into the raw tree. ``resolve_frame_path`` also still accepts the
legacy absolute paths, so un-migrated indexes keep loading.
"""
from __future__ import annotations

import glob as _glob
import os

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


def converted_gt_captures() -> list:
    """Sorted converted-GT capture jsonl files under intermediate/gt/."""
    return sorted(_glob.glob(os.path.join(GT, "*.jsonl")))


def manual_captures() -> list:
    """Sorted manual F11 session capture jsonl files under raw/manual/."""
    return sorted(_glob.glob(os.path.join(RAW_MANUAL, "session*.jsonl")))
