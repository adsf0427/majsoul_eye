"""Guard: no dangling captures/intermediate/gt references remain in code, and the AB
validation CASES point at the live raw/ai_session layout.

Plain-script style: PYTHONPATH=. <auto-python> tests/test_no_stale_gt_refs.py
"""
import os
import re
import glob

# Files that LEGITIMATELY still reference the retired dir symbolically:
#   paths.py  — defines the vestigial GT constant + documents the retirement
#   convert_mjcopilot.py — --out default is paths.GT for the still-runnable legacy CLI
_ALLOW = {os.path.normpath("majsoul_eye/paths.py"),
          os.path.normpath("scripts/data/convert_mjcopilot.py"),
          # prior captures-layout migration tool: names intermediate/gt as its own
          # (historical, completed) destination — not a dangling reference.
          os.path.normpath("scripts/data/migrate_captures_layout.py")}
_PAT = re.compile(r"intermediate.{0,4}gt")


def _code_files():
    for base in ("majsoul_eye", "scripts"):
        for p in glob.glob(os.path.join(base, "**", "*.py"), recursive=True):
            yield p


def test_no_dangling_intermediate_gt_in_code():
    offenders = []
    for p in _code_files():
        if os.path.normpath(p) in _ALLOW:
            continue
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if _PAT.search(line):
                    offenders.append(f"{p}:{i}: {line.strip()}")
    assert not offenders, "stale intermediate/gt references (retired):\n" + "\n".join(offenders)


def test_cases_point_at_raw_ai_session():
    from majsoul_eye.annotate import cases
    for k, c in cases.CASES.items():
        cap = c["capture"].replace("\\", "/")
        assert "intermediate/gt" not in cap, f"{k}: {cap}"
        assert "raw/ai_session" in cap, f"{k}: {cap}"


def test_cases_capture_files_exist_if_captures_present():
    # Data-dependent: only assert existence when the AI capture tree is present
    # (captures/ is gitignored; a fresh clone skips this).
    from majsoul_eye.annotate import cases
    if not os.path.isdir(os.path.join("captures", "raw", "ai_session")):
        return
    missing = [f"{k}: {c['capture']}" for k, c in cases.CASES.items()
               if not os.path.exists(c["capture"])]
    assert not missing, "CASES capture files missing:\n" + "\n".join(missing)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_no_stale_gt_refs OK")
