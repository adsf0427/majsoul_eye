"""Launch Akagi with the majsoul_eye GT recorder installed.

Routes nothing itself — it just injects the tap, then hands off to Akagi's normal
startup. Configure Akagi's MITM as usual and play / 观战 a game with autoplay OFF
(passive capture; see docs/DESIGN.md §3.1, §7). Output is a JSONL capture.

Usage:
    python scripts/record_gt.py --out captures/session1.jsonl
    python scripts/record_gt.py --akagi-dir D:/code/phoenix/Akagi --out captures/s.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # majsoul_eye repo root


def main() -> None:
    ap = argparse.ArgumentParser(description="Record Mahjong Soul GT via Akagi MITM.")
    ap.add_argument(
        "--akagi-dir",
        default=os.environ.get("AKAGI_DIR", str(REPO_ROOT.parent / "Akagi")),
        help="Path to the Akagi repo (default: ../Akagi or $AKAGI_DIR).",
    )
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "captures" / f"session_{datetime.now():%Y%m%d_%H%M%S}.jsonl"),
        help="Output JSONL capture path.",
    )
    ap.add_argument(
        "--screenshots",
        action="store_true",
        help="Also capture window screenshots time-synced to each board change (P2).",
    )
    ap.add_argument("--quiet", type=float, default=0.30,
                    help="Capture once the board has had no events for this many seconds.")
    ap.add_argument("--settle-cap", type=float, default=2.0,
                    help="Force a capture if a burst of events runs this long.")
    args = ap.parse_args()

    akagi_dir = Path(args.akagi_dir).resolve()
    if not (akagi_dir / "run_akagi.py").exists():
        sys.exit(f"Akagi not found at {akagi_dir} (no run_akagi.py). Pass --akagi-dir.")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Make majsoul_eye importable, then Akagi.
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(akagi_dir))
    # Akagi resolves logs/settings relative to CWD.
    os.chdir(akagi_dir)

    from majsoul_eye.capture.akagi_tap import install, recorded_count  # noqa: E402

    # Akagi runs a Textual TUI that owns the terminal — printing to stdout/stderr
    # corrupts it. Route all recorder status to a sidecar log file the user can
    # `tail -f` instead.
    log_path = out_path.with_suffix(out_path.suffix + ".log")

    def log(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%H:%M:%S} {msg}\n")

    # Any uncaught thread exception otherwise prints a traceback to stderr and
    # corrupts Akagi's Textual TUI. Route them to the log file instead.
    import threading

    def _thread_excepthook(a):
        log(f"thread {a.thread.name} error: {a.exc_type.__name__}: {a.exc_value}")

    threading.excepthook = _thread_excepthook

    syncer = None
    if args.screenshots:
        from majsoul_eye.capture.screen import ScreenGrabber  # noqa: E402
        from majsoul_eye.capture.sync import FrameSyncer  # noqa: E402
        frames_dir = out_path.with_suffix("")  # captures/<stem>/  (frames + frames.jsonl)
        frames_dir.mkdir(parents=True, exist_ok=True)
        grabber = ScreenGrabber()
        syncer = FrameSyncer(
            grab=grabber.grab, out_dir=str(frames_dir),
            quiet=args.quiet, settle_cap=args.settle_cap,
        )
        syncer.start()
        log(f"Screenshot sync ON -> {frames_dir} (quiet {args.quiet}s, cap {args.settle_cap}s)")

    install(str(out_path), syncer=syncer)
    log(f"GT recorder installed -> {out_path}")

    # Akagi's bridge/mitm loggers are loguru and inherit loguru's DEFAULT
    # sys.stderr sink (handler id 0), which prints every DEBUG line to the
    # terminal and corrupts the Textual TUI. Drop just that console sink; the
    # explicit per-module file sinks (logs/bridge_*.log etc.) are untouched.
    try:
        from loguru import logger as _loguru
        _loguru.remove(0)
        log("Removed loguru default stderr sink (was corrupting the TUI).")
    except Exception as e:  # already removed / different id
        log(f"loguru stderr sink not removed: {e}")

    log("Starting Akagi. Play/观战 a game (autoplay OFF). Ctrl-C to stop.")

    from akagi.akagi import main as akagi_main  # noqa: E402

    try:
        akagi_main()
    finally:
        if syncer is not None:
            syncer.stop()
            log(f"Frame capture counts: {syncer.counts}")
        log(f"Recorded {recorded_count()} liqi messages -> {out_path}")


if __name__ == "__main__":
    main()
