"""Shared (frame, GT BoardState) loading for the GT-driven annotator and the
dataset builder.

Both ``scripts/annotate/annotate_ai_session.py`` and ``scripts/train/build_dataset.py`` need the
same two things: a ``seq -> BoardState`` map replayed from a capture, and a
``seq -> frame path`` index. This module is the single source of that logic
(previously duplicated between ``scripts/annotate/spike_topdown.py`` and an inline loop in
``build_dataset.py``).
"""
from __future__ import annotations

import json
import os

from majsoul_eye import paths
from majsoul_eye.capture.schema import read_records
from majsoul_eye.capture.sync import RELEVANT_EVENTS
from majsoul_eye.state.replay import Replayer


def build_seq_state(capture: str) -> dict[int, object]:
    """seq -> BoardState snapshot at every board-changing record."""
    rp = Replayer()
    seq_state: dict[int, object] = {}
    for r in read_records(capture):
        rp.apply_record(r)
        if r.mjai and any(ev.get("type") in RELEVANT_EVENTS for ev in r.mjai):
            seq_state[r.seq] = rp.state.copy()
    return seq_state


def load_frames(frames_dir: str, statuses: tuple[str, ...] = ("ok",)) -> dict[int, str]:
    """seq -> resolved image path for frames whose status is in `statuses`.

    Defaults to 'ok' only (the annotator / calibration). build_dataset passes
    ('ok', 'timeout') to keep the same frame set it used before the merge.
    """
    out: dict[int, str] = {}
    path = os.path.join(frames_dir, "frames.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("status") in statuses and d.get("file"):
                seq = d.get("seq", d.get("step"))
                out[seq] = paths.resolve_frame_path(d["file"], frames_dir)
    return out
