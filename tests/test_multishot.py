"""MultiShot (extra-shot scheduler, Task 15) unit tests.

Covers the pure scheduler contract (arm/due, one-shot firing, supersede-on-
rearm cancellation, window=False plans nothing) plus a pin that
`gtframes.load_frames` already ignores `"extra"`-status lines for both the
annotator's default filter and build_dataset's `("ok", "timeout")` filter,
and that the canonical seq->file mapping is unchanged when extra lines are
interleaved with canonical ones. Also tests multishot_window() decision logic
for meld types, operations, and syncing-record semantics.

Plain-script style: PYTHONPATH=. <auto-python> tests/test_multishot.py
(also pytest-compatible).
"""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile

from majsoul_eye.capture.multishot import MultiShot
from majsoul_eye.capture.gtframes import load_frames


# --- MultiShot: arm/due basic ------------------------------------------------

def test_no_shot_before_first_offset():
    ms = MultiShot(offsets=(0.6, 1.2, 2.4))
    ms.arm(1, 100.0, True)
    assert ms.due(100.0) == []
    assert ms.due(100.5) == []


def test_offsets_fire_at_default_thresholds():
    ms = MultiShot()   # default offsets (0.6, 1.2, 2.4)
    ms.arm(1, 0.0, True)
    assert ms.due(0.59) == []
    assert ms.due(0.6) == [(1, 600)]
    assert ms.due(1.19) == []
    assert ms.due(1.2) == [(1, 1200)]
    assert ms.due(2.4) == [(1, 2400)]


def test_multiple_offsets_due_in_same_tick():
    ms = MultiShot(offsets=(0.6, 1.2, 2.4))
    ms.arm(7, 0.0, True)
    assert ms.due(3.0) == [(7, 600), (7, 1200), (7, 2400)]


# --- one-shot firing ----------------------------------------------------

def test_each_offset_fires_exactly_once():
    ms = MultiShot(offsets=(0.6,))
    ms.arm(5, 0.0, True)
    assert ms.due(0.7) == [(5, 600)]
    assert ms.due(0.7) == []          # already fired: not returned again
    assert ms.due(100.0) == []        # still not returned, ever


# --- window=False: no plan ----------------------------------------------

def test_window_false_plans_nothing():
    ms = MultiShot(offsets=(0.6, 1.2, 2.4))
    ms.arm(3, 0.0, False)
    assert ms.due(0.0) == []
    assert ms.due(100.0) == []


# --- supersede on re-arm cancels the old plan ---------------------------

def test_rearm_supersedes_unfired_offsets_of_old_seq():
    ms = MultiShot(offsets=(0.6, 1.2))
    ms.arm(1, 100.0, True)
    assert ms.due(100.7) == [(1, 600)]     # seq=1's first offset fires
    # a new board event arrives before seq=1's 1.2s offset would fire
    ms.arm(2, 100.65, True)
    # seq=1's leftover 1200ms offset must never appear again, even once
    # plenty of real time has passed for it
    assert ms.due(200.0) == [(2, 600), (2, 1200)]


def test_rearm_with_window_false_clears_prior_plan():
    ms = MultiShot(offsets=(0.6,))
    ms.arm(1, 0.0, True)
    ms.arm(2, 0.0, False)     # supersede with a non-window seq
    assert ms.due(10.0) == []


def test_rearm_same_seq_resets_plan():
    # Even re-arming the SAME seq number restarts the plan from its new
    # event_t (arm() never "adds" offsets or accumulates state).
    ms = MultiShot(offsets=(0.6, 1.2))
    ms.arm(1, 0.0, True)
    assert ms.due(0.6) == [(1, 600)]
    ms.arm(1, 10.0, True)
    assert ms.due(10.5) == []
    assert ms.due(10.6) == [(1, 600)]
    assert ms.due(11.2) == [(1, 1200)]


# --- load_frames: "extra" lines are invisible; canonical mapping intact -

def _write_jsonl(path, lines):
    with open(path, "w", encoding="utf-8") as fh:
        for d in lines:
            fh.write(json.dumps(d) + "\n")


def _touch(frames_dir, *names):
    # load_frames -> paths.resolve_frame_path only resolves to the
    # index-relative candidate when the file actually exists on disk;
    # otherwise it falls back to returning the jsonl "file" string
    # unchanged. Touch real (empty) files so this test exercises the real
    # resolution path, matching production frames/ directories.
    os.makedirs(frames_dir, exist_ok=True)
    for name in names:
        open(os.path.join(frames_dir, name), "a", encoding="utf-8").close()


def test_load_frames_ignores_extra_lines_default_statuses():
    with tempfile.TemporaryDirectory() as d:
        _touch(os.path.join(d, "frames"), "000001.png", "000001_dt0600.png",
               "000001_dt1200.png", "000002.png")
        _write_jsonl(os.path.join(d, "frames.jsonl"), [
            {"seq": 1, "file": "frames/000001.png", "status": "ok", "ts": 1.0, "dt": 0.31},
            {"seq": 1, "file": "frames/000001_dt0600.png", "status": "extra", "ts": 1.6, "dt": 0.60},
            {"seq": 1, "file": "frames/000001_dt1200.png", "status": "extra", "ts": 2.2, "dt": 1.21},
            {"seq": 2, "file": "frames/000002.png", "status": "timeout", "ts": 3.0, "dt": 1.0},
        ])
        frames = load_frames(d)   # default statuses=("ok",)
        assert frames == {1: os.path.join(d, "frames/000001.png")}


def test_load_frames_ignores_extra_lines_ok_timeout_statuses():
    with tempfile.TemporaryDirectory() as d:
        _touch(os.path.join(d, "frames"), "000001.png", "000001_dt0600.png",
               "000002.png", "000002_dt2400.png")
        _write_jsonl(os.path.join(d, "frames.jsonl"), [
            {"seq": 1, "file": "frames/000001.png", "status": "ok", "ts": 1.0, "dt": 0.31},
            {"seq": 1, "file": "frames/000001_dt0600.png", "status": "extra", "ts": 1.6, "dt": 0.60},
            {"seq": 2, "file": "frames/000002.png", "status": "timeout", "ts": 3.0, "dt": 1.0},
            {"seq": 2, "file": "frames/000002_dt2400.png", "status": "extra", "ts": 5.4, "dt": 2.41},
        ])
        frames = load_frames(d, statuses=("ok", "timeout"))
        assert frames == {
            1: os.path.join(d, "frames/000001.png"),
            2: os.path.join(d, "frames/000002.png"),
        }


def test_load_frames_canonical_mapping_unchanged_by_interleaved_extras():
    # Same canonical set/order as if the "extra" lines had never been written,
    # regardless of where among the canonical lines they're interleaved.
    with tempfile.TemporaryDirectory() as d:
        _touch(os.path.join(d, "frames"), "000001.png", "000002.png", "000003.png",
               "000001_dt0600.png", "000002_dt1200.png", "000002_dt2400.png")
        without_extra = [
            {"seq": 1, "file": "frames/000001.png", "status": "ok", "ts": 1.0, "dt": 0.31},
            {"seq": 2, "file": "frames/000002.png", "status": "ok", "ts": 2.0, "dt": 0.28},
            {"seq": 3, "file": "frames/000003.png", "status": "ok", "ts": 3.0, "dt": 0.33},
        ]
        with_extra = [
            without_extra[0],
            {"seq": 1, "file": "frames/000001_dt0600.png", "status": "extra", "ts": 1.6, "dt": 0.60},
            without_extra[1],
            {"seq": 2, "file": "frames/000002_dt1200.png", "status": "extra", "ts": 2.8, "dt": 1.19},
            {"seq": 2, "file": "frames/000002_dt2400.png", "status": "extra", "ts": 4.0, "dt": 2.39},
            without_extra[2],
        ]
        _write_jsonl(os.path.join(d, "frames.jsonl"), without_extra)
        baseline = load_frames(d)
        _write_jsonl(os.path.join(d, "frames.jsonl"), with_extra)
        interleaved = load_frames(d)
        assert interleaved == baseline
        assert set(interleaved) == {1, 2, 3}


# --- multishot_window: test the decision logic ---------------------------

def _load_autoplay():
    """Load autoplay_ai as a module to access multishot_window and related funcs."""
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "capture", "autoplay_ai.py")
    spec = importlib.util.spec_from_file_location("autoplay_ai_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _MockRecord:
    """Minimal GTRecord-like object for testing."""
    def __init__(self, mjai=None, raw_liqi=None, syncing=False, seat=0):
        self.mjai = mjai
        self.raw_liqi = raw_liqi
        self.syncing = syncing
        self.seat = seat


def test_multishot_window_none_record():
    """None record (no GT) → False."""
    mod = _load_autoplay()
    assert mod.multishot_window(None) is False


def test_multishot_window_mjai_meld_types():
    """Record with mjai containing meld types (pon, chi, etc.) → True."""
    mod = _load_autoplay()
    rec = _MockRecord(mjai=[{"type": "pon", "tile": "5m"}])
    assert mod.multishot_window(rec) is True

    rec = _MockRecord(mjai=[{"type": "chi", "tiles": ["1m", "2m", "3m"]}])
    assert mod.multishot_window(rec) is True

    rec = _MockRecord(mjai=[{"type": "daiminkan", "tile": "5p"}])
    assert mod.multishot_window(rec) is True

    rec = _MockRecord(mjai=[{"type": "ankan", "tile": "2s"}])
    assert mod.multishot_window(rec) is True

    rec = _MockRecord(mjai=[{"type": "kakan", "tile": "6m"}])
    assert mod.multishot_window(rec) is True

    rec = _MockRecord(mjai=[{"type": "nukidora"}])
    assert mod.multishot_window(rec) is True


def test_multishot_window_dahai_only_no_buttons():
    """Record with only dahai (no meld) and no button-offering operation → False."""
    mod = _load_autoplay()
    # dahai with no operation
    rec = _MockRecord(
        mjai=[{"type": "dahai", "pai": "5m"}],
        raw_liqi={"data": {"data": {"operation": None}}}
    )
    assert mod.multishot_window(rec) is False


def test_multishot_window_operation_type_1_dahai():
    """Record with operation type 1 (dahai, no button) → False."""
    mod = _load_autoplay()
    rec = _MockRecord(
        mjai=[{"type": "dahai", "pai": "5m"}],
        raw_liqi={
            "data": {"data": {"operation": {
                "seat": 0,
                "operationList": [{"type": 1}]
            }}}
        },
        seat=0
    )
    assert mod.multishot_window(rec) is False


def test_multishot_window_operation_type_9_ron():
    """Record with operation type 9 (ron, has button) → True."""
    mod = _load_autoplay()
    rec = _MockRecord(
        mjai=[{"type": "dahai", "pai": "5m"}],
        raw_liqi={
            "data": {"data": {"operation": {
                "seat": 0,
                "operationList": [{"type": 9}]
            }}}
        },
        seat=0
    )
    assert mod.multishot_window(rec) is True


def test_multishot_window_operation_mixed_types():
    """Record with mixed operation types (some have buttons) → True."""
    mod = _load_autoplay()
    # Types 1 and 9: dahai + ron; ron is button-worthy
    rec = _MockRecord(
        mjai=[{"type": "dahai", "pai": "5m"}],
        raw_liqi={
            "data": {"data": {"operation": {
                "seat": 0,
                "operationList": [{"type": 1}, {"type": 9}]
            }}}
        },
        seat=0
    )
    assert mod.multishot_window(rec) is True


def test_multishot_window_syncing_record_with_meld():
    """Syncing records (reconnect replays) should NOT arm extra shots, even with meld."""
    mod = _load_autoplay()
    # Syncing record with a meld event: should return False (stale data, no shot)
    rec = _MockRecord(
        mjai=[{"type": "pon", "tile": "5m"}],
        syncing=True
    )
    assert mod.multishot_window(rec) is False


def test_multishot_window_syncing_record_with_operation():
    """Syncing records block extra shots even if there's a button operation."""
    mod = _load_autoplay()
    rec = _MockRecord(
        mjai=[{"type": "dahai", "pai": "5m"}],
        raw_liqi={
            "data": {"data": {"operation": {
                "seat": 0,
                "operationList": [{"type": 9}]  # ron button
            }}}
        },
        syncing=True,
        seat=0
    )
    assert mod.multishot_window(rec) is False


def test_multishot_window_empty_mjai_no_operation():
    """Record with empty mjai and no operation → False."""
    mod = _load_autoplay()
    rec = _MockRecord(mjai=[], raw_liqi={"data": {"data": {}}})
    assert mod.multishot_window(rec) is False


def test_multishot_window_no_mjai():
    """Record with None mjai and no operation → False."""
    mod = _load_autoplay()
    rec = _MockRecord(mjai=None, raw_liqi={"data": {"data": {}}})
    assert mod.multishot_window(rec) is False


def test_multishot_cancel_method():
    """Test the explicit cancel() method."""
    ms = MultiShot(offsets=(0.6, 1.2))
    ms.arm(1, 100.0, True)
    # Verify it was armed
    assert ms.due(100.6) == [(1, 600)]
    # Re-arm and cancel explicitly
    ms.arm(2, 101.0, True)
    ms.cancel()
    # Should fire nothing
    assert ms.due(200.0) == []


def test_multishot_sorted_offsets():
    """Test that offsets are sorted ascending at init."""
    # Provide offsets out of order
    ms = MultiShot(offsets=(2.4, 0.6, 1.2))
    ms.arm(1, 0.0, True)
    # due() should return in ascending order
    result = ms.due(3.0)
    assert result == [(1, 600), (1, 1200), (1, 2400)]


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    _main()
