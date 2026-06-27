"""Deterministic tests for the debounce-to-quiet capture logic (no live client)."""

import os
import tempfile

import numpy as np

from majsoul_eye.capture.sync import FrameSyncer, frame_diff


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt

    def advance(self, dt):
        self.t += dt


def _img(v):
    return np.full((4, 4, 3), v, dtype=np.uint8)


def _mk(grab, clock, **kw):
    d = tempfile.mkdtemp()
    kw.setdefault("quiet", 0.30)
    kw.setdefault("settle_cap", 2.0)
    s = FrameSyncer(grab=grab, out_dir=d, now=clock.now, sleep=clock.sleep, **kw)
    os.makedirs(s.frames_dir, exist_ok=True)
    return s


def _pump(s, clock, n=5):
    """Tick _maybe_capture a few times (stability confirm needs prime+confirm)."""
    for _ in range(n):
        if s._maybe_capture():
            return True
        clock.advance(s.poll)
    return False


def test_captures_after_quiet():
    clock = FakeClock()
    s = _mk(lambda: _img(100), clock)
    s.on_event(5)                           # t=0
    assert s._maybe_capture() is False      # not quiet yet
    clock.advance(0.31)
    assert _pump(s, clock) is True          # quiet + picture stable -> capture
    assert s._fulfilled == 5 and s.counts["ok"] == 1
    clock.advance(1.0)
    assert s._maybe_capture() is False      # same step not recaptured


def test_waits_while_events_streaming():
    clock = FakeClock()
    s = _mk(lambda: _img(1), clock)
    s.on_event(5)                           # t=0
    clock.advance(0.2); s.on_event(6)       # reset quiet at t=0.2
    clock.advance(0.2)                      # t=0.4, only 0.2 since last event
    assert s._maybe_capture() is False
    clock.advance(0.2)                      # t=0.6, 0.4 since last event >= quiet
    assert _pump(s, clock) is True
    assert s._fulfilled == 6


def test_settle_cap_forces_capture_during_long_burst():
    clock = FakeClock()
    s = _mk(lambda: _img(1), clock, settle_cap=1.0)
    s.on_event(5)                           # pending_since=0
    captured = False
    for _ in range(30):                     # stream events so quiet never holds
        clock.advance(0.1)
        s.on_event(s._pending_key + 1)
        if s._maybe_capture():
            captured = True
            break
    assert captured and s.counts["ok"] == 1


def test_new_event_during_grab_retries():
    clock = FakeClock()
    s = _mk(lambda: _img(5), clock)

    def grab():
        s.on_event(99)                      # a newer event lands mid-grab
        return _img(5)

    s.grab = grab
    s.on_event(5); clock.advance(0.31)
    assert s._maybe_capture() is False      # superseded -> not fulfilled
    assert s._fulfilled != 5


def test_no_capture_when_grab_none():
    clock = FakeClock()
    s = _mk(lambda: None, clock)
    s.on_event(5); clock.advance(0.31)
    assert s._maybe_capture() is True
    assert s.counts["no_capture"] == 1


def test_note_relevance_gate():
    clock = FakeClock()
    s = _mk(lambda: _img(1), clock)
    s.note([{"type": "heartbeat"}], 5)
    assert s._pending_key is None
    s.note([{"type": "dahai", "actor": 0, "pai": "1m"}], 7)
    assert s._pending_key == 7


def test_waits_for_picture_to_stabilize():
    # discard animation: frames change, then settle. Capture only once still.
    clock = FakeClock()
    seq = [_img(0), _img(50), _img(100), _img(200), _img(200), _img(200)]
    it = iter(seq)
    s = _mk(lambda: next(it, _img(200)), clock)
    s.on_event(5)
    clock.advance(0.31)                     # event-quiet, but picture still moving
    assert _pump(s, clock, n=10) is True
    assert s.counts["ok"] == 1 and s._fulfilled == 5


def test_frame_diff():
    assert frame_diff(_img(10), _img(10)) == 0.0
    assert frame_diff(_img(10), _img(20)) == 10.0
    assert frame_diff(np.zeros((2, 2, 3)), np.zeros((3, 3, 3))) > 1e8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_sync OK")
