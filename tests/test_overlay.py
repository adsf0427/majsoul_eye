"""Dependency-light tests for the live detection overlay's pure builders
(no browser, no ultralytics). Mirrors the plain-script style of the other suites."""
import json
import os
import sys

from types import SimpleNamespace

from majsoul_eye.capture import overlay as ov  # noqa: E402


def _hbb(xyxy, tile, cls, score):
    return SimpleNamespace(xyxy=tuple(xyxy), tile=tile, cls=cls, score=score, poly=None)


def _obb(xyxy, pts, tile, cls, score):
    return SimpleNamespace(xyxy=tuple(xyxy), tile=tile, cls=cls, score=score,
                           poly=tuple((float(x), float(y)) for x, y in pts))


def test_detections_to_ops_hbb_and_obb():
    dets = [
        _hbb([10, 20, 50, 60], "5m", 4, 0.91),
        _obb([8, 10, 30, 40], [[10, 10], [30, 12], [28, 40], [8, 38]], "1z", 27, 0.80),
    ]
    ops = ov.detections_to_ops(dets)
    assert ops[0] == {"kind": "rect", "xyxy": [10.0, 20.0, 50.0, 60.0],
                      "label": "5m", "score": 0.91}
    assert ops[1]["kind"] == "poly"
    assert ops[1]["pts"] == [[10.0, 10.0], [30.0, 12.0], [28.0, 40.0], [8.0, 38.0]]
    assert ops[1]["label"] == "1z" and ops[1]["score"] == 0.80


def test_detections_to_ops_empty():
    assert ov.detections_to_ops([]) == []


def test_render_js_embeds_ops_and_clears():
    ops = ov.detections_to_ops([_hbb([1, 2, 3, 4], "9p", 17, 0.5)])
    js = ov.render_js(ops, "cid")
    # references the canvas by id, clears, and embeds the ops as a JSON literal
    assert json.dumps("cid") in js
    assert "clearRect" in js
    assert json.dumps(ops) in js


def test_render_js_empty_is_valid_and_clears():
    js = ov.render_js([], "cid")
    assert "clearRect" in js
    assert "[]" in js          # empty ops array still emitted


def test_render_js_escapes_label():
    # a label with a quote must not break out of the JS string literal
    ops = [{"kind": "rect", "xyxy": [0, 0, 1, 1], "label": 'a"b', "score": 0.1}]
    js = ov.render_js(ops, "cid")
    assert json.dumps(ops) in js          # json.dumps handles the escaping


def test_inject_js_sets_backing_store_to_shot_dims():
    js = ov.inject_js("cid", 1280, 720)
    assert json.dumps("cid") in js
    assert "createElement" in js and "appendChild" in js
    assert "pointerEvents" in js and "9999999" in js
    assert "c.width=1280" in js and "c.height=720" in js


def test_hide_show_js():
    assert "'hidden'" in ov.hide_canvas_js("cid") and json.dumps("cid") in ov.hide_canvas_js("cid")
    assert "'visible'" in ov.show_canvas_js("cid") and json.dumps("cid") in ov.show_canvas_js("cid")


def test_canvas_id_constant():
    assert ov.OVERLAY_CANVAS_ID == "majsoul_eye_overlay"


def _png_bytes(w=128, h=72):
    import cv2
    import numpy as np
    ok, buf = cv2.imencode(".png", np.zeros((h, w, 3), np.uint8))
    assert ok
    return buf.tobytes()


class _FakeDetector:
    def __init__(self, dets):
        self.dets = dets
        self.calls = 0

    def predict(self, bgr):
        self.calls += 1
        return self.dets


def test_overlay_tick_injects_once_then_renders():
    js_log = []
    det = _FakeDetector([_hbb([1, 2, 3, 4], "5m", 4, 0.9)])
    o = ov.DetectionOverlay(capture_png=_png_bytes, eval_js=js_log.append,
                            weights="unused", canvas_id="cid", detector=det)
    o._tick()
    # first tick: inject (sized to the 128x72 png) THEN render
    assert len(js_log) == 2
    assert "createElement" in js_log[0] and "c.width=128" in js_log[0] and "c.height=72" in js_log[0]
    assert "clearRect" in js_log[1]
    o._tick()
    # second tick: render only (no re-inject)
    assert len(js_log) == 3 and "createElement" not in js_log[2]
    assert det.calls == 2


def test_overlay_tick_skips_when_no_png():
    js_log = []
    det = _FakeDetector([])
    o = ov.DetectionOverlay(capture_png=lambda: None, eval_js=js_log.append,
                            weights="unused", canvas_id="cid", detector=det)
    o._tick()
    assert js_log == [] and det.calls == 0


def test_overlay_start_stop_runs_ticks():
    import time
    js_log = []
    det = _FakeDetector([])
    o = ov.DetectionOverlay(capture_png=_png_bytes, eval_js=js_log.append,
                            weights="unused", fps=50, canvas_id="cid", detector=det)
    o.start()
    time.sleep(0.2)
    o.stop()
    assert det.calls >= 1                     # the loop ran at least once
    assert not o._thread.is_alive()           # stop() joined the daemon


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_overlay OK")
