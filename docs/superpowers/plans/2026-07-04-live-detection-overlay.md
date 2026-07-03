# Live Detection Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional, real-time visualizer to `scripts/capture/autoplay_ai.py` that draws the trained tile detector's boxes onto the live Mahjong Soul browser via an injected canvas, at ~10–20 fps, while capture continues.

**Architecture:** A new self-contained module `majsoul_eye/capture/overlay.py` holds pure JS/op builders plus a `DetectionOverlay` daemon-thread orchestrator driven by two injected callables (`capture_png`, `eval_js`) so it stays browser-agnostic and testable. `autoplay_ai.py` gains `--overlay` flags, injects the canvas once, upgrades its screenshot helper to a "clean" capture that hides the canvas during the shot (so dataset PNGs + detector input stay pristine), and starts/stops the overlay.

**Tech Stack:** Python 3.12 (`auto` conda env), ultralytics YOLO (`TileDetector`), OpenCV (`cv2`), NumPy, Playwright via MahjongCopilot's `GameBrowser` (CDP screenshots + serial action queue), CUDA (RTX 5080).

## Global Constraints

- Run everything from the repo root `D:\code\phoenix\majsoul_eye` with `PYTHONPATH=.` and the `auto` env python: `PY=C:/Users/zsx/miniforge3/envs/auto/python.exe`.
- Tests are **plain scripts** (no pytest): a file of `test_*()` functions ending with a `__main__` block that calls each and prints `test_<name> OK`. Use bare `assert`. Skip heavy/optional paths by `print("  (skip ...)")` + `return`.
- `recognize/` must stay Akagi-free. Import direction is one-way: `capture → recognize`, never the reverse. `overlay.py` lives in `capture/` and may import `recognize`.
- The 38-class taxonomy order is frozen — do not touch `tiles.py`.
- `overlay.py` must carry **no** import of MahjongCopilot, Akagi, or Playwright — it only receives callables. `ultralytics`/`torch` load lazily (only when the detector is actually built).
- `autoplay_ai.py` is a plain-stdout script (NOT the Akagi TUI), so `print(..., flush=True)` for status is fine and matches existing style.
- With `--overlay` absent, `autoplay_ai.py` behavior must be byte-identical to today.
- Fixed canvas element id shared by module + wiring: `OVERLAY_CANVAS_ID = "majsoul_eye_overlay"`.

---

### Task 1: Pure JS/op builders in `overlay.py`

Pure, browser-free functions: convert detections to draw-ops, render the draw JS, build the one-time canvas-inject JS, and build the hide/show JS used by the clean-capture path. All unit-tested without a browser or ultralytics.

**Files:**
- Create: `majsoul_eye/capture/overlay.py`
- Test: `tests/test_overlay.py`

**Interfaces:**
- Consumes: `majsoul_eye.recognize.detector.Detection` (fields `xyxy: tuple`, `tile: str`, `cls: int`, `score: float`, `poly: tuple|None`) — only in tests here; the pure functions take already-built `Detection` objects.
- Produces (relied on by Task 2 and Task 3):
  - `OVERLAY_CANVAS_ID: str = "majsoul_eye_overlay"`
  - `detections_to_ops(dets: list) -> list[dict]`
  - `render_js(ops: list[dict], canvas_id: str) -> str`
  - `inject_js(canvas_id: str, shot_w: int, shot_h: int) -> str`
  - `hide_canvas_js(canvas_id: str) -> str`
  - `show_canvas_js(canvas_id: str) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_overlay.py`:

```python
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_overlay OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. "$PY" tests/test_overlay.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'majsoul_eye.capture.overlay'`

- [ ] **Step 3: Write minimal implementation**

Create `majsoul_eye/capture/overlay.py` with the pure builders (the `DetectionOverlay` class is added in Task 2):

```python
"""Live detection-overlay renderer for the AI-autoplay capture path.

Draws the tile detector's boxes onto the live browser via an injected <canvas>.
Browser-agnostic: it takes two callables — ``capture_png()`` (clean screenshot)
and ``eval_js(js)`` (run JS on the page thread) — so it carries NO MahjongCopilot,
Akagi, or Playwright import and its pure builders are unit-testable. ``ultralytics``
/ ``torch`` load lazily via ``recognize.TileDetector`` only when a detector is built.

The canvas backing store is sized to the screenshot's pixels and CSS-scaled to the
full viewport, so detector boxes (already in screenshot px) draw verbatim.
"""
from __future__ import annotations

import json

OVERLAY_CANVAS_ID = "majsoul_eye_overlay"


def detections_to_ops(dets: list) -> list:
    """One draw-op per Detection. OBB (``poly`` set) -> a polygon op; HBB -> a rect op.
    Coordinates are the detector's screenshot pixels, passed through as floats."""
    ops = []
    for d in dets:
        if getattr(d, "poly", None) is not None:
            ops.append({"kind": "poly",
                        "pts": [[float(x), float(y)] for x, y in d.poly],
                        "label": d.tile, "score": float(d.score)})
        else:
            x0, y0, x1, y1 = d.xyxy
            ops.append({"kind": "rect",
                        "xyxy": [float(x0), float(y0), float(x1), float(y1)],
                        "label": d.tile, "score": float(d.score)})
    return ops


def render_js(ops: list, canvas_id: str) -> str:
    """JS that clears the canvas and strokes every op + label. Ops are embedded as a
    JSON literal so labels/coords are escaped by ``json.dumps`` (no injection risk)."""
    cid = json.dumps(canvas_id)
    payload = json.dumps(ops)
    return (
        "(() => {"
        f"const c = document.getElementById({cid}); if (!c) return;"
        "const ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height);"
        "ctx.lineWidth = 2; ctx.font = '16px monospace'; ctx.textBaseline = 'bottom';"
        "ctx.strokeStyle = 'lime'; ctx.fillStyle = 'lime';"
        f"for (const op of {payload}) {{"
        "  let lx, ly;"
        "  if (op.kind === 'rect') {"
        "    const [x0, y0, x1, y1] = op.xyxy;"
        "    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); lx = x0; ly = y0;"
        "  } else {"
        "    const p = op.pts; ctx.beginPath(); ctx.moveTo(p[0][0], p[0][1]);"
        "    for (let i = 1; i < p.length; i++) ctx.lineTo(p[i][0], p[i][1]);"
        "    ctx.closePath(); ctx.stroke(); lx = p[0][0]; ly = p[0][1];"
        "  }"
        "  ctx.fillText(op.label + ' ' + op.score.toFixed(2), lx, ly - 2);"
        "}"
        "})()"
    )


def inject_js(canvas_id: str, shot_w: int, shot_h: int) -> str:
    """Create (once) a fixed, full-viewport, click-through, top canvas whose backing
    store equals the screenshot dimensions. Idempotent: reuses an existing element."""
    cid = json.dumps(canvas_id)
    return (
        "(() => {"
        f"let c = document.getElementById({cid});"
        "if (!c) {"
        "  c = document.createElement('canvas');"
        f"  c.id = {cid};"
        "  c.style.position = 'fixed'; c.style.left = '0'; c.style.top = '0';"
        "  c.style.width = '100vw'; c.style.height = '100vh';"
        "  c.style.zIndex = '9999999'; c.style.pointerEvents = 'none';"
        "  document.body.appendChild(c);"
        "}"
        f"c.width={int(shot_w)}; c.height={int(shot_h)};"
        "})()"
    )


def hide_canvas_js(canvas_id: str) -> str:
    """Make the overlay canvas non-painted (retains its drawn pixels) for a clean shot."""
    cid = json.dumps(canvas_id)
    return f"(() => {{const c = document.getElementById({cid}); if (c) c.style.visibility = 'hidden';}})()"


def show_canvas_js(canvas_id: str) -> str:
    """Restore the overlay canvas after a clean shot."""
    cid = json.dumps(canvas_id)
    return f"(() => {{const c = document.getElementById({cid}); if (c) c.style.visibility = 'visible';}})()"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. "$PY" tests/test_overlay.py`
Expected: PASS — prints `test_overlay OK`

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/capture/overlay.py tests/test_overlay.py
git commit -m "feat(overlay): pure JS/op builders for live detection overlay"
```

---

### Task 2: `DetectionOverlay` orchestrator

Add the daemon-thread class to `overlay.py`. A single `_tick()` (capture → decode → inject-once → detect → render) is directly unit-tested with fakes; `start()`/`stop()` manage a throttled loop. The detector is injectable so the test needs no ultralytics/weights.

**Files:**
- Modify: `majsoul_eye/capture/overlay.py` (append the class)
- Test: `tests/test_overlay.py` (append tests)

**Interfaces:**
- Consumes: `detections_to_ops`, `render_js`, `inject_js` (Task 1); `majsoul_eye.recognize.TileDetector` (lazy, only if no detector injected).
- Produces (relied on by Task 3):
  - `DetectionOverlay(capture_png, eval_js, weights, device="cuda", fps=12, conf=0.25, canvas_id=OVERLAY_CANVAS_ID, detector=None)`
  - `.canvas_id: str`, `.start() -> None`, `.stop() -> None`, `._tick() -> None`
  - `capture_png: () -> bytes|None` (PNG bytes); `eval_js: (str) -> None` (fire-and-forget).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_overlay.py` (before the `__main__` block):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. "$PY" tests/test_overlay.py`
Expected: FAIL — `AttributeError: module 'majsoul_eye.capture.overlay' has no attribute 'DetectionOverlay'`

- [ ] **Step 3: Write minimal implementation**

Append to `majsoul_eye/capture/overlay.py`:

```python
import threading
import time


class DetectionOverlay:
    """Runs a throttled detect+draw loop on a daemon thread, drawing detector boxes
    onto the injected canvas. Decoupled from the game/WS loop: it only calls the two
    injected callables (both must be thread-safe w.r.t. the page).

    ``capture_png`` MUST return a screenshot with the overlay canvas hidden (see
    ``hide_canvas_js``) so detection runs on clean pixels and dataset frames stay clean.
    """

    def __init__(self, capture_png, eval_js, weights, device="cuda", fps=12,
                 conf=0.25, canvas_id=OVERLAY_CANVAS_ID, detector=None):
        self.capture_png = capture_png
        self.eval_js = eval_js
        self.canvas_id = canvas_id
        self.fps = fps
        self._weights = weights
        self._device = device
        self._conf = conf
        self._detector = detector          # injectable for tests; else built lazily
        self._injected = False
        self._stop = False
        self._thread = None

    def _ensure_detector(self):
        if self._detector is None:
            from ..recognize import TileDetector      # lazy: pulls ultralytics/torch
            self._detector = TileDetector(self._weights, device=self._device, conf=self._conf)
        return self._detector

    def _tick(self):
        png = self.capture_png()
        if not png:
            return
        import cv2
        import numpy as np
        bgr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return
        if not self._injected:
            h, w = bgr.shape[:2]
            self.eval_js(inject_js(self.canvas_id, w, h))
            self._injected = True
        dets = self._ensure_detector().predict(bgr)
        self.eval_js(render_js(detections_to_ops(dets), self.canvas_id))

    def _run(self):
        period = 1.0 / max(1e-3, self.fps)
        while not self._stop:
            t0 = time.time()
            try:
                self._tick()
            except Exception as e:                    # never let the loop die
                print(f"  [overlay] tick error: {type(e).__name__}: {e}", flush=True)
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    def start(self):
        self._ensure_detector()                       # load weights once, up front
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="det-overlay", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. "$PY" tests/test_overlay.py`
Expected: PASS — prints `test_overlay OK`

- [ ] **Step 5: Commit**

```bash
git add majsoul_eye/capture/overlay.py tests/test_overlay.py
git commit -m "feat(overlay): DetectionOverlay daemon-thread detect+draw loop"
```

---

### Task 3: Wire the overlay into `autoplay_ai.py`

Add `--overlay*` flags, upgrade `screenshot_png` to a clean (hide-during-shot) capture shared by both the dataset path and the overlay, and start/stop the overlay. Off ⇒ unchanged behavior. Verified by an argparse smoke test + the full existing suite; live behavior is a documented manual check.

**Files:**
- Modify: `scripts/capture/autoplay_ai.py`
- Test: `tests/test_overlay.py` (append one wiring smoke test)

**Interfaces:**
- Consumes: `majsoul_eye.capture.overlay` — `DetectionOverlay`, `OVERLAY_CANVAS_ID`, `hide_canvas_js`, `show_canvas_js` (Tasks 1–2).
- Produces: CLI flags `--overlay`, `--detector-weights`, `--overlay-fps`, `--overlay-conf`, `--overlay-device`.

- [ ] **Step 1: Add the CLI flags.** In `scripts/capture/autoplay_ai.py`, in the argparse block (immediately after the `--height` line, ~line 95), add:

```python
    ap.add_argument("--overlay", action="store_true",
                    help="Draw the tile detector's boxes live onto the browser (visualizer; off by default).")
    ap.add_argument("--detector-weights", default="majsoul_eye/recognize/tile_detector.pt",
                    help="Detector weights for --overlay (pass weights/detector/tile_detector_obb.pt for rotated OBB polys).")
    ap.add_argument("--overlay-fps", type=float, default=12.0, help="Overlay redraw rate (Hz).")
    ap.add_argument("--overlay-conf", type=float, default=0.25, help="Detector confidence threshold for the overlay.")
    ap.add_argument("--overlay-device", default="cuda", help="Torch device for the overlay detector (cuda/cpu).")
```

- [ ] **Step 2: Resolve overlay module + canvas id.** Immediately after `url = args.url or SERVERS[args.server]` (~line 98), add:

```python
    from majsoul_eye.capture import overlay as overlay_mod   # light: no ultralytics until detector built
    overlay_canvas_id = overlay_mod.OVERLAY_CANVAS_ID if args.overlay else None
```

- [ ] **Step 3: Upgrade `screenshot_png` to a clean capture.** Replace the existing `screenshot_png` body (currently ~lines 209-224) with the version below. When `overlay_canvas_id` is `None` (overlay off), it is behaviorally identical to today; when set, it hides the canvas for the duration of the shot:

```python
    def screenshot_png():
        cdp = cdp_holder[0]
        if cdp is None:
            return None
        rq: queue.Queue = queue.Queue()
        cid = overlay_canvas_id
        def _do():
            try:
                if cid:
                    browser.page.evaluate(overlay_mod.hide_canvas_js(cid))
                try:
                    res = cdp.send("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
                    data = base64.b64decode(res["data"])
                finally:
                    if cid:
                        browser.page.evaluate(overlay_mod.show_canvas_js(cid))
                rq.put(data)
            except Exception:
                rq.put(None)
        browser._action_queue.put(_do)
        try:
            return rq.get(True, 5)
        except Exception:
            return None
```

- [ ] **Step 4: Construct + start the overlay.** Immediately before `print("Watching. Ctrl-C to stop.\n", flush=True)` (~line 372), add:

```python
    overlay = None
    if args.overlay:
        eval_js = lambda js: browser._action_queue.put(lambda: browser.page.evaluate(js))
        overlay = overlay_mod.DetectionOverlay(
            capture_png=screenshot_png, eval_js=eval_js,
            weights=args.detector_weights, device=args.overlay_device,
            fps=args.overlay_fps, conf=args.overlay_conf,
            canvas_id=overlay_canvas_id,
        )
        print(f"[overlay] loading detector {args.detector_weights} on {args.overlay_device} …", flush=True)
        overlay.start()
        print(f"[overlay] live @ {args.overlay_fps:g} fps", flush=True)
```

- [ ] **Step 5: Stop the overlay on exit.** In the `finally:` block (~line 467), as its first statement (before `if game_raw_fh is not None:`), add:

```python
        if overlay is not None:
            overlay.stop()
```

- [ ] **Step 6: Write the wiring smoke test.** Append to `tests/test_overlay.py` (before the `__main__` block). It imports the script module and asserts the flags exist without launching a browser:

```python
def test_autoplay_ai_exposes_overlay_flags():
    import argparse
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "capture", "autoplay_ai.py")
    spec = importlib.util.spec_from_file_location("autoplay_ai_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # rebuild the parser the way main() does, up to parse — assert our flags are present
    seen = {}
    real_parse = argparse.ArgumentParser.parse_args
    def capture_parse(self, *a, **k):
        for act in self._actions:
            seen[tuple(act.option_strings)] = act
        raise SystemExit(0)                       # stop before it touches the network
    argparse.ArgumentParser.parse_args = capture_parse
    try:
        try:
            mod.main()
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = real_parse
    flat = {opt for opts in seen for opt in opts}
    for flag in ("--overlay", "--detector-weights", "--overlay-fps", "--overlay-conf", "--overlay-device"):
        assert flag in flat, f"missing flag {flag}"
```

- [ ] **Step 7: Run the overlay suite.**

Run: `PYTHONPATH=. "$PY" tests/test_overlay.py`
Expected: PASS — prints `test_overlay OK`

- [ ] **Step 8: Run the full existing suite (no regressions).**

Run:
```bash
PYTHONPATH=. "$PY" tests/test_tiles.py && PYTHONPATH=. "$PY" tests/test_detector.py && PYTHONPATH=. "$PY" tests/test_overlay.py
```
Expected: each prints its `test_<name> OK` line, no traceback.

- [ ] **Step 9: Manual integration check (documented, not automated).**

From the repo root, with a Majsoul game open in the launched browser (dry-run is fine — no clicking):
```bash
"$PY" scripts/capture/autoplay_ai.py --overlay --overlay-device cuda
```
Confirm: (a) green boxes/labels track tiles in real time; (b) the console shows `[overlay] live @ 12 fps`; (c) saved PNGs under the run's `game*/frames/` contain **no** overlay boxes (open one and check). For rotated boxes on 河/副露, re-run with `--detector-weights weights/detector/tile_detector_obb.pt`.

- [ ] **Step 10: Commit.**

```bash
git add scripts/capture/autoplay_ai.py tests/test_overlay.py
git commit -m "feat(capture): optional live detection overlay in autoplay_ai (--overlay)"
```

---

## Self-Review

**Spec coverage:**
- §4.1 module + pure helpers → Task 1. `DetectionOverlay` → Task 2. §4.2 wiring/flags → Task 3.
- §5 clean-capture (visibility toggle, shared by both paths) → Task 3 Step 3 + `hide/show_canvas_js` (Task 1). Canvas properties (fixed/pointer-events/z-index, backing store = shot dims) → `inject_js` (Task 1), asserted in test.
- §6 coordinate mapping (boxes drawn verbatim in shot px) → `detections_to_ops` + `inject_js` sizing (Task 1).
- §7 error handling (daemon thread, try/except loop, gated by `--overlay`, no change when off) → Task 2 `_run`, Task 3 Steps 3–5.
- §8 testing (unit HBB/OBB + render + integration) → Tasks 1–3 tests + Task 3 Step 9.
- §9 defaults (fps 12, device cuda, default weights) → Task 3 Step 1.

**Placeholder scan:** none — all steps carry concrete code/commands.

**Type consistency:** `detections_to_ops`/`render_js`/`inject_js`/`hide_canvas_js`/`show_canvas_js` and `DetectionOverlay(capture_png, eval_js, weights, device, fps, conf, canvas_id, detector)` are named identically across the module, tests, and the wiring. `OVERLAY_CANVAS_ID` is the single shared id. `capture_png -> bytes|None` and `eval_js -> None` match usage in both the class and `autoplay_ai.py`.
