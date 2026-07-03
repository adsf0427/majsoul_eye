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
import threading
import time

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
        f"const W={int(shot_w)}, H={int(shot_h)};"
        "if (c.width !== W) c.width = W; if (c.height !== H) c.height = H;"
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
        h, w = bgr.shape[:2]
        self.eval_js(inject_js(self.canvas_id, w, h))   # idempotent + non-destructive; re-creates the canvas after a page reload
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
        det = self._ensure_detector()                  # load weights once, up front
        import numpy as np
        det.predict(np.zeros((64, 64, 3), np.uint8))   # warm-up: trip device errors here (caught by caller) instead of 12x/s per tick
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="det-overlay", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
