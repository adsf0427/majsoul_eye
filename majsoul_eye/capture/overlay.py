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
