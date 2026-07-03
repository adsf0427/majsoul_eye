# Live detection overlay for `autoplay_ai.py`

**Date:** 2026-07-03
**Status:** Approved design, pending implementation plan
**Branch:** `feat/live-detection-overlay`

## 1. Goal

Add an **optional** real-time visualizer to the AI-autoplay capture path
(`scripts/capture/autoplay_ai.py`) that draws the trained tile detector's boxes
directly onto the live Mahjong Soul browser window while data is being captured.
Purpose: eyeball the trained detector's quality against real gameplay, live,
without a separate offline step.

Chosen approach (user decisions):
- **Option A — browser-injected canvas** (MahjongCopilot-style), because that
  capture path already owns a Playwright Chromium page.
- **Continuous** redraw (~10–20 fps), independent of the once-per-turn dataset
  capture cadence. GPU is available (RTX 5080, 16 GB, CUDA, ultralytics 8.4.84 in
  the `auto` env), so imgsz-1280 inference at this rate is comfortable.

## 2. Constraints & context

- `autoplay_ai.py` runs in the `auto` conda env as a **plain-stdout script** (NOT
  the Akagi Textual TUI), so `print()` for status/warnings is fine.
- It owns a `game.browser.GameBrowser` (MahjongCopilot) that holds the Playwright
  page on its **own thread with a serial action queue**. All page access must be
  marshalled via `browser._action_queue.put(...)`. Screenshots already go through
  a thread-safe `screenshot_png()` helper that uses a CDP session
  (`Page.captureScreenshot`, `captureBeyondViewport=False`).
- **Pollution problem:** an injected overlay `<canvas>` is part of the page DOM,
  so `Page.captureScreenshot` captures it too. Because the overlay runs *while the
  dataset is being captured*, unmanaged overlay pixels would (a) be baked into the
  saved training PNGs and (b) pollute the detector's own next input. This must be
  prevented (see §5).
- `recognize/` must stay Akagi-free; the overlay may import `recognize` (one-way:
  `capture → recognize`), never the reverse.
- Detector API (`majsoul_eye/recognize/detector.py`): `TileDetector(weights,
  device="cpu", conf=0.25, imgsz=1280).predict(bgr) -> list[Detection]`.
  `Detection.xyxy` is in **input-image (screenshot) pixels**; `Detection.poly` is
  4 rotated corners for OBB weights, else `None`. `Detection.tile` is the class
  name, `Detection.score` the confidence. ultralytics is lazy-imported.

## 3. Non-goals (YAGNI)

- No overlay for the human `record_gt.py` path (that would need a separate OS
  window; out of scope here).
- No changes to `recognize/` or to detector weights.
- No new GT/AI-recommendation overlay — this visualizes **vision detections only**.
- No persistence/recording of the overlay itself.

## 4. Architecture

One new module plus thin, flag-gated wiring.

### 4.1 New module `majsoul_eye/capture/overlay.py`

A self-contained, browser-agnostic `DetectionOverlay`, driven by injected
callables so it carries no MahjongCopilot/Akagi/Playwright import and is testable:

```python
class DetectionOverlay:
    def __init__(
        self,
        capture_png,          # () -> bytes | None   clean PNG (visibility-toggled)
        eval_js,              # (str) -> None        run JS on the page thread (fire-and-forget)
        weights,              # str                  detector weights path
        device="cuda",
        fps=12,
        conf=0.25,
        viewport=(1280, 720), # CSS viewport (w, h)
    ): ...
    def start(self): ...      # spins a daemon thread; injects the canvas once
    def stop(self): ...       # signals the thread to exit
```

**Thread loop** (throttled to `fps`, each iteration wrapped in try/except →
print+continue):
1. `png = capture_png()`; if `None`, idle and retry.
2. `bgr = cv2.imdecode(...)`.
3. `dets = detector.predict(bgr)`.
4. `ops = detections_to_ops(dets)`.
5. `eval_js(render_js(ops, canvas_id))`.

**Pure helper functions** (no browser; unit-tested):
- `detections_to_ops(dets) -> list[dict]` — one op per detection:
  `{"kind": "poly"|"rect", "pts"/"xyxy": ..., "label": tile, "score": score}`.
  Uses `poly` when present (OBB), else `xyxy` (HBB).
- `render_js(ops, canvas_id) -> str` — JS that grabs the canvas by id,
  `clearRect`s, and strokes each op + label. Labels are escaped.
- `INJECT_JS(canvas_id, w, h) -> str` — one-time canvas creation (see §5).

### 4.2 Wiring in `autoplay_ai.py` (all behind `if args.overlay`)

New flags:
- `--overlay` — enable the visualizer (default off).
- `--detector-weights` — default `majsoul_eye/recognize/tile_detector.pt`
  (production HBB). Pass `weights/detector/tile_detector_obb.pt` for rotated OBB
  polys.
- `--overlay-fps` (default 12), `--overlay-conf` (default 0.25),
  `--overlay-device` (default `cuda`).

Wiring steps:
1. After the CDP session is created, inject the canvas once via
   `browser._action_queue.put(lambda: browser.page.evaluate(INJECT_JS(...)))`.
2. Replace the raw CDP capture used by both paths with a **clean** capture
   (§5) that hides→captures→shows the canvas atomically. The existing dataset
   `maybe_screenshot()` uses this same clean capture so saved PNGs stay pristine.
3. `eval_js = lambda js: browser._action_queue.put(lambda: browser.page.evaluate(js))`.
4. Construct `DetectionOverlay(...)`, `start()` it after the page is up, `stop()`
   it in the `finally` block.

With `--overlay` absent, behavior is identical to today: no canvas is injected,
and the clean-capture wrapper degrades to a plain capture (nothing to hide).

## 5. Clean-capture mechanism

Keeps both the saved dataset frames and the detector's own input free of overlay
pixels, while still letting the user see continuous boxes.

- **Injected canvas:** `position:fixed; left:0; top:0; z-index:9999999;
  pointer-events:none;` appended to `document.body`, with a random id. Backing
  store set to the **screenshot's** pixel dimensions; CSS-sized to the full
  viewport (`width:100vw; height:100vh`). Drawing therefore happens in
  screenshot-pixel coordinates and the browser scales the canvas to the viewport —
  no manual dpr/scale math, robust to zoom. Both surfaces are 16:9, so no aspect
  distortion.
- **Clean capture:** a single browser-thread action performs, in order,
  `canvas.style.visibility='hidden'` → CDP `captureScreenshot` →
  `canvas.style.visibility='visible'`. `visibility:hidden` **retains** the drawn
  pixels (unlike `clearRect`), so the boxes snap back instantly after each shot;
  they persist between detection ticks and only blink during the ~20–40 ms capture
  window. Both the overlay loop and the once-per-turn dataset capture route
  through this.
- Because everything (inject, draw, hide/capture/show) is serialized on the
  browser action queue, there is no torn state between a draw and a capture.

**Known tradeoff:** a brief per-capture flicker of the boxes. Acceptable for a dev
tool. If it proves objectionable, the fallback (not in scope) is a separate
transparent OS overlay window, which never touches the captured surface.

## 6. Coordinate mapping

Detector boxes are already in screenshot pixels; the canvas backing store equals
the screenshot dimensions (§5), so boxes/polys are drawn verbatim with no
transform. The browser's CSS scaling of the canvas to the viewport handles any
device-pixel-ratio difference.

## 7. Error handling & safety

- The overlay is a **separate daemon thread**; it never touches the WS/game loop
  and reaches the page only through the thread-safe action queue.
- Every loop iteration is wrapped in try/except; a detector or page-eval error is
  printed and the loop continues. If the page/CDP is gone, `capture_png()` returns
  `None` and the loop idles.
- The feature is fully gated by `--overlay`. Off ⇒ no behavioral change and no
  detector/ultralytics load.

## 8. Testing

- **Unit** (`tests/test_overlay.py`, `auto` env, no browser):
  - `detections_to_ops` maps HBB (`xyxy`) and OBB (`poly`) detections to the right
    op kinds and payloads.
  - `render_js` emits one draw per op, escapes labels, and no-ops on empty input.
  - Coordinate values pass through unchanged (canvas == screenshot px).
- **Integration** (manual): run `autoplay_ai.py --overlay` against a live game;
  confirm boxes track tiles in real time and that saved dataset PNGs contain no
  overlay pixels.

## 9. Open tuning knobs (non-blocking defaults chosen)

- `--overlay-fps` default **12** (RTX 5080 handles more; 12 balances smoothness vs
  flicker duty-cycle).
- Optional tiny corner HUD (fps + detection count) — deferred unless requested.
