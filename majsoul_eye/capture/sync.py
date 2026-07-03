"""Screenshot ↔ GT time-sync (docs/DESIGN.md §3.2 — the top correctness risk).

Protocol events fire BEFORE the animation finishes; we must NOT block Akagi's
MITM thread, so capture is asynchronous.

Strategy: **debounce-to-quiet**. Each board-changing event updates a single
"pending key" and resets a quiet timer. A worker captures ONE frame once no new
event has arrived for `quiet` seconds (the board action has paused — typically
hero's turn), tagging it with the latest key. A `settle_cap` forces a capture if
a burst runs long.

The key is the recorder's **global monotonic record `seq`** — NOT Majsoul's
`last_op_step`, which RESETS every kyoku (so `<step>.png` filenames collide and
later rounds overwrite earlier ones). `seq` is unique across the whole game and
maps to a reconstructable state (replay records up to that seq).

Why debounce vs "settle each event + drop if superseded": against fast (computer)
play ~90% of captures get superseded (straddle) and whole rounds yield nothing;
an animated table cloth also never satisfies a pixel-stability check. Event-quiet
keys off game events, not pixels — robust to both.

The decision lives in :meth:`_maybe_capture`, with injected ``grab``/``now``/
``sleep`` so it is unit-testable without a client.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Callable, Optional

import numpy as np

from .roi_diff import roi_diff

# MJAI events that change the visible board → reset the quiet timer.
RELEVANT_EVENTS = frozenset({
    "start_kyoku", "tsumo", "dahai", "chi", "pon",
    "daiminkan", "ankan", "kakan", "reach", "reach_accepted", "dora", "nukidora",
})


def frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference (0 == identical). Exported for diagnostics."""
    if a.shape != b.shape:
        return 1e9
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


class FrameSyncer:
    def __init__(
        self,
        grab: Callable[[], Optional[np.ndarray]],
        out_dir: str,
        *,
        quiet: float = 0.30,          # seconds of no board events == "settled"
        settle_cap: float = 2.0,      # force a capture if a burst runs this long
        poll: float = 0.05,
        confirm_stable: bool = True,  # also require the *picture* to stop moving
        diff_thresh: float = 3.0,     # frame-diff below this == still (discard animation done)
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        on_pair: Optional[Callable[[int, Optional[str], str], None]] = None,
    ):
        self.grab = grab
        self.out_dir = out_dir
        self.frames_dir = os.path.join(out_dir, "frames")
        self.quiet = quiet
        self.settle_cap = settle_cap
        self.poll = poll
        self.confirm_stable = confirm_stable
        self.diff_thresh = diff_thresh
        self.now = now
        self.sleep = sleep
        self.on_pair = on_pair

        self._lock = threading.Lock()
        self._pending_key: Optional[int] = None    # latest board-changing seq not yet captured
        self._in_deal_window = False                # start_kyoku..first-dahai: skip captures
        self._last_event_t: float = 0.0
        self._pending_since: float = 0.0
        self._fulfilled: Optional[int] = None
        self._ref: Optional[np.ndarray] = None      # previous grab, for stability confirm
        self._ref_key: Optional[int] = None
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._index_fh = None
        self.counts = {"ok": 0, "no_capture": 0, "error": 0}

    # --- producer (MITM thread) --------------------------------------------

    def on_event(self, key: int) -> None:
        """Register a board-changing event with global key `key` (record seq)."""
        if key is None:
            return
        with self._lock:
            t = self.now()
            self._last_event_t = t
            if self._pending_key is None or key > self._pending_key:
                if self._pending_key is None or self._pending_key == self._fulfilled:
                    self._pending_since = t        # start of a new pending burst
                self._pending_key = key

    def note(self, mjai_list, key) -> None:
        """Called per liqi message from the recorder; resets the quiet timer iff
        the message carried a board-changing event. `key` is the record seq."""
        if key is None or not mjai_list:
            return
        types = [ev.get("type") for ev in mjai_list]
        # The deal-in animation (~2-3s) plays from `start_kyoku` until the first
        # `dahai`; a frame in that window shows an unsorted/incomplete hero hand
        # that won't match GT, so never capture it. (The annotator drops the same
        # window again via state.is_deal_window — this just avoids wasting a grab.)
        if "start_kyoku" in types:
            self._in_deal_window = True
        if self._in_deal_window:
            if "dahai" in types:
                self._in_deal_window = False       # first discard settled the board
            else:
                return                             # inside the deal window → skip
        if any(t in RELEVANT_EVENTS for t in types):
            self.on_event(key)

    # --- worker -------------------------------------------------------------

    def start(self) -> None:
        os.makedirs(self.frames_dir, exist_ok=True)
        # Truncate (match GTWriter "w") so a restart yields a clean GT↔frames pair.
        self._index_fh = open(os.path.join(self.out_dir, "frames.jsonl"), "w", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="frame-syncer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=max(2.0, self.settle_cap + 1))
        if self._index_fh:
            self._index_fh.close()
            self._index_fh = None

    def _run(self) -> None:
        while not self._stop:
            # NOTHING here may raise: an uncaught thread exception prints to stderr
            # and corrupts Akagi's TUI.
            try:
                self._maybe_capture()
            except Exception:
                pass
            self.sleep(self.poll)

    def _maybe_capture(self) -> bool:
        """If the board has been event-quiet for `quiet` (or `settle_cap` exceeded),
        capture one frame for the pending key. Returns True if it captured."""
        with self._lock:
            key = self._pending_key
            last_evt = self._last_event_t
            since = self._pending_since
            if key is None or key == self._fulfilled:
                return False
        t = self.now()
        quiet_enough = (t - last_evt) >= self.quiet
        capped = (t - since) >= self.settle_cap
        if not (quiet_enough or capped):
            return False

        try:
            frame = self.grab()
        except Exception:
            frame = None

        with self._lock:
            if self._pending_key != key:   # superseded while grabbing → retry next tick
                return False

        if frame is None:
            with self._lock:
                self._fulfilled = key
            self._ref = None
            self._record(key, None, "no_capture")
            return True

        # Confirm the picture has stopped moving (discard animation finished), using
        # the table-ROI diff so the animated cloth border/2D HUD can't defeat it.
        # Runs on the `capped` path too (one confirmation) so a long burst still
        # waits out the sweep instead of grabbing mid-animation. On the capped path
        # the pending key is *itself* a moving target (that's why it never went
        # quiet), so the comparison there ignores key identity — it just needs two
        # consecutive capped-path grabs to agree, then captures with whatever key is
        # current. Off the capped path (a real quiet-triggered capture) we still
        # gate on the ref belonging to this same key, so a fresh key after an
        # interruption starts a fresh prime instead of comparing across events.
        if self.confirm_stable:
            same_target = capped or self._ref_key == key
            if same_target and self._ref is not None and roi_diff(frame, self._ref) <= self.diff_thresh:
                pass  # stable
            else:
                self._ref, self._ref_key = frame, key
                return False

        with self._lock:
            if self._pending_key != key:
                return False
            self._fulfilled = key
        self._ref = self._ref_key = None
        path = self._save(key, frame)
        self._record(key, path, "ok")
        return True

    # --- io -----------------------------------------------------------------

    def _save(self, key: int, frame: np.ndarray) -> str:
        path = os.path.join(self.frames_dir, f"{key:06d}.png")
        try:
            import cv2  # type: ignore
            cv2.imwrite(path, frame)
        except Exception:
            np.save(path.replace(".png", ".npy"), frame)
            path = path.replace(".png", ".npy")
        return path

    def _record(self, key: int, path: Optional[str], status: str) -> None:
        self.counts[status] = self.counts.get(status, 0) + 1
        if self._index_fh:
            # Store index-relative ("frames/NNNNNN.png") so the frame dir stays portable.
            rel = ("frames/" + os.path.basename(path)) if path else None
            self._index_fh.write(json.dumps({"seq": key, "file": rel, "status": status, "ts": time.time()}) + "\n")
            self._index_fh.flush()
        if self.on_pair:
            self.on_pair(key, path, status)
