"""Extra-shot scheduler for uncertain-timing capture windows (Task 15).

Some board-changing events land the canonical quiet-debounce screenshot at an
uncertain point on an animation timeline — a meld's forced-dahai handoff, or
an action-button offer whose buttons appear/vanish on their own clock. Rather
than gate the canonical frame differently (which risks breaking the existing
sync contract), `MultiShot` plans a few EXTRA screenshots at fixed offsets
after the triggering event, purely additive to (never instead of) the
canonical `{seq:06d}.png` capture. No pixel-stability gating here — the point
is to sample fixed points on the animation timeline, not to wait for it to
settle.

Pure and clock-injected: every time value is a caller-supplied float
(typically `time.time()`), so this is unit-testable with no browser/client.
"""
from __future__ import annotations


class MultiShot:
    """Tracks at most one outstanding extra-shot plan, keyed to the latest
    armed seq. `arm()` always supersedes whatever plan (fired or not)
    preceded it: a new board-changing seq means any leftover offsets from the
    OLD seq no longer make sense (they would land on the NEW board state
    tagged with the wrong seq), so they are dropped, not carried over or
    fired late.
    """

    def __init__(self, offsets: tuple[float, ...] = (0.6, 1.2, 2.4)):
        self._offsets = tuple(sorted(offsets))   # sorted ascending, promised by due() docstring
        self._seq: int | None = None
        self._event_t: float = 0.0
        self._pending_ms: list[int] = []   # planned offsets (ms), this seq only, not yet fired

    def arm(self, seq: int, event_t: float, window: bool) -> None:
        """New pending seq. `window=False` plans no extra shots for this seq
        (e.g. a plain dahai with no meld/button ambiguity). Any call — even
        re-arming the same seq — supersedes/cancels whatever the previous
        call had left un-fired."""
        self._seq = seq
        self._event_t = event_t
        self._pending_ms = [round(o * 1000) for o in self._offsets] if window else []

    def cancel(self) -> None:
        """Explicitly cancel the current plan (clears pending offsets). Safe to call
        even if no plan is armed. Use this instead of arm(0, ..., False) for readability
        when the intent is purely to cancel, not to arm a new seq."""
        self._pending_ms = []

    def due(self, now: float) -> list[tuple[int, int]]:
        """`(seq, planned_ms)` pairs whose offset has elapsed since the armed
        `event_t`. Each pair is returned exactly once (removed from the plan
        here), in ascending offset order."""
        if not self._pending_ms:
            return []
        elapsed_ms = (now - self._event_t) * 1000.0
        # Tiny epsilon: `now`/`event_t` are wall-clock floats subtracted from each
        # other, so an intended exact-boundary elapsed time (e.g. 0.6s) can land a
        # few ULPs under the target due to float error. Without slack that shot
        # would simply fire one tick later in practice (harmless — the main loop
        # polls every ~5ms) but it makes boundary behavior needlessly flaky to
        # reason about and test.
        fired = [ms for ms in self._pending_ms if elapsed_ms >= ms - 1e-6]
        if not fired:
            return []
        fired_set = set(fired)
        self._pending_ms = [ms for ms in self._pending_ms if ms not in fired_set]
        return [(self._seq, ms) for ms in fired]
