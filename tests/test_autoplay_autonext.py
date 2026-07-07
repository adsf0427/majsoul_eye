"""auto_next_flow (autoplay_ai game-end "one more game" loop) unit tests.

Regression target (observed live 2026-07-05): the game-end sequence is
  終局ranking(確認) -> pt/achievement(確認) -> missions(もう一局 + 確認)
  -> rematch dialog(はい / いいえ) -> matchmaking(authGame).
Two real bugs the flow must not repeat:
  (a) the ranking screen's blue 2/3/4位 rank bars false-positived the blue
      "next" guard, so the old flow "clicked next" on the ranking screen and
      returned success — no game ever started (watchdog fired 210s later);
  (b) the rematch DIALOG (はい/いいえ) was never handled at all, so even a correct
      もう一局 click stranded the flow.
Screen identity is decided by button CO-PRESENCE, not a single color box:
  - settlement (ranking/pt): 確認 present, もう一局 absent   -> click 確認
  - missions:                確認 AND もう一局 present         -> click もう一局
  - rematch dialog:          はい AND いいえ present, 確認 ABSENT -> click はい (done)
  - lobby main menu:         stop, never click

Run: PYTHONPATH=. python tests/test_autoplay_autonext.py   (also pytest-compatible)
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "autoplay_ai", _ROOT / "scripts" / "capture" / "autoplay_ai.py")
autoplay_ai = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(autoplay_ai)

# fallback click points the flow uses when a guard reports present (16x9 logical)
XY = {"confirm": (14.35, 8.45), "rematch": (12.21, 8.32),
      "dialog_yes": (6.90, 5.90), "dialog_no": (9.30, 5.90)}


class FakeUI:
    """Scripted screen sequence. Each screen is a set of visible button kinds;
    clicking a button may transition to another screen index (on_<kind>)."""

    def __init__(self, screens):
        self.screens = screens
        self.idx = 0
        self.clicks = []            # (kind, (x16,y9)) in click order

    def _cur(self):
        return self.screens[self.idx]

    def button_guard(self, kind):
        present = kind in self._cur().get("buttons", ())
        return present, (0.5 if present else 0.0), XY[kind]

    def main_menu_visible(self):
        on = bool(self._cur().get("menu"))
        return on, (5.0 if on else 80.0)

    def click_at(self, xy, kind=None):
        self.clicks.append((kind, xy))
        cur = self._cur()
        if kind and f"on_{kind}" in cur:
            self.idx = cur[f"on_{kind}"]
        yield ("click", xy)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t


def drive(ui, timeout=90.0):
    clock = FakeClock()
    state = {"active": True, "started": clock.t, "clicked_next": False, "failed": False}
    gen = autoplay_ai.auto_next_flow(
        button_guard=ui.button_guard,
        main_menu_visible=ui.main_menu_visible,
        click_at=ui.click_at,
        delay_step=lambda s: ("delay", s),
        timeout=timeout,
        state=state,
        now=clock.now,
        log=lambda *_: None,
    )
    for step in gen:
        if isinstance(step, tuple) and step[0] == "delay":
            clock.t += step[1]
    return state


def _kinds(ui):
    return [k for k, _ in ui.clicks]


def test_full_sequence_ranking_pt_missions_dialog():
    ui = FakeUI([
        {"buttons": ("confirm",), "on_confirm": 1},                 # 0 ranking
        {"buttons": ("confirm",), "on_confirm": 2},                 # 1 pt/achievement
        {"buttons": ("confirm", "rematch"), "on_rematch": 3},       # 2 missions
        {"buttons": ("dialog_yes", "dialog_no"), "on_dialog_yes": 4},  # 3 rematch dialog
        {},                                                          # 4 matchmaking (blank)
    ])
    state = drive(ui)
    assert _kinds(ui) == ["confirm", "confirm", "rematch", "dialog_yes"], ui.clicks
    assert state["clicked_next"] is True
    assert state["failed"] is False


def test_missions_clicks_rematch_never_confirm():
    # The whole point: on the missions screen (both buttons) never click 確認 —
    # that exits to the lobby. Must click もう一局.
    ui = FakeUI([
        {"buttons": ("confirm", "rematch"), "on_rematch": 1},
        {"buttons": ("dialog_yes", "dialog_no"), "on_dialog_yes": 2},
        {},
    ])
    state = drive(ui)
    assert _kinds(ui) == ["rematch", "dialog_yes"], ui.clicks
    assert state["clicked_next"] is True


def test_ranking_blue_bar_does_not_trigger_rematch():
    # Ranking shows 確認 only (measured: no もう一局 in the button row). Even if a
    # stray blue "rematch" were briefly reported, co-presence with 確認 gates it;
    # here rematch is simply absent so we must click 確認, not treat it as missions.
    ui = FakeUI([
        {"buttons": ("confirm",), "on_confirm": 1},   # ranking: confirm only
        {"buttons": ("confirm", "rematch"), "on_rematch": 2},
        {"buttons": ("dialog_yes", "dialog_no"), "on_dialog_yes": 3},
        {},
    ])
    state = drive(ui)
    assert _kinds(ui) == ["confirm", "rematch", "dialog_yes"], ui.clicks
    assert state["clicked_next"] is True


def test_confirms_uncapped():
    ui = FakeUI([
        {"buttons": ("confirm",), "on_confirm": 1},
        {"buttons": ("confirm",), "on_confirm": 2},
        {"buttons": ("confirm",), "on_confirm": 3},
        {"buttons": ("confirm", "rematch"), "on_rematch": 4},
        {"buttons": ("dialog_yes", "dialog_no"), "on_dialog_yes": 5},
        {},
    ])
    state = drive(ui)
    assert _kinds(ui) == ["confirm", "confirm", "confirm", "rematch", "dialog_yes"], ui.clicks


def test_dialog_not_triggered_while_confirm_present():
    # A screen that shows はい/いいえ-like colors AND 確認 must be treated as a
    # settlement screen (click 確認), never as the dialog. The dialog has NO 確認.
    ui = FakeUI([
        {"buttons": ("confirm", "dialog_yes", "dialog_no"), "on_confirm": 1},
        {},
    ])
    state = drive(ui)
    assert _kinds(ui) == ["confirm"], ui.clicks


def test_dialog_click_verified_and_retried():
    # First はい click misses (dialog still up); retry, then it clears.
    ui = FakeUI([
        {"buttons": ("dialog_yes", "dialog_no"), "on_dialog_yes": 1},
        {"buttons": ("dialog_yes", "dialog_no"), "on_dialog_yes": 2},
        {},
    ])
    state = drive(ui)
    assert _kinds(ui) == ["dialog_yes", "dialog_yes"], ui.clicks
    assert state["clicked_next"] is True


def test_lobby_visible_stops_without_clicking():
    ui = FakeUI([{"menu": True, "buttons": ("confirm", "rematch")}])
    state = drive(ui)
    assert ui.clicks == [], ui.clicks
    assert state["failed"] is True
    assert state["active"] is False
    assert state["clicked_next"] is False


def test_lobby_after_confirm_stops_flow():
    ui = FakeUI([
        {"buttons": ("confirm",), "on_confirm": 1},
        {"menu": True},
    ])
    state = drive(ui)
    assert _kinds(ui) == ["confirm"], ui.clicks
    assert state["failed"] is True
    assert state["clicked_next"] is False


def test_timeout_marks_failed():
    ui = FakeUI([{}])
    state = drive(ui, timeout=10.0)
    assert ui.clicks == [], ui.clicks
    assert state["failed"] is True
    assert state["active"] is False


def _drive_with_debug(ui, debug, timeout=90.0):
    clock = FakeClock()
    state = {"active": True, "started": clock.t, "clicked_next": False, "failed": False}
    gen = autoplay_ai.auto_next_flow(
        button_guard=ui.button_guard, main_menu_visible=ui.main_menu_visible,
        click_at=ui.click_at, delay_step=lambda s: ("delay", s),
        timeout=timeout, state=state, now=clock.now, log=lambda *_: None, debug=debug)
    for step in gen:
        if isinstance(step, tuple) and step[0] == "delay":
            clock.t += step[1]
    return state


def test_debug_hook_fires_each_iteration_without_changing_decisions():
    # The diagnostic dump runs once per loop iteration and must NOT alter the click path.
    calls = []
    ui = FakeUI([
        {"buttons": ("confirm",), "on_confirm": 1},
        {"buttons": ("confirm", "rematch"), "on_rematch": 2},
        {"buttons": ("dialog_yes", "dialog_no"), "on_dialog_yes": 3},
        {},
    ])
    state = _drive_with_debug(ui, debug=lambda: calls.append(1))
    assert _kinds(ui) == ["confirm", "rematch", "dialog_yes"], ui.clicks
    assert state["clicked_next"] is True
    assert len(calls) >= 3           # at least one dump per advancing screen


def test_debug_hook_exception_never_breaks_flow():
    # A failing dump (e.g. disk/screenshot error) must be swallowed, not strand the flow.
    def boom():
        raise RuntimeError("dump failed")
    ui = FakeUI([
        {"buttons": ("confirm",), "on_confirm": 1},
        {"buttons": ("dialog_yes", "dialog_no"), "on_dialog_yes": 2},
        {},
    ])
    state = _drive_with_debug(ui, debug=boom)
    assert _kinds(ui) == ["confirm", "dialog_yes"], ui.clicks
    assert state["clicked_next"] is True


class FracUI:
    """Screens defined by raw guard *fractions*; presence applies the REAL module-level
    thresholds (autoplay_ai.BUTTON_GUARDS[kind]['min_frac']). Lets us replay recorded
    end-game frames (captures/raw/ai_session4/run_2/_autonext_debug) through the real gate,
    so a threshold regression fails here instead of live on a burner account."""

    def __init__(self, screens):
        self.screens = screens
        self.idx = 0
        self.clicks = []

    def button_guard(self, kind):
        frac = self.screens[self.idx].get(kind, 0.0)
        spec = autoplay_ai.BUTTON_GUARDS[kind]
        return frac >= spec["min_frac"], frac, tuple(spec["fallback"])

    def main_menu_visible(self):
        return False, 80.0

    def click_at(self, xy, kind=None):
        self.clicks.append((kind, xy))
        nxt = self.screens[self.idx].get("_on", {}).get(kind)
        if nxt is not None:
            self.idx = nxt
        yield ("click", xy)


def _drive_frac(ui, timeout=90.0):
    clock = FakeClock()
    state = {"active": True, "started": clock.t, "clicked_next": False, "failed": False}
    gen = autoplay_ai.auto_next_flow(
        button_guard=ui.button_guard, main_menu_visible=ui.main_menu_visible,
        click_at=ui.click_at, delay_step=lambda s: ("delay", s),
        timeout=timeout, state=state, now=clock.now, log=lambda *_: None)
    for step in gen:
        if isinstance(step, tuple) and step[0] == "delay":
            clock.t += step[1]
    return state


# Real recorded (dialog_yes, dialog_no) frac pairs, tagged by whether they are the actual
# はい/いいえ modal, from ai_session4 run_2+run_3+run_5 _autonext_debug logs. confirm is absent
# on all of these. This is the ground truth the gate must reproduce.
_REAL_DIALOG_FRACS = [(0.2068, 0.2264)]                         # pinned across 10 dialogs / 3 runs
_FALSE_DIALOGLIKE_FRACS = [
    (0.0715, 0.8676),   # run_2 game6 i1 — faint art + full rank bars (original stuck frame)
    (0.0801, 0.1765),   # run_2 game2 i6 — settling screen
    (0.0486, 0.8676),   # run_3 game3 i1 — the frame that made run_3 give up
    (0.0447, 0.0759),   # matchmaking-retry frame (post-click)
    (0.1284, 0.8676),   # run_5 game1 i1 — GOLD skin art 0.13 (would pass a 0.12 threshold!)
    (0.3882, 0.0002),   # run_5 game4 i0 — art 0.39 > real button, bars not yet rendered
]


def test_is_rematch_dialog_matches_recorded_ground_truth():
    # The pixel-free decision must fire on every real dialog and no false positive.
    for yf, nf in _REAL_DIALOG_FRACS:
        assert autoplay_ai.is_rematch_dialog(yf, nf, False) is True, (yf, nf)
    for yf, nf in _FALSE_DIALOGLIKE_FRACS:
        assert autoplay_ai.is_rematch_dialog(yf, nf, False) is False, (yf, nf)
    # 確認 present always vetoes the dialog (settlement/missions screen), even if symmetric.
    assert autoplay_ai.is_rematch_dialog(0.2068, 0.2264, True) is False


def test_ai_session4_all_false_dialoglike_frames_rejected():
    # Drive the whole flow on each recorded false positive (incl. run_5's gold-skin 0.13 and
    # 0.39 art that DEFEAT any single yellow threshold): must never click はい on the ranking.
    for yf, nf in _FALSE_DIALOGLIKE_FRACS:
        ui = FracUI([{"dialog_yes": yf, "dialog_no": nf, "confirm": 0.0, "rematch": 0.0}])
        state = _drive_frac(ui, timeout=6.0)
        assert ui.clicks == [], (yf, nf, ui.clicks)


def test_ai_session4_real_dialog_frac_clicks_hai_once():
    # The REAL rematch dialog (dialog_yes 0.2068, dialog_no 0.2264, symmetric, confirm absent):
    # click はい exactly once; matchmaking (all guards gone) -> success, no retry loop.
    ui = FracUI([
        {"dialog_yes": 0.2068, "dialog_no": 0.2264, "confirm": 0.0, "rematch": 0.0,
         "_on": {"dialog_yes": 1}},
        {},                                                     # matchmaking: はい gone
    ])
    state = _drive_frac(ui)
    assert [k for k, _ in ui.clicks] == ["dialog_yes"], ui.clicks
    assert state["clicked_next"] is True


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    _main()
