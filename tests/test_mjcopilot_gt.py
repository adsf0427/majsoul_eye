"""make_capturing_game_state + gt_fields — no MahjongCopilot needed (fake GameState).

Plain-script style: PYTHONPATH=. <auto-python> tests/test_mjcopilot_gt.py
"""
from majsoul_eye.capture.mjcopilot_gt import make_capturing_game_state, gt_fields


class FakeGameState:
    """Stand-in for MahjongCopilot's GameState: derives mjai into
    self.mjai_pending_input_msgs, which the real class also does."""
    def __init__(self, bot):
        self.bot = bot
        self.seat = 0
        self.mjai_pending_input_msgs = []          # traced -> becomes a CapList

    def input(self, msg):
        for ev in msg["events"]:
            self.mjai_pending_input_msgs.append(ev)
        return None

    def reset_pending(self):
        self.mjai_pending_input_msgs = []          # GameState flushes between turns


def test_drain_returns_new_events_each_call():
    gs, drain = make_capturing_game_state(FakeGameState, object())
    gs.input({"events": [{"type": "tsumo", "pai": "5m"}]})
    assert drain() == [{"type": "tsumo", "pai": "5m"}]
    assert drain() == []                            # nothing new since last drain
    gs.input({"events": [{"type": "dahai", "pai": "1p"}]})
    assert drain() == [{"type": "dahai", "pai": "1p"}]


def test_drain_is_deepcopied_isolated_from_later_mutation():
    gs, drain = make_capturing_game_state(FakeGameState, object())
    ev = {"type": "start_kyoku", "tehais": [["1m", "2m"]]}
    gs.input({"events": [ev]})
    out = drain()
    ev["tehais"][0].append("MUTATED")               # GameState mutates the hand in place
    assert out[0]["tehais"][0] == ["1m", "2m"]       # captured copy is frozen


def test_bot_still_sees_events_capList_transparency():
    gs, drain = make_capturing_game_state(FakeGameState, object())
    gs.input({"events": [{"type": "dahai", "pai": "9s"}]})
    # the underlying list the bot reads is still populated
    assert list(gs.mjai_pending_input_msgs) == [{"type": "dahai", "pai": "9s"}]


def test_survives_pending_reset_between_turns():
    gs, drain = make_capturing_game_state(FakeGameState, object())
    gs.input({"events": [{"type": "tsumo", "pai": "5m"}]})
    assert drain() == [{"type": "tsumo", "pai": "5m"}]
    gs.reset_pending()                              # new empty CapList installed
    gs.input({"events": [{"type": "dahai", "pai": "5m"}]})
    assert drain() == [{"type": "dahai", "pai": "5m"}]


def test_gt_fields():
    assert gt_fields({"method": ".lq.ActionPrototype",
                      "data": {"name": "ActionDiscardTile"}}) == (
        ".lq.ActionPrototype", "ActionDiscardTile")
    # non-ActionPrototype: action_name is None even if data has a name
    assert gt_fields({"method": ".lq.FastTest.authGame",
                      "data": {"name": "x"}}) == (".lq.FastTest.authGame", None)
    assert gt_fields(None) == (None, None)
    assert gt_fields({"method": ".lq.ActionPrototype", "data": None}) == (
        ".lq.ActionPrototype", None)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_mjcopilot_gt OK")
