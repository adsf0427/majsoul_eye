"""annotate_frame emits hud_boxes; iter_hud_boxes flattens them; score-anim
window predicate flags reach frames."""
import numpy as np

from majsoul_eye.annotate import build_homographies, annotate_frame
from majsoul_eye.annotate.frame import iter_hud_boxes
from majsoul_eye.state.replay import BoardState, is_score_anim_window

s = BoardState(hero_seat=0, bakaze="E", kyoku=1, oya=0, in_round=True,
               scores=[25000] * 4, left_tile_count=64)
img = np.zeros((1080, 1920, 3), np.uint8)
hom = build_homographies(1920, 1080)
rec = annotate_frame(img, s, hom)
assert "hud_boxes" in rec
hb = list(iter_hud_boxes(rec))
names = {b.name for b in hb}
assert "round_label" in names and "seat_wind_self" in names
# black frame -> numeric fields have no ink -> unreliable, never wrong-text
for b in hb:
    if b.name == "score_self":
        assert b.reliable is False
    if b.name == "round_label":
        assert b.text == "E1"

assert is_score_anim_window(BoardState(last_event="reach_accepted"))
assert is_score_anim_window(BoardState(last_event="reach"))
assert not is_score_anim_window(BoardState(last_event="dahai"))

# Regression: real records BUNDLE a riichi declaration with its forced dahai
# ([reach, dahai]) into ONE record -> last_event is overwritten to "dahai", but
# last_event_types still carries "reach", so the window must still fire True.
assert is_score_anim_window(
    BoardState(last_event="dahai", last_event_types=frozenset({"reach", "dahai"}))
)
# No event applied this record at all -> both signals empty -> False.
assert not is_score_anim_window(BoardState(last_event=None, last_event_types=frozenset()))


def test_score_anim_window_bundling_integration():
    """Pins Replayer.apply_record wiring (not just the predicate in isolation):
    feed a tiny synthetic capture where ONE record carries [reach, dahai] MJAI
    events -- the real-world shape Majsoul sends for a riichi declaration -- and
    assert the resulting state's last_event is the misleading "dahai" while
    is_score_anim_window still correctly reports True via last_event_types."""
    from majsoul_eye.state.replay import Replayer, is_score_anim_window

    class _BundledReachRecord:
        raw_liqi = None
        mjai = [
            {"type": "reach", "actor": 0},
            {"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": True},
        ]

    rp = Replayer(hero_seat=0)
    rp.apply({"type": "start_game", "id": 0})
    rp.apply({
        "type": "start_kyoku", "bakaze": "E", "dora_marker": "1m", "honba": 0,
        "kyoku": 1, "kyotaku": 0, "oya": 0, "scores": [25000] * 4,
        "tehais": [
            ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "9s"],
            ["?"] * 13, ["?"] * 13, ["?"] * 13,
        ],
    })
    rp.apply_record(_BundledReachRecord())
    assert rp.state.last_event == "dahai"                      # the misleading old signal
    assert rp.state.last_event_types == frozenset({"reach", "dahai"})
    assert is_score_anim_window(rp.state) is True              # bundling-proof


test_score_anim_window_bundling_integration()
print("test_hud_frame OK")
