"""Tests for the EXPERIMENTAL opponent hand-row tile-back annotator
(majsoul_eye/annotate/backs.py): pure slot geometry (anchor sides, meld
shrink+bias, slot indexing) plus the state-driven skip rules (hero seat,
holding seats, opt-in emission through annotate_frame/iter_tile_boxes).
Synthetic BoardStates only — no capture data needed.
"""
import numpy as np

from majsoul_eye.annotate import pipeline as P
from majsoul_eye.annotate._backs_manual import BACK_DRAWN_QUADS, BACK_SLOT_QUADS
from majsoul_eye.annotate.backs import (BACK_ROWS, _anchor_coord, _meld_k, _tpl_pitch,
                                        back_boxes, drawn_quad, generate_back_boxes)
from majsoul_eye.state.replay import Replayer

HOM = P.build_homographies(1920, 1080)
HINV = HOM["H_full_inv"]


def _along(pt_fw, pos):
    return pt_fw[0] if BACK_ROWS[pos]["along"] == "x" else pt_fw[1]


def test_plain13_reproduces_manual_templates():
    for pos in (1, 2, 3):
        boxes = generate_back_boxes(pos, 13, 0, HINV)
        assert len(boxes) == 13
        assert [b["slot"] for b in boxes] == list(range(13))
        for b, tpl in zip(boxes, BACK_SLOT_QUADS[pos]):
            assert np.allclose(b["poly_fullwarp"], tpl, atol=0.11), (pos, b["slot"])
        # the manual templates came out at one physical tile width per seat
        assert abs(_tpl_pitch(pos) - 71.2) < 0.5, (pos, _tpl_pitch(pos))


def test_meld_shrink_keeps_players_left_anchor():
    for pos in (1, 2, 3):
        boxes = generate_back_boxes(pos, 10, 1, HINV)   # one meld -> 10-tile row
        assert len(boxes) == 10
        a = _anchor_coord(pos)
        # anchored extreme of slot 0 unchanged (stretch is about the anchor)
        anchored = [abs(_along(p, pos) - a) for p in boxes[0]["poly_fullwarp"]]
        tpl0 = [abs(_along(p, pos) - a) for p in BACK_SLOT_QUADS[pos][0]]
        assert abs(min(anchored) - min(tpl0)) < 0.11
        # moving-end outer edge = template outer edge of slot 9, stretched by k
        k = _meld_k(pos, 10, 1)
        outer_tpl = max(abs(_along(p, pos) - a) for p in BACK_SLOT_QUADS[pos][9])
        outer_now = max(abs(_along(p, pos) - a) for p in boxes[9]["poly_fullwarp"])
        assert abs(outer_now - outer_tpl * k) < 0.11, (pos, outer_now, outer_tpl * k)
        assert (k > 1.0) == (BACK_ROWS[pos]["meld_bias"] > 0)


def test_drawn_quad_rides_the_moving_end():
    for pos in (1, 2, 3):
        base = np.float32(drawn_quad(pos, 13, 0, HINV))
        tpl = np.float32(P.fullwarp_to_original(BACK_DRAWN_QUADS[pos], HINV))
        assert np.allclose(base, tpl, atol=0.11)          # 13-row: exactly as clicked
        # shorter row: the slot rides the moving end back toward the anchor.
        # Assert in FULLWARP space (a perspective map is not shape-preserving in
        # original px — the nearer position legitimately renders bigger).
        short_fw = np.float32(P.original_to_fullwarp(drawn_quad(pos, 10, 1, HINV), HOM["H_full"]))
        base_fw = np.float32(BACK_DRAWN_QUADS[pos])
        ai = 0 if BACK_ROWS[pos]["along"] == "x" else 1
        d = (short_fw.mean(0) - base_fw.mean(0))
        toward_anchor = d[ai] if BACK_ROWS[pos]["anchor"] == "high" else -d[ai]
        pitch = _tpl_pitch(pos)
        # ~3 tiles closer to the anchor, minus the small meld stretch; cross unchanged
        assert 2.4 * pitch < toward_anchor < 3.2 * pitch, (pos, toward_anchor / pitch)
        assert abs(d[1 - ai]) < 0.6, (pos, d)


def test_bad_row_n_yields_nothing():
    assert generate_back_boxes(2, 0, 0, HINV) == []
    assert generate_back_boxes(2, 14, 0, HINV) == []
    assert generate_back_boxes(0, 13, 0, HINV) == []    # hero pos has no row model


def _state(counts=(13, 13, 13, 13)):
    rp = Replayer(hero_seat=0)
    for ev in ({"type": "start_game", "id": 0},
               {"type": "start_kyoku", "bakaze": "E", "dora_marker": "1m", "honba": 0,
                "kyoku": 1, "kyotaku": 0, "oya": 0,
                "scores": [25000] * 4,
                "tehais": [["1m"] * 13, ["?"] * 13, ["?"] * 13, ["?"] * 13]}):
        rp.apply(ev)
    st = rp.state
    st.concealed_counts = list(counts)
    return st


def test_back_boxes_skips_holding_seat_and_flags():
    img = np.zeros((1080, 1920, 3), np.uint8)
    st = _state((13, 13, 14, 13))                        # seat2 = across = pos2 holding
    rec, flags = back_boxes(img, st, HOM)
    assert rec["2"] == []
    assert "pos2:backs_holding" in flags
    assert len(rec["1"]) == 13 and len(rec["3"]) == 13
    # black frame -> every emitted box fails the live-fill check
    assert all(b.get("reliable") is False for b in rec["1"] + rec["3"])


def test_annotate_frame_backs_opt_in():
    from majsoul_eye.annotate.frame import annotate_frame, iter_tile_boxes
    img = np.zeros((1080, 1920, 3), np.uint8)
    st = _state()
    rec = annotate_frame(img, st, HOM)                   # default: no backs
    assert "back_boxes" not in rec
    assert all(b.zone != "oppback" for b in iter_tile_boxes(rec))
    rec = annotate_frame(img, st, HOM, backs=True)
    assert set(rec["back_boxes"]) == {"1", "2", "3"}
    ob = [b for b in iter_tile_boxes(rec) if b.zone == "oppback"]
    assert len(ob) == 39 and all(b.tile == "back" for b in ob)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} OK")
    print("all backs tests passed")
