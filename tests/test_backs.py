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


def test_melded_row_is_the_first_slots_unchanged():
    # a melded row does not re-space: it's exactly the first row_n templates
    # (meld_bias 0 -> k=1 for every seat; STATUS §1.49).
    for pos in (1, 2, 3):
        boxes = generate_back_boxes(pos, 10, 1, HINV)   # one meld -> 10-tile row
        assert len(boxes) == 10
        assert _meld_k(pos, 10, 1) == 1.0
        for b, tpl in zip(boxes, BACK_SLOT_QUADS[pos]):
            assert np.allclose(b["poly_fullwarp"], tpl, atol=0.11), (pos, b["slot"])


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


def test_back_boxes_labels_holding_row_plus_drawn():
    img = np.zeros((1080, 1920, 3), np.uint8)
    st = _state((13, 13, 14, 13))                        # seat2 = across = pos2 holding
    rec, flags = back_boxes(img, st, HOM)
    assert "backs_holding" not in " ".join(flags)        # holding is no longer skipped
    # settled seats: 13 row backs; holding seat: 13 static row + 1 drawn = 14
    assert len(rec["1"]) == 13 and len(rec["3"]) == 13
    assert len(rec["2"]) == 14
    drawn = [b for b in rec["2"] if b.get("drawn")]
    assert len(drawn) == 1 and drawn[0]["slot"] == 13
    # black frame -> every emitted box fails the live-fill check
    assert all(b.get("reliable") is False for b in rec["1"] + rec["2"] + rec["3"])


def test_holding_row_matches_settled_geometry():
    # a holding seat's static row (n-1) reuses the exact settled templates; the
    # drawn box sits past the moving end (slot index n-1).
    img = np.zeros((1080, 1920, 3), np.uint8)
    st = _state((14, 13, 13, 13))                        # seat1 = right = pos1 holding, 0 meld
    rec, _ = back_boxes(img, st, HOM)
    row = [b for b in rec["1"] if not b.get("drawn")]
    assert len(row) == 13
    for b, tpl in zip(row, BACK_SLOT_QUADS[1]):
        assert np.allclose(b["poly_fullwarp"], tpl, atol=0.11)


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


_TILE_STATS, _FELT_STATS = (100.0, 40.0), (30.0, 5.0)


def _quad_keyed_patch_stats(felt_quads):
    """Stub for backs._patch_stats keyed on WHICH quad is probed (robust to
    call order and to whether the drawn slot is probed at all): quads in
    `felt_quads` read like bare felt, everything else like a tile back."""
    felt = [np.float32(q) for q in felt_quads]

    def stub(img, quad):
        q = np.float32(quad)
        if any(f.shape == q.shape and np.allclose(q, f, atol=0.5) for f in felt):
            return _FELT_STATS
        return _TILE_STATS
    return stub


def test_sorting_suspect_condition_b_removed():
    # Condition-B signature: every ROW slot AND the drawn slot read like tiles,
    # only the empty reference reads like felt. The old "drawn slot occupied"
    # verdict false-fired on 253/256 firings of a dark-skin game and 17.6% of
    # eligible frames dataset-wide, and build_dataset drops the WHOLE frame
    # (spec 2026-07-10). A fully tile-like row must NOT be called mid-sort.
    import majsoul_eye.annotate.backs as B_mod
    img = np.zeros((8, 8, 3), np.uint8)                  # stub ignores pixels
    empty_q = P.fullwarp_to_original(B_mod._drawn_fw(1, 13, 0, 1.15), HINV)
    orig = B_mod._patch_stats
    try:
        B_mod._patch_stats = _quad_keyed_patch_stats([empty_q])
        assert B_mod.sorting_suspect(img, 1, 13, 0, HINV) is False
    finally:
        B_mod._patch_stats = orig


def test_sorting_suspect_condition_a_survives():
    # Condition-A signature: one ROW slot reads like the empty-felt reference
    # -> the row really is mid-compaction; the gate must still fire.
    import majsoul_eye.annotate.backs as B_mod
    img = np.zeros((8, 8, 3), np.uint8)
    k = B_mod._meld_k(1, 13, 0)
    gap_q = P.fullwarp_to_original(B_mod._stretch_quad(BACK_SLOT_QUADS[1][6], 1, k), HINV)
    empty_q = P.fullwarp_to_original(B_mod._drawn_fw(1, 13, 0, 1.15), HINV)
    orig = B_mod._patch_stats
    try:
        B_mod._patch_stats = _quad_keyed_patch_stats([gap_q, empty_q])
        assert B_mod.sorting_suspect(img, 1, 13, 0, HINV) is True
    finally:
        B_mod._patch_stats = orig


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} OK")
    print("all backs tests passed")
