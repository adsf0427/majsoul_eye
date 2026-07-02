"""Named validation cases for the annotator (AB set).

A frozen ``case -> {capture, seq, note}`` map of hand-picked seqs that exercise
the hard geometry: every meld type (chi / pon / daiminkan / ankan / kakan) and a
riichi-sideways discard in each of the 4 river orientations.

Relocated here from ``scripts/annotate/spike_topdown.py`` so both the top-down
visualization spike and ``scripts/annotate/build_case_annotations.py`` share one
fixture instead of one script importing it from another.

Seat mapping (screen pos from hero): ai_run_3_game1 hero=3 → self3/right0/across1/left2;
                                     ai_run_3_game3 hero=1 → self1/right2/across3/left0.
"""
from __future__ import annotations

_GT = "captures/intermediate/gt"

CASES: dict[str, dict] = {
    "rivers_full":     {"capture": f"{_GT}/ai_run_3_game1.jsonl", "seq": 1458, "note": "rivers[13,12,12,13] + melds s0/s1/s2"},
    "A_chi_daiminkan": {"capture": f"{_GT}/ai_run_3_game1.jsonl", "seq": 124,  "note": "right(s0): chi+daiminkan"},
    "B_daiminkan_pon": {"capture": f"{_GT}/ai_run_3_game1.jsonl", "seq": 140,  "note": "right(s0): chi+daiminkan; left(s2): pon,pon"},
    "E_ankan":         {"capture": f"{_GT}/ai_run_3_game1.jsonl", "seq": 390,  "note": "left(s2): ankan; across(s1): pon"},
    "Z_longchain":     {"capture": f"{_GT}/ai_run_3_game1.jsonl", "seq": 1458, "note": "across(s1): pon,pon; left(s2): chi,pon"},
    "C_kakan_single":  {"capture": f"{_GT}/ai_run_3_game3.jsonl", "seq": 118,  "note": "right(s2): kakan; left(s0): pon"},
    "D_kakan_multi":   {"capture": f"{_GT}/ai_run_3_game3.jsonl", "seq": 118,  "note": "kakan + neighbour pon"},
    # Riichi: the declaring discard is rendered SIDEWAYS in the river, so each seat
    # exercises the sideways tile in a different river orientation. One case per
    # screen position (self/right/across/left) so all 4 rotations are covered.
    "F_riichi_self":   {"capture": f"{_GT}/ai_run_3_game1.jsonl", "seq": 1456, "note": "self(s3): riichi @river idx10, E4"},
    "G_riichi_right":  {"capture": f"{_GT}/ai_run_3_game1.jsonl", "seq": 726,  "note": "right(s0): riichi @river idx8, E2 (left s2 also riichi)"},
    "H_riichi_across": {"capture": f"{_GT}/ai_run_3_game3.jsonl", "seq": 782,  "note": "across(s3): riichi @river idx11, E4 (deep river)"},
    "I_riichi_left":   {"capture": f"{_GT}/ai_run_3_game1.jsonl", "seq": 748,  "note": "left(s2): riichi @river idx12, E2 (right s0 also riichi)"},
}
