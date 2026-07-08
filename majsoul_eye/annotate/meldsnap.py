"""Per-round meld-snap consensus (Phase 2, STATUS §1.51).

The meld strip is physically fixed within a kyoku, so the correct rigid-snap offset
(d_along, d_cross) is ~constant per (game, bakaze, kyoku, honba, screen-pos). Per-frame
snapping (snap_meld_strip) occasionally mislocks a single frame (flicker) or is gated
off on a low-feature frame (falls back to the raw template). This module measures every
frame's per-frame snap, takes a robust per-round consensus, and returns a per-seq
override so annotate_frame can place every frame of a round at the same, voted offset.

Only safe AFTER the Phase-1 corner recalibration (else large consistent mislock blocks
could out-vote the truth)."""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from majsoul_eye.annotate import pipeline as P
from majsoul_eye.annotate.seatgt import seat_gt
from majsoul_eye.state.replay import check_invariants, is_call_window, is_deal_window

CLUSTER_TOL = 12.0      # px; (d_along,d_cross) within this (both axes) = same cluster
MIN_ROUND_FRAMES = 3    # need >= this many confident frames to trust a round
MIN_ROUND_CONF = 0.55   # winning cluster must hold >= this fraction of total score
MIN_FEATURES = 2        # a frame's snap counts only if n_along + n_cross >= this


def _wmean(vals, weights):
    tw = float(sum(weights))
    return float(sum(v * w for v, w in zip(vals, weights)) / tw) if tw > 0 else 0.0


def round_meld_consensus(samples):
    """samples: list[(d_along, d_cross, score, n_features)] for ONE (kyoku, pos).
    Returns (d_along, d_cross, conf) — the score-weighted centre of the dominant
    (d_along, d_cross) cluster — or None if too few confident frames / no dominant
    cluster (caller then marks that round's meld boxes unreliable)."""
    conf = [(da, dc, sc) for da, dc, sc, n in samples if n >= MIN_FEATURES and sc > 0]
    if len(conf) < MIN_ROUND_FRAMES:
        return None
    best_w, best = -1.0, []
    for da0, dc0, _ in conf:
        members = [(da, dc, sc) for da, dc, sc in conf
                   if abs(da - da0) <= CLUSTER_TOL and abs(dc - dc0) <= CLUSTER_TOL]
        w = sum(sc for _, _, sc in members)
        if w > best_w:
            best_w, best = w, members
    total = sum(sc for _, _, sc in conf)
    if total <= 0 or best_w / total < MIN_ROUND_CONF:
        return None
    da_c = _wmean([m[0] for m in best], [m[2] for m in best])
    dc_c = _wmean([m[1] for m in best], [m[2] for m in best])
    return (round(da_c, 1), round(dc_c, 1), round(best_w / total, 3))


def measure_meld_snaps(img, state, hom):
    """{pos: (d_along, d_cross, score, n_features)} for each screen pos 0..3 whose seat
    has melds. Warp + meld masks + snap_meld_strip only (no river/hand/dora/hud)."""
    Hinv = hom["H_full_inv"]
    full = P.warp_to_full(img, hom["H_full"], hom["full_size"])
    hsv = cv2.cvtColor(full, cv2.COLOR_BGR2HSV)
    mw = P.tile_face_mask(hsv=hsv)
    mb = P.tile_back_mask(hsv=hsv)
    out = {}
    for pos in range(4):
        _, _, melds, _ = seat_gt(state, pos)
        if not melds:
            continue
        boxes = P.generate_meld_boxes_v2(pos, melds, Hinv)
        if not boxes:
            continue
        da, dc, diag = P.snap_meld_strip(mw, mb, boxes, pos)
        out[pos] = (da, dc, diag["score"], diag["n_along"] + diag["n_cross"])
    return out


def game_meld_overrides(seq_states, seq_frames, hom):
    """One game -> {seq: {pos: (d_along, d_cross) | None}}. Measures every settled frame,
    groups per (bakaze,kyoku,honba,pos), consensuses, and returns per-seq overrides.
    pos maps to None when that round has no confident consensus (annotate_frame then
    marks those meld boxes unreliable). Only pos with melds appear. Frames dropped by
    the build (deal/call window, invariant) are not measured (they never emit boxes)."""
    per_seq = {}
    by_round = defaultdict(list)
    for seq in sorted(seq_states):
        if seq not in seq_frames:
            continue
        st = seq_states[seq]
        if is_deal_window(st) or is_call_window(st) or check_invariants(st):
            continue
        img = cv2.imread(seq_frames[seq])
        if img is None:
            continue
        if img.shape[1] != 1920:
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
        m = measure_meld_snaps(img, st, hom)
        if not m:
            continue
        per_seq[seq] = m
        kk = (st.bakaze, st.kyoku, st.honba)
        for pos, s in m.items():
            by_round[(kk, pos)].append(s)
    consensus = {k: round_meld_consensus(v) for k, v in by_round.items()}
    overrides = {}
    for seq, m in per_seq.items():
        st = seq_states[seq]
        kk = (st.bakaze, st.kyoku, st.honba)
        ov = {}
        for pos in m:
            c = consensus.get((kk, pos))
            ov[pos] = (c[0], c[1]) if c is not None else None
        overrides[seq] = ov
    return overrides
