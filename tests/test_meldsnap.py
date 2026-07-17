"""Unit tests for round_meld_consensus (pure) + a smoke test for game_meld_overrides."""
import glob
import os

import pytest

from majsoul_eye.annotate.meldsnap import round_meld_consensus


def test_majority_cluster_wins_over_single_outlier():
    # 17 frames agree at (0.8, 24), one flips to (51, -0.5) -> consensus is the majority
    samples = [(0.8, 24.0, 2.6, 5)] * 17 + [(51.0, -0.5, 2.9, 6)]
    r = round_meld_consensus(samples)
    assert r is not None
    assert abs(r[0] - 0.8) < 2.0 and abs(r[1] - 24.0) < 2.0, r
    assert r[2] > 0.9, r  # conf


def test_no_feature_frames_are_ignored():
    # score > 0 but n_features < MIN_FEATURES (2): dropped ONLY by the feature gate, so a
    # round of these has too few confident samples -> None. (If the n-gate were removed,
    # all 8 would survive as confident and consensus would return a value, not None.)
    assert round_meld_consensus([(0.0, 0.0, 3.0, 1)] * 8) is None


def test_too_few_confident_frames_returns_none():
    assert round_meld_consensus([(1.0, 1.0, 2.0, 4)] * 2) is None


def test_ambiguous_split_returns_none():
    # 3 at (0,0) vs 3 at (46,0): no cluster reaches MIN_ROUND_CONF (0.55) -> None (safe)
    samples = [(0.0, 0.0, 2.0, 4)] * 3 + [(46.0, 0.0, 2.0, 4)] * 3
    assert round_meld_consensus(samples) is None


def test_cross_axis_consensus():
    # Same d_along (1.0), two d_cross groups 44px apart. Correct 2-D clustering keeps them
    # separate and returns the majority dc=24. A broken ALONG-ONLY clustering would merge
    # all 14 (same da) and average dc to ~11.4 -> caught by the tight tolerance.
    samples = [(1.0, 24.0, 4.0, 5)] * 10 + [(1.0, -20.0, 4.0, 5)] * 4
    r = round_meld_consensus(samples)
    assert r is not None
    assert abs(r[1] - 24.0) < 3.0, r
    assert abs(r[0] - 1.0) < 1.0, r


def test_game_meld_overrides_smoke():
    from collections import defaultdict
    from majsoul_eye.annotate import build_homographies
    from majsoul_eye.annotate.meldsnap import game_meld_overrides
    from majsoul_eye.capture.gtframes import build_seq_state, load_frames
    caps = sorted(glob.glob("captures/raw/ai_session/run_*/game*/game*.jsonl"))
    if not caps:
        # The AI-session capture corpus lives only on the dev machine
        # (gitignored); a clean clone / release gate cannot exercise this.
        pytest.skip("local capture corpus unavailable")
    hom = build_homographies(1920, 1080)
    exercised = 0
    for cap in caps:
        ss = build_seq_state(cap)
        fr = load_frames(os.path.dirname(cap))
        ov = game_meld_overrides(ss, fr, hom)
        by_round = defaultdict(set)
        by_round_n = defaultdict(int)
        for seq, per_pos in ov.items():
            st = ss[seq]
            for pos, val in per_pos.items():
                if val is not None:
                    key = (st.bakaze, st.kyoku, st.honba, pos)
                    by_round[key].add(val)
                    by_round_n[key] += 1
        # within a (kyoku,pos) round every non-None override must be IDENTICAL
        for key, vals in by_round.items():
            assert len(vals) == 1, f"{key} has non-uniform override {vals}"
            if by_round_n[key] >= 2:
                exercised += 1
        if exercised >= 3:            # enough multi-frame rounds actually checked; stop
            break
    assert exercised >= 1, "smoke test never exercised a multi-frame meld round (vacuous)"
    print("game_meld_overrides smoke OK: exercised", exercised, "multi-frame rounds")


if __name__ == "__main__":
    for _n, _f in sorted(list(globals().items())):
        if _n.startswith("test_") and callable(_f):
            _f()
            print(_n, "OK")
    print("all meldsnap tests passed")
