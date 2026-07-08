"""Unit tests for round_meld_consensus (pure) + a smoke test for game_meld_overrides."""
import glob
import os

from majsoul_eye.annotate.meldsnap import round_meld_consensus


def test_majority_cluster_wins_over_single_outlier():
    # 17 frames agree at (0.8, 24), one flips to (51, -0.5) -> consensus is the majority
    samples = [(0.8, 24.0, 2.6, 5)] * 17 + [(51.0, -0.5, 2.9, 6)]
    r = round_meld_consensus(samples)
    assert r is not None
    assert abs(r[0] - 0.8) < 2.0 and abs(r[1] - 24.0) < 2.0, r
    assert r[2] > 0.9, r  # conf


def test_no_feature_frames_are_ignored():
    # n_features < MIN_FEATURES (2) are dropped; too few real samples -> None
    assert round_meld_consensus([(0.0, 0.0, 0.0, 1)] * 8) is None


def test_too_few_confident_frames_returns_none():
    assert round_meld_consensus([(1.0, 1.0, 2.0, 4)] * 2) is None


def test_ambiguous_split_returns_none():
    # 3 at (0,0) vs 3 at (46,0): no cluster reaches MIN_ROUND_CONF (0.55) -> None (safe)
    samples = [(0.0, 0.0, 2.0, 4)] * 3 + [(46.0, 0.0, 2.0, 4)] * 3
    assert round_meld_consensus(samples) is None


def test_cross_axis_consensus():
    # occasional cross flip: 10 at dc=24, 1 at dc=-0.5 -> consensus dc ~24
    samples = [(1.0, 24.0, 4.0, 5)] * 10 + [(1.0, -0.5, 3.0, 6)]
    r = round_meld_consensus(samples)
    assert r is not None and abs(r[1] - 24.0) < 2.0, r


def test_game_meld_overrides_smoke():
    from majsoul_eye.annotate import build_homographies
    from majsoul_eye.annotate.meldsnap import game_meld_overrides
    from majsoul_eye.capture.gtframes import build_seq_state, load_frames
    caps = glob.glob("captures/raw/ai_session/run_*/game*/game*.jsonl")
    assert caps, "no AI captures found — run from repo root"
    cap = sorted(caps)[0]
    ss = build_seq_state(cap)
    fr = load_frames(os.path.dirname(cap))
    ov = game_meld_overrides(ss, fr, build_homographies(1920, 1080))
    # every override value is a dict {pos: (da,dc) | None}; within one kyoku+pos all
    # non-None overrides are IDENTICAL (that is the whole point — one offset per round).
    from collections import defaultdict
    by_round = defaultdict(set)
    for seq, per_pos in ov.items():
        st = ss[seq]
        for pos, val in per_pos.items():
            if val is not None:
                by_round[(st.bakaze, st.kyoku, st.honba, pos)].add(val)
    for key, vals in by_round.items():
        assert len(vals) == 1, f"{key} has non-uniform override {vals}"
    print("game_meld_overrides smoke OK:", len(ov), "frames")


if __name__ == "__main__":
    for _n, _f in sorted(list(globals().items())):
        if _n.startswith("test_") and callable(_f):
            _f()
            print(_n, "OK")
    print("all meldsnap tests passed")
