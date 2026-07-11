"""Sanma (3P) adaptation units: mode flag, HUD gating, geometry switch, nuki boxes.

Screen-layout ground truth (STATUS §1.59): actors 0-2 are the E1 chair indices,
chair 3 (the would-be north seat) renders empty all game, so the 4P ring
(hero+rel)%4 holds unchanged and the phantom's rel is (3-hero)%4.
"""
import numpy as np

from majsoul_eye.state.replay import Replayer, BoardState
from majsoul_eye.annotate.hud import field_texts
from majsoul_eye.annotate import pipeline as P


def _kyoku(scores, **kw):
    ev = {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0,
          "kyotaku": 0, "oya": 0, "scores": scores,
          "dora_marker": "1p", "tehais": [["?"] * 13] * 4}
    ev.update(kw)
    return ev


def _play(rp: Replayer, *events: dict) -> None:
    for ev in events:
        rp.apply(ev)


def test_sanma_flag_from_start_kyoku():
    rp = Replayer()
    _play(rp, {"type": "start_game", "id": 1}, _kyoku([35000, 35000, 35000, 0]))
    assert rp.state.sanma is True
    assert rp.state.scores == [35000, 35000, 35000, 0]

    rp4 = Replayer()
    _play(rp4, {"type": "start_game", "id": 1}, _kyoku([25000] * 4))
    assert rp4.state.sanma is False

    # REGRESSION (real capture, ai_session/run_8/game6): a 4P seat CAN start a
    # kyoku at exactly 0 points — the conservation identity (sum+1000*kyotaku:
    # 100000 vs 105000) must not misread it as sanma.
    rp0 = Replayer()
    _play(rp0, {"type": "start_game", "id": 1},
          _kyoku([19600, 47000, 33400, 0], kyoku=3, oya=2))
    assert rp0.state.sanma is False

    # sanma renchan with a pending riichi stick in the pot still conserves 105000
    rp3 = Replayer()
    _play(rp3, {"type": "start_game", "id": 1},
          _kyoku([36000, 33000, 35000, 0], kyoku=1, honba=1, kyotaku=1))
    assert rp3.state.sanma is True


def test_sanma_flag_from_unpadded_scores():
    rp = Replayer()
    _play(rp, {"type": "start_game", "id": 0}, _kyoku([35000, 35000, 35000]))
    assert rp.state.sanma is True
    assert rp.state.scores == [35000, 35000, 35000, 0]   # padded to NUM_PLAYERS


def test_sanma_flag_from_nukidora_and_sticky():
    rp = Replayer()
    _play(rp, {"type": "start_game", "id": 2}, _kyoku([35000, 35000, 35000, 0]),
          {"type": "tsumo", "actor": 0, "pai": "?"},
          {"type": "nukidora", "actor": 0, "pai": "N"})
    assert rp.state.sanma is True and rp.state.nukidora[0] == 1
    # sticky across kyoku (next start_kyoku keeps the flag even if scores drift)
    _play(rp, _kyoku([36000, 34000, 35000, 0], kyoku=2, oya=1))
    assert rp.state.sanma is True
    assert rp.state.copy().sanma is True                 # survives copy()


def test_field_texts_sanma_gating():
    st = BoardState(hero_seat=1, bakaze="E", kyoku=2, oya=1, in_round=True,
                    scores=[35000, 34000, 36000, 0], sanma=True)
    t = field_texts(st)
    # hero=1 -> phantom chair rel = (3-1)%4 = 2 = across: no score_across label
    assert "score_across" not in t
    assert t["score_self"] == "34000" and t["score_right"] == "36000"
    assert t["score_left"] == "35000"
    assert t["seat_wind_self"] == "E"                    # (1-1)%3
    st.oya = 2
    assert field_texts(st)["seat_wind_self"] == "W"      # (1-2)%3 = 2 -> "ESW"[2]

    st4 = BoardState(hero_seat=1, bakaze="E", kyoku=2, oya=2, in_round=True,
                     scores=[25000] * 4, sanma=False)
    t4 = field_texts(st4)
    assert "score_across" in t4 and t4["seat_wind_self"] == "N"   # (1-2)%4 = 3


def test_set_sanma_swaps_in_place():
    grid_ref = P.DISCARD_GRID                            # identity must survive
    four = P.DISCARD_GRID[1]["dcol"]
    P.set_sanma(True)
    assert P.DISCARD_GRID is grid_ref
    assert P.DISCARD_GRID[1]["dcol"] == (0.0, 72.72)     # 3P side-river pitch
    assert P.MELD_STRIP2[0]["corner"] == (2388.0, 1843.5)
    P.set_sanma(True)                                    # idempotent
    assert P.DISCARD_GRID[1]["dcol"] == (0.0, 72.72)
    P.set_sanma(False)
    assert P.DISCARD_GRID[1]["dcol"] == four == (0.0, 74.88)
    assert P.MELD_STRIP2[0]["corner"] == (2388.2, 1889.5)


def test_generate_nukidora_boxes():
    hom = P.build_homographies(1920, 1080)
    assert P.generate_nukidora_boxes(0, 0, hom["H_full_inv"]) == []
    boxes = P.generate_nukidora_boxes(0, 3, hom["H_full_inv"])
    assert len(boxes) == 3
    assert all(b["tile"] == "N" and b["nuki"] and not b["sideways"] for b in boxes)
    xs = [np.mean(np.float32(b["poly_fullwarp"])[:, 0]) for b in boxes]
    assert xs[0] > xs[1] > xs[2]                         # self pile grows leftward
    assert abs((xs[0] - xs[1]) - 72.61) < 0.1
    # original-space quads exist and are 4-point
    assert all(len(b["poly_original"]) == 4 for b in boxes)
    # right pile grows downward
    ys = [np.mean(np.float32(b["poly_fullwarp"])[:, 1])
          for b in P.generate_nukidora_boxes(1, 2, hom["H_full_inv"])]
    assert ys[1] - ys[0] > 80


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    _main()
