"""Tests for per-frame annotation orchestration (majsoul_eye.annotate.frame).

annotate_frame's box GEOMETRY comes from GT + calibration (no image needed), so a
black frame exercises structure + the reliability gate (nothing renders -> every
box reliable=False). iter_tile_boxes / crop_box are the seam build_dataset uses.
"""
import numpy as np

from majsoul_eye.state.replay import BoardState, Meld, RiverTile
from majsoul_eye.annotate import build_homographies, annotate_frame, iter_tile_boxes, crop_box
from majsoul_eye.annotate.frame import AnnBox, crop_quad
from majsoul_eye.tiles import NAME_TO_ID
from majsoul_eye.normalize import locate_fullscreen
from majsoul_eye.coords import dora_slot, MAX_DORA

HOM = build_homographies(1920, 1080)


def _state():
    s = BoardState(hero_seat=0, bakaze="E", kyoku=1, honba=0, last_actor=1)
    s.rivers[0] = [RiverTile("1m"), RiverTile("2p")]
    s.melds[0] = [Meld("pon", 1, ["P", "P", "P"], called_pai="P")]
    s.hero_hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
    s.dora_markers = ["E"]
    return s


def test_annotate_frame_shape_and_reliability():
    rec = annotate_frame(np.zeros((1080, 1920, 3), np.uint8), _state(), HOM)
    assert set(rec) >= {"hero_seat", "kyoku", "discard_slots", "meld_boxes", "hand_boxes", "dora_boxes", "flags"}
    assert rec["hero_seat"] == 0 and rec["kyoku"] == "E1"
    assert set(rec["discard_slots"]) == {"0", "1", "2", "3"}
    assert len(rec["discard_slots"]["0"]) == 2          # self river = rivers[0]
    assert len(rec["meld_boxes"]["0"]) == 3             # pon -> 3 display cells
    # black frame renders nothing -> reliability gate marks every box unreliable
    assert all(not s.get("reliable", True) for s in rec["discard_slots"]["0"])


def test_iter_tile_boxes_zones_and_types():
    rec = annotate_frame(np.zeros((1080, 1920, 3), np.uint8), _state(), HOM)
    boxes = list(iter_tile_boxes(rec))
    assert {"river", "meld", "hand"} <= {b.zone for b in boxes}
    for b in boxes:
        assert b.tile in NAME_TO_ID
        assert (b.poly_original is not None) == (b.zone in ("river", "meld"))
        assert (b.px_box is not None) == (b.zone in ("hand", "dora"))


def test_reliable_propagation():
    rec = {"discard_slots": {"0": [
                {"tile": "1m", "face_poly_original": [[0, 0], [10, 0], [10, 10], [0, 10]], "riichi": False},
                {"tile": "2p", "face_poly_original": [[0, 0], [10, 0], [10, 10], [0, 10]],
                 "riichi": False, "reliable": False}],
            "1": [], "2": [], "3": []},
           "meld_boxes": {"0": [], "1": [], "2": [], "3": []}, "hand_boxes": [], "dora_boxes": []}
    a, b = list(iter_tile_boxes(rec))
    assert a.reliable is True and b.reliable is False
    assert a.sideways is False


def _dora_state():
    s = BoardState(hero_seat=0, bakaze="E", kyoku=1, honba=0, last_actor=1)
    s.dora_markers = ["E"]          # 1 revealed -> slots 1..4 are face-down backs
    return s


def test_dora_back_reliable_on_skinned_back():
    # Paint the 4 face-down dora slots a NON-orange (blue) skin colour.
    img = np.zeros((1080, 1920, 3), np.uint8)
    region = locate_fullscreen(img)
    for i in range(1, MAX_DORA):
        x1, y1, x2, y2 = region.norm_to_px(dora_slot(i))
        img[y1:y2, x1:x2] = (200, 40, 40)     # BGR bright blue
    rec = annotate_frame(img, _dora_state(), HOM)
    backs = [d for d in rec["dora_boxes"] if d.get("back")]
    assert len(backs) == 4
    assert all(d.get("reliable", True) for d in backs)   # skin back is rendered -> reliable

    # A black frame (nothing rendered) must still drop the back slots.
    black = annotate_frame(np.zeros((1080, 1920, 3), np.uint8), _dora_state(), HOM)
    black_backs = [d for d in black["dora_boxes"] if d.get("back")]
    assert black_backs and all(not d.get("reliable", True) for d in black_backs)


def test_crop_box_sizes():
    img = np.zeros((1080, 1920, 3), np.uint8)
    quad = AnnBox("river", "1m", "tile", [[100, 100], [160, 100], [160, 180], [100, 180]], None, False, True)
    px = AnnBox("hand", "1m", "tile", None, [100, 100, 160, 180], False, True)
    assert crop_box(img, quad).shape == (64, 64, 3)
    assert crop_box(img, px).shape == (64, 64, 3)
    assert crop_box(img, quad, size=96).shape == (96, 96, 3)
    assert crop_quad(img, quad.poly_original, 48).shape == (48, 48, 3)


def test_meld_snap_override_shifts_and_flags():
    import glob
    import numpy as np
    from majsoul_eye.annotate import annotate_frame, build_homographies
    from majsoul_eye.capture.gtframes import build_seq_state, load_frames
    import cv2, os

    # find a settled frame with an opponent meld (some captures have none, e.g. a
    # short/empty game; scan captures in sorted order for the first that qualifies)
    ss = fr = seq = None
    for cap in sorted(glob.glob("captures/raw/ai_session/run_*/game*/game*.jsonl")):
        ss_c = build_seq_state(cap); fr_c = load_frames(os.path.dirname(cap))
        seq_c = next((s for s in sorted(ss_c) if s in fr_c
                      and any(ss_c[s].melds[seat] for seat in range(4))), None)
        if seq_c is not None:
            ss, fr, seq = ss_c, fr_c, seq_c
            break
    assert seq is not None, "no capture with a settled meld frame found"
    hom = build_homographies(1920, 1080)
    img = cv2.imread(fr[seq])
    if img.shape[1] != 1920:
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
    # which pos has melds
    from majsoul_eye.annotate.seatgt import seat_gt
    pos = next(p for p in range(4) if seat_gt(ss[seq], p)[2])

    # override (0,0) => boxes exactly at the template (no snap); a given (da,dc) shifts them
    base = annotate_frame(img, ss[seq], hom, meld_snap_override={pos: (0.0, 0.0)})
    shifted = annotate_frame(img, ss[seq], hom, meld_snap_override={pos: (10.0, 0.0)})
    b0 = np.float32(base["meld_boxes"][str(pos)][0]["poly_fullwarp"])
    b1 = np.float32(shifted["meld_boxes"][str(pos)][0]["poly_fullwarp"])
    from majsoul_eye.annotate import pipeline as P
    along = np.array(P.MELD_STRIP2[pos]["along"])
    moved = (b1 - b0).mean(axis=0)
    assert abs(float(np.dot(moved, along)) - 10.0) < 0.5, moved  # moved +10 along
    assert base["meld_boxes"][str(pos)][0]["snap"] == (0.0, 0.0)

    # override None => template + reliable False + low_round_conf flag
    lc = annotate_frame(img, ss[seq], hom, meld_snap_override={pos: None})
    assert all(b.get("reliable") is False for b in lc["meld_boxes"][str(pos)])
    assert any(f == f"pos{pos}:meld:low_round_conf" for f in lc["flags"])
    print("meld_snap_override OK")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_annotate_frame OK")
