"""Pure-geometry unit tests for the precise annotation pipeline (majsoul_eye.annotate.pipeline).

No image needed — these pin the homography round-trip and the GT->box generators
so the moved calibration can't silently drift.
"""
import numpy as np

from majsoul_eye.annotate.pipeline import (
    build_homographies, transform_points, generate_discard_slots,
    meld_display_cells, generate_meld_boxes_v2, river_sideways_index,
)

HOM = build_homographies(1920, 1080)


def test_homography_roundtrip():
    pts = np.float32([[500, 300], [1400, 300], [960, 540], [700, 800]])
    full = transform_points(pts, HOM["H_full"])
    back = transform_points(full, HOM["H_full_inv"])
    assert np.allclose(pts, back, atol=1e-3), (pts, back)


def test_discard_slots_count_and_rows():
    river = [{"pai": p, "tsumogiri": False, "riichi": False}
             for p in ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m"]]   # 8 -> 2 rows (6/row)
    slots = generate_discard_slots(0, river, HOM["H_full_inv"])
    assert len(slots) == len(river)
    assert [s["tile"] for s in slots] == [r["pai"] for r in river]
    for s in slots:
        for k in ("poly_original", "face_poly_original", "poly_fullwarp", "row", "col"):
            assert k in s, k
    rows = [s["row"] for s in slots]
    assert rows == sorted(rows) and min(rows) == 1 and max(rows) == 2


def test_face_poly_inside_full():
    s = generate_discard_slots(2, [{"pai": "5p", "tsumogiri": False, "riichi": False}],
                               HOM["H_full_inv"])[0]
    full = np.asarray(s["poly_fullwarp"], float)
    face = np.asarray(s["face_poly_fullwarp"], float)
    assert full[:, 0].min() - 1e-6 <= face[:, 0].min() and face[:, 0].max() <= full[:, 0].max() + 1e-6
    assert full[:, 1].min() - 1e-6 <= face[:, 1].min() and face[:, 1].max() <= full[:, 1].max() + 1e-6


def test_meld_display_cells_compositions():
    chi = {"type": "chi", "tiles": ["4s", "5s", "6s"], "from_seat": 3, "called_pai": "4s", "added_pai": ""}
    c = meld_display_cells(chi, 0)
    assert len(c) == 3 and sum(x["sideways"] for x in c) == 1

    pon = {"type": "pon", "tiles": ["P", "P", "P"], "from_seat": 1, "called_pai": "P", "added_pai": ""}
    assert len(meld_display_cells(pon, 0)) == 3

    dk = {"type": "daiminkan", "tiles": ["2p"] * 4, "from_seat": 2, "called_pai": "2p", "added_pai": ""}
    assert len(meld_display_cells(dk, 0)) == 4

    ak = {"type": "ankan", "tiles": ["3m"] * 4, "from_seat": 0, "called_pai": "", "added_pai": ""}
    a = meld_display_cells(ak, 0)
    labels = [x["label"] for x in a]
    assert labels[0] == "back" and labels[-1] == "back" and labels.count("back") == 2

    kk = {"type": "kakan", "tiles": ["7s"] * 4, "from_seat": 1, "called_pai": "7s", "added_pai": "7s"}
    k = meld_display_cells(kk, 0)
    assert any("stacked" in x for x in k)


def test_generate_meld_boxes_v2_counts():
    melds = [{"type": "pon", "tiles": ["P", "P", "P"], "from_seat": 1, "called_pai": "P", "added_pai": ""},
             {"type": "ankan", "tiles": ["3m"] * 4, "from_seat": 0, "called_pai": "", "added_pai": ""}]
    boxes = generate_meld_boxes_v2(0, melds, HOM["H_full_inv"])
    assert len(boxes) == 7                                   # pon 3 + ankan 4 line cells
    assert all("poly_original" in b and "tile" in b for b in boxes)
    assert sum(1 for b in boxes if b["tile"] == "back") == 2


def test_river_sideways_index():
    riv = [{"riichi": False, "called": False}] * 3 + [{"riichi": True, "called": False}] \
        + [{"riichi": False, "called": False}]
    assert river_sideways_index(riv) == 3
    assert river_sideways_index([{"riichi": False, "called": False}] * 4) is None
    # riichi tile claimed away -> the NEXT visible discard renders sideways
    claimed = [{"riichi": False, "called": False}, {"riichi": True, "called": True},
               {"riichi": False, "called": False}]
    assert river_sideways_index(claimed) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_annotate_pipeline OK")
