"""Tests for grid-based 河 auto-labeling (WHERE = perspective grid, WHAT = GT order).

Real frames still tune the per-seat quads (coords.RIVER_QUADS, T2 calibration);
these pin the grid math + GT-order assignment.
"""

import cv2
import numpy as np

from majsoul_eye.coords import RIVER_QUADS
from majsoul_eye.normalize import locate_fullscreen
from majsoul_eye.state.replay import BoardState, RiverTile
from majsoul_eye.label.river import RiverGrid, label_river, detect_river_blocks, assign_from_gt


def _grid():
    return RiverGrid((0.40, 0.50), (0.58, 0.50), (0.58, 0.68), (0.40, 0.68))


def test_grid_cells_monotonic_inbounds():
    g = _grid()
    assert g.capacity == 18
    c0, c5, c6, c17 = g.cell_box(0), g.cell_box(5), g.cell_box(6), g.cell_box(17)
    assert abs(c0.x0 - 0.40) < 1e-3 and abs(c0.y0 - 0.50) < 1e-3
    assert c5.x0 > c0.x0 and abs(c5.y0 - c0.y0) < 1e-3      # same row, further right
    assert c6.y0 > c0.y0 and abs(c6.x0 - c0.x0) < 1e-3      # next row, same column
    assert abs(c17.x1 - 0.58) < 1e-3 and abs(c17.y1 - 0.68) < 1e-3
    for i in range(18):
        b = g.cell_box(i)
        assert 0.40 - 1e-6 <= b.x0 < b.x1 <= 0.58 + 1e-6
        assert 0.50 - 1e-6 <= b.y0 < b.y1 <= 0.68 + 1e-6


def test_riichi_widens_cell():
    g = _grid()
    assert g.cell_box(0, riichi=True).w > g.cell_box(0, riichi=False).w


def test_label_river_grid_order():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    region = locate_fullscreen(frame)
    s = BoardState(hero_seat=0)
    s.rivers[0] = [RiverTile("1m"), RiverTile("2m"), RiverTile("3m")]
    g = RiverGrid(*RIVER_QUADS["self"])
    samples, ok = label_river(region, s, "self", g)
    assert ok and [x.label for x in samples] == ["1m", "2m", "3m"]
    assert all(x.zone == "river" and x.tile_class is not None for x in samples)
    for x in samples:
        x0, y0, x1, y1 = x.px_box
        assert 0 <= x0 < x1 <= 1920 and 0 <= y0 < y1 <= 1080


def test_overflow_flagged_and_capped():
    s = BoardState(hero_seat=0)
    s.rivers[0] = [RiverTile("1m")] * 19          # > capacity 18
    g = RiverGrid(*RIVER_QUADS["self"])
    samples, ok = label_river(locate_fullscreen(np.zeros((1080, 1920, 3), np.uint8)), s, "self", g)
    assert not ok and len(samples) == g.capacity


def test_assign_from_gt_counts():
    pairs, ok = assign_from_gt([(0, 0, 1, 1)] * 3, ["1m", "2m", "3m"])
    assert ok and [t for _, t in pairs] == ["1m", "2m", "3m"]
    pairs, ok = assign_from_gt([(0, 0, 1, 1)] * 2, ["1m", "2m", "3m"])
    assert not ok and len(pairs) == 2


def test_called_tile_excluded_from_gt():
    s = BoardState(hero_seat=0)
    s.rivers[0] = [RiverTile("1m"), RiverTile("2m", called=True)]
    assert [t.pai for t in s.visible_river(0)] == ["1m"]


def test_detect_river_blocks_assigns_seats():
    frame = np.full((1080, 1920, 3), 20, np.uint8)
    centers = {"self": (0.50, 0.60), "across": (0.50, 0.21), "left": (0.34, 0.39), "right": (0.66, 0.39)}
    for cx, cy in centers.values():
        x0, y0 = int((cx - 0.065) * 1920), int((cy - 0.075) * 1080)
        x1, y1 = int((cx + 0.065) * 1920), int((cy + 0.075) * 1080)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (220, 220, 220), -1)
    blocks = detect_river_blocks(frame)
    assert set(blocks) == {"self", "across", "left", "right"}
    for pos, (cx, cy) in centers.items():
        bcx, bcy = blocks[pos]["center"]
        assert abs(bcx - cx) < 0.03 and abs(bcy - cy) < 0.03


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_river OK")
