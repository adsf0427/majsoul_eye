"""Precise fullwarp annotation pipeline (top-down homography + data-calibrated
discard grid + composition-aware melds + per-frame mask snap).

- ``pipeline``  — pure geometry + image-evidence core (moved verbatim from the
  root ``mahjong_relative_annotation_pipeline.py``; that name still re-exports it).
- ``seatgt``    — GT plumbing (``seat_gt`` / ``SEAT_POS``).
- ``frame``     — per-frame orchestration: ``annotate_frame`` (full record),
  ``iter_tile_boxes`` / ``AnnBox`` / ``crop_box`` for crop+label consumers.
"""
from majsoul_eye.annotate import pipeline  # noqa: F401
from majsoul_eye.annotate.pipeline import build_homographies  # noqa: F401
from majsoul_eye.annotate.frame import (  # noqa: F401
    annotate_frame,
    iter_tile_boxes,
    crop_box,
    crop_quad,
    AnnBox,
)
