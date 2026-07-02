"""Precise fullwarp annotation pipeline (top-down homography + data-calibrated
discard grid + composition-aware melds + per-frame mask snap).

- ``pipeline``  — pure geometry + image-evidence core (moved verbatim from the
  former root ``mahjong_relative_annotation_pipeline.py``, now removed).
- ``seatgt``    — GT plumbing (``seat_gt`` / ``_screen_to_seat`` / ``SEAT_POS``).
- ``cases``     — the named AB validation seqs (``CASES``).
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
