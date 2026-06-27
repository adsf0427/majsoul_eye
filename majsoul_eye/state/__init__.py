"""Offline game-state reconstruction from captured GT.

Pure, Akagi-free: consumes MJAI events (+ raw-liqi extras) from a capture and
rebuilds the full 4-player board state per tick — the structured 场况 that the
labeler turns into bounding-box / classification labels.
"""

from .replay import BoardState, Meld, RiverTile, Replayer

__all__ = ["BoardState", "Meld", "RiverTile", "Replayer"]
