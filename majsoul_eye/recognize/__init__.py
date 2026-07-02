"""Runtime recognition models (the shipped recognizer — Akagi-free)."""

from .classifier import TileNet, TileClassifier, preprocess, INPUT
from .detector import TileDetector, Detection   # ultralytics imported lazily inside

__all__ = ["TileNet", "TileClassifier", "preprocess", "INPUT",
           "TileDetector", "Detection"]
