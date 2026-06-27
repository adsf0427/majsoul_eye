"""Auto-labeling: turn a synced (frame, BoardState) pair into training labels.

`Akagi GT = WHAT` + `coords/normalize = WHERE` → label samples, zero hand-drawing.
"""

from .autolabel import LabelSample, label_frame, save_classification_crops, to_yolo_lines

__all__ = ["LabelSample", "label_frame", "save_classification_crops", "to_yolo_lines"]
