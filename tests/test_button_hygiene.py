"""Dataset hygiene: a frame whose buttons could not be labeled (count_mismatch)
must not enter the DETECTOR dataset at all.

Root cause it guards (STATUS §1.55): `hud_emit` skips unreliable boxes but the
frame image was still written, so a visibly-rendered button became a YOLO
background negative. Measured: 1009 such frames in v5's train split vs 1102
frames carrying real button labels -> the detector learned to suppress buttons
on any background it had seen dropped (val recall 0/92 on rendered-but-dropped
buttons, vs 99.3% on labeled ones).
"""
import importlib.util

_spec = importlib.util.spec_from_file_location("bd", "scripts/train/build_dataset.py")
bd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bd)


def _rec(hud_boxes):
    return {"hud_boxes": hud_boxes}


# --- the predicate ----------------------------------------------------------
# count_mismatch => GT says a button is on screen but we have no box for it.
assert bd.has_unlabeled_buttons(_rec([
    {"name": "btn_chi", "px_box": [10, 20, 30, 40], "reliable": False,
     "flag": "count_mismatch"},
    {"name": "btn_skip", "px_box": None, "reliable": False,
     "flag": "count_mismatch"},
])) is True

# A frame with no buttons offered at all is clean.
assert bd.has_unlabeled_buttons(_rec([])) is False
assert bd.has_unlabeled_buttons({}) is False

# Properly labeled buttons are clean.
assert bd.has_unlabeled_buttons(_rec([
    {"name": "btn_chi", "px_box": [10, 20, 30, 40]},
    {"name": "btn_skip", "px_box": [50, 20, 70, 40]},
])) is False

# A numeric HUD field marked unreliable (score animation, low ink) is NOT a
# button drop — those boxes are simply omitted, nothing is left unlabeled on
# screen that the detector must learn as an object.
assert bd.has_unlabeled_buttons(_rec([
    {"name": "score_self", "px_box": [10, 20, 30, 40], "reliable": False},
    {"name": "btn_chi", "px_box": [50, 20, 70, 40]},
])) is False

# score_anim marks button boxes unreliable WITHOUT a count_mismatch flag; the
# buttons are still located, so the frame stays usable for the detector.
assert bd.has_unlabeled_buttons(_rec([
    {"name": "btn_riichi", "px_box": [10, 20, 30, 40], "reliable": False},
])) is False

# --- the guard actually excludes the frame from the YOLO set ----------------
# hud_emit already drops the boxes; the frame image must go too.
lines, crops = bd.hud_emit(_rec([
    {"name": "btn_chi", "px_box": [10, 20, 30, 40], "reliable": False,
     "flag": "count_mismatch"},
]), None, 1920, 1080, False)
assert lines == [], f"count_mismatch buttons must emit no label lines, got {lines}"

print("test_button_hygiene OK")
