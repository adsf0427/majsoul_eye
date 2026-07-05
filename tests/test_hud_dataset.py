"""hud_emit: reliable HUD boxes -> 55-class YOLO lines + rotated padded reader
crops; unreliable/no-text boxes emit no crop; buttons emit label only."""
import numpy as np

import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "build_dataset", pathlib.Path("scripts/train/build_dataset.py"))
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

from majsoul_eye.hud import HUD_NAME_TO_ID

frame = np.full((1080, 1920, 3), 40, np.uint8)
rec = {"hud_boxes": [
    {"name": "score_self", "px_box": [900, 460, 1000, 500], "text": "25000"},
    {"name": "score_across", "px_box": [900, 300, 1000, 335], "text": "24000"},
    {"name": "btn_pon", "px_box": [1200, 740, 1360, 790]},
    {"name": "wall_count", "px_box": [925, 385, 995, 415], "text": "余64",
     "reliable": False},
]}
lines, crops = bd.hud_emit(rec, frame, 1920, 1080, obb=False)
assert len(lines) == 3                                   # unreliable dropped
assert lines[0].startswith(f"{HUD_NAME_TO_ID['score_self']} ")
assert any(l.startswith(f"{HUD_NAME_TO_ID['btn_pon']} ") for l in lines)
assert len(crops) == 2                                   # buttons: no crop
relpath, img, meta = crops[0]
assert meta == {"file": relpath, "name": "score_self", "text": "25000", "pad": 0.15}
assert relpath.startswith("score_self/")
# 180° field comes out rotated-to-upright: crop of across (35px tall box +pad)
_, img2, meta2 = crops[1]
assert meta2["name"] == "score_across" and img2.shape[0] > 0
print("test_hud_dataset OK")
