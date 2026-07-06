"""Detector aug-config tests: the corrected defaults (fliplr off, hsv_v boosted)
and overridability. No torch/ultralytics/GPU — build_train_kwargs is pure."""

import os

from scripts.train.train_detector import build_parser, build_train_kwargs


def test_aug_defaults_turn_off_fliplr_and_boost_value():
    args = build_parser().parse_args(["--data", "d.yaml"])
    kw, aug = build_train_kwargs(args, device=0)
    assert kw["fliplr"] == 0.0        # directional tiles: no mirror flip
    assert kw["flipud"] == 0.0        # never wanted on a top-down board
    assert kw["hsv_v"] == 0.5         # brightness / dora-glow proxy (was YOLO 0.4)
    assert kw["hsv_s"] == 0.7         # unchanged default, now explicit
    assert kw["mosaic"] == 1.0
    assert kw["close_mosaic"] == 10
    assert kw["data"] == "d.yaml"
    assert "project" not in kw        # not set unless --project given
    assert aug["fliplr"] == 0.0 and "data" not in aug   # aug is the sub-dict only


def test_aug_overridable_from_cli():
    args = build_parser().parse_args(
        ["--data", "d.yaml", "--fliplr", "0.5", "--hsv-v", "0.9", "--project", "runs/x"])
    kw, _ = build_train_kwargs(args, device="0,1")
    assert kw["fliplr"] == 0.5
    assert kw["hsv_v"] == 0.9
    assert kw["device"] == "0,1"
    # absolute: ultralytics nests a RELATIVE project under runs/<task>/ (get_save_dir)
    assert kw["project"] == os.path.abspath("runs/x") and os.path.isabs(kw["project"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_train_detector_aug OK")
