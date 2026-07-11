import numpy as np
from types import SimpleNamespace

import majsoul_eye.recognize.hudstate as hudstate
from majsoul_eye.coords import HAND, dora_slot
from majsoul_eye.normalize import BoardRegion
from majsoul_eye.recognize.assemble import assemble_with_evidence
from majsoul_eye.recognize.detector import Detection
from majsoul_eye.recognize.evidence import CandidatePolicy
from majsoul_eye.tiles import NAME_TO_ID, TILE_NAMES

REGION = BoardRegion(0, 0, 1920, 1080, 1920, 1080)


def det_for_box(tile, box, score=0.9):
    x0, y0, x1, y1 = box
    return Detection((x0, y0, x1, y1), tile, tile, NAME_TO_ID[tile], score)


class StubClassifier:
    def predict_proba(self, crops):
        out = np.zeros((len(crops), len(TILE_NAMES)), np.float32)
        out[:, NAME_TO_ID["1m"]] = 0.65
        out[:, NAME_TO_ID["2m"]] = 0.25
        out[:, NAME_TO_ID["3m"]] = 0.10
        return out


def test_hand_and_dora_keep_original_detection_boxes():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    hand_box = REGION.norm_to_px(HAND.slot_box(0))
    dora_box = REGION.norm_to_px(dora_slot(0))
    result = assemble_with_evidence(
        [det_for_box("1m", hand_box), det_for_box("5s", dora_box)], REGION,
        frame_bgr=frame)
    hand = next(f for f in result.fields if f.field_key == "hand:0")
    dora = next(f for f in result.fields if f.field_key == "dora:0")
    assert hand.detections[0].xyxy == tuple(float(v) for v in hand_box)
    assert dora.detections[0].tile == "5s"


def test_candidates_require_named_calibration():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    hand_box = REGION.norm_to_px(HAND.slot_box(0))
    dets = [det_for_box("1m", hand_box)]
    off = assemble_with_evidence(dets, REGION, frame_bgr=frame,
                                 tile_classifier=StubClassifier(),
                                 candidate_policy=CandidatePolicy(None, 3))
    assert off.fields[0].candidates == []
    on = assemble_with_evidence(dets, REGION, frame_bgr=frame,
                                tile_classifier=StubClassifier(),
                                candidate_policy=CandidatePolicy("tile-temp-v1", 3))
    assert [(c.value, round(c.confidence, 2)) for c in on.fields[0].candidates] == [
        ("1m", 0.65), ("2m", 0.25), ("3m", 0.10)]


def test_hud_scores_use_numeric_relative_seats_zero_through_three():
    original = hudstate.assemble_hud
    hudstate.assemble_hud = lambda dets, reader, frame: {
        "scores": [25000, 26000, 24000, 25000],
    }
    try:
        dets = [SimpleNamespace(name=name) for name in
                ("score_self", "score_right", "score_across", "score_left")]
        result = hudstate.assemble_hud_with_evidence(dets, None, None)
    finally:
        hudstate.assemble_hud = original
    assert [(field.field_key, field.value) for field in result.fields] == [
        ("round.scores.0", 25000), ("round.scores.1", 26000),
        ("round.scores.2", 24000), ("round.scores.3", 25000),
    ]


def test_hud_class_probabilities_only_expose_classification_heads():
    import torch

    from majsoul_eye.hud import ROUND_CLASSES, WIND_CLASSES
    from majsoul_eye.recognize.hudreader import HudReader

    class FixedHead:
        def __init__(self, size):
            self.logits = torch.arange(size, dtype=torch.float32)[None]

        def __call__(self, _tensor):
            return self.logits

    reader = HudReader.__new__(HudReader)
    reader.device = "cpu"
    reader.round = FixedHead(len(ROUND_CLASSES))
    reader.wind = FixedHead(len(WIND_CLASSES))
    crop = np.zeros((32, 64, 3), np.uint8)

    round_names, round_probabilities = reader.class_probabilities(crop, "round_label")
    wind_names, wind_probabilities = reader.class_probabilities(crop, "seat_wind_self")
    assert round_names == list(ROUND_CLASSES)
    assert wind_names == list(WIND_CLASSES)
    assert np.isclose(round_probabilities.sum(), 1.0)
    assert np.isclose(wind_probabilities.sum(), 1.0)
    assert int(round_probabilities.argmax()) == len(ROUND_CLASSES) - 1
    assert int(wind_probabilities.argmax()) == len(WIND_CLASSES) - 1

    try:
        reader.class_probabilities(crop, "score_self")
    except ValueError as exc:
        assert "no class distribution" in str(exc)
    else:
        raise AssertionError("numeric CTC fields must not expose class probabilities")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_recognition_evidence OK")
