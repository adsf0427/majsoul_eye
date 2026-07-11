import hashlib

import cv2
import numpy as np

from majsoul_eye.recognize.runtime import RecognitionContext, RecognitionRuntime, RuntimeFailure
from majsoul_eye.what_cut.schema import parse_what_cut_draft


class FakeDetector:
    def predict(self, image): return []


class FakeClassifier:
    def predict_proba(self, crops): return np.empty((0, 38), np.float32)


class FakeHudReader:
    pass


def runtime():
    instance = RecognitionRuntime.__new__(RecognitionRuntime)
    instance.manifest = type("M", (), {
        "manifest_version": "test-v1", "layout_id": "majsoul-desktop-16x9-v1",
        "support_status": "experimental", "manifest_sha256": "a" * 64,
        "raw": {"layout": {"minWidth": 1280, "minHeight": 720,
                            "aspectRatio": 16 / 9, "aspectTolerance": 0.02},
                "candidates": {"topK": 3, "calibrationVersion": None}},
        "assets": {"detector": type("A", (), {"sha256": "b" * 64})(),
                   "classifier": type("A", (), {"sha256": "c" * 64})(),
                   "hudReader": type("A", (), {"sha256": "d" * 64})()},
    })()
    instance.detector = FakeDetector()
    instance.classifier = FakeClassifier()
    instance.hud_reader = FakeHudReader()
    instance.eye_revision = "eye-rev"
    return instance


def png(width=1280, height=720):
    ok, data = cv2.imencode(".png", np.zeros((height, width, 3), np.uint8))
    assert ok
    return data.tobytes()


def test_experimental_layout_requires_explicit_internal_opt_in():
    body = png()
    context = RecognitionContext("req", "draft", hashlib.sha256(body).hexdigest(),
                                 "majsoul-desktop-16x9-v1", False, None)
    try:
        runtime().recognize_bytes(body, context)
    except RuntimeFailure as exc:
        assert exc.code == "UNSUPPORTED_LAYOUT"
    else:
        raise AssertionError("experimental layout must reject default caller")


def test_recognize_returns_parseable_draft_even_with_structure_issues():
    body = png()
    context = RecognitionContext("req", "draft", hashlib.sha256(body).hexdigest(),
                                 "majsoul-desktop-16x9-v1", True, "image-ref")
    data = runtime().recognize_bytes(body, context)
    assert data["schemaVersion"] == 1
    assert parse_what_cut_draft(data["draft"])["draftId"] == "draft"
    assert data["recognizer"]["supportStatus"] == "experimental"


def test_reconstruct_failure_has_frozen_null_fields():
    body = png()
    context = RecognitionContext("req", "draft", hashlib.sha256(body).hexdigest(),
                                 "majsoul-desktop-16x9-v1", True, "image-ref")
    draft = runtime().recognize_bytes(body, context)["draft"]
    data = runtime().reconstruct_draft(draft, draft["revision"])
    assert data["ok"] is False
    assert data["issues"] and data["issues"][0]["severity"] == "blocking"
    assert data["mjai"] is None and data["heroSeatAbs"] is None
    assert data["fabricated"] is None and data["selectedHistory"] is None
    assert data["decision"] is None
    assert data["historyBaseline"] == []


def test_original_byte_digest_must_match_context():
    context = RecognitionContext("req", "draft", "0" * 64,
                                 "majsoul-desktop-16x9-v1", True, None)
    try:
        runtime().recognize_bytes(png(), context)
    except RuntimeFailure as exc:
        assert exc.code == "INVALID_IMAGE_DIGEST"
    else:
        raise AssertionError("mismatched screenshot digest must be rejected")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_recognition_runtime OK")
