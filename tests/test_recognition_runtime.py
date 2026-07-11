import hashlib

import cv2
import numpy as np

from majsoul_eye.recognize.runtime import RecognitionContext, RecognitionRuntime, RuntimeFailure
from majsoul_eye.what_cut.schema import parse_what_cut_draft
from test_assemble import _dora_dets, _gt, _hand_dets, _meld_dets, _river_dets

H13 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
       "1p", "2p", "3p", "4p"]


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


def _board_runtime():
    """Runtime whose detector emits a coherent full board (forward fixture
    geometry from test_assemble): hero H13 + drawn 5p, rivers
    [[9p], [], [S], [W]], seat-2 pon of P claimed from relative seat 3."""
    dets = (_hand_dets(H13, drawn="5p")
            + _dora_dets(["5s"])
            + _river_dets(0, ["9p"])
            + _river_dets(2, ["S"])
            + _river_dets(3, ["W"])
            + _meld_dets(2, [_gt("pon", ["P", "P", "P"], called="P",
                                 from_seat_rel=3, seat=2)]))

    class BoardDetector:
        def predict(self, image): return dets

    instance = runtime()
    instance.detector = BoardDetector()
    return instance


def test_success_chain_recognizes_reconstructs_and_decides():
    body = png(1920, 1080)
    context = RecognitionContext("req", "draft", hashlib.sha256(body).hexdigest(),
                                 "majsoul-desktop-16x9-v1", True, None)
    instance = _board_runtime()
    data = instance.recognize_bytes(body, context)
    assert data["issues"] == []
    draft = parse_what_cut_draft(data["draft"])
    ghosts = draft["historyOverrides"]["ghostDiscards"]
    assert [g["id"] for g in ghosts] == ["ghost:2:0"]
    assert ghosts[0]["ownerRelSeat"] == 1 and ghosts[0]["pai"] == "P"
    # baseline sync visibly rewrote the build-time default: seat 2's post-pon
    # discard is a FORCED tedashi (build default was inferred/False)
    assert draft["players"][2]["rivers"][0]["tsumogiri"]["source"] == "forced"
    assert draft["players"][2]["rivers"][0]["tsumogiri"]["baselineSource"] == "forced"

    result = instance.reconstruct_draft(draft, draft["revision"])
    assert result["ok"] is True, result["issues"]
    assert result["mjai"] is not None and result["heroSeatAbs"] is not None
    assert result["fabricated"] is not None
    assert result["selectedHistory"] is not None
    assert result["selectedHistory"]["solverVersion"] == "hidden-history-v1"
    assert [(item["itemKind"], item["itemId"]) for item in result["historyBaseline"]] == [
        ("river", "river:0:0"), ("river", "river:2:0"),
        ("river", "river:3:0"), ("ghost", "ghost:2:0")]
    decision = result["decision"]
    assert decision is not None
    assert decision["actorRelSeat"] == 0 and decision["kind"] == "action"
    assert "5p" in decision["legalDiscards"] and decision["candidateCount"] >= 14


def test_reconstruct_with_stale_revision_reports_stale_revision():
    body = png(1920, 1080)
    context = RecognitionContext("req", "draft", hashlib.sha256(body).hexdigest(),
                                 "majsoul-desktop-16x9-v1", True, None)
    instance = _board_runtime()
    draft = instance.recognize_bytes(body, context)["draft"]
    data = instance.reconstruct_draft(draft, draft["revision"] + 1)
    assert data["ok"] is False
    assert data["issues"][0]["code"] == "STALE_REVISION"
    assert data["mjai"] is None and data["historyBaseline"] == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_recognition_runtime OK")
