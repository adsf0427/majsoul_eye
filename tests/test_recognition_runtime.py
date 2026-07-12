import dataclasses
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
        "raw": {"layout": {"minFrameWidth": 640, "minFrameHeight": 360,
                            "minBoardWidth": 1280, "minBoardHeight": 720,
                            "anchorToleranceCanon": 30.0, "maxResidualCanon": 8.0,
                            "minHandInliers": 4, "clipToleranceFrac": 0.005},
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


def context(body, *, allow=True, rect=(0, 0, 1280, 720), ref="image-ref"):
    return RecognitionContext("req", "draft", hashlib.sha256(body).hexdigest(),
                              "majsoul-desktop-16x9-v1", allow, ref, rect)


def test_experimental_layout_requires_explicit_internal_opt_in():
    body = png()
    try:
        runtime().recognize_bytes(body, context(body, allow=False))
    except RuntimeFailure as exc:
        assert exc.code == "UNSUPPORTED_LAYOUT"
    else:
        raise AssertionError("experimental layout must reject default caller")


def test_recognize_returns_parseable_draft_even_with_structure_issues():
    # A calibrated rect (the manual fallback) stands in for the landmark fit, so
    # this still exercises draft-building on a frame with nothing recognizable.
    body = png()
    data = runtime().recognize_bytes(body, context(body))
    assert data["schemaVersion"] == 1
    assert parse_what_cut_draft(data["draft"])["draftId"] == "draft"
    assert data["recognizer"]["supportStatus"] == "experimental"


def test_unlocalizable_frame_is_rejected_rather_than_drafted():
    """No landmarks and no calibration => we cannot know where the board is.

    Guessing (what locate_fullscreen did) does not fail loudly; it silently maps
    every tile into the wrong zone. Refusing is the only honest answer.
    """
    body = png()
    try:
        runtime().recognize_bytes(body, context(body, rect=None))
    except RuntimeFailure as exc:
        assert exc.code == "LOCALIZATION_FAILED"
    else:
        raise AssertionError("an un-locatable frame must not yield a draft")


def test_aspect_alone_no_longer_rejects_a_screenshot():
    """The 16:9 +-2% gate is gone: a 2.17:1 phone frame is a first-class input."""
    body = png(2868, 1320)
    data = runtime().recognize_bytes(body, context(body, rect=(260, 0, 2347, 1320)))
    assert data["schemaVersion"] == 1


def test_board_cropped_out_of_frame_is_named_as_such():
    body = png(1280, 720)
    try:  # board sticks 200px off the right edge
        runtime().recognize_bytes(body, context(body, rect=(200, 0, 1280, 720)))
    except RuntimeFailure as exc:
        assert exc.code == "BOARD_CLIPPED"
        assert "right" in str(exc)
    else:
        raise AssertionError("a cropped board must be rejected, and say so")


def test_board_rendered_too_small_is_named_as_such():
    body = png(1280, 720)
    try:
        runtime().recognize_bytes(body, context(body, rect=(0, 0, 900, 506)))
    except RuntimeFailure as exc:
        assert exc.code == "BOARD_TOO_SMALL"
    else:
        raise AssertionError("a board below the readable floor must be rejected")


def test_thumbnail_is_rejected_before_inference():
    body = png(320, 180)
    try:
        runtime().recognize_bytes(body, context(body, rect=None))
    except RuntimeFailure as exc:
        assert exc.code == "IMAGE_TOO_SMALL"
    else:
        raise AssertionError("a thumbnail must be rejected on the cheap path")


def test_reconstruct_failure_has_frozen_null_fields():
    body = png()
    draft = runtime().recognize_bytes(body, context(body))["draft"]
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


def _board_dets():
    """A coherent full board in CANONICAL 1920x1080 px (forward fixture geometry
    from test_assemble): hero H13 + drawn 5p, rivers [[9p], [], [S], [W]],
    seat-2 pon of P claimed from relative seat 3."""
    return (_hand_dets(H13, drawn="5p")
            + _dora_dets(["5s"])
            + _river_dets(0, ["9p"])
            + _river_dets(2, ["S"])
            + _river_dets(3, ["W"])
            + _meld_dets(2, [_gt("pon", ["P", "P", "P"], called="P",
                                 from_seat_rel=3, seat=2)]))


def _shift(dets, k, ox, oy):
    """Re-render the same board under img = k * canon + (ox, oy).

    This is exactly what a different device does to the picture — nothing about
    the BOARD changes, only where it lands in the frame.
    """
    moved = []
    for det in dets:
        x0, y0, x1, y1 = det.xyxy
        poly = (tuple((x * k + ox, y * k + oy) for x, y in det.poly)
                if det.poly else None)
        moved.append(dataclasses.replace(
            det, xyxy=(x0 * k + ox, y0 * k + oy, x1 * k + ox, y1 * k + oy),
            poly=poly))
    return moved


def _board_runtime(dets=None):
    payload = _board_dets() if dets is None else dets

    class BoardDetector:
        def predict(self, image): return payload

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


def _semantic(draft):
    """The draft minus everything that legitimately differs between two frames
    of the same board (image dims, evidence pixel boxes)."""
    return {key: value for key, value in draft.items()
            if key not in ("source", "evidence")}


def test_the_same_board_reads_identically_from_a_phone_and_from_a_desktop():
    """THE regression guard.

    A 2.17:1 phone screenshot is the same board, drawn smaller and pushed right.
    If the runtime ever again assumes "the frame IS the board" (as
    locate_fullscreen did, which is how wide support was silently lost), the tiles
    land in the wrong zones and this draft stops matching. Note the failure mode
    it guards is SILENT: a mis-located board yields a full, plausible, wrong
    board — not an error.
    """
    desktop_body = png(1920, 1080)
    desktop = _board_runtime().recognize_bytes(desktop_body, context(
        desktop_body, rect=None))

    k = 1320 / 1080                                   # centered 16:9 on a 2868-wide screen
    phone_body = png(2868, 1320)
    phone_dets = _shift(_board_dets(), k, (2868 - k * 1920) / 2, 0)
    phone = _board_runtime(phone_dets).recognize_bytes(phone_body, context(
        phone_body, rect=None))

    assert phone["issues"] == [] and desktop["issues"] == []
    assert _semantic(phone["draft"]) == _semantic(desktop["draft"])


def test_a_board_under_browser_chrome_reads_identically_too():
    """Windowed capture: the table is inset and smaller, nothing else changes."""
    desktop_body = png(1920, 1080)
    desktop = _board_runtime().recognize_bytes(desktop_body, context(
        desktop_body, rect=None))

    k = 0.75
    window_body = png(1700, 1000)
    window_dets = _shift(_board_dets(), k, 110, 140)
    window = _board_runtime(window_dets).recognize_bytes(window_body, context(
        window_body, rect=None))

    assert window["issues"] == []
    assert _semantic(window["draft"]) == _semantic(desktop["draft"])


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
