from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

from majsoul_eye.annotate import pipeline as P
from majsoul_eye.normalize import BoardRegion, clipped_sides, locate_anchor
from majsoul_eye.recognize.accuracy_gate import verify_layout_support
from majsoul_eye.recognize.assemble import assemble_with_evidence
from majsoul_eye.recognize.classifier import TileClassifier
from majsoul_eye.recognize.detector import TileDetector
from majsoul_eye.recognize.evidence import CandidatePolicy
from majsoul_eye.recognize.hudreader import HudReader
from majsoul_eye.recognize.manifest import load_model_manifest, verify_model_assets
from majsoul_eye.recognize.mode import detect_mode
from majsoul_eye.state.decision import analyze_hero_decision
from majsoul_eye.state.reconstruct import reconstruct
from majsoul_eye.what_cut.adapter import draft_to_observed
from majsoul_eye.what_cut.from_recognition import (
    DraftBuildContext, apply_history_baseline, build_recognized_draft,
)
from majsoul_eye.what_cut.schema import (
    FabricatedHistoryV1, RecognizeWhatCutData, ReconstructWhatCutData,
    WhatCutDraftV1, WhatCutIssueV1, WhatCutRecognizerV1,
    parse_what_cut_draft,
)


@dataclass(frozen=True)
class RecognitionContext:
    request_id: str
    draft_id: str
    image_sha256: str
    layout_id: str
    allow_experimental: bool
    image_ref: str | None
    # User-supplied board rect (ox, oy, bw, bh) in source-image px, from the
    # manual-calibration fallback. Overrides the landmark fit when present.
    board_rect: tuple[int, int, int, int] | None = None
    # User-supplied player count: "auto" | "3p" | "4p". A forced mode picks the
    # GEOMETRY; it never buys truth — every conservation check still runs, so a
    # wrong override still blocks rather than producing a board.
    board_mode: str = "auto"


class RuntimeFailure(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fabricated(result, baseline, observed) -> FabricatedHistoryV1:
    assert result.selected_history is not None
    operations = result.selected_history["operations"]
    hero_draws = sum(1 for op in operations if op["kind"] == "draw"
                     and op["actorRelSeat"] == 0)
    if observed.drawn_tile is not None:
        hero_draws = max(0, hero_draws - 1)
    return {"defaultedRoundFields": list(result.fabricated.get("defaults", [])),
            "heroHiddenDrawCount": hero_draws,
            "opponentUnknownDrawCount": sum(1 for op in operations
                if op["kind"] == "draw" and op["actorRelSeat"] != 0 and op["pai"] == "?"),
            "inferredRiverCount": sum(1 for item in baseline
                if item["itemKind"] == "river" and item["baselineSource"] == "inferred"),
            "inferredGhostCount": sum(1 for item in baseline
                if item["itemKind"] == "ghost" and item["baselineSource"] == "inferred")}


def _failed_reconstruct(revision: int, issues: list[WhatCutIssueV1]) -> ReconstructWhatCutData:
    return {"schemaVersion": 1, "revision": revision, "ok": False,
            "issues": issues, "mjai": None, "heroSeatAbs": None,
            "fabricated": None, "historyBaseline": [],
            "selectedHistory": None, "decision": None}


class RecognitionRuntime:
    @classmethod
    def from_manifest(cls, manifest_path: str, *, device: str, eye_revision: str,
                      evaluation_mode: bool = False,
                      detector_factory=TileDetector,
                      classifier_factory=TileClassifier,
                      hud_reader_factory=HudReader):
        manifest = load_model_manifest(manifest_path)
        verify_model_assets(manifest)
        if not evaluation_mode:
            verify_layout_support(manifest)
        self = cls.__new__(cls)
        self.manifest, self.eye_revision = manifest, eye_revision
        inference = manifest.raw["inference"]
        self.detector = detector_factory(manifest.assets["detector"].path,
                                         device=device,
                                         conf=inference["detectorConf"],
                                         imgsz=inference["imgsz"])
        self.classifier = classifier_factory(manifest.assets["classifier"].path,
                                             device=device)
        self.hud_reader = hud_reader_factory(manifest.assets["hudReader"].path,
                                             device=device)
        return self

    def warmup(self) -> None:
        frame = np.zeros((720, 1280, 3), np.uint8)
        tile = np.zeros((64, 64, 3), np.uint8)
        self.detector.predict(frame)
        self.classifier.predict_proba([tile])
        self.hud_reader.read(np.zeros((32, 128, 3), np.uint8), "score_self")
        self.hud_reader.read(tile, "round_label")
        self.hud_reader.read(tile, "seat_wind_self")

    def metadata(self) -> WhatCutRecognizerV1:
        assets = self.manifest.assets
        return {"manifestVersion": self.manifest.manifest_version,
                "layoutId": self.manifest.layout_id,
                "detectorSha": assets["detector"].sha256,
                "classifierSha": assets["classifier"].sha256,
                "hudReaderSha": assets["hudReader"].sha256,
                "eyeRevision": self.eye_revision,
                "supportStatus": self.manifest.support_status}

    def _decode_and_validate(self, image_bytes: bytes, context: RecognitionContext):
        if context.layout_id != self.manifest.layout_id:
            raise RuntimeFailure("UNSUPPORTED_LAYOUT", "layoutId is not loaded")
        if self.manifest.support_status == "experimental" and not context.allow_experimental:
            raise RuntimeFailure("UNSUPPORTED_LAYOUT", "layout is experimental")
        digest = hashlib.sha256(image_bytes).hexdigest()
        if (len(context.image_sha256) != 64
                or any(ch not in "0123456789abcdef" for ch in context.image_sha256)
                or digest != context.image_sha256):
            raise RuntimeFailure("INVALID_IMAGE_DIGEST",
                                 "X-Image-SHA256 does not match original bytes")
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeFailure("INVALID_IMAGE", "cannot decode image")
        height, width = image.shape[:2]
        layout = self.manifest.raw["layout"]
        if width < layout["minFrameWidth"] or height < layout["minFrameHeight"]:
            raise RuntimeFailure("IMAGE_TOO_SMALL", "image is below minimum dimensions")
        return image

    def _locate(self, image, dets, context: RecognitionContext):
        """Find the board, then judge whether enough of it is actually in frame.

        The aspect of the SCREENSHOT says nothing — a phone letterboxes the 16:9
        table between HUD columns, a window insets it under chrome. What matters
        is (a) can we pin the board, and (b) is all of it present.
        """
        layout = self.manifest.raw["layout"]
        if context.board_rect is not None:              # user-calibrated: trust it
            ox, oy, bw, bh = context.board_rect
            region = BoardRegion(ox, oy, bw, bh, image.shape[1], image.shape[0])
        else:
            found = locate_anchor(image, dets,
                                  tol=layout["anchorToleranceCanon"])
            if found is None or found.residual > layout["maxResidualCanon"]:
                raise RuntimeFailure(
                    "LOCALIZATION_FAILED",
                    "cannot locate the board from this screenshot")
            if found.hand_inliers < layout["minHandInliers"]:
                # The hand row is what pins the scale (the center panel alone is a
                # 270/1920 baseline). No hand row also means no what-cut question.
                raise RuntimeFailure("HAND_NOT_VISIBLE",
                                     "the hero hand row is not fully visible")
            region = found.region

        if (region.bw < layout["minBoardWidth"]
                or region.bh < layout["minBoardHeight"]):
            raise RuntimeFailure("BOARD_TOO_SMALL",
                                 "the board is rendered too small to read")
        sides = clipped_sides(region, layout["clipToleranceFrac"])
        if sides:
            raise RuntimeFailure("BOARD_CLIPPED",
                                 f"the board is cropped: {', '.join(sides)}")
        return region

    def recognize_bytes(self, image_bytes: bytes,
                        context: RecognitionContext) -> RecognizeWhatCutData:
        image = self._decode_and_validate(image_bytes, context)
        candidates = self.manifest.raw["candidates"]
        policy = CandidatePolicy(candidates["calibrationVersion"], candidates["topK"])
        dets = self.detector.predict(image)
        region = self._locate(image, dets, context)
        mode = detect_mode(dets, region, context.board_mode)
        geom = P.geometry_for(mode.sanma)
        assembly = assemble_with_evidence(
            dets, region,
            frame_bgr=image, hud_reader=self.hud_reader,
            tile_classifier=self.classifier, candidate_policy=policy,
            geom=geom, phantom_rel=mode.phantom_rel)
        draft = build_recognized_draft(
            assembly, DraftBuildContext(context.draft_id, context.image_ref,
                                        context.image_sha256, image.shape[1], image.shape[0]),
            self.metadata())
        issues = list(mode.issues)
        issues += [{"code": "RECOGNITION_STRUCTURE", "severity": "blocking",
                    "fieldPath": None, "evidenceIds": [],
                    "messageKey": "whatCut.issue.RECOGNITION_STRUCTURE",
                    "params": {"message": message}}
                   for message in assembly.issues]
        adapted = draft_to_observed(draft)
        issues.extend(adapted.issues)
        if adapted.observed is not None:
            rebuilt = reconstruct(adapted.observed, adapted.overrides)
            if rebuilt.ok:
                apply_history_baseline(draft, rebuilt.history_baseline)
            else:
                issues.extend(rebuilt.issues)
        return {"schemaVersion": 1, "draft": draft, "issues": issues,
                "recognizer": self.metadata()}

    def reconstruct_draft(self, draft: WhatCutDraftV1,
                          revision: int) -> ReconstructWhatCutData:
        parsed = parse_what_cut_draft(draft)
        if parsed["revision"] != revision:
            return _failed_reconstruct(revision, [{
                "code": "STALE_REVISION", "severity": "blocking",
                "fieldPath": "revision", "evidenceIds": [],
                "messageKey": "whatCut.issue.STALE_REVISION",
                "params": {"draftRevision": parsed["revision"],
                           "requestRevision": revision}}])
        adapted = draft_to_observed(parsed)
        if adapted.observed is None:
            return _failed_reconstruct(revision, adapted.issues)
        rebuilt = reconstruct(adapted.observed, adapted.overrides)
        if not rebuilt.ok:
            return _failed_reconstruct(revision, rebuilt.issues)
        hero_abs = next(event["id"] for event in rebuilt.events
                        if event["type"] == "start_game")
        decision = analyze_hero_decision(adapted.observed, rebuilt.selected_history)
        issues = list(decision.issues)
        return {"schemaVersion": 1, "revision": revision, "ok": True,
                "issues": issues, "mjai": rebuilt.events,
                "heroSeatAbs": hero_abs,
                "fabricated": _fabricated(rebuilt, rebuilt.history_baseline,
                                           adapted.observed),
                "historyBaseline": rebuilt.history_baseline,
                "selectedHistory": rebuilt.selected_history,
                "decision": decision.decision}
