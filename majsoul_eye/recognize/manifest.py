from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass


class ManifestError(RuntimeError):
    def __init__(self, code: str, message: str, asset: str | None = None):
        super().__init__(message)
        self.code, self.asset = code, asset


@dataclass(frozen=True)
class ModelAsset:
    name: str
    path: str
    sha256: str
    required: bool


@dataclass(frozen=True)
class LoadedModelManifest:
    path: str
    raw: dict
    manifest_sha256: str
    assets: dict[str, ModelAsset]

    @property
    def manifest_version(self): return self.raw["manifestVersion"]
    @property
    def layout_id(self): return self.raw["layout"]["layoutId"]
    @property
    def support_status(self): return self.raw["supportStatus"]


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model_manifest(path: str) -> LoadedModelManifest:
    absolute = os.path.abspath(path)
    raw_bytes = open(absolute, "rb").read()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict) or set(raw) != {"schemaVersion", "manifestVersion",
            "layout", "models", "inference", "candidates", "supportStatus",
            "goldenGate"}:
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "manifest keys drift")
    if raw.get("schemaVersion") != 1:
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "manifest schemaVersion must be 1")
    if not isinstance(raw.get("manifestVersion"), str) or not raw["manifestVersion"]:
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "manifestVersion must be non-empty")
    if raw.get("supportStatus") not in ("experimental", "supported"):
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "invalid supportStatus")
    layout = raw.get("layout") or {}
    # The board is found by landmark fit, not assumed from the frame's aspect, so
    # the contract pins the FIT's quality bars and the BOARD's minimum size — the
    # frame floor is only a cheap "this is not a thumbnail" guard before inference.
    expected_layout = {"layoutId": "majsoul-desktop-16x9-v1",
                       "minFrameWidth": 640, "minFrameHeight": 360,
                       "minBoardWidth": 1280, "minBoardHeight": 720,
                       "anchorToleranceCanon": 30.0, "maxResidualCanon": 8.0,
                       "minHandInliers": 4, "clipToleranceFrac": 0.005}
    if any(layout.get(key) != value for key, value in expected_layout.items()):
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "desktop layout contract drift")
    if set(layout) != set(expected_layout):
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "layout keys drift")
    inference = raw.get("inference")
    if (not isinstance(inference, dict) or set(inference) != {"detectorConf", "imgsz"}
            or not isinstance(inference["detectorConf"], (int, float))
            or not 0 < inference["detectorConf"] <= 1
            or type(inference["imgsz"]) is not int or inference["imgsz"] < 640):
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "invalid fixed inference settings")
    candidates = raw.get("candidates")
    if (not isinstance(candidates, dict)
            or set(candidates) != {"topK", "calibrationVersion", "autoReplace"}
            or type(candidates["topK"]) is not int or candidates["topK"] < 0
            or candidates["autoReplace"] is not False
            or (candidates["calibrationVersion"] is not None
                and not isinstance(candidates["calibrationVersion"], str))):
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "invalid candidate policy")
    gate = raw.get("goldenGate")
    if (not isinstance(gate, dict)
            or set(gate) != {"datasetVersion", "comparisonVersion",
                             "reportPath", "reportChecksumPath"}
            or not all(isinstance(gate[key], str) and gate[key] for key in gate)
            or gate["comparisonVersion"] != "what-cut-semantic-v1"):
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "invalid golden gate paths")
    root = os.path.dirname(absolute)
    for key in ("reportPath", "reportChecksumPath"):
        resolved_report = os.path.abspath(os.path.join(root, gate[key]))
        if os.path.commonpath([root, resolved_report]) != root:
            raise ManifestError("MODEL_MANIFEST_MISMATCH",
                                f"{key} escapes manifest directory")
    models = raw.get("models")
    if not isinstance(models, dict) or set(models) != {"detector", "classifier", "hudReader"}:
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "model asset keys drift")
    assets = {}
    for name in ("detector", "classifier", "hudReader"):
        spec = models[name]
        if (not isinstance(spec, dict) or set(spec) != {"path", "sha256", "required"}
                or not isinstance(spec["path"], str) or not spec["path"]
                or spec["required"] is not True
                or not isinstance(spec["sha256"], str) or len(spec["sha256"]) != 64
                or any(ch not in "0123456789abcdef" for ch in spec["sha256"])):
            raise ManifestError("MODEL_MANIFEST_MISMATCH", "invalid model asset", name)
        resolved = os.path.abspath(os.path.join(root, spec["path"]))
        try:
            confined = os.path.commonpath([root, resolved]) == root
        except ValueError:
            confined = False
        if not confined:
            raise ManifestError("MODEL_MANIFEST_MISMATCH", "asset escapes manifest directory", name)
        assets[name] = ModelAsset(name, resolved, spec["sha256"], bool(spec["required"]))
    return LoadedModelManifest(absolute, raw, hashlib.sha256(raw_bytes).hexdigest(), assets)


def verify_model_assets(manifest: LoadedModelManifest) -> None:
    for name, asset in manifest.assets.items():
        if not os.path.isfile(asset.path):
            if asset.required:
                raise ManifestError("MODEL_MANIFEST_MISMATCH", "required model missing", name)
            continue
        if _sha256(asset.path) != asset.sha256:
            raise ManifestError("MODEL_MANIFEST_MISMATCH", "model SHA-256 mismatch", name)
