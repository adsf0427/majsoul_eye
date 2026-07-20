import hashlib
import json
import os
import tempfile

from majsoul_eye.recognize.manifest import (
    ManifestError, load_model_manifest, verify_model_assets,
)


def manifest(asset, digest=None):
    spec = {"path": asset, "required": True}
    if digest is not None:
        spec["sha256"] = digest  # legacy shape: tolerated, ignored
    return {"schemaVersion": 1, "manifestVersion": "test-v1",
            "layout": {"layoutId": "majsoul-desktop-16x9-v1",
                       "minFrameWidth": 640, "minFrameHeight": 360,
                       "minBoardWidth": 1208, "minBoardHeight": 680,
                       "anchorToleranceCanon": 30.0, "maxResidualCanon": 8.0,
                       "maxResidualCanonRelaxed": 16.0,
                       "relaxedResidualMinInliers": 12,
                       "minHandInliers": 4, "clipToleranceFrac": 0.005},
            "models": {name: dict(spec)
                       for name in ("detector", "classifier", "hudReader")},
            "inference": {"detectorConf": 0.25, "imgsz": 1280},
            "candidates": {"topK": 3, "calibrationVersion": None,
                           "autoReplace": False},
            "supportStatus": "experimental",
            "modes": ["4p", "3p"],
            "goldenGate": {"datasetVersion": "majsoul-desktop-16x9-gold-v1",
                           "comparisonVersion": "what-cut-semantic-v2",
                           "reportPath": "model-manifest.internal-v1.accuracy.json",
                           "reportChecksumPath": "model-manifest.internal-v1.accuracy.json.sha256"}}


def test_manifest_resolves_relative_asset_and_verifies_presence():
    with tempfile.TemporaryDirectory() as td:
        asset = os.path.join(td, "detector.pt")
        open(asset, "wb").write(b"model")
        path = os.path.join(td, "manifest.json")
        open(path, "w", encoding="utf-8").write(json.dumps(manifest("detector.pt")))
        loaded = load_model_manifest(path)
        verify_model_assets(loaded)
        assert loaded.assets["detector"].path == asset


def test_missing_required_model_is_stable_manifest_error():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "manifest.json")
        open(path, "w", encoding="utf-8").write(json.dumps(manifest("detector.pt")))
        try:
            verify_model_assets(load_model_manifest(path))
        except ManifestError as exc:
            assert exc.code == "MODEL_MANIFEST_MISMATCH"
            assert exc.asset == "detector"
        else:
            raise AssertionError("missing required model must fail readiness")


def test_legacy_sha_field_is_tolerated_and_ignored():
    # Digest comparison was retired (lite policy): a legacy manifest that still
    # carries sha256 keys loads, and mismatched bytes are NOT an error.
    with tempfile.TemporaryDirectory() as td:
        open(os.path.join(td, "detector.pt"), "wb").write(b"wrong")
        path = os.path.join(td, "manifest.json")
        open(path, "w", encoding="utf-8").write(
            json.dumps(manifest("detector.pt", hashlib.sha256(b"model").hexdigest())))
        verify_model_assets(load_model_manifest(path))


def test_manifest_rejects_candidate_autoreplace_and_asset_escape():
    with tempfile.TemporaryDirectory() as td:
        raw = manifest("../outside.pt", "0" * 64)
        raw["candidates"]["autoReplace"] = True
        path = os.path.join(td, "manifest.json")
        open(path, "w", encoding="utf-8").write(json.dumps(raw))
        try:
            load_model_manifest(path)
        except ManifestError as exc:
            assert exc.code == "MODEL_MANIFEST_MISMATCH"
        else:
            raise AssertionError("autoReplace manifest must fail")
        raw["candidates"]["autoReplace"] = False
        open(path, "w", encoding="utf-8").write(json.dumps(raw))
        try:
            load_model_manifest(path)
        except ManifestError as exc:
            assert exc.code == "MODEL_MANIFEST_MISMATCH"
            assert exc.asset == "detector"
        else:
            raise AssertionError("asset path escape must fail")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_model_manifest OK")


def test_manifest_rejects_bad_modes_declarations():
    with tempfile.TemporaryDirectory() as td:
        asset = os.path.join(td, "detector.pt")
        open(asset, "wb").write(b"model")
        digest = hashlib.sha256(b"model").hexdigest()
        for bad in ([], ["3p"], ["4p", "4p"], ["4p", "2p"], "4p", None):
            raw = manifest("detector.pt", digest)
            raw["modes"] = bad
            path = os.path.join(td, "manifest.json")
            open(path, "w", encoding="utf-8").write(json.dumps(raw))
            try:
                load_model_manifest(path)
            except ManifestError as exc:
                assert exc.code == "MODEL_MANIFEST_MISMATCH"
            else:
                raise AssertionError(f"modes {bad!r} must be rejected")
        raw = manifest("detector.pt", digest)
        raw["modes"] = ["4p"]
        path = os.path.join(td, "manifest.json")
        open(path, "w", encoding="utf-8").write(json.dumps(raw))
        assert load_model_manifest(path).modes == ("4p",)
