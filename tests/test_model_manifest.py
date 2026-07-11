import hashlib
import json
import os
import tempfile

from majsoul_eye.recognize.manifest import (
    ManifestError, load_model_manifest, verify_model_assets,
)


def manifest(asset, digest):
    return {"schemaVersion": 1, "manifestVersion": "test-v1",
            "layout": {"layoutId": "majsoul-desktop-16x9-v1",
                       "minWidth": 1280, "minHeight": 720,
                       "aspectRatio": 1.7777777777777777,
                       "aspectTolerance": 0.02},
            "models": {name: {"path": asset, "sha256": digest,
                               "required": True}
                       for name in ("detector", "classifier", "hudReader")},
            "inference": {"detectorConf": 0.25, "imgsz": 1280},
            "candidates": {"topK": 3, "calibrationVersion": None,
                           "autoReplace": False},
            "supportStatus": "experimental",
            "goldenGate": {"datasetVersion": "majsoul-desktop-16x9-gold-v1",
                           "comparisonVersion": "what-cut-semantic-v1",
                           "reportPath": "model-manifest.internal-v1.accuracy.json",
                           "reportChecksumPath": "model-manifest.internal-v1.accuracy.json.sha256"}}


def test_manifest_resolves_relative_asset_and_verifies_sha():
    with tempfile.TemporaryDirectory() as td:
        asset = os.path.join(td, "detector.pt")
        open(asset, "wb").write(b"model")
        digest = hashlib.sha256(b"model").hexdigest()
        path = os.path.join(td, "manifest.json")
        open(path, "w", encoding="utf-8").write(json.dumps(manifest("detector.pt", digest)))
        loaded = load_model_manifest(path)
        verify_model_assets(loaded)
        assert loaded.assets["detector"].path == asset


def test_mismatched_sha_is_stable_manifest_error():
    with tempfile.TemporaryDirectory() as td:
        open(os.path.join(td, "detector.pt"), "wb").write(b"wrong")
        path = os.path.join(td, "manifest.json")
        open(path, "w", encoding="utf-8").write(json.dumps(manifest("detector.pt", "0" * 64)))
        try:
            verify_model_assets(load_model_manifest(path))
        except ManifestError as exc:
            assert exc.code == "MODEL_MANIFEST_MISMATCH"
            assert exc.asset == "detector"
        else:
            raise AssertionError("bad model SHA must fail readiness")


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
