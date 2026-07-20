from majsoul_eye.recognize.accuracy_gate import (
    build_accuracy_report, evaluate_gate_metrics, verify_layout_support,
)
from majsoul_eye.recognize.manifest import ManifestError


def fake_manifest(modes=("4p", "3p"), sha="a" * 64, dataset="gold-v1"):
    return type("M", (), {
        "support_status": "supported", "manifest_sha256": sha,
        "modes": tuple(modes),
        "raw": {"goldenGate": {"datasetVersion": dataset,
                                 "comparisonVersion": "what-cut-semantic-v2"}},
    })()


def passing_section():
    return evaluate_gate_metrics(
        modified_fields=[0] * 89 + [2] * 11,
        game_ids=[f"g{i // 5}" for i in range(100)],
        structurally_entered=[True] * 95 + [False] * 5,
        near_duplicate_pairs=[])


def test_gate_rejects_small_or_non_independent_set():
    report = evaluate_gate_metrics(
        modified_fields=[0] * 50, game_ids=[f"g{i // 5}" for i in range(50)],
        structurally_entered=[True] * 50, near_duplicate_pairs=[])
    assert report["passed"] is False
    assert "effectiveScreenshots 50 < 100" in report["failures"]
    assert "distinctGames 10 < 20" in report["failures"]


def test_gate_enforces_all_user_cost_thresholds():
    edits = [0] * 89 + [3] * 11
    report = evaluate_gate_metrics(
        modified_fields=edits, game_ids=[f"g{i // 5}" for i in range(100)],
        structurally_entered=[True] * 94 + [False] * 6,
        near_duplicate_pairs=[])
    assert report["passed"] is False
    assert "structuralEntryRate 0.940000 < 0.950000" in report["failures"]
    assert "p90ModifiedFields 3 > 2" in report["failures"]


def test_gate_passes_only_complete_de_duplicated_target():
    report = passing_section()
    assert report["passed"] is True
    assert report["medianModifiedFields"] == 0
    assert report["p90ModifiedFields"] == 2


def test_conventional_even_median_does_not_round_down():
    report = evaluate_gate_metrics(
        modified_fields=[0] * 50 + [1] * 50,
        game_ids=[f"g{i // 5}" for i in range(100)],
        structurally_entered=[True] * 100, near_duplicate_pairs=[])
    assert report["medianModifiedFields"] == 0.5
    assert report["passed"] is False


def test_supported_report_rejects_non_finite_metric():
    report = build_accuracy_report(
        manifest_sha256="a" * 64, dataset_version="gold-v1",
        modes={"4p": passing_section(), "3p": passing_section()})
    report["modes"]["3p"]["structuralEntryRate"] = float("nan")
    try:
        verify_layout_support(fake_manifest(), report)
    except ManifestError as exc:
        assert exc.code == "MODEL_MANIFEST_MISMATCH"
    else:
        raise AssertionError("NaN report metric must never promote a layout")


def test_supported_requires_every_declared_mode_to_pass():
    # A passing 4p-only report must not promote a manifest that also serves 3p.
    report = build_accuracy_report(
        manifest_sha256="a" * 64, dataset_version="gold-v1",
        modes={"4p": passing_section()})
    try:
        verify_layout_support(fake_manifest(("4p", "3p")), report)
    except ManifestError as exc:
        assert "modes do not match" in str(exc)
    else:
        raise AssertionError("missing 3p section must never promote a 3p manifest")
    # And a failing 3p section blocks even when 4p is perfect.
    failing = evaluate_gate_metrics(
        modified_fields=[0] * 30, game_ids=[f"g{i // 5}" for i in range(30)],
        structurally_entered=[True] * 30, near_duplicate_pairs=[])
    report = build_accuracy_report(
        manifest_sha256="a" * 64, dataset_version="gold-v1",
        modes={"4p": passing_section(), "3p": failing})
    assert report["passed"] is False
    assert any(failure.startswith("mode 3p:") for failure in report["failures"])
    try:
        verify_layout_support(fake_manifest(("4p", "3p")), report)
    except ManifestError as exc:
        assert exc.code == "MODEL_MANIFEST_MISMATCH"
    else:
        raise AssertionError("failing 3p section must never promote the manifest")


def test_supported_accepts_fully_passing_two_mode_report():
    report = build_accuracy_report(
        manifest_sha256="a" * 64, dataset_version="gold-v1",
        modes={"4p": passing_section(), "3p": passing_section()})
    assert report["passed"] is True
    verify_layout_support(fake_manifest(), report)


def test_semantic_diff_excludes_ids_evidence_baseline_and_provenance():
    from scripts.eval.eval_what_cut_goldens import semantic_fields
    from test_what_cut_schema import minimal_draft
    a = minimal_draft(); b = minimal_draft()
    b["draftId"] = "other"; b["revision"] = 99
    b["evidence"] = [{"id": "e", "bbox": [0.0, 0.0, 1.0, 1.0],
                      "polygon": None, "zone": "hand"}]
    mark = b["players"][0]["rivers"][0]["tsumogiri"]
    mark.update({"source": "inferred", "baselineValue": False,
                 "baselineSource": "forced"})
    assert semantic_fields(a) == semantic_fields(b)
    b["players"][0]["rivers"][0]["pai"] = "8p"
    assert len(set(semantic_fields(a)) ^ set(semantic_fields(b))) == 2


def test_hamming_threshold_is_exactly_four():
    from scripts.eval.eval_what_cut_goldens import hamming64
    assert hamming64(0, 0b1111) == 4
    assert hamming64(0, 0b11111) == 5


def test_clean_recognition_with_reconstruction_conflict_is_not_structural_entry():
    from copy import deepcopy

    from scripts.eval.eval_what_cut_goldens import score_runtime_sample
    from test_what_cut_schema import minimal_draft

    expected = minimal_draft()
    recognized_draft = deepcopy(expected)
    recognized_draft["players"][0]["rivers"][0]["pai"] = "8p"

    class ConflictingRuntime:
        reconstructed = None

        def recognize_bytes(self, image_bytes, context):
            assert image_bytes == b"image"
            return {"draft": recognized_draft, "issues": []}

        def reconstruct_draft(self, draft, revision):
            self.reconstructed = (draft, revision)
            return {
                "ok": False,
                "issues": [{"code": "RECONSTRUCTION_CONFLICT",
                            "severity": "blocking"}],
            }

    runtime = ConflictingRuntime()
    entered, modified = score_runtime_sample(
        runtime, b"image", object(), expected)
    assert entered is False
    assert modified == 1  # Diff remains recognition-vs-GT, not reconstruction output.
    assert runtime.reconstructed == (recognized_draft, recognized_draft["revision"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_what_cut_accuracy_gate OK")
