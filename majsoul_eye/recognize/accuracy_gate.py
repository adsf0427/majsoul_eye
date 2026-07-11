from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from typing import TypedDict

from majsoul_eye.recognize.manifest import LoadedModelManifest, ManifestError

MIN_SCREENSHOTS = 100
MIN_GAMES = 20
MIN_STRUCTURAL_ENTRY_RATE = 0.95
MAX_MEDIAN_MODIFIED_FIELDS = 0
MAX_P90_MODIFIED_FIELDS = 2
NEAR_DUPLICATE_HAMMING_MAX = 4
COMPARISON_VERSION = "what-cut-semantic-v1"


class AccuracyReportV1(TypedDict):
    schemaVersion: int
    datasetVersion: str
    comparisonVersion: str
    manifestSha256: str
    thresholds: dict[str, int | float]
    rawScreenshots: int
    effectiveScreenshots: int
    distinctGames: int
    nearDuplicatePairs: int
    structuralEntryRate: float
    medianModifiedFields: float
    p90ModifiedFields: int
    nearDuplicateSamplePairs: list[list[str]]
    passed: bool
    failures: list[str]


def _nearest_rank(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def evaluate_gate_metrics(*, manifest_sha256: str, dataset_version: str,
                          modified_fields: list[int], game_ids: list[str],
                          structurally_entered: list[bool],
                          near_duplicate_pairs: list[tuple[str, str]]) -> AccuracyReportV1:
    if not (len(modified_fields) == len(game_ids) == len(structurally_entered)):
        raise ValueError("golden metric vectors must have equal length")
    count = len(modified_fields)
    games = len(set(game_ids))
    entry_rate = sum(structurally_entered) / count if count else 0.0
    median = float(statistics.median(modified_fields)) if count else 0.0
    p90 = _nearest_rank(modified_fields, 0.9) if count else 0
    failures = []
    if count < MIN_SCREENSHOTS: failures.append(f"effectiveScreenshots {count} < {MIN_SCREENSHOTS}")
    if games < MIN_GAMES: failures.append(f"distinctGames {games} < {MIN_GAMES}")
    if near_duplicate_pairs: failures.append(f"nearDuplicatePairs {len(near_duplicate_pairs)} > 0")
    if entry_rate < MIN_STRUCTURAL_ENTRY_RATE:
        failures.append(f"structuralEntryRate {entry_rate:.6f} < {MIN_STRUCTURAL_ENTRY_RATE:.6f}")
    if median > MAX_MEDIAN_MODIFIED_FIELDS:
        failures.append(f"medianModifiedFields {median} > {MAX_MEDIAN_MODIFIED_FIELDS}")
    if p90 > MAX_P90_MODIFIED_FIELDS:
        failures.append(f"p90ModifiedFields {p90} > {MAX_P90_MODIFIED_FIELDS}")
    return {"schemaVersion": 1, "datasetVersion": dataset_version,
            "comparisonVersion": COMPARISON_VERSION,
            "thresholds": {"minScreenshots": MIN_SCREENSHOTS,
                           "minGames": MIN_GAMES,
                           "minStructuralEntryRate": MIN_STRUCTURAL_ENTRY_RATE,
                           "maxMedianModifiedFields": MAX_MEDIAN_MODIFIED_FIELDS,
                           "maxP90ModifiedFields": MAX_P90_MODIFIED_FIELDS,
                           "nearDuplicateHammingMax": NEAR_DUPLICATE_HAMMING_MAX},
            "manifestSha256": manifest_sha256, "effectiveScreenshots": count,
            "rawScreenshots": count,
            "distinctGames": games, "nearDuplicatePairs": len(near_duplicate_pairs),
            "structuralEntryRate": entry_rate, "medianModifiedFields": median,
            "p90ModifiedFields": p90,
            "nearDuplicateSamplePairs": [list(pair) for pair in near_duplicate_pairs],
            "passed": not failures, "failures": failures}


def load_accuracy_report(manifest: LoadedModelManifest) -> AccuracyReportV1:
    gate = manifest.raw["goldenGate"]
    root = os.path.dirname(manifest.path)
    report_path = os.path.abspath(os.path.join(root, gate["reportPath"]))
    checksum_path = os.path.abspath(os.path.join(root, gate["reportChecksumPath"]))
    if (os.path.commonpath([root, report_path]) != root
            or os.path.commonpath([root, checksum_path]) != root):
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "accuracy report path escapes manifest")
    if not os.path.isfile(report_path) or not os.path.isfile(checksum_path):
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "supported layout lacks report/checksum")
    report_bytes = open(report_path, "rb").read()
    expected = open(checksum_path, encoding="ascii").read().strip()
    if (len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected)
            or hashlib.sha256(report_bytes).hexdigest() != expected):
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "accuracy report checksum mismatch")
    return json.loads(report_bytes)


def _validate_report(report: dict) -> None:
    expected = {"schemaVersion", "datasetVersion", "comparisonVersion",
        "manifestSha256", "thresholds",
        "rawScreenshots", "effectiveScreenshots", "distinctGames",
        "nearDuplicatePairs", "structuralEntryRate", "medianModifiedFields",
        "p90ModifiedFields", "nearDuplicateSamplePairs", "passed", "failures"}
    numeric_counts = ("rawScreenshots", "effectiveScreenshots", "distinctGames",
                      "nearDuplicatePairs", "p90ModifiedFields")
    if (not isinstance(report, dict) or set(report) != expected
            or report.get("schemaVersion") != 1
            or not isinstance(report.get("datasetVersion"), str)
            or not report["datasetVersion"]
            or report.get("comparisonVersion") != COMPARISON_VERSION
            or report.get("thresholds") != {
                "minScreenshots": MIN_SCREENSHOTS, "minGames": MIN_GAMES,
                "minStructuralEntryRate": MIN_STRUCTURAL_ENTRY_RATE,
                "maxMedianModifiedFields": MAX_MEDIAN_MODIFIED_FIELDS,
                "maxP90ModifiedFields": MAX_P90_MODIFIED_FIELDS,
                "nearDuplicateHammingMax": NEAR_DUPLICATE_HAMMING_MAX}
            or not isinstance(report.get("manifestSha256"), str)
            or len(report["manifestSha256"]) != 64
            or any(ch not in "0123456789abcdef" for ch in report["manifestSha256"])
            or any(type(report.get(key)) is not int or report[key] < 0
                   for key in numeric_counts)
            or not isinstance(report.get("medianModifiedFields"), (int, float))
            or not math.isfinite(report["medianModifiedFields"])
            or not isinstance(report.get("structuralEntryRate"), (int, float))
            or not math.isfinite(report["structuralEntryRate"])
            or not 0 <= report["structuralEntryRate"] <= 1
            or type(report.get("passed")) is not bool
            or not isinstance(report.get("failures"), list)
            or any(not isinstance(value, str) for value in report["failures"])
            or report["passed"] != (len(report["failures"]) == 0)
            or report["rawScreenshots"] < report["effectiveScreenshots"]
            or not isinstance(report.get("nearDuplicateSamplePairs"), list)
            or len(report["nearDuplicateSamplePairs"]) != report["nearDuplicatePairs"]
            or any(not isinstance(pair, list) or len(pair) != 2
                   or any(not isinstance(sample_id, str) or not sample_id
                          for sample_id in pair)
                   for pair in report["nearDuplicateSamplePairs"])):
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "invalid accuracy report shape")


def verify_layout_support(manifest: LoadedModelManifest,
                          report: AccuracyReportV1 | None = None) -> None:
    if manifest.support_status == "experimental":
        return
    report = report or load_accuracy_report(manifest)
    _validate_report(report)
    if (not report["passed"] or report["effectiveScreenshots"] < MIN_SCREENSHOTS
            or report["distinctGames"] < MIN_GAMES
            or report["nearDuplicatePairs"] != 0
            or report["structuralEntryRate"] < MIN_STRUCTURAL_ENTRY_RATE
            or report["medianModifiedFields"] > MAX_MEDIAN_MODIFIED_FIELDS
            or report["p90ModifiedFields"] > MAX_P90_MODIFIED_FIELDS):
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "supported layout lacks passing report")
    if report["manifestSha256"] != manifest.manifest_sha256:
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "accuracy report was produced by another manifest")
    if report["datasetVersion"] != manifest.raw["goldenGate"]["datasetVersion"]:
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "accuracy dataset version mismatch")
    if report["comparisonVersion"] != manifest.raw["goldenGate"]["comparisonVersion"]:
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "accuracy comparison version mismatch")
