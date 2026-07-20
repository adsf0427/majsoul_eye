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
# v2: the semantic diff gained the sanma mode fields (nPlayers, phantomRelSeat,
# nukiCount) — a mode misread now counts as edits instead of being invisible.
COMPARISON_VERSION = "what-cut-semantic-v2"
# Board modes a manifest may declare. The layout is mode-agnostic (localization
# is shared); the GATE is not — accuracy must be proven per mode.
KNOWN_MODES = ("4p", "3p")


class AccuracyModeReport(TypedDict):
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


class AccuracyReportV2(TypedDict):
    """One report per manifest, one section per DECLARED mode.

    A manifest is `supported` only when every mode it declares passes its own
    section — a 4p-only golden run must never promote a manifest that also
    serves sanma."""
    schemaVersion: int
    datasetVersion: str
    comparisonVersion: str
    manifestSha256: str
    modes: dict[str, AccuracyModeReport]
    passed: bool
    failures: list[str]


def _nearest_rank(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def evaluate_gate_metrics(*, modified_fields: list[int], game_ids: list[str],
                          structurally_entered: list[bool],
                          near_duplicate_pairs: list[tuple[str, str]]) -> AccuracyModeReport:
    """One MODE's metrics -> its report section. Thresholds are identical per
    mode: sanma buys no discount."""
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
    return {"thresholds": {"minScreenshots": MIN_SCREENSHOTS,
                           "minGames": MIN_GAMES,
                           "minStructuralEntryRate": MIN_STRUCTURAL_ENTRY_RATE,
                           "maxMedianModifiedFields": MAX_MEDIAN_MODIFIED_FIELDS,
                           "maxP90ModifiedFields": MAX_P90_MODIFIED_FIELDS,
                           "nearDuplicateHammingMax": NEAR_DUPLICATE_HAMMING_MAX},
            "effectiveScreenshots": count,
            "rawScreenshots": count,
            "distinctGames": games, "nearDuplicatePairs": len(near_duplicate_pairs),
            "structuralEntryRate": entry_rate, "medianModifiedFields": median,
            "p90ModifiedFields": p90,
            "nearDuplicateSamplePairs": [list(pair) for pair in near_duplicate_pairs],
            "passed": not failures, "failures": failures}


def build_accuracy_report(*, manifest_sha256: str, dataset_version: str,
                          modes: dict[str, AccuracyModeReport]) -> AccuracyReportV2:
    if not modes or any(mode not in KNOWN_MODES for mode in modes):
        raise ValueError(f"report modes must be a non-empty subset of {KNOWN_MODES}")
    failures = [f"mode {mode}: {failure}"
                for mode in sorted(modes) for failure in modes[mode]["failures"]]
    return {"schemaVersion": 2, "datasetVersion": dataset_version,
            "comparisonVersion": COMPARISON_VERSION,
            "manifestSha256": manifest_sha256,
            "modes": {mode: modes[mode] for mode in sorted(modes)},
            "passed": not failures, "failures": failures}


def load_accuracy_report(manifest: LoadedModelManifest) -> AccuracyReportV2:
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


def _validate_mode_report(section: dict) -> None:
    expected = {"thresholds",
        "rawScreenshots", "effectiveScreenshots", "distinctGames",
        "nearDuplicatePairs", "structuralEntryRate", "medianModifiedFields",
        "p90ModifiedFields", "nearDuplicateSamplePairs", "passed", "failures"}
    numeric_counts = ("rawScreenshots", "effectiveScreenshots", "distinctGames",
                      "nearDuplicatePairs", "p90ModifiedFields")
    if (not isinstance(section, dict) or set(section) != expected
            or section.get("thresholds") != {
                "minScreenshots": MIN_SCREENSHOTS, "minGames": MIN_GAMES,
                "minStructuralEntryRate": MIN_STRUCTURAL_ENTRY_RATE,
                "maxMedianModifiedFields": MAX_MEDIAN_MODIFIED_FIELDS,
                "maxP90ModifiedFields": MAX_P90_MODIFIED_FIELDS,
                "nearDuplicateHammingMax": NEAR_DUPLICATE_HAMMING_MAX}
            or any(type(section.get(key)) is not int or section[key] < 0
                   for key in numeric_counts)
            or not isinstance(section.get("medianModifiedFields"), (int, float))
            or not math.isfinite(section["medianModifiedFields"])
            or not isinstance(section.get("structuralEntryRate"), (int, float))
            or not math.isfinite(section["structuralEntryRate"])
            or not 0 <= section["structuralEntryRate"] <= 1
            or type(section.get("passed")) is not bool
            or not isinstance(section.get("failures"), list)
            or any(not isinstance(value, str) for value in section["failures"])
            or section["passed"] != (len(section["failures"]) == 0)
            or section["rawScreenshots"] < section["effectiveScreenshots"]
            or not isinstance(section.get("nearDuplicateSamplePairs"), list)
            or len(section["nearDuplicateSamplePairs"]) != section["nearDuplicatePairs"]
            or any(not isinstance(pair, list) or len(pair) != 2
                   or any(not isinstance(sample_id, str) or not sample_id
                          for sample_id in pair)
                   for pair in section["nearDuplicateSamplePairs"])):
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "invalid accuracy report shape")


def _validate_report(report: dict) -> None:
    expected = {"schemaVersion", "datasetVersion", "comparisonVersion",
                "manifestSha256", "modes", "passed", "failures"}
    if (not isinstance(report, dict) or set(report) != expected
            or report.get("schemaVersion") != 2
            or not isinstance(report.get("datasetVersion"), str)
            or not report["datasetVersion"]
            or report.get("comparisonVersion") != COMPARISON_VERSION
            or not isinstance(report.get("manifestSha256"), str)
            or len(report["manifestSha256"]) != 64
            or any(ch not in "0123456789abcdef" for ch in report["manifestSha256"])
            or not isinstance(report.get("modes"), dict) or not report["modes"]
            or any(mode not in KNOWN_MODES for mode in report["modes"])
            or type(report.get("passed")) is not bool
            or not isinstance(report.get("failures"), list)
            or any(not isinstance(value, str) for value in report["failures"])
            or report["passed"] != (len(report["failures"]) == 0)
            or report["passed"] != all(section.get("passed") is True
                                       for section in report["modes"].values())):
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "invalid accuracy report shape")
    for section in report["modes"].values():
        _validate_mode_report(section)


def _verify_mode_section(section: AccuracyModeReport) -> bool:
    return (section["passed"] and section["effectiveScreenshots"] >= MIN_SCREENSHOTS
            and section["distinctGames"] >= MIN_GAMES
            and section["nearDuplicatePairs"] == 0
            and section["structuralEntryRate"] >= MIN_STRUCTURAL_ENTRY_RATE
            and section["medianModifiedFields"] <= MAX_MEDIAN_MODIFIED_FIELDS
            and section["p90ModifiedFields"] <= MAX_P90_MODIFIED_FIELDS)


def verify_layout_support(manifest: LoadedModelManifest,
                          report: AccuracyReportV2 | None = None) -> None:
    if manifest.support_status == "experimental":
        return
    report = report or load_accuracy_report(manifest)
    _validate_report(report)
    # Every DECLARED mode needs its own passing section — no more, no less. A
    # report proving only 4p must not promote a manifest that also serves 3p,
    # and a stray section for an undeclared mode means report/manifest drift.
    if set(report["modes"]) != set(manifest.modes):
        raise ManifestError("MODEL_MANIFEST_MISMATCH",
                            "accuracy report modes do not match manifest modes")
    if not report["passed"] or not all(
            _verify_mode_section(report["modes"][mode]) for mode in report["modes"]):
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "supported layout lacks passing report")
    if report["manifestSha256"] != manifest.manifest_sha256:
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "accuracy report was produced by another manifest")
    if report["datasetVersion"] != manifest.raw["goldenGate"]["datasetVersion"]:
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "accuracy dataset version mismatch")
    if report["comparisonVersion"] != manifest.raw["goldenGate"]["comparisonVersion"]:
        raise ManifestError("MODEL_MANIFEST_MISMATCH", "accuracy comparison version mismatch")
