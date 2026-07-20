from __future__ import annotations

import argparse
import hashlib
import json
import os

import cv2
import numpy as np

from majsoul_eye.recognize.accuracy_gate import (
    COMPARISON_VERSION, NEAR_DUPLICATE_HAMMING_MAX, build_accuracy_report,
    evaluate_gate_metrics,
)
from majsoul_eye.recognize.manifest import load_model_manifest
from majsoul_eye.recognize.runtime import (
    RecognitionContext, RecognitionRuntime, RuntimeFailure,
)
from majsoul_eye.what_cut.schema import parse_what_cut_draft


def dhash64(image) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def hamming64(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def resolve_dataset_path(root: str, relative: str) -> str:
    resolved = os.path.abspath(os.path.join(root, relative))
    if os.path.commonpath([root, resolved]) != root:
        raise SystemExit(f"golden path escapes dataset root: {relative}")
    return resolved


def semantic_fields(draft: dict) -> list[tuple[str, object]]:
    fields = []
    round_ = draft["round"]
    # Mode identity fields (comparison v2): a 3p board read as 4p — or a missed
    # phantom seat / nuki pile — must surface as edits, not stay invisible.
    fields.append(("nPlayers", draft.get("nPlayers", 4)))
    fields.append(("round.phantomRelSeat", round_.get("phantomRelSeat")))
    for seat, player in enumerate(draft["players"]):
        fields.append((f"player.{seat}.nukiCount", player.get("nukiCount", 0)))
    for key in ("gameLength", "bakaze", "kyoku", "honba", "kyotaku",
                "leftTileCount", "seatWindSelf"):
        fields.append((f"round.{key}", round_[key]))
    for index, score in enumerate(round_["scores"]):
        fields.append((f"round.scores.{index}", score))
    for index, tile in enumerate(draft["doraMarkers"]):
        fields.append((f"dora.{index}.pai", tile["pai"]))
    for seat, player in enumerate(draft["players"]):
        fields.append((f"player.{seat}.reach", player["reach"]))
        fields.append((f"player.{seat}.concealedCount", player["concealedCount"]))
        if player["hand"] is not None:
            for index, tile in enumerate(player["hand"]):
                fields.append((f"player.{seat}.hand.{index}.pai", tile["pai"]))
        fields.append((f"player.{seat}.drawn.pai",
                       player["drawnTile"]["pai"] if player["drawnTile"] else None))
        for index, discard in enumerate(player["rivers"]):
            fields.extend(((f"player.{seat}.river.{index}.pai", discard["pai"]),
                           (f"player.{seat}.river.{index}.sideways", discard["sideways"]),
                           (f"player.{seat}.river.{index}.tsumogiri",
                            discard["tsumogiri"]["value"])))
        for index, meld in enumerate(player["melds"]):
            fields.extend(((f"player.{seat}.meld.{index}.type", meld["type"]),
                           (f"player.{seat}.meld.{index}.tiles", tuple(meld["tiles"])),
                           (f"player.{seat}.meld.{index}.calledPai", meld["calledPai"]),
                           (f"player.{seat}.meld.{index}.addedPai", meld["addedPai"]),
                           (f"player.{seat}.meld.{index}.fromOffset", meld["fromOffset"])))
    for index, ghost in enumerate(draft["historyOverrides"]["ghostDiscards"]):
        fields.extend(((f"ghost.{index}.ownerRelSeat", ghost["ownerRelSeat"]),
                       (f"ghost.{index}.pai", ghost["pai"]),
                       (f"ghost.{index}.beforeMeld",
                        tuple(ghost["beforeMeldId"].split(":")[-2:])),
                       (f"ghost.{index}.tsumogiri", ghost["tsumogiri"]["value"])))
    return sorted(fields)


def modification_count(actual: dict, expected: dict) -> int:
    left, right = dict(semantic_fields(actual)), dict(semantic_fields(expected))
    return sum(left.get(key) != right.get(key) for key in set(left) | set(right))


def score_runtime_sample(runtime, image_bytes: bytes, context,
                         expected: dict) -> tuple[bool, int]:
    recognized = runtime.recognize_bytes(image_bytes, context)
    actual = recognized["draft"]
    rebuilt = runtime.reconstruct_draft(actual, actual["revision"])
    issues = [*recognized["issues"], *rebuilt["issues"]]
    entered = bool(rebuilt["ok"]) and not any(
        issue["severity"] == "blocking" for issue in issues)
    # Accuracy edits deliberately compare the raw recognized draft with GT. The
    # reconstruction pass gates structural entry but never rewrites this diff.
    return entered, modification_count(actual, expected)


def load_rows(path: str) -> list[dict]:
    rows = []
    sample_ids = set()
    for line_number, line in enumerate(open(path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"schemaVersion", "datasetVersion", "sampleId", "gameId",
                    "imagePath", "imageSha256", "expectedDraftPath",
                    "independentOfTraining"}
        if set(row) != required or row["schemaVersion"] != 1:
            raise SystemExit(f"invalid golden row {line_number}")
        if row["independentOfTraining"] is not True:
            raise SystemExit(f"golden row {line_number} is not held out from training")
        if not all(isinstance(row[key], str) and row[key] for key in
                   ("datasetVersion", "sampleId", "gameId", "imagePath",
                    "imageSha256", "expectedDraftPath")):
            raise SystemExit(f"golden row {line_number} has an empty string field")
        if row["sampleId"] in sample_ids:
            raise SystemExit(f"duplicate sampleId at golden row {line_number}")
        sample_ids.add(row["sampleId"])
        if (len(row["imageSha256"]) != 64
                or any(ch not in "0123456789abcdef" for ch in row["imageSha256"])):
            raise SystemExit(f"invalid imageSha256 at golden row {line_number}")
        rows.append(row)
    return rows


def parse_mode_goldens(specs: list[str]) -> dict[str, str]:
    """Repeatable ``--goldens MODE=PATH`` -> {mode: path}. Every mode the
    manifest declares must be given exactly once — a report missing a declared
    mode can never promote the manifest, so refusing early is kinder."""
    out = {}
    for spec in specs:
        mode, _, path = spec.partition("=")
        if mode not in ("4p", "3p") or not path:
            raise SystemExit(f"--goldens must be 4p=PATH or 3p=PATH, got {spec!r}")
        if mode in out:
            raise SystemExit(f"duplicate --goldens for mode {mode}")
        out[mode] = path
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--goldens", required=True, action="append",
                        help="MODE=PATH, repeat per declared mode (4p=..., 3p=...)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eye-revision", required=True)
    args = parser.parse_args()
    manifest = load_model_manifest(args.manifest)
    if manifest.raw["goldenGate"]["comparisonVersion"] != COMPARISON_VERSION:
        raise SystemExit("manifest comparisonVersion is unsupported")
    manifest_root = os.path.dirname(manifest.path)
    report_path = os.path.abspath(os.path.join(
        manifest_root, manifest.raw["goldenGate"]["reportPath"]))
    checksum_path = os.path.abspath(os.path.join(
        manifest_root, manifest.raw["goldenGate"]["reportChecksumPath"]))
    if os.path.abspath(args.out) != report_path:
        raise SystemExit("--out must equal manifest goldenGate.reportPath")
    runtime = RecognitionRuntime.from_manifest(
        args.manifest, device=args.device, eye_revision=args.eye_revision,
        evaluation_mode=True)
    mode_goldens = parse_mode_goldens(args.goldens)
    if set(mode_goldens) != set(manifest.modes):
        raise SystemExit(f"--goldens modes {sorted(mode_goldens)} must equal "
                         f"manifest modes {sorted(manifest.modes)}")
    mode_sections = {}
    for mode, goldens_path in sorted(mode_goldens.items()):
        rows = load_rows(goldens_path)
        hashes = []
        entered, edits, game_ids = [], [], []
        dataset_versions = {row["datasetVersion"] for row in rows}
        if len(dataset_versions) > 1:
            raise SystemExit("golden rows must share one datasetVersion")
        dataset_version = next(iter(dataset_versions),
                               manifest.raw["goldenGate"]["datasetVersion"])
        if dataset_version != manifest.raw["goldenGate"]["datasetVersion"]:
            raise SystemExit("golden datasetVersion does not match manifest")
        root = os.path.dirname(os.path.abspath(goldens_path))
        for row in rows:
            image_path = resolve_dataset_path(root, row["imagePath"])
            image_bytes = open(image_path, "rb").read()
            if hashlib.sha256(image_bytes).hexdigest() != row["imageSha256"]:
                raise SystemExit(f"image SHA mismatch: {row['sampleId']}")
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise SystemExit(f"cannot decode image: {row['sampleId']}")
            hashes.append((row["sampleId"], dhash64(image)))
            expected = parse_what_cut_draft(json.load(open(
                resolve_dataset_path(root, row["expectedDraftPath"]), encoding="utf-8")))
            # board_mode stays "auto": mode detection is part of what the gate
            # measures — a forced mode would grade an easier product than ships.
            context = RecognitionContext(row["sampleId"], f"eval:{row['sampleId']}",
                                         row["imageSha256"], manifest.layout_id, True, None)
            try:
                sample_entered, modified = score_runtime_sample(
                    runtime, image_bytes, context, expected)
                entered.append(sample_entered)
                edits.append(modified)
            except RuntimeFailure:
                entered.append(False)
                edits.append(len(semantic_fields(expected)))
            game_ids.append(row["gameId"])
        near = []
        for left in range(len(hashes)):
            for right in range(left + 1, len(hashes)):
                if hamming64(hashes[left][1], hashes[right][1]) <= NEAR_DUPLICATE_HAMMING_MAX:
                    near.append((hashes[left][0], hashes[right][0]))
        section = evaluate_gate_metrics(
            modified_fields=edits, game_ids=game_ids,
            structurally_entered=entered, near_duplicate_pairs=near)
        section["rawScreenshots"] = len(rows)
        section["nearDuplicateSamplePairs"] = [list(pair) for pair in near]
        mode_sections[mode] = section
    report = build_accuracy_report(
        manifest_sha256=manifest.manifest_sha256,
        dataset_version=manifest.raw["goldenGate"]["datasetVersion"],
        modes=mode_sections)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    open(report_path, "w", encoding="utf-8").write(payload)
    open(checksum_path, "w", encoding="ascii").write(
        hashlib.sha256(payload.encode("utf-8")).hexdigest() + "\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
