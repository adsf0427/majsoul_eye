"""Machine-generate what-cut golden sets from HELD-OUT GT capture games.

The gate's goldens were meant to be hand-labeled; that never scaled past 16
frames, so the accuracy gate has never actually run. GT capture sessions make
hand labels unnecessary: every frame comes with a complete liqi-replayed
BoardState, so the EXPECTED draft is computed from ground truth — never from
the recognizer under test. Sanma sessions get first-class goldens this way,
and the same sweep backfills 4p (the gate finally has data in both modes).

Independence is the hard constraint, not frame count: the detector was trained
on these same sessions, so ONLY games named in the dataset's ``games.json``
``val`` list (held out whole from every training stage) are eligible. Today
that is 5×4p + 1×3p games — far below the gate's 20-games-per-mode floor, so
the produced report documents real held-out accuracy but cannot yet promote
the manifest. Graduation needs future capture sessions reserved as val.

Expected-draft contract (why this mirrors the recognizer, not raw GT):
- Projection runs through the SAME ``draft_from_observed`` the worker uses.
- ``concealedCount`` is nulled on every seat: assembly never reports it (the
  app derives it later), so a GT-filled count would be a fake edit per seat.
- The single-frame-unobservable history marks (tsumogiri baselines, ghost
  order) come from the production reconstruct solver run on the GT observed
  state — recognition accuracy is what the gate measures, not the solver's
  inference-vs-fate gap, which is identical on both sides of the diff.

Frame eligibility: in-round, not the deal window, not a pending call, not a
score-animation window, hero has a drawn tile (a what-cut decision point),
GT observed passes check_observed, and the production reconstruct accepts it
with zero blocking issues. dhash near-duplicates (Hamming <= 4) are dropped at
build time so the gate's nearDuplicatePairs check measures the recognizer's
dataset, not this builder's sampling.

Usage (GPU not required — no recognition runs here):
    python scripts/eval/build_sanma_goldens.py \
        --dataset datasets/v6 --out datasets/goldens-heldout-v2 \
        --dataset-version majsoul-gold-heldout-v2 --per-game 24
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2  # noqa: E402

from majsoul_eye.capture.gtframes import build_seq_state, load_frames  # noqa: E402
from majsoul_eye.state.observe import check_observed, observed_from_board  # noqa: E402
from majsoul_eye.state.reconstruct import reconstruct  # noqa: E402
from majsoul_eye.state.replay import (  # noqa: E402
    is_call_pending, is_deal_window, is_score_anim_window,
)
from majsoul_eye.what_cut.adapter import draft_to_observed  # noqa: E402
from majsoul_eye.what_cut.from_recognition import (  # noqa: E402
    apply_history_baseline, draft_from_observed,
)
from majsoul_eye.what_cut.schema import parse_what_cut_draft  # noqa: E402

from eval_what_cut_goldens import dhash64, hamming64  # noqa: E402

MANUAL_SOURCE = {"kind": "manual", "imageRef": None, "imageHash": None,
                 "width": None, "height": None}


def expected_draft_for(state, sample_id: str) -> dict | None:
    """GT BoardState -> the draft a PERFECT recognizer would emit, or None if
    this frame is not a clean single-frame position (then it is no golden)."""
    observed = observed_from_board(state, include_hud=True)
    if observed.violations or check_observed(observed):
        return None
    draft = draft_from_observed(observed, draft_id=sample_id,
                                source=copy.deepcopy(MANUAL_SOURCE), recognizer=None)
    # Assembly never reports concealed counts (the app derives them later);
    # a GT-filled count would be one fake edit per opponent on every golden.
    for player in draft["players"]:
        player["concealedCount"] = None
    adapted = draft_to_observed(parse_what_cut_draft(draft))
    if adapted.observed is None:
        return None
    rebuilt = reconstruct(adapted.observed, adapted.overrides)
    if not rebuilt.ok or any(issue["severity"] == "blocking" for issue in rebuilt.issues):
        return None
    apply_history_baseline(draft, rebuilt.history_baseline)
    return draft


def eligible_seqs(seq_state) -> list[int]:
    out = []
    for seq, state in sorted(seq_state.items()):
        if (not state.in_round or state.hero_seat < 0 or not state.drawn_tile
                or is_deal_window(state) or is_call_pending(state)
                or is_score_anim_window(state)):
            continue
        out.append(seq)
    return out


def spaced(items: list[int], count: int) -> list[int]:
    if len(items) <= count:
        return items
    step = len(items) / count
    return [items[int(index * step)] for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="versioned dataset dir whose games.json names the held-out val games")
    parser.add_argument("--out", required=True, help="golden dataset output dir")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--per-game", type=int, default=24)
    args = parser.parse_args()

    games_manifest = json.load(open(os.path.join(args.dataset, "games.json"),
                                    encoding="utf-8"))
    by_name = {game["name"]: game for game in games_manifest["games"]}
    val_names = games_manifest["val"]
    missing = [name for name in val_names if name not in by_name]
    if missing:
        raise SystemExit(f"val games missing from games.json: {missing}")

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "drafts"), exist_ok=True)
    rows = {"4p": [], "3p": []}
    kept_hashes: list[int] = []
    stats = {"eligible": 0, "gt_rejected": 0, "near_duplicate": 0, "no_frame": 0}

    for name in val_names:
        game = by_name[name]
        seq_state = build_seq_state(game["capture"])
        frames = load_frames(game["frames_dir"])
        candidates = eligible_seqs(seq_state)
        stats["eligible"] += len(candidates)
        # Oversample before dedup/GT-rejection so the per-game budget survives.
        for seq in spaced(candidates, args.per_game * 3):
            if len([r for mode in rows.values() for r in mode
                    if r["gameId"] == name]) >= args.per_game:
                break
            if seq not in frames:
                stats["no_frame"] += 1
                continue
            state = seq_state[seq]
            sample_id = f"{name}:{seq}"
            expected = expected_draft_for(state, sample_id)
            if expected is None:
                stats["gt_rejected"] += 1
                continue
            image = cv2.imread(frames[seq])
            if image is None:
                stats["no_frame"] += 1
                continue
            digest = dhash64(image)
            if any(hamming64(digest, kept) <= 4 for kept in kept_hashes):
                stats["near_duplicate"] += 1
                continue
            kept_hashes.append(digest)
            mode = "3p" if expected["nPlayers"] == 3 else "4p"
            extension = os.path.splitext(frames[seq])[1] or ".png"
            image_rel = f"images/{sample_id.replace(':', '_')}{extension}"
            draft_rel = f"drafts/{sample_id.replace(':', '_')}.json"
            shutil.copyfile(frames[seq], os.path.join(args.out, image_rel))
            image_sha = hashlib.sha256(
                open(os.path.join(args.out, image_rel), "rb").read()).hexdigest()
            with open(os.path.join(args.out, draft_rel), "w", encoding="utf-8") as fh:
                json.dump(expected, fh, ensure_ascii=False, indent=1)
                fh.write("\n")
            rows[mode].append({
                "schemaVersion": 1, "datasetVersion": args.dataset_version,
                "sampleId": sample_id, "gameId": name,
                "imagePath": image_rel, "imageSha256": image_sha,
                "expectedDraftPath": draft_rel,
                # Load-bearing honesty: only whole-game held-out val games are
                # walked above, so every row really is independent of training.
                "independentOfTraining": True,
            })

    for mode, mode_rows in rows.items():
        out_path = os.path.join(args.out, f"goldens-{mode}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fh:
            for row in mode_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        games = len({row["gameId"] for row in mode_rows})
        print(f"[{mode}] {len(mode_rows)} goldens from {games} held-out games "
              f"-> {out_path}")
    print(f"stats: {stats}")


if __name__ == "__main__":
    main()
