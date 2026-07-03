"""Consistency-scorer core (pure; no model weights needed)."""
import numpy as np

from majsoul_eye.annotate.consistency import (
    verdict_from_probs, is_empty_felt, BoxVerdict, TAU,
)
from majsoul_eye.tiles import NAME_TO_ID, TILE_NAMES


def _one_hot(name, p=0.99):
    row = np.full(38, (1.0 - p) / 37, np.float32)
    row[NAME_TO_ID[name]] = p
    return row


def test_agree_is_ok():
    v = verdict_from_probs(_one_hot("8s"), "8s")
    assert v.ok and v.pred == "8s" and v.reason == "" and v.conf > 0.9
    print("test_agree_is_ok OK")


def test_confident_mismatch_is_bad():
    # classifier is sure it's 3p, GT says 8s -> bad (mismatch, low P(gt))
    v = verdict_from_probs(_one_hot("3p"), "8s")
    assert not v.ok and v.pred == "3p" and v.reason == "mismatch"
    print("test_confident_mismatch_is_bad OK")


def test_mismatch_but_gt_still_plausible_is_ok():
    # top1 != gt, but P(gt) >= TAU -> keep (avoid deleting on weak classifier calls)
    row = np.full(38, 0.0, np.float32)
    row[NAME_TO_ID["5p"]] = 0.45
    row[NAME_TO_ID["5pr"]] = 0.55   # top1 = 5pr, but gt=5p has conf 0.45... make gt pass:
    row[NAME_TO_ID["5p"]] = 0.50
    row = row / row.sum()
    v = verdict_from_probs(row, "5p", tau=0.30)
    assert v.ok, (v.pred, v.conf)
    print("test_mismatch_but_gt_still_plausible_is_ok OK")


def test_empty_felt_detection():
    felt = np.zeros((64, 64, 3), np.uint8)          # flat -> no tile face
    felt[:, :] = (90, 60, 40)
    tile = np.zeros((64, 64, 3), np.uint8)
    tile[8:56, 8:56] = 240                          # bright face
    assert is_empty_felt(felt)
    assert not is_empty_felt(tile)
    print("test_empty_felt_detection OK")


if __name__ == "__main__":
    test_agree_is_ok()
    test_confident_mismatch_is_bad()
    test_mismatch_but_gt_still_plausible_is_ok()
    test_empty_felt_detection()
    print("ALL test_consistency (core) OK")
