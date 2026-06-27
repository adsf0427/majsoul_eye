"""Bag (multiset) scoring for the mycv baseline — position-agnostic and free of
any dependency on our own coordinate calibration (so it is fair to mycv).

For one zone we compare the multiset of predicted tile names to the multiset of
GT tile names:

* ``correct`` = size of the multiset intersection (right class, ignoring order).
* ``recall`` (a.k.a. ``end_to_end``) = correct / n_gt — fraction of GT tiles mycv
  recognized (this is the end-to-end number).
* ``precision`` = correct / n_pred — of the tiles mycv emitted, the fraction with
  the right class. NOTE: this is PRECISION, not "classification accuracy" — it does
  not measure recall, and within a seat-bag it can mask reciprocal swaps (two
  errors that exchange classes both score as correct). Treat it as an
  upper-bound-leaning precision (verified by the adversarial review panel).
* over/under-detection is exposed via n_pred vs n_gt.

A ``strict`` pass keeps red fives distinct (5mr≠5m); a ``lenient`` pass collapses
red→normal, so the gap between them isolates red-five confusion from suit/rank
errors.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..tiles import red_to_normal


@dataclass
class ZoneTally:
    n_gt: int = 0
    n_pred: int = 0
    correct: int = 0           # strict multiset intersection
    correct_lenient: int = 0   # red fives collapsed to normal
    confusions: Counter = field(default_factory=Counter)  # (gt_extra, pred_extra) name pairs (diagnostic)

    def add(self, other: "ZoneTally") -> None:
        self.n_gt += other.n_gt
        self.n_pred += other.n_pred
        self.correct += other.correct
        self.correct_lenient += other.correct_lenient
        self.confusions.update(other.confusions)

    @property
    def recall(self) -> float:
        """correct / n_gt — the end-to-end number (fraction of GT recognized)."""
        return self.correct / self.n_gt if self.n_gt else float("nan")

    # backwards-friendly alias; recall IS the end-to-end metric
    end_to_end = recall

    @property
    def precision(self) -> float:
        """correct / n_pred — of what mycv emitted, fraction with the right class.
        This is PRECISION, not classification accuracy (see module docstring)."""
        return self.correct / self.n_pred if self.n_pred else float("nan")

    @property
    def recall_lenient(self) -> float:
        return self.correct_lenient / self.n_gt if self.n_gt else float("nan")

    # alias kept for callers/json that referenced the old name
    end_to_end_lenient = recall_lenient

    def summary(self) -> str:
        return (f"n_gt={self.n_gt} n_pred={self.n_pred} correct={self.correct} "
                f"| recall(e2e)={self.recall:.3f} precision={self.precision:.3f} "
                f"| lenient_recall={self.recall_lenient:.3f}")


def bag_tally(pred_names: list[str], gt_names: list[str]) -> ZoneTally:
    pc, gc = Counter(pred_names), Counter(gt_names)
    inter = pc & gc
    pc_l = Counter(red_to_normal(n) for n in pred_names)
    gc_l = Counter(red_to_normal(n) for n in gt_names)
    # diagnostic: what GT classes were missed vs what extra classes were predicted
    missed = gc - pc          # in GT, not matched
    extra = pc - gc           # predicted, not in GT
    confusions: Counter = Counter()
    for name, c in missed.items():
        confusions[("miss", name)] += c
    for name, c in extra.items():
        confusions[("extra", name)] += c
    return ZoneTally(
        n_gt=len(gt_names),
        n_pred=len(pred_names),
        correct=sum(inter.values()),
        correct_lenient=sum((pc_l & gc_l).values()),
        confusions=confusions,
    )
