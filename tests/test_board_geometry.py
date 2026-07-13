"""Guard: the RECOGNITION path never touches the process-global geometry.

``annotate.pipeline.set_sanma()`` swaps DISCARD_GRID / DISCARD_ROW_OFFSETS /
MELD_STRIP2 in place, process-wide. That is fine for the offline annotator and
the calibration tools (one frame, one thread), but the recognition worker is a
long-lived server that serves both modes from a thread pool. A per-request
set_sanma() there would corrupt a CONCURRENT request of the other mode — and
silently, because the wrong table does not raise, it just reads a different,
entirely plausible, WRONG board. Recognition therefore takes a BoardGeometry
argument and the globals stay out of it.

Plain-script style: PYTHONPATH=. <python> tests/test_board_geometry.py
"""
import glob
import os
import re

from majsoul_eye.annotate import pipeline as P

_RECOGNITION_TREES = ("majsoul_eye/recognize", "majsoul_eye/state",
                      "majsoul_eye/what_cut", "majsoul_eye/worker")
_SET_SANMA = re.compile(r"\bset_sanma\b")


def test_recognition_path_never_calls_set_sanma():
    offenders = []
    for tree in _RECOGNITION_TREES:
        for path in glob.glob(os.path.join(tree, "**", "*.py"), recursive=True):
            with open(path, encoding="utf-8") as fh:
                if _SET_SANMA.search(fh.read()):
                    offenders.append(path)
    assert not offenders, (
        f"{offenders} call set_sanma(); the recognition path must pass a "
        f"BoardGeometry explicitly — a global swap races across concurrent "
        f"requests of the other mode and silently produces a wrong board")


def test_frozen_geometries_survive_a_global_swap():
    """The whole point of the value object: a set_sanma(True) ANYWHERE in the
    process (a test, a calibration script) must not reach into GEOMETRY_4P.
    If _frozen() aliased the live dicts instead of the pristine snapshot, this
    would poison the recogniser's 4P geometry for the rest of the process."""
    before = {seat: dict(cfg) for seat, cfg in P.GEOMETRY_4P.discard_grid.items()}
    was_active = P._SANMA_ACTIVE
    try:
        P.set_sanma(True)
        assert P.DISCARD_GRID[0]["dcol"] == P.DISCARD_GRID_3P[0]["dcol"], \
            "precondition: set_sanma really did swap the globals"
        after = {seat: dict(cfg) for seat, cfg in P.GEOMETRY_4P.discard_grid.items()}
        assert after == before, "GEOMETRY_4P aliased the live tables"
    finally:
        P.set_sanma(was_active)


def test_the_two_modes_really_differ():
    # If they were accidentally the same object/values, every guard above would
    # pass vacuously and sanma would silently be read with 4P geometry.
    assert P.GEOMETRY_4P.discard_grid[1]["dcol"] != P.GEOMETRY_3P.discard_grid[1]["dcol"]
    assert P.GEOMETRY_4P.meld_strip2[0]["corner"] != P.GEOMETRY_3P.meld_strip2[0]["corner"]
    assert P.GEOMETRY_4P.nuki_strip is None      # 4P has no north pile
    assert P.GEOMETRY_3P.nuki_strip is not None
    assert P.geometry_for(False) is P.GEOMETRY_4P
    assert P.geometry_for(True) is P.GEOMETRY_3P


def test_geometry_is_immutable():
    import dataclasses
    try:
        P.GEOMETRY_4P.sanma = True
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("BoardGeometry must be frozen")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_board_geometry OK")
