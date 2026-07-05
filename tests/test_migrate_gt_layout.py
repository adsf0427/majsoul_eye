"""migrate_gt_into_gamedir: sibling-shape AI GT jsonls move INTO their frames dirs,
and datasets/*/games.json capture paths are rewritten to the nested form.

Plain-script style: PYTHONPATH=. <auto-python> tests/test_migrate_gt_layout.py
"""
import importlib.util
import json
import os
import tempfile


def _load():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "data",
                        "migrate_gt_into_gamedir.py")
    spec = importlib.util.spec_from_file_location("migrate_gt_layout", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mk_tree(d):
    """A raw tree with: sibling multi-game, sibling single-game, already-nested,
    a manual-style session (must be ignored), and a jsonl with no dir twin."""
    ai = os.path.join(d, "captures", "raw", "ai_session")
    os.makedirs(os.path.join(ai, "run_3", "game1", "frames"))
    open(os.path.join(ai, "run_3", "game1.jsonl"), "w").close()          # sibling -> move
    os.makedirs(os.path.join(ai, "run_2"))
    open(os.path.join(d, "captures", "raw", "ai_session", "run_2.jsonl"), "w").close()  # single-game -> move
    os.makedirs(os.path.join(ai, "run_9", "game1"))
    open(os.path.join(ai, "run_9", "game1", "game1.jsonl"), "w").close() # already nested -> untouched
    open(os.path.join(ai, "run_7.jsonl"), "w").close()                   # no dir twin -> left in place
    man = os.path.join(d, "captures", "raw", "manual")
    os.makedirs(os.path.join(man, "session5"))
    open(os.path.join(man, "session5.jsonl"), "w").close()               # manual -> NEVER touched
    return ai


def test_plan_moves_sibling_shapes_only():
    mod = _load()
    with tempfile.TemporaryDirectory() as d:
        _mk_tree(d)
        raw = os.path.join(d, "captures", "raw")
        moves = mod.plan_moves(raw)
        rel = sorted((os.path.relpath(s, raw).replace(os.sep, "/"),
                      os.path.relpath(t, raw).replace(os.sep, "/")) for s, t in moves)
        assert rel == [
            ("ai_session/run_2.jsonl", "ai_session/run_2/run_2.jsonl"),
            ("ai_session/run_3/game1.jsonl", "ai_session/run_3/game1/game1.jsonl"),
        ], rel


def test_apply_moves_and_manifest_rewrite_idempotent():
    mod = _load()
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        try:
            os.chdir(d)
            _mk_tree(d)
            ds = os.path.join("datasets", "v9")
            os.makedirs(ds)
            manifest = {"val": "ai_run_3_game1", "games": [
                {"name": "ai_run_3_game1", "dir": "g", "kind": "ai",
                 "capture": "captures/raw/ai_session/run_3/game1.jsonl",
                 "frames_dir": "captures/raw/ai_session/run_3/game1"},
                {"name": "session5", "dir": "s", "kind": "manual",
                 "capture": "captures/raw/manual/session5.jsonl",
                 "frames_dir": "captures/raw/manual/session5"},
            ]}
            with open(os.path.join(ds, "games.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f)

            raw = os.path.join("captures", "raw")
            n_moved = mod.apply_moves(mod.plan_moves(raw))
            assert n_moved == 2
            assert os.path.exists(os.path.join(raw, "ai_session", "run_3", "game1", "game1.jsonl"))
            assert not os.path.exists(os.path.join(raw, "ai_session", "run_3", "game1.jsonl"))
            assert os.path.exists(os.path.join(raw, "ai_session", "run_2", "run_2.jsonl"))

            n_rw = mod.rewrite_manifests("datasets", apply=True)
            assert n_rw == 1
            m = json.load(open(os.path.join(ds, "games.json"), encoding="utf-8"))
            assert m["games"][0]["capture"] == "captures/raw/ai_session/run_3/game1/game1.jsonl"
            assert m["games"][0]["frames_dir"] == "captures/raw/ai_session/run_3/game1"  # unchanged
            assert m["games"][1]["capture"] == "captures/raw/manual/session5.jsonl"      # untouched

            # idempotent: a second pass finds nothing
            assert mod.plan_moves(raw) == []
            assert mod.rewrite_manifests("datasets", apply=True) == 0
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_migrate_gt_layout OK")
