"""Migration path-planning: b64 game dir -> new-layout targets.

Plain-script style: PYTHONPATH=. <auto-python> tests/test_migrate_ai.py
"""
import os
import tempfile
import importlib.util


def _load():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "data", "migrate_ai_to_gtrecord.py")
    spec = importlib.util.spec_from_file_location("migrate_ai", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plan_targets_multi_game():
    mod = _load()
    gd = "captures/raw/ai_session/run_3/game1"
    t = mod.plan_targets(gd)
    assert t["name"] == "ai_run_3_game1"
    assert t["gt_path"].replace(os.sep, "/").endswith("run_3/game1.jsonl")
    assert t["wire_dest"].replace(os.sep, "/").endswith("run_3/game1/liqi.jsonl")
    assert t["index_path"].replace(os.sep, "/").endswith("run_3/game1/frames.jsonl")


def test_plan_targets_single_game_run():
    mod = _load()
    gd = "captures/raw/ai_session/run_1"
    t = mod.plan_targets(gd)
    assert t["name"] == "ai_run_1"
    assert t["gt_path"].replace(os.sep, "/").endswith("ai_session/run_1.jsonl")
    assert t["wire_dest"].replace(os.sep, "/").endswith("run_1/liqi.jsonl")


def test_is_migrated_needs_all_three_outputs():
    mod = _load()
    with tempfile.TemporaryDirectory() as d:
        gd = os.path.join(d, "run_3", "game1")
        os.makedirs(gd)
        # legacy: only the b64 wire frames.jsonl -> NOT migrated
        open(os.path.join(gd, "frames.jsonl"), "w").close()
        assert mod.is_migrated(gd) is False
        # partial: wire renamed + GTRecord written, but index (frames.jsonl) missing -> NOT migrated
        os.rename(os.path.join(gd, "frames.jsonl"), os.path.join(gd, "liqi.jsonl"))
        open(os.path.join(d, "run_3", "game1.jsonl"), "w").close()
        assert mod.is_migrated(gd) is False
        # complete: index written too -> migrated
        open(os.path.join(gd, "frames.jsonl"), "w").close()
        assert mod.is_migrated(gd) is True


def test_find_game_dirs_detects_legacy_and_partial():
    mod = _load()
    import json as _json
    with tempfile.TemporaryDirectory() as d:
        ai = os.path.join(d, "ai_session")
        # legacy dir: b64 frames.jsonl
        leg = os.path.join(ai, "run_3", "game1")
        os.makedirs(leg)
        with open(os.path.join(leg, "frames.jsonl"), "w") as f:
            f.write(_json.dumps({"seq": 1, "ts": 0.0, "b64": "AAAA"}) + "\n")
        # partial dir: wire already renamed to liqi.jsonl, outputs incomplete
        part = os.path.join(ai, "run_4", "game1")
        os.makedirs(part)
        open(os.path.join(part, "liqi.jsonl"), "w").close()
        # a fully-migrated screenshot-index frames.jsonl must NOT be mistaken for wire
        with open(os.path.join(part, "frames.jsonl"), "w") as f:
            f.write(_json.dumps({"seq": 9, "file": "frames/000009.png", "status": "ok"}) + "\n")
        found = set(os.path.abspath(p) for p in mod.find_game_dirs(ai))
        assert os.path.abspath(leg) in found
        assert os.path.abspath(part) in found     # partial dir is re-picked
        # _looks_like_wire distinguishes a b64 wire line from an index line
        assert mod._looks_like_wire(os.path.join(leg, "frames.jsonl")) is True
        assert mod._looks_like_wire(os.path.join(part, "frames.jsonl")) is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_migrate_ai OK")
