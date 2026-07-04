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


def test_already_migrated_detected():
    mod = _load()
    with tempfile.TemporaryDirectory() as d:
        gd = os.path.join(d, "run_3", "game1")
        os.makedirs(gd)
        # no liqi.jsonl yet, has b64 frames.jsonl -> NOT migrated
        open(os.path.join(gd, "frames.jsonl"), "w").close()
        assert mod.is_migrated(gd) is False
        # after rename, liqi.jsonl exists -> migrated
        os.rename(os.path.join(gd, "frames.jsonl"), os.path.join(gd, "liqi.jsonl"))
        open(os.path.join(d, "run_3", "game1.jsonl"), "w").close()
        assert mod.is_migrated(gd) is True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_migrate_ai OK")
