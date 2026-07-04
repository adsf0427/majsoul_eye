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


class _FakeRec:
    def __init__(self, seq):
        self._seq = seq
    def to_json_line(self):
        import json as _json
        return _json.dumps({"seq": self._seq, "mjai": []})


def test_migrate_one_legacy_window(tmp=None):
    """Fresh legacy dir (b64 frames.jsonl present): migrate_one writes GTRecord,
    renames wire -> liqi.jsonl, writes the index. Exercises _atomic_write +
    _render_records + the rename branch."""
    mod = _load()
    import json as _json
    with tempfile.TemporaryDirectory() as d:
        gd = os.path.join(d, "run_3", "game1")
        os.makedirs(os.path.join(gd, "frames"))
        with open(os.path.join(gd, "frames.jsonl"), "w") as f:      # legacy b64 wire
            f.write(_json.dumps({"seq": 1, "ts": 0.0, "b64": "AAAA"}) + "\n")
        open(os.path.join(gd, "frames", "000001.png"), "w").close()

        t = mod.migrate_one(gd, [_FakeRec(1)], [{"seq": 1}])

        # GTRecord written with schema header + record line
        with open(t["gt_path"], encoding="utf-8") as f:
            gt_lines = [l.strip() for l in f if l.strip()]
        assert gt_lines[0] == '{"_schema": 1}'
        assert _json.loads(gt_lines[1])["seq"] == 1
        # wire renamed to liqi.jsonl; frames.jsonl is now the index (no b64)
        assert os.path.exists(os.path.join(gd, "liqi.jsonl"))
        with open(os.path.join(gd, "frames.jsonl"), encoding="utf-8") as f:
            idx = _json.loads(f.readline())
        assert idx == {"seq": 1, "file": "frames/000001.png", "status": "ok"}
        assert mod.is_migrated(gd) is True
        # no leftover temp files
        assert not os.path.exists(t["gt_path"] + ".tmp")


def test_migrate_one_resumed_window():
    """Resumed dir (wire already renamed to liqi.jsonl, gt_path partial, index
    missing): migrate_one must SKIP the rename (no frames.jsonl to rename) and
    still complete the index. Exercises the resumed branch."""
    mod = _load()
    import json as _json
    with tempfile.TemporaryDirectory() as d:
        gd = os.path.join(d, "run_4", "game1")
        os.makedirs(os.path.join(gd, "frames"))
        open(os.path.join(gd, "liqi.jsonl"), "w").close()            # wire already renamed
        open(os.path.join(d, "run_4", "game1.jsonl"), "w").close()   # partial gt_path
        open(os.path.join(gd, "frames", "000002.png"), "w").close()
        assert mod.is_migrated(gd) is False                          # index missing

        t = mod.migrate_one(gd, [_FakeRec(2)], [{"seq": 2}])         # must NOT raise on missing frames.jsonl

        assert os.path.exists(os.path.join(gd, "liqi.jsonl"))        # untouched
        with open(os.path.join(gd, "frames.jsonl"), encoding="utf-8") as f:
            assert _json.loads(f.readline()) == {"seq": 2, "file": "frames/000002.png", "status": "ok"}
        with open(t["gt_path"], encoding="utf-8") as f:
            assert f.readline().strip() == '{"_schema": 1}'          # rewritten atomically
        assert mod.is_migrated(gd) is True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_migrate_ai OK")
