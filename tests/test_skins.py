"""Dependency-light tests for the skin-swap MITM helper — no browser, no mitmproxy needed.

Covers the pure seams of majsoul_eye.capture.skins (env/path resolution, argv, config text,
seed no-clobber, port wait) + the --skins CLI surface. Plain-script style (no pytest):
  PYTHONPATH=. <auto-python> tests/test_skins.py
"""
import os
import socket
import tempfile

from majsoul_eye.capture import skins  # noqa: E402


def test_env_and_path_resolution():
    d = skins.env_dir("auto")
    assert d.name == "auto" and d.parent.name == "envs"          # sibling env of the running interp
    assert skins.mitmdump_exe("auto").endswith(os.path.join("auto", "Scripts", "mitmdump.exe")
                                                if os.name == "nt" else os.path.join("auto", "bin", "mitmdump"))
    assert skins.env_python("auto").endswith("python.exe" if os.name == "nt" else "python")
    assert skins.DEFAULT_ENV == "auto"
    bs = skins.builder_script()
    assert bs.endswith(os.path.join("scripts", "capture", "build_skin_config.py")) and os.path.exists(bs)


def test_build_cmd():
    cmd = skins.build_cmd("/x/mitmdump", 23410, "/conf")
    assert cmd[:5] == ["/x/mitmdump", "-p", "23410", "-s", "addons.py"]
    assert "--set" in cmd and "confdir=/conf" in cmd and "ssl_insecure=true" in cmd


def test_settings_text_offline_vs_online():
    on = skins.settings_yaml_text(mod=True, offline=True)
    assert "mod: true" in on and "helper: false" in on and "auto_update: false" in on
    off = skins.settings_yaml_text(mod=True, offline=False)
    assert "auto_update: true" in off
    assert "auto_update: false" in skins.mod_seed_yaml_text(offline=True)


def test_seed_config_creates_and_does_not_clobber():
    with tempfile.TemporaryDirectory() as d:
        # pre-existing settings.mod.yaml must be preserved (mod.py owns it / persisted skin state)
        cfg = os.path.join(d, "config")
        os.makedirs(cfg)
        sentinel = "config:\n  character: 200042  # user choice\n"
        with open(os.path.join(cfg, "settings.mod.yaml"), "w", encoding="utf-8") as f:
            f.write(sentinel)
        skins.seed_config(d, offline=True)
        # settings.yaml always (re)written with the mod plugin enabled
        assert os.path.exists(os.path.join(cfg, "settings.yaml"))
        with open(os.path.join(cfg, "settings.yaml"), encoding="utf-8") as f:
            assert "mod: true" in f.read()
        # settings.mod.yaml left untouched
        with open(os.path.join(cfg, "settings.mod.yaml"), encoding="utf-8") as f:
            assert f.read() == sentinel

    with tempfile.TemporaryDirectory() as d:
        skins.seed_config(d, offline=True)                       # absent -> minimal offline seed created
        with open(os.path.join(d, "config", "settings.mod.yaml"), encoding="utf-8") as f:
            assert "auto_update: false" in f.read()


def test_wait_port():
    # nothing listening on an unlikely port -> False fast
    assert skins.wait_port(59999, timeout=0.5) is False
    # a listening socket -> True
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert skins.wait_port(port, timeout=2.0) is True
    finally:
        srv.close()

    # a proc that has already exited -> stop immediately (False)
    class _Dead:
        def poll(self):
            return 0
    assert skins.wait_port(59998, timeout=5.0, proc=_Dead()) is False


def test_skinproxy_props():
    sp = skins.SkinProxy("/some/MajsoulMax", port=23410, confdir="/cf")
    assert sp.proxy_str == "http://127.0.0.1:23410"
    assert sp.cert_path == os.path.join(os.path.abspath("/cf"), skins.CERT_NAME)   # confdir is abspath'd
    assert sp.randomize is None                                  # manual mode by default


def test_autoplay_ai_exposes_skins_flags():
    import argparse
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "capture", "autoplay_ai.py")
    spec = importlib.util.spec_from_file_location("autoplay_ai_skins_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen = {}
    real_parse = argparse.ArgumentParser.parse_args

    def capture_parse(self, *a, **k):
        for act in self._actions:
            seen[tuple(act.option_strings)] = act
        raise SystemExit(0)
    argparse.ArgumentParser.parse_args = capture_parse
    try:
        try:
            mod.main()
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = real_parse
    flat = {opt for opts in seen for opt in opts}
    for f in ("--skins", "--skins-port", "--skins-env", "--skins-dir",
              "--skins-randomize", "--skins-slots", "--skins-all-seats"):
        assert f in flat, f"missing flag {f}"
    defaults = {opts[0]: act.default for opts, act in seen.items()}
    assert defaults["--skins"] is False                          # opt-in
    assert defaults["--skins-port"] == 23410
    assert defaults["--skins-slots"] == "7,6,8"                   # 牌面(13) excluded by default


def test_patcher_roundtrip():
    """The tracked mod.py patcher must stay in sync with the (gitignored) vendored mod.py:
    reversing its hunks then re-applying must reproduce the current file byte-for-byte.
    Skipped if MajsoulMax isn't checked out locally (it's gitignored)."""
    import importlib.util
    here = os.path.dirname(__file__)
    mod_py = os.path.join(here, "..", "_external", "MajsoulMax", "plugin", "mod.py")
    if not os.path.exists(mod_py):
        print("  (skip test_patcher_roundtrip: _external/MajsoulMax absent)")
        return
    p_path = os.path.join(here, "..", "scripts", "capture", "patch_majsoulmax.py")
    spec = importlib.util.spec_from_file_location("patch_majsoulmax_test", p_path)
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)

    cur = open(mod_py, encoding="utf-8").read()
    assert P.MARKER in cur, "working-copy mod.py is not patched (run scripts/capture/patch_majsoulmax.py)"
    pristine = cur
    for _label, old, new in P.HUNKS:
        assert new in pristine
        pristine = pristine.replace(new, old, 1)
    assert P.MARKER not in pristine
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "plugin"))
        mp = os.path.join(d, "plugin", "mod.py")
        open(mp, "w", encoding="utf-8").write(pristine)
        assert P.ensure_patched(d) is True
        assert open(mp, encoding="utf-8").read() == cur          # apply reproduces the patched file
        assert P.ensure_patched(d) is False                      # idempotent


def test_patcher_protobuf7():
    """MajsoulMax's addon must not use the protobuf<4 `including_default_value_fields` kwarg —
    under a modern protobuf it throws on every parse (no unlock). ensure_protobuf7 must leave none
    and be idempotent. Skipped if MajsoulMax isn't checked out locally."""
    import importlib.util
    here = os.path.dirname(__file__)
    mjmax = os.path.join(here, "..", "_external", "MajsoulMax")
    if not os.path.exists(os.path.join(mjmax, "liqi_new.py")):
        print("  (skip test_patcher_protobuf7: _external/MajsoulMax absent)")
        return
    p_path = os.path.join(here, "..", "scripts", "capture", "patch_majsoulmax.py")
    spec = importlib.util.spec_from_file_location("patch_majsoulmax_pb7", p_path)
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    P.ensure_protobuf7(mjmax)                                   # make sure it's applied
    assert P.ensure_protobuf7(mjmax) == []                     # idempotent: second call changes nothing
    for rel in P.PROTOBUF7_FILES:
        f = os.path.join(mjmax, rel)
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                assert P.OLD_KW not in fh.read(), f"{rel} still uses the removed protobuf kwarg"


def test_patcher_res_framing():
    """mod.py's modified-message write-back must preserve the EMPTY method_name block (0a 00).
    BaseMessage is proto3, so a plain SerializeToString of a rewritten RES silently DROPS field 1
    (empty string, no presence) — but Majsoul's native frames always carry it, and downstream
    POSITIONAL parsers (MahjongCopilot liqi.py:157, i.e. our autoplay tap) assert on it. mod.py
    rewrites the authGame RES unconditionally, so without this patch --skins drops authGame in the
    tap -> GameState keeps self.seat=0 -> Mortal never reacts (the 2026-07-05 autoplay-dead bug).
    Skipped if MajsoulMax isn't checked out locally."""
    import importlib.util
    import sys
    here = os.path.dirname(__file__)
    mjmax = os.path.abspath(os.path.join(here, "..", "_external", "MajsoulMax"))
    if not os.path.exists(os.path.join(mjmax, "plugin", "mod.py")):
        print("  (skip test_patcher_res_framing: _external/MajsoulMax absent)")
        return
    p_path = os.path.join(here, "..", "scripts", "capture", "patch_majsoulmax.py")
    spec = importlib.util.spec_from_file_location("patch_majsoulmax_rf", p_path)
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    P.ensure_res_framing(mjmax)                                 # apply if not present
    assert P.ensure_res_framing(mjmax) is False                # idempotent
    with open(os.path.join(mjmax, "plugin", "mod.py"), encoding="utf-8") as fh:
        assert P.RES_FRAMING_MARKER in fh.read()

    # Functional: the framing the patch installs must be byte-identical to Majsoul's native wire.
    sys.path.insert(0, mjmax)
    try:
        try:
            import liqi_new                                     # noqa: vendored, needs protobuf
            from proto import basic_pb2
        except Exception as e:                                  # pragma: no cover - env without protobuf
            print(f"  (skip res-framing functional check: {type(e).__name__}: {e})")
            return
        payload = b"\x08\x01"                                   # arbitrary inner protobuf
        original = bytes([3, 7, 0]) + liqi_new.toProtobuf(      # RES frame as Majsoul sends it
            [{"id": 1, "type": "string", "data": b""},
             {"id": 2, "type": "string", "data": payload}])
        assert original[3:5] == b"\x0a\x00"                     # empty method_name block present
        blk = basic_pb2.BaseMessage()
        blk.ParseFromString(original[3:])
        assert not blk.SerializeToString().startswith(b"\x0a")  # THE BUG: proto3 drops field 1
        fixed = original[:3] + liqi_new.toProtobuf(             # what the patched write-back emits
            [{"id": 1, "type": "string", "data": blk.method_name.encode()},
             {"id": 2, "type": "string", "data": blk.data}])
        assert fixed == original                                # byte-identical to native framing
    finally:
        sys.path.remove(mjmax)


def test_patcher_max_data_idempotent():
    """ensure_max_data must be idempotent and leave mod.py loading proto/max_data.yaml.
    Skipped if MajsoulMax isn't checked out locally."""
    import importlib.util
    here = os.path.dirname(__file__)
    mjmax = os.path.join(here, "..", "_external", "MajsoulMax")
    if not os.path.exists(os.path.join(mjmax, "plugin", "mod.py")):
        print("  (skip test_patcher_max_data_idempotent: _external/MajsoulMax absent)")
        return
    p_path = os.path.join(here, "..", "scripts", "capture", "patch_majsoulmax.py")
    spec = importlib.util.spec_from_file_location("patch_majsoulmax_md", p_path)
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    P.ensure_max_data(mjmax)                                    # apply if not present
    assert P.ensure_max_data(mjmax) is False                   # idempotent
    with open(os.path.join(mjmax, "plugin", "mod.py"), encoding="utf-8") as fh:
        assert P.MAX_DATA_MARKER in fh.read()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_skins OK")
