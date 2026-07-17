"""The worker must be startable from the PACKAGE alone (release payloads prune
scripts/), via ``python -m majsoul_eye.worker``. See STATUS.md §1.67."""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_serve_main_is_in_the_package():
    from majsoul_eye.worker.serve import main
    assert callable(main)


def test_module_entry_parses_args():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    res = subprocess.run(
        [sys.executable, "-m", "majsoul_eye.worker", "--help"],
        capture_output=True, text=True, timeout=120, env=env,
        cwd=str(REPO_ROOT))
    assert res.returncode == 0, res.stderr[-500:]
    assert "--manifest" in res.stdout
    assert "--eye-revision" in res.stdout


def test_dev_wrapper_still_delegates():
    src = (REPO_ROOT / "scripts" / "recognize" / "serve_worker.py").read_text()
    assert "from majsoul_eye.worker.serve import main" in src
