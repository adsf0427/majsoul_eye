"""Real-screenshot golden: the 16 user shots must keep reading the same board.

Why this file exists
--------------------
The wide-phone path was built, measured at 16/16, written up in STATUS -- and then
silently un-wired by a later refactor, because NOTHING loaded ``samples/``. The
evidence lived in prose. This turns it into a test.

The images themselves are 48 MB and stay out of git (``.gitignore``: ``samples/``);
the EXPECTATIONS are committed. Without the images the test skips loudly rather
than passing quietly, so a fresh clone is never told it verified something it did
not. Regenerate with::

    PYTHONPATH=. python tests/test_samples_golden.py --update
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

GOLDEN = Path(__file__).with_name("samples_golden.json")
SAMPLES = Path(__file__).resolve().parent.parent / "samples"
MANIFEST = "majsoul_eye/recognize/model-manifest.internal-v1.json"


def _images():
    if not SAMPLES.is_dir():
        return []
    return sorted(p for p in SAMPLES.iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg"))


def _read(path, runtime):
    import cv2
    import numpy as np

    from majsoul_eye.recognize.runtime import RecognitionContext

    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    data = runtime.recognize_bytes(body, RecognitionContext(
        "golden", "golden", digest, runtime.manifest.layout_id, True, None))
    draft = data["draft"]
    image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    from majsoul_eye.normalize import locate_anchor
    region = locate_anchor(image, runtime.detector.predict(image)).region

    hero = draft["players"][0]
    return {
        "sha256": digest,
        "board": [region.ox, region.oy, region.bw, region.bh],
        "issues": [issue["code"] for issue in data["issues"]],
        "hand": [tile["pai"] for tile in hero["hand"]],
        "drawn": (hero["drawnTile"] or {}).get("pai") if hero["drawnTile"] else None,
        "rivers": [len(player["rivers"]) for player in draft["players"]],
        "melds": [len(player["melds"]) for player in draft["players"]],
        "dora": [tile["pai"] for tile in draft["doraMarkers"]],
        "round": [draft["round"][key] for key in
                  ("gameLength", "bakaze", "kyoku", "honba", "kyotaku",
                   "leftTileCount", "seatWindSelf")],
        "scores": draft["round"]["scores"],
    }


def _runtime():
    from majsoul_eye.recognize.runtime import RecognitionRuntime
    return RecognitionRuntime.from_manifest(MANIFEST, device="cpu",
                                            eye_revision="golden",
                                            evaluation_mode=True)


def test_real_user_screenshots_still_read_the_same_board():
    images = _images()
    if not images:
        print("SKIP test_samples_golden: samples/ is absent (48 MB, out of git)")
        return
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert set(golden) == {p.name for p in images}, (
        "samples/ and the golden disagree on which screenshots exist")

    runtime = _runtime()
    for path in images:
        want = golden[path.name]
        got = _read(path, runtime)
        assert got["sha256"] == want["sha256"], f"{path.name}: the image itself changed"
        # The board rect is a fit, so allow sub-percent jitter; everything the user
        # would actually SEE must be exact.
        for index, (a, b) in enumerate(zip(got["board"], want["board"])):
            assert abs(a - b) <= max(4, 0.005 * want["board"][2]), \
                f"{path.name}: board rect drifted {got['board']} vs {want['board']}"
        for key in ("issues", "hand", "drawn", "rivers", "melds", "dora",
                    "round", "scores"):
            assert got[key] == want[key], f"{path.name}: {key} changed"
    print(f"test_samples_golden OK ({len(images)} real screenshots)")


def _update():
    images = _images()
    if not images:
        raise SystemExit("samples/ is absent; nothing to record")
    runtime = _runtime()
    GOLDEN.write_text(
        json.dumps({p.name: _read(p, runtime) for p in images},
                   indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"wrote {GOLDEN} ({len(images)} screenshots)")


if __name__ == "__main__":
    if "--update" in sys.argv:
        _update()
    else:
        test_real_user_screenshots_still_read_the_same_board()
