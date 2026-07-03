"""purge_occlusion_frames: verdicts on the PROPER saved crops + drop routing.

The fix under test: purge classifies build_dataset's saved crops/<tile>/<seq>_<ci>.png
(the perspective-warped crop_box outputs) instead of re-cropping the raw YOLO AABB.
Frames are synthesised with saved crops whose top-left pixel encodes the stub
classifier's prediction (StubClf reads crop[0,0,0] as the class id)."""
import os, glob, importlib.util, shutil, sys
import numpy as np, cv2

_spec = importlib.util.spec_from_file_location("poc", "scripts/data/purge_occlusion_frames.py")
poc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(poc)

from majsoul_eye.tiles import NAME_TO_ID, TILE_NAMES


class StubClf:
    """Reads each crop's top-left pixel (blue channel) as the predicted class id."""
    def predict_proba(self, crops):
        out = np.zeros((len(crops), 38), np.float32)
        for i, c in enumerate(crops):
            out[i, int(c[0, 0, 0])] = 1.0
        return out


def _write_frame(ds, seq, boxes):
    """boxes: list of (gt_name, pred_name, sideways). Every box gets a YOLO line (distinct
    cx); non-sideways boxes also get a saved crop crops/<gt>/<seq>_<ci>.png (ci = per-frame
    non-sideways counter, as build_dataset names them) whose [0,0,0] pixel = NAME_TO_ID[pred]."""
    os.makedirs(f"{ds}/yolo/images", exist_ok=True)
    os.makedirs(f"{ds}/yolo/labels", exist_ok=True)
    img = np.full((100, 100, 3), 240, np.uint8)
    lines, ci = [], 0
    for k, (gt, pred, sideways) in enumerate(boxes):
        cx = 0.1 + 0.1 * k
        lines.append(f"{NAME_TO_ID[gt]} {cx:.3f} 0.5 0.05 0.05")
        if not sideways:
            cdir = f"{ds}/crops/{gt}"; os.makedirs(cdir, exist_ok=True)
            crop = np.full((16, 16, 3), 200, np.uint8)
            crop[0, 0, 0] = NAME_TO_ID[pred]
            cv2.imwrite(f"{cdir}/{seq}_{ci:03d}.png", crop)
            ci += 1
    cv2.imwrite(f"{ds}/yolo/images/{seq}.png", img)
    open(f"{ds}/yolo/labels/{seq}.txt", "w").write("\n".join(lines) + "\n")


def _run_apply(tmpdir):
    orig_cls, orig_argv = poc.TileClassifier, sys.argv
    poc.TileClassifier = lambda: StubClf()
    sys.argv = ["purge_occlusion_frames.py", "--datasets-dir", tmpdir, "--apply",
                "--tau", "0.5", "--max-bad", "2"]
    try:
        poc.main()
    finally:
        poc.TileClassifier, sys.argv = orig_cls, orig_argv


def test_plan_frame_routing(tmpdir="scratch_purge_t1"):
    shutil.rmtree(tmpdir, ignore_errors=True)
    ds = f"{tmpdir}/precise_fake"
    _write_frame(ds, "000001", [("8s", "8s", False)])            # good
    _write_frame(ds, "000002", [("8s", "3p", False)])            # bad (mismatch)
    clf = StubClf()
    d1 = poc.plan_frame(ds, "000001", clf, 0.5, 2)
    d2 = poc.plan_frame(ds, "000002", clf, 0.5, 2)
    assert d1[0] == "keep", d1
    assert d2[0] in ("drop_boxes", "drop_frame"), d2
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("test_plan_frame_routing OK")


def test_drop_boxes_surgical_clean_map(tmpdir="scratch_purge_t2"):
    """No sideways -> #crops == #yolo lines -> surgically drop ONLY the bad box's line and
    ONLY the bad crop file; the good same-class box/crop survive."""
    shutil.rmtree(tmpdir, ignore_errors=True)
    ds = f"{tmpdir}/precise_fake"; seq = "000003"
    _write_frame(ds, seq, [("8s", "8s", False), ("8s", "3p", False)])   # box0 good, box1 bad
    _run_apply(tmpdir)
    assert os.path.exists(f"{ds}/crops/8s/{seq}_000.png"), "good crop must survive"
    assert not os.path.exists(f"{ds}/crops/8s/{seq}_001.png"), "bad crop must be deleted"
    kept = [ln for ln in open(f"{ds}/yolo/labels/{seq}.txt").read().splitlines() if ln.strip()]
    assert len(kept) == 1 and kept[0].split()[1] == "0.100", kept   # box0 (cx0.1) survives
    assert os.path.exists(f"{ds}/yolo/images/{seq}.png"), "frame image must survive drop_boxes"
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("test_drop_boxes_surgical_clean_map OK")


def test_sideways_frame_conservative_drop(tmpdir="scratch_purge_t3"):
    """A sideways box makes #crops != #yolo lines -> crop<->line order unrecoverable, so the
    whole DETECTOR frame is dropped; only the bad crop is deleted, good crops survive."""
    shutil.rmtree(tmpdir, ignore_errors=True)
    ds = f"{tmpdir}/precise_fake"; seq = "000004"
    _write_frame(ds, seq, [("8s", "8s", False), ("5m", "5m", True), ("3p", "7s", False)])
    _run_apply(tmpdir)
    assert not os.path.exists(f"{ds}/yolo/images/{seq}.png"), "detector image must be dropped"
    assert not os.path.exists(f"{ds}/yolo/labels/{seq}.txt"), "detector label must be dropped"
    assert os.path.exists(f"{ds}/crops/8s/{seq}_000.png"), "good crop must survive"
    assert not os.path.exists(f"{ds}/crops/3p/{seq}_001.png"), "bad crop must be deleted"
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("test_sideways_frame_conservative_drop OK")


def test_idempotent(tmpdir="scratch_purge_t4"):
    """Applying twice equals applying once (no crash, stable state)."""
    shutil.rmtree(tmpdir, ignore_errors=True)
    ds = f"{tmpdir}/precise_fake"; seq = "000005"
    _write_frame(ds, seq, [("8s", "8s", False), ("8s", "3p", False)])
    _run_apply(tmpdir)
    label_after_1 = open(f"{ds}/yolo/labels/{seq}.txt").read()
    _run_apply(tmpdir)                                   # second pass must be a no-op
    assert open(f"{ds}/yolo/labels/{seq}.txt").read() == label_after_1, "second apply changed state"
    assert os.path.exists(f"{ds}/crops/8s/{seq}_000.png")
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("test_idempotent OK")


if __name__ == "__main__":
    test_plan_frame_routing()
    test_drop_boxes_surgical_clean_map()
    test_sideways_frame_conservative_drop()
    test_idempotent()
    print("ALL test_purge_occlusion OK")
