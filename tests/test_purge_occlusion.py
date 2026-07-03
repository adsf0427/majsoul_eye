"""purge_occlusion_frames: routing + apply/idempotency on a tiny synthetic dataset."""
import os, glob, importlib.util
import numpy as np, cv2

# load the script as a module
_spec = importlib.util.spec_from_file_location("poc", "scripts/data/purge_occlusion_frames.py")
poc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(poc)

from majsoul_eye.tiles import NAME_TO_ID


class StubClf:
    def __init__(self, mapping): self.mapping = mapping   # crop-tag byte -> tile name
    def predict_proba(self, crops):
        out = np.zeros((len(crops), 38), np.float32)
        for i, c in enumerate(crops):
            out[i, NAME_TO_ID[self.mapping[int(c[0, 0, 0])]]] = 1.0
        return out


def _write(ds, seq, boxes):
    """boxes: list of (gt_cls_id, tag_byte). Writes image (tag encoded at each box's own
    top-left pixel, i.e. crop-local [0,0,0] once plan_frame crops it out via the box's
    normalized coords) + label. One box per row, all at distinct positions.

    NOTE: deviates from the brief's verbatim `img[0, 0] = tag` (always the full image's
    origin), which is never inside the parsed crop here (cy is fixed at 0.5, so the box
    never reaches row 0) -> plan_frame's crop always sampled background (240) at [0,0,0],
    making StubClf.predict_proba KeyError(240) regardless of implementation. Fixed by
    writing the tag at the box's actual top-left pixel using the same coord math as
    _read_boxes, so it lands at the crop's local [0,0,0] as the classifier expects.
    """
    os.makedirs(f"{ds}/yolo/images", exist_ok=True)
    os.makedirs(f"{ds}/yolo/labels", exist_ok=True)
    img = np.full((100, 100, 3), 240, np.uint8)
    lines = []
    for k, (cls, tag) in enumerate(boxes):
        cx, cy, bw, bh = 0.1 + 0.1 * k, 0.5, 0.05, 0.05
        x0 = max(0, int((cx - bw / 2) * 100)); y0 = max(0, int((cy - bh / 2) * 100))
        img[y0, x0] = tag
        lines.append(f"{cls} {cx:.3f} {cy} {bw} {bh}")
    cv2.imwrite(f"{ds}/yolo/images/{seq}.png", img)
    open(f"{ds}/yolo/labels/{seq}.txt", "w").write("\n".join(lines) + "\n")


def test_plan_frame_routing(tmpdir="scratch_purge_test"):
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    ds = f"{tmpdir}/precise_fake"
    # good frame: gt 8s, classifier sees 8s
    _write(ds, "000001", [(NAME_TO_ID["8s"], NAME_TO_ID["8s"])])
    # bad frame: gt 8s, classifier sees 3p
    _write(ds, "000002", [(NAME_TO_ID["8s"], NAME_TO_ID["3p"])])
    clf = StubClf({NAME_TO_ID["8s"]: "8s", NAME_TO_ID["3p"]: "3p"})
    d1 = poc.plan_frame(f"{ds}/yolo/images/000001.png", f"{ds}/yolo/labels/000001.txt", clf, 0.5, 2)
    d2 = poc.plan_frame(f"{ds}/yolo/images/000002.png", f"{ds}/yolo/labels/000002.txt", clf, 0.5, 2)
    assert d1[0] == "keep", d1
    assert d2[0] in ("drop_boxes", "drop_frame"), d2
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("test_plan_frame_routing OK")


def test_drop_boxes_partial_class_keeps_good_crop(tmpdir="scratch_purge_test2"):
    """Regression for the over-deletion bug: a frame with TWO 8s boxes where only
    ONE is bad must NOT glob-delete both crops/8s/<seq>_*.png files — only the whole
    class is safe to glob-delete when EVERY box of that class in the frame is bad.
    Here only box index 1 is bad, so both crop files (good 000 + bad 001) must be
    left untouched, and only the bad box's YOLO label line must be dropped."""
    import shutil, sys
    shutil.rmtree(tmpdir, ignore_errors=True)
    ds = f"{tmpdir}/precise_fake"
    seq = "000003"
    # two 8s boxes: box0 good (classifier sees 8s), box1 bad (classifier sees 3p)
    _write(ds, seq, [(NAME_TO_ID["8s"], NAME_TO_ID["8s"]), (NAME_TO_ID["8s"], NAME_TO_ID["3p"])])

    # synthetic crops/ dir mimicking build_dataset's per-frame global counter naming
    os.makedirs(f"{ds}/crops/8s", exist_ok=True)
    good_crop = f"{ds}/crops/8s/{seq}_000.png"
    bad_crop = f"{ds}/crops/8s/{seq}_001.png"
    cv2.imwrite(good_crop, np.full((10, 10, 3), 200, np.uint8))
    cv2.imwrite(bad_crop, np.full((10, 10, 3), 50, np.uint8))

    stub = StubClf({NAME_TO_ID["8s"]: "8s", NAME_TO_ID["3p"]: "3p"})
    orig_clf_cls = poc.TileClassifier
    orig_argv = sys.argv
    poc.TileClassifier = lambda: stub
    sys.argv = ["purge_occlusion_frames.py", "--datasets-dir", tmpdir, "--apply",
                "--tau", "0.5", "--max-bad", "2"]
    try:
        poc.main()
    finally:
        poc.TileClassifier = orig_clf_cls
        sys.argv = orig_argv

    assert os.path.exists(good_crop), "good 8s crop must survive when only 1 of 2 boxes is bad"
    label_path = f"{ds}/yolo/labels/{seq}.txt"
    kept_lines = [ln for ln in open(label_path, encoding="utf-8").read().splitlines() if ln.strip()]
    assert len(kept_lines) == 1, kept_lines
    # the surviving line is box0's (cx=0.100); box1's (cx=0.200) must be gone
    assert kept_lines[0].split()[1] == "0.100", kept_lines

    shutil.rmtree(tmpdir, ignore_errors=True)
    print("test_drop_boxes_partial_class_keeps_good_crop OK")


if __name__ == "__main__":
    test_plan_frame_routing()
    test_drop_boxes_partial_class_keeps_good_crop()
    print("ALL test_purge_occlusion OK")
