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


if __name__ == "__main__":
    test_plan_frame_routing()
    print("ALL test_purge_occlusion OK")
