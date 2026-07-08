"""Browse / clean the YOLO detector dataset in the FiftyOne GUI.

The detector dataset's train.txt/val.txt hold REPO-ROOT-RELATIVE POSIX image
paths (see build_detector_dataset.py). FiftyOne's built-in YOLOv5 importer would
mis-resolve those against data.yaml's `path:`, so we build the fo.Dataset by
hand: read the image lists, derive each label file (images/->labels/, .png->.txt),
and convert normalized YOLO (cx,cy,w,h) boxes to FiftyOne top-left [x,y,w,h].

Run FROM THE REPO ROOT (paths in the lists are relative to it):

    # sanity-check load without opening a browser
    PYTHONPATH=. $PY scripts/inspect/fiftyone_view.py --check
    # launch the GUI (filter by class, tag bad frames, then export a clean set)
    PYTHONPATH=. $PY scripts/inspect/fiftyone_view.py

Cleaning workflow: in the GUI, tag junk samples (default tag key you pick, e.g.
"reject"); then re-run with --export-clean <dir> to write a fresh detector-style
list of everything NOT tagged, ready to retrain.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

import yaml


def load_names(data_yaml: Path) -> dict:
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = d["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    return {int(k): v for k, v in names.items()}


def label_path_for(img: Path) -> Path:
    """.../yolo/images/000028.png -> .../yolo/labels/000028.txt"""
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def game_of(rel: str) -> str:
    """datasets/precise_ai_run_1/yolo/images/000028.png -> precise_ai_run_1.

    Flat review sets name frames ``<game>__<seq>.png`` (many games in one dir), so
    a ``__`` in the stem carries the game — prefer it over the (shared) parent dir."""
    stem = Path(rel).stem
    if "__" in stem:
        return stem.rsplit("__", 1)[0]
    parts = Path(rel).parts
    if "yolo" in parts:
        i = parts.index("yolo")
        if i > 0:
            return parts[i - 1]
    return parts[1] if len(parts) > 1 else "unknown"


def read_labels(label_file: Path, names: dict):
    """Parse one YOLO label file -> a FiftyOne label object, auto-detecting the
    row format: 5-field HBB rows -> fo.Detections; 9-field OBB rows (cls x1 y1
    .. x4 y4, build_dataset.py --obb) -> fo.Polylines (closed quads, so the
    App draws the true rotated box). A build emits one uniform format per file."""
    import fiftyone as fo
    dets, polys = [], []
    if not label_file.exists():
        return fo.Detections(detections=[])
    for line in label_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        f = line.split()
        cls = int(f[0])
        label = names.get(cls, str(cls))
        if len(f) == 9:
            pts = [(float(f[i]), float(f[i + 1])) for i in range(1, 9, 2)]
            polys.append(fo.Polyline(label=label, points=[pts], closed=True, filled=False))
        else:
            cx, cy, w, h = (float(x) for x in f[1:5])
            dets.append(fo.Detection(
                label=label,
                bounding_box=[cx - w / 2.0, cy - h / 2.0, w, h],  # top-left, normalized
            ))
    if polys:
        return fo.Polylines(polylines=polys)
    return fo.Detections(detections=dets)


def build_split(list_file: Path, names: dict, split: str, root: Path):
    import fiftyone as fo
    samples, missing = [], 0
    for line in list_file.read_text().splitlines():
        rel = line.strip()
        if not rel:
            continue
        img = (root / rel)
        if not img.exists():
            missing += 1
            continue
        s = fo.Sample(filepath=str(img.resolve()))
        s["ground_truth"] = read_labels(label_path_for(img), names)
        s["game"] = game_of(rel)
        s["split"] = split          # data field, NOT a tag — tags are reserved for user reject-marking
        samples.append(s)
    return samples, missing


def _repair_zero_count(ds):
    """Self-heal a mongod unclean-shutdown artifact (diagnosed 2026-07-07).

    FiftyOne's embedded mongod gets hard-killed on Windows session exit; a
    recently-created sample collection then keeps a WiredTiger METADATA count of
    0 even though its documents survived via the journal. find()/iteration/
    $count aggregation all see the samples, but the fast `count` command — which
    len(dataset) and the App's grid use — reports 0, so the GUI renders empty
    ('backs_sample' showed len 0 with 68 iterable samples). `validate`
    recomputes and repairs the metadata count in place."""
    if len(ds) > 0:
        return ds
    if next(iter(ds), None) is None:
        return ds                                    # genuinely empty
    import fiftyone.core.odm as foo
    foo.get_db_conn().command("validate", ds._doc.sample_collection_name)
    print(f"  repaired zero-count metadata on '{ds.name}' "
          f"(mongod unclean-shutdown artifact) -> {len(ds)} samples")
    return ds


def load_dataset(root: Path, data_yaml: Path, name: str, rebuild: bool = False):
    import fiftyone as fo
    # Reuse the persisted dataset so sample tags (e.g. "reject") survive across
    # runs — a fresh build would wipe them. --rebuild re-imports from disk (use
    # after editing labels on disk, e.g. via CVAT).
    if fo.dataset_exists(name) and not rebuild:
        ds = _repair_zero_count(fo.load_dataset(name))
        print(f"loaded existing FiftyOne dataset '{name}' ({len(ds)} samples; --rebuild to re-import from disk)")
        return ds, 0

    names = load_names(data_yaml)
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = root / d.get("path", ".")

    # Reuse-and-clear rather than delete_dataset(): on some envs (protobuf 4.x)
    # delete_dataset() crashes in its annotation-run cleanup. clear() just drops
    # the samples, which is all a rebuild needs.
    if fo.dataset_exists(name):
        ds = fo.load_dataset(name)
        ds.clear()
    else:
        ds = fo.Dataset(name)

    total_missing = 0
    for split, key in (("train", "train"), ("val", "val")):
        lf = base / d[key]
        if not lf.exists():
            print(f"! {split} list not found: {lf}")
            continue
        samples, missing = build_split(lf, names, split, root)
        ds.add_samples(samples)
        total_missing += missing
        print(f"  {split}: {len(samples)} samples ({missing} missing image files skipped)")

    ds.persistent = True
    return ds, total_missing


def print_stats(ds):
    from collections import Counter
    n_det = 0
    cls = Counter()
    per_game = Counter()
    for s in ds:
        per_game[s["game"]] += 1
        gt = s["ground_truth"]
        items = gt.detections if hasattr(gt, "detections") else gt.polylines
        for det in items:
            n_det += 1
            cls[det.label] += 1
    print(f"\nsamples: {len(ds)}   detections: {n_det}")
    print("per-game:", dict(sorted(per_game.items())))
    print("class distribution (sorted by count):")
    for k, v in sorted(cls.items(), key=lambda kv: -kv[1]):
        print(f"   {k:>5}: {v}")


def export_clean(ds, out_dir: Path, reject_tag: str, root: Path):
    """Write a fresh detector-style split (train.txt/val.txt) of the samples
    NOT tagged `reject_tag`, with repo-root-relative image paths preserved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = {"train": [], "val": []}
    dropped = 0
    for s in ds:
        if reject_tag in s.tags:
            dropped += 1
            continue
        split = s["split"] if s.has_field("split") else "train"
        rel = Path(s.filepath).resolve().relative_to(root.resolve()).as_posix()
        kept[split].append(rel)
    for split, lines in kept.items():
        (out_dir / f"{split}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"  wrote {out_dir / (split + '.txt')}  ({len(lines)} kept)")
    print(f"dropped {dropped} samples tagged '{reject_tag}'")
    print(f"point a new data.yaml's train/val at these lists to retrain on the cleaned set.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets/detector/data.yaml", help="detector data.yaml")
    ap.add_argument("--name", default="majsoul_eye_detector", help="FiftyOne dataset name")
    ap.add_argument("--root", default=".", help="repo root the list paths are relative to")
    ap.add_argument("--check", action="store_true", help="build + print stats, do not launch the GUI")
    ap.add_argument("--rebuild", action="store_true", help="re-import from disk, discarding tags (use after editing labels)")
    ap.add_argument("--port", type=int, default=5252)  # 5151 (FiftyOne default) is in Windows' reserved 5068-5167 range
    ap.add_argument("--export-clean", metavar="DIR", help="write train/val lists of samples NOT tagged --reject-tag")
    ap.add_argument("--reject-tag", default="reject", help="sample tag marking junk to drop on export")
    args = ap.parse_args()

    root = Path(args.root)
    ds, missing = load_dataset(root, Path(args.data), args.name, rebuild=args.rebuild)
    print_stats(ds)
    if missing:
        print(f"\n! {missing} image files referenced by the lists were missing on disk.")

    if args.export_clean:
        export_clean(ds, Path(args.export_clean), args.reject_tag, root)
        return
    if args.check:
        print("\n--check: dataset built OK, not launching GUI.")
        return

    import fiftyone as fo
    session = fo.launch_app(ds, port=args.port)
    print(f"\nFiftyOne running at http://localhost:{args.port}  (Ctrl-C to quit)")
    # wait(-1) = block until Ctrl-C. Plain wait() returns as soon as the browser
    # tab disconnects (incl. slow first connect / a reload), which read as "the
    # program quit by itself" on Windows.
    session.wait(-1)


if __name__ == "__main__":
    main()
