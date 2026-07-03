"""Package frames + their YOLO labels into a CVAT-importable dataset (YOLO 1.1),
so you can FIX bounding boxes in CVAT and write them back with cvat_import.py.

Frames are renamed  <game>__<stem>.png/.txt  so the round-trip is unambiguous
(the same stem e.g. 000028 exists in many games). cvat_import.py parses that
prefix to write each corrected label back to datasets/<game>/yolo/labels/.

Run FROM THE REPO ROOT:

    # a whole game
    PYTHONPATH=. $PY scripts/inspect/cvat_export.py --game precise_ai_run_1 --out cvat_pkg --zip
    # several games, capped
    PYTHONPATH=. $PY scripts/inspect/cvat_export.py --game precise_session5 --game precise_session6 --out cvat_pkg --limit 200 --zip
    # a hand-picked frame list (repo-root-relative image paths, e.g. FiftyOne rejects)
    PYTHONPATH=. $PY scripts/inspect/cvat_export.py --frames-list bad.txt --out cvat_pkg --zip

Then in CVAT (localhost:8080): Create task -> upload cvat_pkg.zip (or the images),
set the label list from obj.names, fix boxes, Export as "YOLO 1.1".
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import yaml


def load_names(data_yaml: Path) -> list[str]:
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = d["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return list(names)


def game_of(rel: Path) -> str:
    parts = rel.parts
    if "yolo" in parts:
        i = parts.index("yolo")
        if i > 0:
            return parts[i - 1]
    return parts[1] if len(parts) > 1 else "unknown"


def label_for(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def collect(root: Path, games: list[str], frames_list: Path | None, limit: int | None):
    """Yield (game, img_path, label_path) tuples."""
    items = []
    if frames_list:
        for line in frames_list.read_text(encoding="utf-8").splitlines():
            rel = line.strip()
            if rel:
                p = root / rel
                items.append((game_of(Path(rel)), p, label_for(p)))
    for g in games:
        img_dir = root / "datasets" / g / "yolo" / "images"
        for img in sorted(img_dir.glob("*.png")):
            items.append((g, img, label_for(img)))
    if limit:
        items = items[:limit]
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", action="append", default=[], help="dataset game dir name (repeatable)")
    ap.add_argument("--frames-list", type=Path, help="file of repo-root-relative image paths")
    ap.add_argument("--out", type=Path, required=True, help="output package directory")
    ap.add_argument("--data", type=Path, default=Path("datasets/detector/data.yaml"))
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--zip", action="store_true", help="also produce <out>.zip for CVAT upload")
    args = ap.parse_args()

    if not args.game and not args.frames_list:
        ap.error("give --game and/or --frames-list")

    names = load_names(args.data)
    items = collect(args.root, args.game, args.frames_list, args.limit)
    if not items:
        raise SystemExit("no frames matched")

    out = args.out
    data_dir = out / "obj_train_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_lines, copied, no_label = [], 0, 0
    for game, img, lbl in items:
        stem = f"{game}__{img.stem}"
        shutil.copy2(img, data_dir / f"{stem}.png")
        if lbl.exists():
            shutil.copy2(lbl, data_dir / f"{stem}.txt")
        else:
            (data_dir / f"{stem}.txt").write_text("", encoding="utf-8")
            no_label += 1
        train_lines.append(f"obj_train_data/{stem}.png")
        copied += 1

    (out / "obj.names").write_text("\n".join(names) + "\n", encoding="utf-8")
    (out / "obj.data").write_text(
        f"classes = {len(names)}\ntrain = train.txt\nnames = obj.names\nbackup = backup/\n",
        encoding="utf-8",
    )
    (out / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

    print(f"packaged {copied} frames into {out}/  ({no_label} had no existing label -> empty)")
    print(f"classes: {len(names)}")
    if args.zip:
        zip_path = shutil.make_archive(str(out), "zip", root_dir=str(out))
        print(f"zip: {zip_path}  (upload this to CVAT)")


if __name__ == "__main__":
    main()
