"""Write CVAT-corrected YOLO labels back into datasets/<game>/yolo/labels/.

Feed it the zip (or extracted dir) you Exported from CVAT as "YOLO 1.1". Frames
were namespaced <game>__<stem> by cvat_export.py, so each corrected label file is
routed back to datasets/<game>/yolo/labels/<stem>.txt.  Non-destructive by default:
--dry-run shows what WOULD change; run without it to actually overwrite.

Run FROM THE REPO ROOT:

    PYTHONPATH=. $PY scripts/inspect/cvat_import.py cvat_export.zip --dry-run
    PYTHONPATH=. $PY scripts/inspect/cvat_import.py cvat_export.zip

Use --target-root DIR to write into a mirror tree (DIR/<game>/yolo/labels/...)
instead of the live datasets/ — handy for a safe diff before committing.
"""
from __future__ import annotations
import argparse
import tempfile
import zipfile
from pathlib import Path

META = {"obj.names", "obj.data", "train.txt", "test.txt", "val.txt"}


def find_label_files(root: Path):
    """All YOLO label .txt under the export, excluding metadata lists."""
    out = []
    for p in root.rglob("*.txt"):
        if p.name in META:
            continue
        out.append(p)
    return out


def target_for(name_stem: str, root: Path, target_root: Path | None) -> tuple[str, str, Path] | None:
    """precise_ai_run_1__000028 -> datasets/precise_ai_run_1/yolo/labels/000028.txt"""
    if "__" not in name_stem:
        return None
    game, stem = name_stem.split("__", 1)
    base = (target_root / game) if target_root else (root / "datasets" / game)
    return game, stem, base / "yolo" / "labels" / f"{stem}.txt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", type=Path, help="CVAT 'YOLO 1.1' export (.zip or extracted dir)")
    ap.add_argument("--root", type=Path, default=Path("."), help="repo root")
    ap.add_argument("--target-root", type=Path, default=None, help="write into DIR/<game>/... instead of live datasets/")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-new", action="store_true", help="also write labels for frames not already in the dataset")
    args = ap.parse_args()

    tmp = None
    src = args.export
    if args.export.is_file() and args.export.suffix == ".zip":
        tmp = tempfile.mkdtemp(prefix="cvat_import_")
        with zipfile.ZipFile(args.export) as z:
            z.extractall(tmp)
        src = Path(tmp)

    label_files = find_label_files(src)
    if not label_files:
        raise SystemExit(f"no label .txt files found under {src}")

    changed = created = skipped = unmatched = 0
    for lf in label_files:
        t = target_for(lf.stem, args.root, args.target_root)
        if t is None:
            unmatched += 1
            print(f"  ? unmatched (no '<game>__<stem>' name): {lf.name}")
            continue
        game, stem, dst = t
        new = lf.read_text(encoding="utf-8")
        exists = dst.exists()
        if not exists and not args.allow_new:
            skipped += 1
            print(f"  - skip (target not in dataset, use --allow-new): {game}/{stem}")
            continue
        old = dst.read_text(encoding="utf-8") if exists else None
        if old == new:
            continue
        action = "update" if exists else "create"
        print(f"  {action}: datasets/{game}/yolo/labels/{stem}.txt "
              f"({len((old or '').splitlines())} -> {len(new.splitlines())} boxes)")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(new, encoding="utf-8")
        if exists:
            changed += 1
        else:
            created += 1

    verb = "would change" if args.dry_run else "changed"
    print(f"\n{verb}: {changed} updated, {created} created, {skipped} skipped, {unmatched} unmatched "
          f"(of {len(label_files)} label files)")
    if args.dry_run:
        print("dry-run: nothing written. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
