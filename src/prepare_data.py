"""
Build a balanced, reproducible train/val/test split from the PlantVillage dataset.

Why this script exists:
- The full dataset (~20k images) is too big for fast CPU experiments, so we take a
  fixed number of images per class (default 100).
- We split BEFORE any training so the test set is never seen during training or
  tuning (prevents data leakage).
- A fixed random seed makes the sample reproducible: same seed -> same images.

Usage (from the project folder):
    python src/prepare_data.py
    python src/prepare_data.py --per-class 100 --seed 42 --source "C:\\path\\to\\PlantVillage"
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path(r"C:\Users\krish\datasets\plantvillage\PlantVillage")
DEFAULT_DEST = PROJECT_ROOT / "data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")

# The Kaggle upload contains a duplicate nested "PlantVillage" folder with the same
# 15 classes. Skipping it avoids counting images twice and leaking test images into train.
SKIP_FOLDERS = {"PlantVillage"}


def list_class_folders(source: Path) -> list[Path]:
    folders = [p for p in source.iterdir() if p.is_dir() and p.name not in SKIP_FOLDERS]
    return sorted(folders, key=lambda p: p.name)


def list_images(folder: Path) -> list[Path]:
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images, key=lambda p: p.name)  # sort first so shuffling is reproducible


def split_counts(n: int, train_frac: float, val_frac: float) -> tuple[int, int, int]:
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced PlantVillage split.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source folder not found: {args.source}")

    class_folders = list_class_folders(args.source)
    if not class_folders:
        raise SystemExit(f"No class folders found in {args.source}")

    # Start clean so re-running never mixes old and new splits.
    for split in SPLITS:
        shutil.rmtree(args.dest / split, ignore_errors=True)

    rng = random.Random(args.seed)
    n_train, n_val, n_test = split_counts(args.per_class, args.train_frac, args.val_frac)
    manifest_rows = []

    print(f"Source: {args.source}")
    print(f"Classes found: {len(class_folders)}  |  per class: {args.per_class} "
          f"-> train {n_train} / val {n_val} / test {n_test}\n")

    for folder in class_folders:
        images = list_images(folder)
        if len(images) < args.per_class:
            raise SystemExit(f"{folder.name} has only {len(images)} images (< {args.per_class}).")

        rng.shuffle(images)
        chosen = images[: args.per_class]
        assignments = (
            [("train", p) for p in chosen[:n_train]]
            + [("val", p) for p in chosen[n_train : n_train + n_val]]
            + [("test", p) for p in chosen[n_train + n_val :]]
        )

        for split, src in assignments:
            out_dir = args.dest / split / folder.name
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out_dir / src.name)
            manifest_rows.append({"split": split, "class": folder.name, "file": src.name})

        print(f"  {folder.name:<45} available {len(images):>5}  -> copied {len(chosen)}")

    manifest_path = args.dest / "split_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class", "file"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    totals = {s: sum(1 for r in manifest_rows if r["split"] == s) for s in SPLITS}
    print(f"\nDone. train={totals['train']}  val={totals['val']}  test={totals['test']}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
