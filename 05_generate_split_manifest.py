"""
Generate train/valid/test split manifests WITHOUT redistributing raw images.

Each manifest row records the split, the relative image path, the label
(severity class), and a SHA-256 hash of the image file. This lets anyone who
has independently obtained the (non-redistributable) raw images verify that
their local copy matches the exact split used in the paper, and lets
reviewers audit the split composition without ever needing the images
themselves.

Usage:
    python 05_generate_split_manifest.py \
        --dataset-root /path/to/Basal_dataset_v4 \
        --labels-csv severity_pseudolabels_all_splits.csv \
        --out-dir data/splits
"""
import argparse
import csv
import hashlib
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def find_image(images_dir: Path, stem: str):
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = list(images_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=Path,
                     help="Root folder containing train/, valid/, test/ subfolders")
    ap.add_argument("--labels-csv", required=True, type=Path,
                     help="Output of 01_generate_severity_labels.py "
                          "(severity_pseudolabels_all_splits.csv)")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    # df["image"] is stored as "<split>/<stem>" (see notebook cell 9)
    df["split"] = df["image"].apply(lambda s: s.split("/", 1)[0])
    df["stem"] = df["image"].apply(lambda s: s.split("/", 1)[1])

    for split in sorted(df["split"].unique()):
        split_df = df[df["split"] == split]
        images_dir = args.dataset_root / split / "images"

        rows = []
        missing = 0
        for _, row in split_df.iterrows():
            img_path = find_image(images_dir, row["stem"])
            if img_path is None:
                missing += 1
                rows.append({
                    "split": split,
                    "relative_path": None,
                    "filename": row["stem"],
                    "label": row.get("severity_class_rule"),
                    "sha256": None,
                    "status": "MISSING_AT_MANIFEST_TIME",
                })
                continue
            rows.append({
                "split": split,
                "relative_path": str(img_path.relative_to(args.dataset_root)),
                "filename": img_path.name,
                "label": row.get("severity_class_rule"),
                "sha256": sha256_of_file(img_path),
                "status": "OK",
            })

        out_path = args.out_dir / f"split_manifest_{split}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"[{split}] wrote {len(rows)} rows ({missing} missing) -> {out_path}")


if __name__ == "__main__":
    main()
