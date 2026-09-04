"""
Generate severity pseudo-labels (Healthy/Mild/Moderate/Severe) from YOLO
detection box annotations.

Extracted and parameterized from BSR_YOLO_ML_pipeline_v11.ipynb (cells 0-19).
Logic is unchanged from the notebook; only the hardcoded local paths have
been replaced with --config / --dataset-root arguments.

Usage:
    python 01_generate_severity_labels.py \
        --config configs/severity_labeling.yaml \
        --dataset-root /path/to/Basal_dataset_v4
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# Class-name -> feature mapping (data.yaml `names`). Confirm these match
# your dataset's data.yaml before running — see configs/severity_labeling.yaml.
FRUITING_BODY_CLASS_NAMES = {"mushroom", "white-button"}
ROT_CLASS_NAME = "rot"


def load_class_names(data_yaml_path):
    with open(data_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]
    return names


def resolve_split_dirs(data_yaml_path, split, dataset_root):
    with open(data_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    split_key = {"train": "train", "val": "val", "valid": "val", "test": "test"}[split]
    images_dir = Path(cfg[split_key])
    if not images_dir.is_absolute() or not images_dir.exists():
        folder_name = {"train": "train", "val": "valid", "test": "test"}[split_key]
        candidate = Path(dataset_root) / folder_name / "images"
        if candidate.exists():
            images_dir = candidate
    labels_dir = Path(str(images_dir).replace("images", "labels"))
    return images_dir, labels_dir


def parse_label_file(label_path, class_names):
    """Class-name matching ("mushroom"/"white-button" -> fruiting body,
    "rot" -> rot instance) is hardcoded to this project's data.yaml
    `names` list. Update FRUITING_BODY_CLASS_NAMES / ROT_CLASS_NAME below
    if your class names differ."""
    fruiting_body_count = 0
    rot_instance_count = 0
    rot_area_ratio = 0.0
    box_count = 0

    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                w = float(parts[3])
                h = float(parts[4])
                area = w * h  # YOLO w*h is already normalized to image area

                if cls_id < 0 or cls_id >= len(class_names):
                    continue
                cls_name = class_names[cls_id]

                box_count += 1
                if cls_name in FRUITING_BODY_CLASS_NAMES:
                    fruiting_body_count += 1
                elif cls_name == ROT_CLASS_NAME:
                    rot_instance_count += 1
                    rot_area_ratio += area

    return {
        "box_count": box_count,
        "fruiting_body_count": fruiting_body_count,
        "rot_instance_count": rot_instance_count,
        "rot_area_ratio": rot_area_ratio,
    }


def rule_based_class(feat, moderate_min, severe_min, severe_fruiting_count):
    """Anchored to MPOB stem-rot percentage bands."""
    fb, rot_n, area = feat["fruiting_body_count"], feat["rot_instance_count"], feat["rot_area_ratio"]
    if fb == 0 and rot_n == 0:
        return "Healthy"
    if area >= severe_min or fb >= severe_fruiting_count:
        return "Severe"
    if area >= moderate_min:
        return "Moderate"
    return "Mild"


def compute_severity_score(feat, norm_stats, alpha, beta, gamma):
    def norm(key):
        lo, hi = norm_stats[key]
        return 0.0 if hi - lo < 1e-9 else (feat[key] - lo) / (hi - lo)
    return (alpha * norm("rot_area_ratio")
            + beta * norm("fruiting_body_count")
            + gamma * norm("rot_instance_count"))


def quartile_class(scores):
    """Dataset-own quartiles (nearest-rank on sorted scores, matching the
    original notebook exactly rather than numpy's interpolated percentile)."""
    sorted_scores = sorted(scores)
    n = len(sorted_scores)

    def pct(p):
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return sorted_scores[idx]

    q1, q2, q3 = pct(0.25), pct(0.50), pct(0.75)

    def classify(score):
        if score <= q1:
            return "Healthy"
        if score <= q2:
            return "Mild"
        if score <= q3:
            return "Moderate"
        return "Severe"

    return classify, (q1, q2, q3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dataset-root", type=Path, default=None,
                     help="Overrides configs/severity_labeling.yaml: data.dataset_root")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(42)

    dataset_root = args.dataset_root or cfg["data"]["dataset_root"]
    assert dataset_root, "dataset_root must be set via --dataset-root or the config file"
    dataset_root = Path(dataset_root)
    data_yaml = dataset_root / "data.yaml"

    class_names = load_class_names(data_yaml)
    splits = cfg["data"]["splits_to_process"]
    image_exts = set(cfg["data"]["image_extensions"])

    split_dirs = [(s, *resolve_split_dirs(data_yaml, s, dataset_root)) for s in splits]
    for split_name, images_dir, labels_dir in split_dirs:
        status = "OK" if labels_dir.exists() else "MISSING"
        print(f"[{split_name:6s}] images: {images_dir}  |  labels: {labels_dir}  ({status})")

    records = []
    for split_name, images_dir, labels_dir in split_dirs:
        if not labels_dir.exists():
            print(f"Skipping split '{split_name}' -- labels dir not found: {labels_dir}")
            continue
        if images_dir.exists():
            for p in sorted(images_dir.iterdir()):
                if p.suffix.lower() in image_exts:
                    records.append({
                        "image": f"{split_name}/{p.stem}",
                        "label_path": labels_dir / f"{p.stem}.txt",
                        "source": split_name,
                    })

    assert records, "No images/labels found -- check dataset_root / data.yaml"

    rows = []
    for rec in records:
        feats = parse_label_file(rec["label_path"], class_names)
        feats["image"] = rec["image"]
        feats["source"] = rec["source"]
        rows.append(feats)

    fw = cfg["feature_weights"]
    rt = cfg["rule_thresholds"]

    norm_stats = {
        key: (min(r[key] for r in rows), max(r[key] for r in rows))
        for key in ("rot_area_ratio", "fruiting_body_count", "rot_instance_count")
    }

    scores = []
    for r in rows:
        r["severity_score"] = compute_severity_score(
            r, norm_stats, fw["alpha_rot_area"], fw["beta_fruiting_body"], fw["gamma_rot_instances"])
        r["severity_class_rule"] = rule_based_class(
            r, rt["moderate_min_rot_area_ratio"], rt["severe_min_rot_area_ratio"],
            rt["severe_fruiting_body_count"])
        scores.append(r["severity_score"])

    lo, hi = min(scores), max(scores)
    for r in rows:
        r["severity_score_norm"] = 0.0 if hi - lo < 1e-9 else (r["severity_score"] - lo) / (hi - lo)

    classify_fn, (q1, q2, q3) = quartile_class(scores)
    for r in rows:
        r["severity_class_quartile"] = classify_fn(r["severity_score"])
    print(f"Quartile cut points: Q1={q1:.4f}  Q2={q2:.4f}  Q3={q3:.4f}")

    df = pd.DataFrame(rows)[[
        "image", "source", "box_count", "fruiting_body_count", "rot_instance_count",
        "rot_area_ratio", "severity_score", "severity_score_norm",
        "severity_class_rule", "severity_class_quartile",
    ]]

    out_csv = cfg["output"]["csv_path"]
    df.to_csv(out_csv, index=False)
    print(f"Saved: {Path(out_csv).resolve()}")
    print(df["severity_class_rule"].value_counts())


if __name__ == "__main__":
    main()
