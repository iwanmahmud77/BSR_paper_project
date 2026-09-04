"""
Train YOLOv8n / YOLOv11n / YOLO26n detectors, each across multiple
independent seeds, and produce the mean +/- std comparison table used in
the manuscript.

Extracted and parameterized from BSR_YOLO_train_v5_multi.ipynb
(cells 4, 6, 8, 10, 16, 17, 19). Training/aggregation logic is unchanged
from the notebook; hardcoded local paths now come from --dataset-root /
configs/yolo_train.yaml.

Usage:
    python 02_train_yolo_detectors.py \
        --config configs/yolo_train.yaml \
        --dataset-root /path/to/Basal_dataset_v4 \
        --models YOLOv8n YOLOv11n YOLO26n
"""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO


def get_params_and_flops(model, imgsz):
    try:
        info = model.info(verbose=False)
        if info is not None:
            _, n_params, _, flops = info
            return n_params, flops
    except Exception:
        pass
    from ultralytics.utils.torch_utils import get_flops, get_num_params
    return get_num_params(model.model), get_flops(model.model, imgsz=imgsz)


def compute_f1_macro(metrics):
    p = np.asarray(metrics.box.p, dtype=float)
    r = np.asarray(metrics.box.r, dtype=float)
    f1_per_class = 2 * p * r / (p + r + 1e-16)
    return float(np.mean(f1_per_class)), f1_per_class


def train_one_model(model_label, seed, model_configs, data_yaml_path, class_names,
                     project_dir, epochs, imgsz, batch, patience, augmentation_cfg,
                     force_retrain=False):
    cfg = model_configs[model_label]
    run_name = f'{cfg["run_base_name"]}_seed{seed}'
    run_dir = os.path.join(project_dir, run_name)
    summary_path = os.path.join(run_dir, "summary.json")

    if os.path.exists(summary_path) and not force_retrain:
        print(f"'{model_label}' seed={seed} already trained — found {summary_path}. Skipping.")
        return

    print(f"\n{'='*60}\nTraining {model_label} (seed={seed}) [{cfg['weights_file']}]\n{'='*60}\n")
    model = YOLO(cfg["weights_file"])

    start = time.time()
    model.train(
        data=str(data_yaml_path), epochs=epochs, imgsz=imgsz, batch=batch, device=0,
        project=project_dir, name=run_name, patience=patience, exist_ok=True,
        seed=seed, workers=0, **augmentation_cfg,
    )
    elapsed_min = (time.time() - start) / 60
    print(f"\n{model_label} (seed={seed}) finished training in {elapsed_min:.1f} minutes")

    metrics = model.val(data=str(data_yaml_path), imgsz=imgsz, split="val")
    n_params, flops = get_params_and_flops(model, imgsz)
    f1_macro, f1_per_class_arr = compute_f1_macro(metrics)

    per_class_map = {class_names[i]: round(float(metrics.box.maps[i]), 4) for i in range(len(class_names))}
    per_class_f1 = {class_names[i]: round(float(f1_per_class_arr[i]), 4) for i in range(len(class_names))}

    summary = {
        "model_label": model_label, "seed": seed, "weights_file": cfg["weights_file"], "run_name": run_name,
        "train_time_min": round(elapsed_min, 2), "params_M": round(n_params / 1e6, 3),
        "flops_G": round(flops, 3), "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4), "f1_macro": round(f1_macro, 4),
        "map50": round(float(metrics.box.map50), 4), "map50_95": round(float(metrics.box.map), 4),
        "inference_ms": round(metrics.speed["inference"], 3),
        "per_class_map50_95": per_class_map, "per_class_f1": per_class_f1,
        "epochs_configured": epochs, "imgsz": imgsz, "batch": batch,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")


def load_all_available_results(model_configs, seeds, project_dir):
    results = defaultdict(list)
    for model_label, cfg in model_configs.items():
        for seed in seeds:
            summary_path = os.path.join(project_dir, f'{cfg["run_base_name"]}_seed{seed}', "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    results[model_label].append(json.load(f))
    return dict(results)


def fmt_mean_std(mean, std, decimals=3):
    return f"{mean:.{decimals}f} \u00b1 {std:.{decimals}f}"


def build_comparison_table(all_results):
    metric_keys = ["precision", "recall", "f1_macro", "map50", "map50_95", "inference_ms", "train_time_min"]
    agg_rows = []
    for model_label, runs in all_results.items():
        row = {"Model": model_label, "N seeds": len(runs)}
        for key in metric_keys:
            vals = np.array([r[key] for r in runs], dtype=float)
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        row["Params (M)"] = runs[0]["params_M"]
        row["FLOPs (G)"] = runs[0]["flops_G"]
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)

    display_cols = {
        "Precision": "precision", "Recall": "recall", "F1 (macro)": "f1_macro",
        "mAP50": "map50", "mAP50-95": "map50_95",
        "Inference (ms/img)": "inference_ms", "Train time (min)": "train_time_min",
    }
    comparison_df = agg_df[["Model", "N seeds", "Params (M)", "FLOPs (G)"]].copy()
    for display_name, key in display_cols.items():
        comparison_df[display_name] = agg_df.apply(
            lambda row: fmt_mean_std(row[f"{key}_mean"], row[f"{key}_std"]), axis=1)
    return agg_df, comparison_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--models", nargs="+", default=None,
                     help="Subset of configs/yolo_train.yaml: models keys to train (default: all)")
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(args.dataset_root or cfg["data"]["dataset_root"])
    assert dataset_root, "dataset_root must be set via --dataset-root or the config file"

    original_yaml_path = dataset_root / "data.yaml"
    with open(original_yaml_path) as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg["names"]

    # Fix image paths to be absolute; keep nc/names as-is.
    data_cfg["train"] = str(dataset_root / "train" / "images")
    data_cfg["val"] = str(dataset_root / "valid" / "images")
    data_cfg["test"] = str(dataset_root / "test" / "images")
    fixed_yaml_path = dataset_root / "data_fixed.yaml"
    with open(fixed_yaml_path, "w") as f:
        yaml.dump(data_cfg, f, default_flow_style=False)
    print(f"Wrote fixed yaml to: {fixed_yaml_path}")

    train_cfg = cfg["training"]
    project_dir = str(dataset_root / "runs" / "detect")
    seeds = cfg["seeds"]["values"]
    model_configs = cfg["models"]
    models_to_train = args.models or list(model_configs.keys())

    for model_label in models_to_train:
        for seed in seeds:
            train_one_model(
                model_label, seed, model_configs, fixed_yaml_path, class_names, project_dir,
                epochs=train_cfg["epochs"], imgsz=train_cfg["imgsz"], batch=train_cfg["batch"],
                patience=train_cfg["patience"], augmentation_cfg=cfg["augmentation"],
                force_retrain=args.force_retrain)

    all_results = load_all_available_results(model_configs, seeds, project_dir)
    for model_label, runs in all_results.items():
        print(f"{model_label}: {len(runs)}/{len(seeds)} seed runs completed")

    agg_df, comparison_df = build_comparison_table(all_results)
    agg_df.to_csv(args.out_dir / "model_comparison_multiseed_raw.csv", index=False)
    comparison_df.to_csv(args.out_dir / "model_comparison_multiseed.csv", index=False)
    print("\nSaved:", args.out_dir / "model_comparison_multiseed.csv")
    print(comparison_df)


if __name__ == "__main__":
    main()
