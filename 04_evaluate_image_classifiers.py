"""
Load the checkpoints saved by 03_train_image_classifiers.py, evaluate on the
held-out test split, and reproduce the comparison table / confusion matrices.

Usage:
    python 04_evaluate_image_classifiers.py \
        --config configs/classifier_train.yaml \
        --checkpoints-dir saved_models \
        --out-dir results
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)
from torch.utils.data import DataLoader
from torchvision import transforms

# Reuse dataset / model-factory code from the training script (03_*.py isn't
# a valid Python module name because it starts with a digit, so we load it
# by file path instead of a normal import).
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "train_classifiers", Path(__file__).parent / "03_train_image_classifiers.py")
train_classifiers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_classifiers)


def evaluate_test_set(model, loader, device, num_classes, class_order):
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    start_time = time.time()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            all_true.extend(labels.numpy())
            all_pred.extend(predictions)
            all_prob.append(probabilities.cpu().numpy())

    inference_time = time.time() - start_time
    y_true, y_pred, y_prob = np.array(all_true), np.array(all_pred), np.vstack(all_prob)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "inference_time": inference_time,
    }
    report = classification_report(
        y_true, y_pred, labels=range(num_classes), target_names=class_order,
        digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    return metrics, report, cm, y_true, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--labels-csv", type=Path, default=None)
    ap.add_argument("--checkpoints-dir", type=Path, default=Path("saved_models"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    with open(args.checkpoints_dir / "meta.json") as f:
        meta = json.load(f)

    class_order = meta["class_order"]
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    num_classes = meta["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = Path(args.dataset_root or cfg["data"]["dataset_root"])
    labels_csv = args.labels_csv or cfg["data"]["labels_csv"]

    df = pd.read_csv(labels_csv)
    classification_df = df[["image", cfg["data"]["split_column"], cfg["data"]["label_column"]]].copy()
    classification_df.columns = ["image", "source", "label"]
    classification_df = classification_df[classification_df["label"].isin(class_order)].copy()
    classification_df["image_path"] = classification_df.apply(
        lambda row: train_classifiers.resolve_image_path(dataset_root, row["source"], row["image"]),
        axis=1)
    classification_df = classification_df.dropna(subset=["image_path"])
    test_df = classification_df[classification_df["source"] == "test"]

    img_cfg = cfg["image"]
    eval_transform = transforms.Compose([
        transforms.Resize((img_cfg["size"], img_cfg["size"])),
        transforms.ToTensor(),
        transforms.Normalize(img_cfg["normalize_mean"], img_cfg["normalize_std"]),
    ])
    test_loader = DataLoader(
        train_classifiers.BSRClassificationDataset(test_df, class_to_idx, eval_transform),
        batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)

    rows = []
    for model_name in meta["models_to_train"]:
        safe = model_name.replace("/", "-")
        model = train_classifiers.create_model(model_name, num_classes)
        model.load_state_dict(torch.load(args.checkpoints_dir / f"{safe}.pth", map_location=device))
        model = model.to(device)

        metrics, report, cm, y_true, y_pred = evaluate_test_set(
            model, test_loader, device, num_classes, class_order)

        print(f"\n{'='*70}\n{model_name}\n{'='*70}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        print("\n" + report)

        np.savetxt(args.out_dir / f"confusion_matrix_{safe}.csv", cm, delimiter=",", fmt="%d")
        rows.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Macro Precision": metrics["macro_precision"],
            "Macro Recall": metrics["macro_recall"],
            "Macro F1": metrics["macro_f1"],
            "Weighted F1": metrics["weighted_f1"],
            "Training Time (s)": meta["training_times"].get(model_name),
            "Test Inference Time (s)": metrics["inference_time"],
        })

    comparison = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
    comparison.to_csv(args.out_dir / "image_classification_comparison.csv", index=False)
    print("\nSaved comparison table to", args.out_dir / "image_classification_comparison.csv")
    print(comparison)


if __name__ == "__main__":
    main()
