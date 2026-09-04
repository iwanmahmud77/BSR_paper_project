"""
Train EfficientNet-B0, MobileNetV2, Swin-Tiny, and ConvNeXt-V2-Tiny on the
severity classification task (Healthy/Mild/Moderate/Severe).

Extracted and parameterized from the "DIRECT IMAGE CLASSIFICATION" section
of BSR_YOLO_ML_pipeline_v11.ipynb (cells 45-60). Training logic (transforms,
optimizer, early stopping, checkpoint restore) is unchanged from the
notebook; hardcoded paths and per-model constants now come from
configs/classifier_train.yaml.

Usage:
    python 03_train_image_classifiers.py --config configs/classifier_train.yaml
"""
import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
                     ".JPG", ".JPEG", ".PNG"]


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_image_path(dataset_root: Path, source: str, image_name: str):
    image_name = str(image_name).strip()
    direct = dataset_root / image_name
    if direct.exists() and direct.is_file():
        return direct

    basename = Path(image_name).name
    stem = Path(basename).stem
    image_dir = dataset_root / str(source) / "images"

    candidate = image_dir / basename
    if candidate.exists() and candidate.is_file():
        return candidate

    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists() and candidate.is_file():
            return candidate

    matches = list(image_dir.glob(f"{stem}.*"))
    for match in matches:
        if match.is_file() and match.suffix.lower() in [e.lower() for e in IMAGE_EXTENSIONS]:
            return match

    return None


class BSRClassificationDataset(Dataset):
    def __init__(self, dataframe, class_to_idx, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(Path(row["image_path"])).convert("RGB")
        label = self.class_to_idx[row["label"]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def create_model(model_name, num_classes):
    if model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "Swin-Tiny":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)

    elif model_name == "ConvNeXt-V2-Tiny":
        try:
            import timm
            model = timm.create_model(
                "convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True, num_classes=num_classes)
        except Exception as e:
            print(f"timm unavailable / failed ({e}); falling back to torchvision ConvNeXt-Tiny (v1).")
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_training:
                loss.backward()
                optimizer.step()
        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def train_one_model(model_name, train_loader, valid_loader, device, num_classes,
                     epochs, lr, weight_decay, patience):
    print(f"\n{'='*70}\nTRAINING: {model_name}\n{'='*70}")
    model = create_model(model_name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
    best_val_acc, best_val_loss, best_state, epochs_no_improve = -1, float("inf"), None, 0
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        valid_loss, valid_acc = run_epoch(model, valid_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}")

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_state = copy.deepcopy(model.state_dict())

        if valid_loss < best_val_loss - 1e-4:
            best_val_loss = valid_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no val_loss improvement for {patience} epochs).")
                break

    model.load_state_dict(best_state)
    training_time = time.time() - start_time
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Training time: {training_time:.1f} seconds")
    return model, history, training_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--labels-csv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("saved_models"))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    dataset_root = Path(args.dataset_root or cfg["data"]["dataset_root"])
    assert dataset_root, "dataset_root must be set via --dataset-root or the config file"
    labels_csv = args.labels_csv or cfg["data"]["labels_csv"]

    class_order = cfg["data"]["class_order"]
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    num_classes = len(class_order)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(labels_csv)
    classification_df = df[["image", cfg["data"]["split_column"], cfg["data"]["label_column"]]].copy()
    classification_df.columns = ["image", "source", "label"]
    classification_df = classification_df[classification_df["label"].isin(class_order)].copy()

    classification_df["image_path"] = classification_df.apply(
        lambda row: resolve_image_path(dataset_root, row["source"], row["image"]), axis=1)
    missing = classification_df["image_path"].isna().sum()
    if missing:
        print(f"WARNING: {missing} images could not be resolved and will be dropped.")
    classification_df = classification_df.dropna(subset=["image_path"])

    img_cfg = cfg["image"]
    aug_cfg = cfg["train_augmentation"]
    train_transform = transforms.Compose([
        transforms.Resize((img_cfg["size"], img_cfg["size"])),
        transforms.RandomHorizontalFlip(p=aug_cfg["random_horizontal_flip_p"]),
        transforms.RandomRotation(aug_cfg["random_rotation_degrees"]),
        transforms.ColorJitter(**aug_cfg["color_jitter"]),
        transforms.ToTensor(),
        transforms.Normalize(img_cfg["normalize_mean"], img_cfg["normalize_std"]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((img_cfg["size"], img_cfg["size"])),
        transforms.ToTensor(),
        transforms.Normalize(img_cfg["normalize_mean"], img_cfg["normalize_std"]),
    ])

    train_df = classification_df[classification_df["source"] == "train"]
    valid_df = classification_df[classification_df["source"] == "valid"]

    train_cfg = cfg["training"]
    train_loader = DataLoader(
        BSRClassificationDataset(train_df, class_to_idx, train_transform),
        batch_size=train_cfg["batch_size"], shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(
        BSRClassificationDataset(valid_df, class_to_idx, eval_transform),
        batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available())

    args.out_dir.mkdir(exist_ok=True, parents=True)
    training_times = {}

    for model_spec in cfg["models"]:
        model_name = model_spec["name"]
        model, history, elapsed = train_one_model(
            model_name, train_loader, valid_loader, device, num_classes,
            epochs=train_cfg["num_epochs"], lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"], patience=train_cfg["patience"])

        training_times[model_name] = elapsed
        safe = model_name.replace("/", "-")
        torch.save(model.state_dict(), args.out_dir / f"{safe}.pth")
        with open(args.out_dir / f"{safe}_history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved: {args.out_dir / f'{safe}.pth'}")

    meta = {
        "models_to_train": [m["name"] for m in cfg["models"]],
        "class_order": class_order,
        "num_classes": num_classes,
        "image_size": img_cfg["size"],
        "seed": cfg.get("seed", 42),
        "training_times": training_times,
    }
    with open(args.out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\nAll models + metadata saved to:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
