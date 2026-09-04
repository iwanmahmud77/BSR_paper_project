# [BSR palm oil project, IJIES journal] — Basal Stem Rot (BSR) severity detection & classification
Title:Proximal Remote Sensing of Oil Palm Basal Stem Rot using Deep Learning Vision Algorithms and Ground-Based Images
## Citation
[We will add citation link once accepted.]


This repository contains the full non-image pipeline used to produce the
tables and figures in the manuscript: YOLO-based lesion detection, rule-based
severity pseudo-labeling, KNN/Random Forest classification on detector
features, and direct image classification with EfficientNet-B0, MobileNetV2,
Swin-Tiny, and ConvNeXt-V2-Tiny.

> **Raw images are not redistributed** [raw data
> sharing agreement restrics public access, only samples of annotated images available for seeing]. Everything needed to
> audit and reproduce the experimental protocol without the raw images —
> split manifests, labeling rules, configs, seeds, and code — is provided
> here.  

## Repository structure

```
configs/                    Final hyperparameters used for every experiment
  severity_labeling.yaml    Feature weights + rule thresholds for pseudo-labels
  yolo_train.yaml           Detector training config (epochs, augmentation, seeds)
  classifier_train.yaml     Image-classifier training config (all 4 backbones)
data/splits/                Train/valid/test manifests (filename + SHA-256, no images)
scripts/
  01_generate_severity_labels.py   Detector labels -> severity_class_rule / _quartile
  02_train_yolo_detectors.py       YOLOv8n / YOLOv11n / YOLO26n, 10 seeds each
  03_train_image_classifiers.py    EfficientNet-B0 / MobileNetV2 / Swin-Tiny / ConvNeXt-V2-Tiny
  04_evaluate_image_classifiers.py Test-set metrics, confusion matrices, comparison table
  05_generate_split_manifest.py    Builds data/splits/*.csv from your local dataset copy
notebooks/                  Original exploratory notebooks (kept for narrative context;
                             scripts/ above are the authoritative, reproducible entry points)
results/                    Output CSVs corresponding to the tables/figures in the paper
```

## Environment

```bash
conda create -n bsr python=3.10 -y
conda activate bsr
pip install -r requirements.txt
```

We additionally provide `requirements-lock.txt` with the exact package
versions used to produce the reported results (`pip freeze` output from the
training environment) — regenerate and commit this from your actual training
machine before submission; do not leave `torch.__version__` / CUDA version
undocumented, since ConvNeXt-V2-Tiny's `timm` weight tag
(`convnextv2_tiny.fcmae_ft_in22k_in1k`) and Swin-Tiny's default weights are
version-sensitive.

Hardware used:  
see Table 7 in the manuscript.

## Getting the images

[raw data (images) sharing agreement restrics public access]

Once you have the images, place them so that `dataset_root` contains
`train/`, `valid/`, `test/` subfolders each with `images/` and `labels/`
(standard YOLO/Ultralytics layout), matching `data.yaml`.

## Reproducing the pipeline end to end

```bash
# 1. Generate severity pseudo-labels from YOLO box annotations
python scripts/01_generate_severity_labels.py \
    --config configs/severity_labeling.yaml \
    --dataset-root /path/to/Basal_dataset_v4

# 2. (Optional, only needed to re-verify against our exact split) build/verify
#    the split manifest against your local image copy
python scripts/05_generate_split_manifest.py \
    --dataset-root /path/to/Basal_dataset_v4 \
    --labels-csv severity_pseudolabels_all_splits.csv \
    --out-dir data/splits
# Diff the sha256 column against data/splits/split_manifest_*.csv already
# committed in this repo to confirm you have the identical split.

# 3. Train the YOLO detectors (YOLOv8n / YOLOv11n / YOLO26n, 10 seeds each)
python scripts/02_train_yolo_detectors.py --config configs/yolo_train.yaml

# 4. Train the four direct image classifiers
python scripts/03_train_image_classifiers.py --config configs/classifier_train.yaml

# 5. Evaluate on the held-out test split and reproduce the comparison table
python scripts/04_evaluate_image_classifiers.py --config configs/classifier_train.yaml
```

Each script writes its outputs (CSV tables, confusion matrices, training
curves) to `results/`.

## Random seeds

- Severity labeling / KNN / Random Forest / classifier training: seed `42`
  (`set_seed()` in `scripts/01_generate_severity_labels.py` and
  `scripts/03_train_image_classifiers.py`; see `configs/*.yaml`).
- YOLO detector training: seeds `1`–`10` independently per architecture
  (`configs/yolo_train.yaml: seeds.values`); we report mean ± std across
  seeds in the paper.

## Non-determinism notes

`torch.backends.cudnn.deterministic = True` /
`torch.backends.cudnn.benchmark = False` are set for the classifiers.
Full bitwise reproducibility across different GPU models/CUDA versions is
not guaranteed; results should match within the reported ± std across seeds.


## License

[We will add manuscript license.]
