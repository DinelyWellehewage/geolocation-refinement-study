# Deep Learning Final Project — Image Geolocation

## Overview

This repository contains the final solution for the **Image Geolocation Challenge**.

Given a 512 × 512 street-level image from one of 12 European countries, the model predicts its geographic coordinates as latitude and longitude.

The final approach combines:

- geographic cluster classification,
- auxiliary country classification,
- and bounded local coordinate refinement on the 3D unit sphere.

The final model contains **4,385,215 parameters**, remaining below the challenge limit of 5,000,000 parameters.

---

## Final Method

The final model is trained in two stages.

### Stage 1 — Geographic Cluster Classification

Training coordinates are first converted from latitude/longitude to 3D unit-sphere coordinates.

KMeans is then applied to the training coordinates to create **128 geographic clusters**.

The Stage 1 model consists of:

- MobileNetV3-Large backbone,
- 128-class geographic cluster prediction head,
- 12-class auxiliary country prediction head.

The MobileNetV3-Large backbone is initialized using publicly available ImageNet pretrained weights.

The country head provides auxiliary supervision during training.

Without residual refinement, inference selects the most likely geographic cluster and uses the corresponding cluster center as the predicted location.

### Stage 2 — Spherical Residual Refinement

A three-dimensional residual head is added to the Stage 1 model.

For a training location with unit-sphere coordinate `true_xyz` and assigned geographic cluster center `center_xyz`, the residual target is:

```text
(true_xyz - center_xyz) / 0.05
```

The residual head predicts three bounded values using `tanh`.

At inference time:

```text
predicted_xyz =
    predicted_cluster_center
    + 0.05 * predicted_residual
```

The resulting vector is normalized back onto the unit sphere and converted to latitude and longitude.

The residual head is zero-initialized. The MobileNetV3 backbone, cluster head, and country head are initialized from the trained Stage 1 K=128 + country checkpoint.

The complete network is then jointly fine-tuned.

This gives the model two complementary localization mechanisms:

1. **Geographic cluster classification** provides coarse localization.
2. **Spherical residual prediction** provides local coordinate refinement around the predicted cluster center.

---

## Validation Protocol

A held-out subset of the provided labeled training data was used during model development.

The split is stratified by country using random seed 42.

The resulting development split contains:

```text
Training images:   9,406
Validation images: 2,352
Total:            11,758
```

KMeans geographic clusters used for validation are fitted only on the training portion of this split.

The hidden public-holdout labels are not used for model selection or validation.

---

## Validation Evaluation

The validation artifacts required by the evaluation scripts are included in the repository:

```text
geo_dataset/train_cluster128.csv
geo_dataset/val_cluster128.csv
geo_dataset/cluster128_centers_xyz.npy
```

### Recreate the Validation Split

The validation split and K=128 cluster assignments can also be recreated with:

```bash
python3 -m src.datasets.split

python3 -m src.create_cluster_variants \
  --clusters 128 \
  --save-files
```

### Evaluate the K=128 + Country Baseline

```bash
python3 -m src.evaluate_validation \
  --checkpoint checkpoints/cluster_country/validation_k128_cw020.pth \
  --num-clusters 128 \
  --num-countries 12 \
  --batch-size 64 \
  --num-workers 4
```

### Evaluate the Final Residual Architecture

```bash
python3 -m src.evaluate_residual_validation \
  --checkpoint checkpoints/cluster_country_residual_xyz/k128_residual_xyz_baseline_init_rw1.pth \
  --num-clusters 128 \
  --num-countries 12 \
  --batch-size 64 \
  --num-workers 4
```

### Evaluate the Residual Contribution

The contribution of the residual itself can be evaluated using the analysis script in `experiments/`:

```bash
python3 -m experiments.evaluate_residual_xyz \
  --checkpoint checkpoints/cluster_country_residual_xyz/k128_residual_xyz_baseline_init_rw1.pth \
  --num-clusters 128 \
  --num-countries 12 \
  --batch-size 64 \
  --num-workers 4
```

---

## Validation Results

| Method | Median (km) ↓ | Mean (km) ↓ | <200 km ↑ | <750 km ↑ |
|---|---:|---:|---:|---:|
| Direct pretrained regression | 284.79 | 467.64 | 37.97% | 81.89% |
| K=80 cluster + country | 126.19 | 459.76 | 60.12% | 80.27% |
| K=96 cluster + country | 122.97 | 479.87 | 59.48% | 78.61% |
| K=128 cluster only | 105.11 | 485.10 | 60.03% | 78.27% |
| K=128 cluster + country | 102.24 | **463.24** | **61.22%** | **79.21%** |
| **K=128 cluster + country + spherical residual** | **97.95** | 467.06 | 60.88% | 78.57% |

Direct coordinate regression produced a median localization error of **284.79 km**.

Replacing direct regression with geographic classification substantially improved the primary metric. Increasing the number of geographic clusters further improved median localization accuracy, with K=128 + auxiliary country supervision reaching **102.24 km**.

Joint spherical residual refinement reduced the median error further to **97.95 km**, the best validation median obtained during model development.

Compared with the K=128 + country baseline:

```text
102.24 km → 97.95 km
```

This is an improvement of **4.29 km**, or approximately **4.2%**, in median haversine error.

The residual model does not improve every secondary metric. The K=128 + country baseline retains a slightly better mean error and slightly higher `<200 km` and `<750 km` accuracy.

The residual model was selected as the final method because **median haversine distance is the primary challenge metric**.

---

## Residual Ablations

Additional experiments were performed to determine whether the improvement originates from the residual prediction itself or from general fine-tuning of the network.

### Frozen Baseline + Residual Head

In this ablation, the MobileNetV3 backbone, cluster head, and country head were frozen while the newly added residual head was trained.

The residual head contains only **3,843 trainable parameters**.

| Method | Median (km) ↓ | Mean (km) ↓ | <200 km ↑ | <750 km ↑ |
|---|---:|---:|---:|---:|
| K=128 + country baseline | 102.24 | 463.24 | 61.22% | 79.21% |
| Frozen baseline + residual | 103.41 | **462.71** | **61.82%** | **79.34%** |
| Jointly fine-tuned residual | **97.95** | 467.06 | 60.88% | 78.57% |

Training only the residual head did not improve the primary median metric.

The best median was obtained when the complete network was jointly fine-tuned.

### Residual Contribution Within the Joint Model

The same jointly fine-tuned checkpoint was evaluated both with and without applying its predicted residual.

| Decoding | Median (km) ↓ | Mean (km) ↓ | <200 km ↑ | <750 km ↑ |
|---|---:|---:|---:|---:|
| Cluster center only | 102.51 | 470.03 | 60.63% | 78.36% |
| Cluster center + residual | **97.95** | **467.06** | **60.88%** | **78.57%** |

For the same trained network, applying the learned spherical residual reduces median error from **102.51 km to 97.95 km**.

Therefore, the improvement is not explained only by changes to the cluster classifier during joint fine-tuning. The learned residual itself provides useful local coordinate refinement.

---

## Final Configuration

### Stage 1 — Baseline Training

| Setting | Value |
|---|---:|
| Geographic clusters | 128 |
| Countries | 12 |
| Epochs | 40 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Weight decay | 0.001 |
| Country loss weight | 0.20 |
| Label smoothing | 0.10 |
| Random seed | 42 |

The backbone is MobileNetV3-Large initialized with publicly available ImageNet pretrained weights.

### Stage 2 — Residual Fine-Tuning

| Setting | Value |
|---|---:|
| Geographic clusters | 128 |
| Countries | 12 |
| Residual dimensions | 3 |
| Residual scale | 0.05 |
| Residual activation | `tanh` |
| Residual initialization | zero |
| Validation fine-tuning epochs | 10 |
| Selected validation epoch | 6 |
| Full-data fine-tuning epochs | 6 |
| Batch size | 64 |
| Learning rate | 0.0001 |
| Weight decay | 0.001 |
| Country loss weight | 0.20 |
| Residual loss weight | 1.0 |
| Label smoothing | 0.10 |
| Random seed | 42 |

Six epochs are used for the final full-data residual fine-tuning because the best validation median during model development was obtained at epoch 6.

---

## Parameter Count

The Stage 1 K=128 + country model contains:

```text
4,381,372 parameters
```

The spherical residual head adds:

```text
3,843 parameters
```

The complete final model therefore contains:

```text
4,385,215 parameters
```

This is below the challenge limit of:

```text
5,000,000 parameters
```

The two training checkpoints should not be interpreted as two models used simultaneously at inference time.

The Stage 1 checkpoint is used only to initialize Stage 2 training. The final Stage 2 checkpoint contains the complete network used for prediction.

---

## Dataset

The code expects the provided challenge dataset to have the following structure:

```text
geo_dataset/
├── train/
│   ├── *.jpg
│   └── ...
├── holdout_public/
│   ├── *.jpg
│   └── ...
└── train_labels.csv
```

`train_labels.csv` contains:

```text
filename,country,iso,lat,lng
```

The 12 countries represented in the challenge are:

- Belarus
- Finland
- France
- Germany
- Iceland
- Italy
- Norway
- Poland
- Spain
- Sweden
- Turkey
- United Kingdom

All geolocation model training and fine-tuning use only the labeled images supplied with the challenge.

No external, scraped, downloaded, or generated geolocation training images are used.

Publicly available ImageNet pretrained weights are used only to initialize the MobileNetV3-Large backbone.

---

## Installation

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Training the Final Model

The final model is trained in three steps:

```text
Create K=128 clusters
        ↓
Stage 1: train cluster + country model
        ↓
Stage 2: jointly fine-tune spherical residual model
        ↓
Final checkpoint
```

## 1. Create Full-Data Geographic Clusters

Create K=128 geographic clusters using all provided labeled training coordinates:

```bash
python3 -m src.create_final_clusters \
  --num-clusters 128 \
  --seed 42
```

This generates:

```text
geo_dataset/final/train_cluster128_full.csv
geo_dataset/final/cluster128_centers_full.npy
```

The geographic clusters are created only from the coordinates contained in the provided challenge training labels.

---

## 2. Train Stage 1 — Full-Data Cluster + Country Model

Run:

```bash
python3 -m src.main_cluster_country_final \
  --num-clusters 128 \
  --num-countries 12 \
  --epochs 40 \
  --batch-size 64 \
  --lr 0.001 \
  --weight-decay 0.001 \
  --country-weight 0.20 \
  --label-smoothing 0.1 \
  --num-workers 4 \
  --seed 42 \
  --checkpoint-name final_k128_imagenet_cw020.pth
```

The resulting Stage 1 checkpoint is:

```text
checkpoints/final/final_k128_imagenet_cw020.pth
```

For SLURM environments:

```bash
sbatch train_final_k128.slurm
```

---

## 3. Train Stage 2 — Full-Data Spherical Residual Model

Initialize Stage 2 from the Stage 1 checkpoint and jointly fine-tune the complete network:

```bash
python3 -m src.main_cluster_country_residual_xyz_final \
  --num-clusters 128 \
  --num-countries 12 \
  --epochs 6 \
  --batch-size 64 \
  --lr 0.0001 \
  --weight-decay 0.001 \
  --country-weight 0.20 \
  --residual-weight 1.0 \
  --label-smoothing 0.1 \
  --num-workers 4 \
  --seed 42 \
  --init-checkpoint checkpoints/final/final_k128_imagenet_cw020.pth \
  --checkpoint-name final_k128_residual_xyz_baseline_init_rw1.pth
```

The resulting final checkpoint is:

```text
checkpoints/final/final_k128_residual_xyz_baseline_init_rw1.pth
```

For SLURM environments:

```bash
sbatch train_final_residual_xyz.slurm
```

Only this final Stage 2 checkpoint is required during inference.

---

## Training Time

The original full-data Stage 1 K=128 + country training run took approximately:

```text
16 minutes 53 seconds
```

using one NVIDIA A40 GPU.

The six-epoch Stage 2 residual fine-tuning run took approximately:

```text
2 minutes 58 seconds
```

using one NVIDIA A40 GPU.

Therefore, reproducing both training stages from scratch takes approximately:

```text
19 minutes 51 seconds
```

on the same hardware, excluding cluster-generation and environment-setup time.

---

# Generating Predictions

Generate the final `predictions.csv` using the Stage 2 checkpoint:

```bash
python3 -m src.predict_holdout_residual_xyz \
  --checkpoint checkpoints/final/final_k128_residual_xyz_baseline_init_rw1.pth \
  --centers geo_dataset/final/cluster128_centers_full.npy \
  --image-dir geo_dataset/holdout_public \
  --output predictions.csv \
  --num-clusters 128 \
  --num-countries 12 \
  --batch-size 64 \
  --num-workers 4
```

If `--output` is omitted, the prediction script defaults to:

```text
predictions.csv
```

The generated file contains exactly **2,400 prediction rows** with the required columns:

```text
filename,pred_lat,pred_lng
```

---

## Output Format

The final prediction file follows the required challenge format:

```text
filename,pred_lat,pred_lng
```

The included `predictions.csv` contains:

- 2,400 prediction rows,
- 2,400 unique filenames,
- one prediction for every image in `holdout_public/`,
- no duplicate filenames,
- no missing holdout images,
- latitude and longitude in decimal degrees,
- no additional index column.

---

## Repository Organization

The final submission pipeline is implemented in `src/`.

Development and ablation runners are stored separately in `experiments/` so that the final training and inference path remains unambiguous.

The two root SLURM scripts correspond directly to the two final training stages.
The `experiments/` directory contains historical development, baseline, and ablation implementations used during model selection. These files are retained to document the experimental process but are not part of the final training or inference pipeline.



```text
.
├── CHALLENGE.md
├── README.md
├── requirements.txt
├── predictions.csv
│
├── train_final_k128.slurm
├── train_final_residual_xyz.slurm
│
├── checkpoints/
│   ├── cluster_country/
│   │   ├── cluster128_imagenet_cluster_only_seed42.pth
│   │   ├── cluster128_imagenet_country_cw020_seed42.pth
│   │   ├── cluster80_imagenet_country_cw020_seed42.pth
│   │   ├── cluster96_imagenet_country_cw020_seed42.pth
│   │   └── validation_k128_cw020.pth
│   │
│   ├── cluster_country_residual_xyz/
│   │   └── k128_residual_xyz_baseline_init_rw1.pth
│   │
│   ├── regression_pretrained/
│   │   └── regression_imagenet_seed42.pth
│   │
│   └── final/
│       ├── final_k128_imagenet_cw020.pth
│       └── final_k128_residual_xyz_baseline_init_rw1.pth
│
├── geo_dataset/
│   ├── train_cluster128.csv
│   ├── val_cluster128.csv
│   ├── cluster128_centers_xyz.npy
│   │
│   └── final/
│       ├── train_cluster128_full.csv
│       └── cluster128_centers_full.npy
│
├── experiments/
└── src/
    ├── create_cluster_variants.py
    ├── create_final_clusters.py
    ├── evaluate_validation.py
    ├── evaluate_residual_validation.py
    ├── main_cluster_country_final.py
    ├── main_cluster_country_residual_xyz_final.py
    ├── predict_holdout_residual_xyz.py
    │
    ├── datasets/
    │   ├── split.py
    │   ├── dataset_cluster_country.py
    │   ├── dataset_cluster_country_residual_xyz.py
    │   ├── dataloaders_cluster_country.py
    │   └── dataloaders_cluster_country_residual_xyz.py
    │
    ├── models/
    │   ├── model_cluster_country.py
    │   └── model_cluster_country_residual_xyz.py
    │
    ├── training/
    │   ├── country_consistency.py
    │   ├── geographic_label_smoothing.py
    │   ├── train_cluster_country.py
    │   └── train_cluster_country_residual_xyz.py
    │
    └── utils/
        └── metrics.py
```

---

## Final Pipeline

The files directly involved in producing the final model and predictions are:

```text
src/create_final_clusters.py

src/main_cluster_country_final.py
        ↓
checkpoints/final/final_k128_imagenet_cw020.pth

src/main_cluster_country_residual_xyz_final.py
        ↓
checkpoints/final/final_k128_residual_xyz_baseline_init_rw1.pth

src/predict_holdout_residual_xyz.py
        ↓
predictions.csv
```

The supporting model, dataset, training, and metric implementations are contained in the corresponding `src/models/`, `src/datasets/`, `src/training/`, and `src/utils/` directories.

---

## Experiment Code

The `experiments/` directory contains development and ablation runners used during model selection.

These scripts are retained to document the experimental process but are not part of the final inference pipeline.

They include:

- `main_cluster_country.py` — geographic cluster + country validation experiments,
- `main_cluster_country_residual_xyz.py` — joint spherical residual experiments,
- `main_cluster_country_residual_xyz_frozen.py` — frozen-baseline residual-head ablation,
- `evaluate_residual_xyz.py` — cluster-center versus residual decoding analysis.

Historical checkpoints from the principal experiments are retained under `checkpoints/`.

---

## Final Model Files

The files required to reproduce the submitted holdout predictions are:

```text
checkpoints/final/final_k128_residual_xyz_baseline_init_rw1.pth
geo_dataset/final/cluster128_centers_full.npy
predictions.csv
```

The Stage 1 checkpoint is additionally retained because it is required to reproduce Stage 2 training:

```text
checkpoints/final/final_k128_imagenet_cw020.pth
```

At inference time, only the final residual checkpoint is loaded as the model. The Stage 1 and Stage 2 checkpoints are **not** used as an ensemble.


## Repository

https://github.com/DinelyWellehewage/deep-learning-final-project-geolocation
