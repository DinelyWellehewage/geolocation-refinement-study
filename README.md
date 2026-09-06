# Deep Learning Final Project — Image Geolocation

## Overview

This repository contains the final solution for the Image Geolocation Challenge.

Given a street-level image from one of 12 European countries, the model predicts its geographic coordinates (latitude and longitude).

The final approach combines geographic cluster classification with bounded local coordinate refinement on the 3D unit sphere.

---

## Final Method

The method is trained in two stages.

### Stage 1 — Geographic Cluster Classification

Training coordinates are converted from latitude/longitude to 3D unit-sphere coordinates.

KMeans is then applied to the training coordinates to create 128 geographic clusters.

The baseline model consists of:

- MobileNetV3-Large backbone
- 128-class geographic cluster prediction head
- 12-class auxiliary country prediction head

The country head provides auxiliary supervision during training.

Without residual refinement, inference selects the most likely geographic cluster and uses its cluster center as the predicted location.

### Stage 2 — Spherical Residual Refinement

A three-dimensional residual head is added to the baseline model.

For a training location with unit-sphere coordinate `true_xyz` and geographic cluster center `center_xyz`, the residual target is:

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

The residual head is zero-initialized, while the MobileNetV3 backbone, cluster head, and country head are initialized from the trained K=128 + country baseline checkpoint.

The complete network is then jointly fine-tuned using the residual objective.

This allows the cluster classifier to provide coarse geographic localization while the residual head performs local coordinate refinement.

---

## Validation Evaluation

A held-out subset of the provided labeled training data was used during model development.

The hidden holdout labels were not used for model selection or evaluation.

### Baseline Evaluation

To recreate the validation split and K=128 geographic clusters:

```bash
python3 -m src.datasets.split

python3 -m src.create_cluster_variants \
  --clusters 128 \
  --save-files
```

Evaluate the K=128 + country baseline:

```bash
python3 -m src.evaluate_validation \
  --checkpoint checkpoints/cluster_country/validation_k128_cw020.pth \
  --num-clusters 128 \
  --num-countries 12 \
  --batch-size 64 \
  --num-workers 4
```

### Residual Evaluation

Evaluate the jointly fine-tuned spherical residual model:

```bash
python3 -m src.evaluate_residual_validation \
  --checkpoint checkpoints/cluster_country_residual_xyz/k128_residual_xyz_baseline_init_rw1.pth \
  --num-clusters 128 \
  --num-countries 12 \
  --batch-size 64 \
  --num-workers 4
```

The contribution of the residual itself can be evaluated using:

```bash
python3 -m src.evaluate_residual_xyz \
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

Increasing the number of geographic clusters substantially improved the primary median localization error.

Direct coordinate regression produced a median error of 284.79 km. Geographic classification with K=128 clusters and auxiliary country supervision reduced this to 102.24 km.

Adding bounded spherical residual refinement and jointly fine-tuning the baseline further reduced the median error to **97.95 km**, the best validation median obtained during model development.

Compared with the K=128 + country baseline, this reduces median error by 4.29 km, or approximately 4.2%.

The residual model does not improve every secondary metric. The baseline retains a slightly better mean error and slightly higher <200 km and <750 km accuracy. The residual model was selected because median haversine distance is the primary challenge metric.

---

## Residual Ablation

Additional experiments were performed to determine where the residual improvement originates.

### Frozen Baseline + Residual Head

The MobileNetV3 backbone, cluster head, and country head were frozen and only the new residual head was trained.

Only 3,843 parameters were trainable in this experiment.

| Method | Median (km) ↓ | Mean (km) ↓ | <200 km ↑ | <750 km ↑ |
|---|---:|---:|---:|---:|
| K=128 + country baseline | 102.24 | 463.24 | 61.22% | 79.21% |
| Frozen baseline + residual | 103.41 | **462.71** | **61.82%** | **79.34%** |
| Jointly fine-tuned residual | **97.95** | 467.06 | 60.88% | 78.57% |

Training only the residual head did not improve the median error. This indicates that joint adaptation of the shared representation and output heads is important for the improvement in the primary metric.

### Residual Contribution Within the Joint Model

The same jointly fine-tuned checkpoint was also evaluated with and without applying its residual prediction.

| Decoding | Median (km) ↓ | Mean (km) ↓ | <200 km ↑ | <750 km ↑ |
|---|---:|---:|---:|---:|
| Cluster center only | 102.51 | 470.03 | 60.63% | 78.36% |
| Cluster center + residual | **97.95** | **467.06** | **60.88%** | **78.57%** |

For the same trained network, applying the learned residual reduces median error from 102.51 km to 97.95 km.

Therefore, the improvement is not explained only by changes to the cluster classifier. The learned spherical residual itself provides useful local coordinate refinement.

---

## Final Configuration

### Baseline Training

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

The baseline uses a MobileNetV3-Large backbone initialized with publicly available ImageNet pretrained weights.

### Residual Fine-Tuning

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

---

## Parameter Count

The final residual model contains:

```text
4,385,215 parameters
```

This is below the challenge limit of 5,000,000 parameters.

The residual head adds only 3,843 parameters to the 4,381,372-parameter K=128 + country baseline.

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

All geolocation model training and fine-tuning use only the labeled images supplied with the challenge.

No additional geolocation images are added to the training dataset.

---

## Installation

Create a Python environment and install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

# Training

## 1. Create Full-Data Geographic Clusters

Create K=128 geographic clusters from all provided labeled training coordinates:

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

---

## 2. Train the Full-Data Baseline

Train the K=128 + country model:

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

The resulting baseline checkpoint is:

```text
checkpoints/final/final_k128_imagenet_cw020.pth
```

For SLURM environments:

```bash
sbatch train_final_k128.slurm
```

---

## 3. Fine-Tune the Full-Data Residual Model

Initialize from the full-data baseline and jointly fine-tune the spherical residual model:

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
  --num-workers 4 \
  --seed 42 \
  --init-checkpoint checkpoints/final/final_k128_imagenet_cw020.pth \
  --checkpoint-name final_k128_residual_xyz_baseline_init_rw1.pth
```

For SLURM environments:

```bash
sbatch train_final_residual_xyz.slurm
```

The resulting final checkpoint is:

```text
checkpoints/final/final_k128_residual_xyz_baseline_init_rw1.pth
```

---

## Training Time

The original full-data K=128 + country training run took approximately:

```text
16 minutes 53 seconds
```

using one NVIDIA A40 GPU.

The six-epoch residual fine-tuning stage took approximately:

```text
2 minutes 58 seconds
```

using one NVIDIA A40 GPU.

---

# Generating Predictions

## Baseline Predictions

The baseline checkpoint can reproduce the original baseline predictions:

```bash
python3 -m src.predict_holdout \
  --checkpoint checkpoints/final/final_k128_imagenet_cw020.pth \
  --centers geo_dataset/final/cluster128_centers_full.npy \
  --image-dir geo_dataset/holdout_public \
  --output predictions.csv
```

## Residual Predictions

The final residual checkpoint can generate the refined predictions:

```bash
python3 -m src.predict_holdout_residual_xyz \
  --checkpoint checkpoints/final/final_k128_residual_xyz_baseline_init_rw1.pth \
  --centers geo_dataset/final/cluster128_centers_full.npy \
  --image-dir geo_dataset/holdout_public \
  --output predictions_residual.csv \
  --num-clusters 128 \
  --num-countries 12 \
  --batch-size 64 \
  --num-workers 4
```

`predictions_residual.csv` contains exactly 2,400 prediction rows with the required columns:

```text
filename,pred_lat,pred_lng
```

---

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── predictions.csv
├── predictions_residual.csv
├── train_final_k128.slurm
├── train_final_residual_xyz.slurm
│
├── checkpoints/
│   └── final/
│       ├── final_k128_imagenet_cw020.pth
│       └── final_k128_residual_xyz_baseline_init_rw1.pth
│
├── geo_dataset/
│   └── final/
│       ├── train_cluster128_full.csv
│       └── cluster128_centers_full.npy
│
└── src/
    ├── create_final_clusters.py
    ├── main_cluster_country_final.py
    ├── main_cluster_country_residual_xyz.py
    ├── main_cluster_country_residual_xyz_final.py
    ├── main_cluster_country_residual_xyz_frozen.py
    ├── evaluate_validation.py
    ├── evaluate_residual_validation.py
    ├── evaluate_residual_xyz.py
    ├── predict_holdout.py
    ├── predict_holdout_residual_xyz.py
    │
    ├── datasets/
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

## Output Format

The prediction files follow the required format:

```text
filename,pred_lat,pred_lng
```

The residual prediction file contains:

- 2,400 prediction rows
- one row for every image in `holdout_public/`
- no duplicate filenames
- no missing holdout images
- latitude and longitude in decimal degrees

---

## Final Model Files

The files required to reproduce the residual predictions are:

```text
checkpoints/final/final_k128_residual_xyz_baseline_init_rw1.pth
geo_dataset/final/cluster128_centers_full.npy
predictions_residual.csv
```

The baseline checkpoint is also retained because it is used to initialize residual fine-tuning:

```text
checkpoints/final/final_k128_imagenet_cw020.pth
```