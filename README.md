
# Deep Learning Final Project — Image Geolocation

## Overview

This repository contains the final solution for the Image Geolocation Challenge.

Given a street-level image from one of 12 European countries, the model predicts its geographic coordinates (latitude and longitude).

The final approach formulates geolocation as geographic cluster classification rather than direct coordinate regression.

---

## Final Method

The training coordinates are first converted from latitude/longitude to 3D coordinates on the unit sphere.

KMeans is then applied to the training coordinates to create 128 geographic clusters.

The model consists of:

- MobileNetV3-Large backbone
- 128-class geographic cluster prediction head
- 12-class auxiliary country prediction head

At inference time, the model predicts a geographic cluster. The corresponding cluster center is converted back from 3D unit-sphere coordinates to latitude and longitude.

The country head is used as auxiliary supervision during training. The final submitted coordinates are obtained from the geographic cluster prediction.


## Validation Results

A held-out subset of the provided labeled training data was used during
model development. The hidden holdout labels were not used for model
selection or evaluation.

| Method | Median (km) ↓ | Mean (km) ↓ | <200 km ↑ | <750 km ↑ |
|---|---:|---:|---:|---:|
| Direct pretrained regression | 284.79 | 467.64 | 37.97% | 81.89% |
| K=80 cluster + country | 126.19 | 459.76 | 60.12% | 80.27% |
| K=96 cluster + country | 122.97 | 479.87 | 59.48% | 78.61% |
| K=128 cluster only | 105.11 | 485.10 | 60.03% | 78.27% |
| **K=128 cluster + country (final)** | **102.26** | **463.24** | **61.22%** | **79.21%** |

Increasing the number of geographic clusters substantially improved the
median localization error. Direct coordinate regression achieved a median
error of 284.79 km, while the final K=128 geographic classification model
reduced it to 102.26 km.

The auxiliary country objective produced a smaller additional improvement
over the K=128 cluster-only model, reducing the median from 105.11 km to
102.26 km and increasing the <200 km accuracy from 60.03% to 61.22%.

The final model was subsequently trained on all 11,758 labeled training
images before generating `predictions.csv` for the 2,400-image hidden
holdout set.

### Final Configuration

| Setting | Value |
|---|---:|
| Geographic clusters | 128 |
| Countries | 12 |
| Epochs | 40 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Weight decay | 0.001 |
| Country loss weight | 0.20 |
| Label smoothing | 0.1 |
| Random seed | 42 |

The final model uses a MobileNetV3-Large backbone initialized with ImageNet pretrained weights.

---

## Parameter Count

The final model contains:

```text
4,381,372 parameters
````

This is below the challenge limit of 5,000,000 parameters.

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

The final model is trained using the provided labeled training data.

No additional geolocation images are added to the training dataset.

---

## Installation

Create a Python environment and install the required packages.

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Training

### 1. Create Geographic Clusters

Before training, create the 128 geographic clusters from the provided training coordinates:

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

The clustering is computed only from the provided training coordinates.

### 2. Train the Final Model

The final model is trained using:

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

For SLURM environments, the same final training configuration is provided in:

```bash
sbatch train_final_k128.slurm
```

The resulting model is stored at:

```text
checkpoints/final/final_k128_imagenet_cw020.pth
```

### Training Time

The final training run on all 11,758 labeled training images took approximately:

```text
16 minutes 53 seconds
```

using one NVIDIA A40 GPU.

---

## Generating `predictions.csv`

The provided trained checkpoint can be used to generate predictions for `holdout_public/`.

```bash
python3 -m src.predict_holdout \
  --checkpoint checkpoints/final/final_k128_imagenet_cw020.pth \
  --centers geo_dataset/final/cluster128_centers_full.npy \
  --image-dir geo_dataset/holdout_public \
  --output predictions.csv
```

The generated file has the required format:

```text
filename,pred_lat,pred_lng
```

The submitted `predictions.csv` contains exactly 2,400 rows, one for every image in `holdout_public/`.

---

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── predictions.csv
├── train_final_k128.slurm
│
├── checkpoints/
│   └── final/
│       └── final_k128_imagenet_cw020.pth
│
├── geo_dataset/
│   └── final/
│       └── cluster128_centers_full.npy
│
└── src/
    ├── create_final_clusters.py
    ├── main_cluster_country_final.py
    ├── predict_holdout.py
    │
    ├── datasets/
    │   ├── dataloaders_cluster_country.py
    │   └── dataset_cluster_country.py
    │
    ├── models/
    │   └── model_cluster_country.py
    │
    ├── training/
    │   ├── country_consistency.py
    │   ├── geographic_label_smoothing.py
    │   └── train_cluster_country.py
    │
    └── utils/
        └── metrics.py
```

---

## Output Format

The final prediction file follows the required format exactly:

```text
filename,pred_lat,pred_lng
```

It contains:

* 2,400 prediction rows
* one row for every image in `holdout_public/`
* no duplicate filenames
* no missing holdout images
* latitude and longitude in decimal degrees

---

## Final Model Files

The files required for reproducing the submitted predictions are:

```text
checkpoints/final/final_k128_imagenet_cw020.pth
geo_dataset/final/cluster128_centers_full.npy
predictions.csv
```


