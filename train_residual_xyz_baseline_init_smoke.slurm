#!/bin/bash

#SBATCH --job-name=resxyz-init-smoke
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/resxyz_init_smoke_%j.out
#SBATCH --error=logs/resxyz_init_smoke_%j.err

set -e

cd ~/deep-learning-final-project-geolocation || exit 1

source .venv-cluster/bin/activate

mkdir -p logs
mkdir -p checkpoints/cluster_country_residual_xyz

python3 -m src.main_cluster_country_residual_xyz \
  --num-clusters 128 \
  --num-countries 12 \
  --epochs 1 \
  --batch-size 64 \
  --lr 0.0001 \
  --weight-decay 0.001 \
  --num-workers 4 \
  --seed 42 \
  --country-weight 0.20 \
  --residual-weight 1.0 \
  --label-smoothing 0.10 \
  --pretrained \
  --init-checkpoint checkpoints/cluster_country/validation_k128_cw020.pth \
  --checkpoint-name smoke_baseline_init_residual_xyz.pth

