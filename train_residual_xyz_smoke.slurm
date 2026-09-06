#!/bin/bash
#SBATCH --job-name=resxyz_smoke
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/resxyz_smoke_%j.out
#SBATCH --error=logs/resxyz_smoke_%j.err

set -e

cd ~/deep-learning-final-project-geolocation

source .venv-cluster/bin/activate

mkdir -p logs

python3 -m src.main_cluster_country_residual_xyz \
  --num-clusters 128 \
  --num-countries 12 \
  --epochs 1 \
  --batch-size 64 \
  --lr 0.001 \
  --weight-decay 0.001 \
  --num-workers 4 \
  --seed 42 \
  --country-weight 0.20 \
  --residual-weight 1.0 \
  --label-smoothing 0.10 \
  --pretrained \
  --checkpoint-name smoke_residual_xyz.pth
  