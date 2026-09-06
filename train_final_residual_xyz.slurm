#!/bin/bash

#SBATCH --job-name=geo-final-resxyz
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/final_resxyz_%j.out
#SBATCH --error=logs/final_resxyz_%j.err

set -e

cd ~/deep-learning-final-project-geolocation || exit 1

source .venv-cluster/bin/activate

mkdir -p logs
mkdir -p checkpoints/final

python3 -m src.main_cluster_country_residual_xyz_final \
  --num-clusters 128 \
  --num-countries 12 \
  --epochs 6 \
  --batch-size 64 \
  --lr 0.0001 \
  --weight-decay 0.001 \
  --country-weight 0.20 \
  --residual-weight 1.0 \
  --label-smoothing 0.10 \
  --num-workers 4 \
  --seed 42 \
  --init-checkpoint checkpoints/final/final_k128_imagenet_cw020.pth \
  --checkpoint-name final_k128_residual_xyz_baseline_init_rw1.pth