#!/bin/bash

#SBATCH --job-name=resxyz-frozen
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/resxyz_frozen_%j.out
#SBATCH --error=logs/resxyz_frozen_%j.err

set -e

cd ~/deep-learning-final-project-geolocation || exit 1

source .venv-cluster/bin/activate

mkdir -p logs
mkdir -p checkpoints/cluster_country_residual_xyz

python3 -m src.main_cluster_country_residual_xyz_frozen \
  --num-clusters 128 \
  --num-countries 12 \
  --epochs 10 \
  --batch-size 64 \
  --lr 0.001 \
  --weight-decay 0.001 \
  --num-workers 4 \
  --seed 42 \
  --init-checkpoint checkpoints/cluster_country/validation_k128_cw020.pth \
  --checkpoint-name k128_residual_xyz_frozen_baseline.pth