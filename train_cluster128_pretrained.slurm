#!/bin/bash

#SBATCH --job-name=geo-k128-pretrain
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

cd ~/deep-learning-final-project-geolocation || exit 1

source .venv-cluster/bin/activate

mkdir -p logs
mkdir -p checkpoints/cluster_country

python3 -m src.main_cluster_country \
  --num-clusters 128 \
  --num-countries 12 \
  --epochs 40 \
  --batch-size 64 \
  --lr 0.001 \
  --weight-decay 0.001 \
  --country-weight 0.0 \
  --consistency-weight 0.0 \
  --smoothing-mode ordinary \
  --label-smoothing 0.1 \
  --pretrained \
  --seed 42 \
  --checkpoint-name cluster128_imagenet_cluster_only_seed42.pth