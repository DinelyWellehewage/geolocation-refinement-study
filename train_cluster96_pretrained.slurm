#!/bin/bash

#SBATCH --job-name=geo-k96-pretrain
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

python -m src.main_cluster_country \
    --num-clusters 96 \
    --num-countries 12 \
    --epochs 40 \
    --batch-size 64 \
    --lr 0.001 \
    --weight-decay 0.001 \
    --num-workers 4 \
    --seed 42 \
    --country-weight 0.20 \
    --consistency-weight 0.0 \
    --smoothing-mode ordinary \
    --label-smoothing 0.1 \
    --pretrained \
    --checkpoint-name cluster96_imagenet_country_cw020_seed42.pth