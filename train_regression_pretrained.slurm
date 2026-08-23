#!/bin/bash

#SBATCH --job-name=geo-reg-pretrain
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

cd ~/deep-learning-final-project-geolocation || exit 1

source .venv-cluster/bin/activate

mkdir -p logs
mkdir -p checkpoints/regression_pretrained

python -m src.main_regression_pretrained \
    --epochs 40 \
    --batch-size 64 \
    --lr 0.001 \
    --weight-decay 0.001 \
    --num-workers 4 \
    --seed 42 \
    --checkpoint-name regression_imagenet_seed42.pth