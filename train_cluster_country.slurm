#!/bin/bash

#SBATCH --job-name=geo-clust-country
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

cd ~/deep-learning-final-project-geolocation

source .venv-cluster/bin/activate

mkdir -p logs
mkdir -p checkpoints/cluster_country

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

python -m src.main_cluster_country \
    --num-clusters 64 \
    --num-countries 12 \
    --epochs 40 \
    --batch-size 64 \
    --lr 0.001 \
    --num-workers 4 \
    --seed 42 \
    --country-weight 0.2 \
    --checkpoint-name cluster64_country_seed42.pth