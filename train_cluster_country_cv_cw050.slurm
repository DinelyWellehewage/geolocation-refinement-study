#!/bin/bash

#SBATCH --job-name=geo-cv-k128-cw050
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

cd ~/deep-learning-final-project-geolocation || exit 1

source .venv-cluster/bin/activate

mkdir -p logs
mkdir -p checkpoints/cv_k128_cw050
mkdir -p results/cv_k128_cw050

for FOLD in 0 1 2 3 4
do
    echo
    echo "========================================"
    echo "STARTING FOLD: $FOLD"
    echo "========================================"

    python3 -m src.main_cluster_country_cv \
        --fold "$FOLD" \
        --num-clusters 128 \
        --num-countries 12 \
        --epochs 40 \
        --batch-size 64 \
        --lr 0.001 \
        --weight-decay 0.001 \
        --country-weight 0.50 \
        --label-smoothing 0.1 \
        --num-workers 4 \
        --seed 42 \
        --experiment-name cv_k128_cw050

    STATUS=$?

    if [ "$STATUS" -ne 0 ]; then
        echo "Training failed for fold $FOLD"
        exit "$STATUS"
    fi

    echo "FINISHED FOLD: $FOLD"
done

echo
echo "========================================"
echo "5-FOLD CV CW050 COMPLETE"
echo "========================================"