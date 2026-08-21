#!/bin/bash

#SBATCH --job-name=geo-cw-pretrain
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

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

for CW in 0.05 0.20
do
    if [ "$CW" = "0.05" ]; then
        NAME="cw005"
    else
        NAME="cw020"
    fi

    echo
    echo "========================================"
    echo "STARTING COUNTRY WEIGHT: $CW"
    echo "========================================"

    python -m src.main_cluster_country \
        --num-clusters 64 \
        --num-countries 12 \
        --epochs 40 \
        --batch-size 64 \
        --lr 0.001 \
        --weight-decay 0.001 \
        --num-workers 4 \
        --seed 42 \
        --country-weight "$CW" \
        --consistency-weight 0.0 \
        --smoothing-mode ordinary \
        --label-smoothing 0.1 \
        --pretrained \
        --checkpoint-name "cluster64_imagenet_country_${NAME}_seed42.pth"

    STATUS=$?

    if [ "$STATUS" -ne 0 ]; then
        echo "Training failed for country weight $CW"
        exit "$STATUS"
    fi

    echo "FINISHED COUNTRY WEIGHT: $CW"
done

echo
echo "========================================"
echo "PRETRAINED COUNTRY WEIGHT SWEEP COMPLETE"
echo "========================================"