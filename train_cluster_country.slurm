#!/bin/bash

#SBATCH --job-name=geo-wd-sweep
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

for WD in 0 0.0001 0.001
do
    if [ "$WD" = "0" ]; then
        NAME="wd0"
    elif [ "$WD" = "0.0001" ]; then
        NAME="wd1e4"
    else
        NAME="wd1e3"
    fi

    echo
    echo "========================================"
    echo "STARTING WEIGHT DECAY: $WD"
    echo "========================================"

    python -m src.main_cluster_country \
        --num-clusters 64 \
        --num-countries 12 \
        --epochs 40 \
        --batch-size 64 \
        --lr 0.001 \
        --weight-decay "$WD" \
        --num-workers 4 \
        --seed 42 \
        --country-weight 0.2 \
        --smoothing-mode ordinary \
        --label-smoothing 0.1 \
        --checkpoint-name "cluster64_country_ls01_${NAME}_seed42.pth"

    STATUS=$?

    if [ "$STATUS" -ne 0 ]; then
        echo "Training failed for weight decay $WD"
        exit "$STATUS"
    fi

    echo "FINISHED WEIGHT DECAY: $WD"
done

echo
echo "========================================"
echo "WEIGHT DECAY SWEEP COMPLETE"
echo "========================================"