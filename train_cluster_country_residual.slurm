#!/bin/bash

#SBATCH --job-name=geo-resid-sweep
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

cd ~/deep-learning-final-project-geolocation || exit 1

source .venv-cluster/bin/activate

mkdir -p logs
mkdir -p checkpoints/cluster_country_residual

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

for RW in 10 100 1000
do
    if [ "$RW" = "10" ]; then
        NAME="rw10"
    elif [ "$RW" = "100" ]; then
        NAME="rw100"
    else
        NAME="rw1000"
    fi

    echo
    echo "========================================"
    echo "STARTING RESIDUAL WEIGHT: $RW"
    echo "========================================"

    python -m src.main_cluster_country_residual \
        --num-clusters 64 \
        --num-countries 12 \
        --epochs 40 \
        --batch-size 64 \
        --lr 0.001 \
        --weight-decay 0.001 \
        --country-weight 0.1 \
        --consistency-weight 0.05 \
        --residual-weight "$RW" \
        --label-smoothing 0.1 \
        --residual-scale 0.06 \
        --num-workers 4 \
        --seed 42 \
        --checkpoint-name "cluster64_country_residual_${NAME}_seed42.pth"

    STATUS=$?

    if [ "$STATUS" -ne 0 ]; then
        echo "Training failed for residual weight $RW"
        exit "$STATUS"
    fi

    echo "FINISHED RESIDUAL WEIGHT: $RW"
done

echo
echo "========================================"
echo "RESIDUAL WEIGHT SWEEP COMPLETE"
echo "========================================"