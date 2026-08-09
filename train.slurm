#!/bin/bash
#SBATCH --job-name=geo-train
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

module load python/3.12-base
source .venv-cluster/bin/activate

echo "Node: $(hostname)"
echo "Python: $(python --version)"

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

python -m src.main --epochs 20 --batch-size 64 --lr 0.001 --num-workers 4
