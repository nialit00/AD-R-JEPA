#!/bin/bash -l
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=70kFT0.1_eval
#SBATCH --chdir=/home/es/es_es/es_nialit00/BEVCar
#SBATCH --output=logs/%x_%j.out


set -e

mkdir -p logs

# Load CUDA
module load devel/cuda/12.4

# Load conda
if ! command -v conda >/dev/null 2>&1; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
  source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
fi

# Activate environment
conda activate bevcar

# Debug info
hostname
pwd
date
which python
python -V
nvidia-smi || true

export PYTHONPATH=$PYTHONPATH:$(pwd)
export WANDB_MODE=offline

# Evaluation
srun -u python eval.py --config configs/eval/eval_bevcar.yaml





