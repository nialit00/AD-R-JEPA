#!/bin/bash -l
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=bevcar_build_ops
#SBATCH --chdir=/home/es/es_es/es_nialit00/BEVCar
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

mkdir -p logs

# Load CUDA (choose the version available on your cluster)
module load devel/cuda/12.4

# Load conda
if ! command -v conda >/dev/null 2>&1; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
  source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
fi

# Activate your BEVCar environment
conda activate bevcar

# Debug info
hostname
pwd
date
which python
python -V
nvidia-smi || true

echo "Building Deformable Attention CUDA ops..."
cd nets/ops
bash make.sh

echo "Build finished successfully."

