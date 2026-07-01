#!/bin/bash -l

set -e

# Conda laden
source ~/miniconda3/etc/profile.d/conda.sh

# Environment aktivieren
conda activate bevcar

# Projektpfad
cd /home/es/es_es/es_nialit00/BEVCar

# CUDA / Debug
module load devel/cuda/12.4
hostname
nvidia-smi

# PYTHONPATH setzen
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Training starten
python train.py --config configs/train/train_bevcar.yaml