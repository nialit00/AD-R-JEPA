#!/bin/bash -l
#SBATCH --job-name=hold_node
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Node warmhalten
sleep 108000
