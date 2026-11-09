#!/bin/bash
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=tp5-captioning
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$HOME/TPIntroductionML/TP5"

set +u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning
set -u

echo "Job $SLURM_JOB_ID sur $SLURM_NODELIST"
nvidia-smi

python -u train.py
