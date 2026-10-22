#!/bin/bash
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=alexnet-full
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set +u
set -eo pipefail

cd "$HOME/TPIntroductionML/TP3"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate deeplearning

set -u

echo "Début du job $SLURM_JOB_ID sur $SLURM_NODELIST"
date
nvidia-smi

python -u train_imagenet.py

echo "Entraînement terminé"
date
