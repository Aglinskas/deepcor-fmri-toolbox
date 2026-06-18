#!/bin/bash
# ===========================================================================
# Example SLURM batch job for DeepCor.
#
# Runs 03_batch_run_subject.py once per subject as a SLURM ARRAY job, so all
# subjects denoise in parallel (one GPU task each).
#
# HOW TO USE
#   1. Edit SUBJECTS / paths in 03_batch_run_subject.py.
#   2. Edit the #SBATCH lines and the CONFIG section below for your cluster.
#   3. Submit:   sbatch 03_slurm_batch_job.sh
#
# See wiki/batch_jobs.md for an explanation.
# ===========================================================================

#SBATCH --job-name=deepcor
#SBATCH --gres=gpu:1                 # DeepCor is much faster on a GPU
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2                  # one task per subject: indices 0..2 (EDIT to match SUBJECTS)
#SBATCH --output=logs/deepcor_%a.out # %a = array index
#SBATCH --error=logs/deepcor_%a.err

# --------------------------------------------------------------------------
# CONFIG  — EDIT THESE
# --------------------------------------------------------------------------
ANALYSIS_NAME=deepcor-run            # groups all outputs under this name

# Hyperparameters. Set LOW here for a quick test of the whole pipeline.
# For a REAL denoising run, use roughly: EPOCHS=100, REPS=20.
export DEEPCOR_EPOCHS=5              # training epochs per ensemble member
export DEEPCOR_REPS=5               # number of models in the ensemble

# --------------------------------------------------------------------------
# Environment  — EDIT to match your setup
# --------------------------------------------------------------------------
mkdir -p logs
source ~/anaconda3/etc/profile.d/conda.sh
conda activate deepcor

echo "HOST: $HOSTNAME"
which python
nvidia-smi
date

# --------------------------------------------------------------------------
# Run DeepCor for this array task's subject.
# --------------------------------------------------------------------------
python 03_batch_run_subject.py ${SLURM_ARRAY_TASK_ID} ${ANALYSIS_NAME}

date
echo "Done."
