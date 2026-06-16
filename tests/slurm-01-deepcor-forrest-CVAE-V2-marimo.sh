#!/bin/bash
#SBATCH --job-name=DeepCor-forrest-V2-marimo
#SBATCH --output=./slurm_outputs/DeepCor-forrest-V2-marimo_out_%a.txt
#SBATCH --error=./slurm_outputs/DeepCor-forrest-V2-marimo_err_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32gb
#SBATCH --partition=medium,sccn
#SBATCH --array=0-1

#rm -rf ../Data/DeepCor-Outputs/*
#rm -rf ./slurm_outputs/*
#rm -rf *.html
#sbatch slurm-01-deepcor-forrest-CVAE-V2-marimo.sh

source /home/aglinska/anaconda3/etc/profile.d/conda.sh
conda activate deepcor

echo "HOSTNAME: $HOSTNAME"
echo "CONDA_PREFIX: $CONDA_PREFIX"
which python
which marimo
nvidia-smi

#marimo export html 02_StudyForrest-advanced-v2_mo.py -o studyforrest-S_${SLURM_ARRAY_TASK_ID}_R1.html --watch -- --s ${SLURM_ARRAY_TASK_ID} --r 1

#marimo export html 02_StudyForrest-advanced-v2_mo.py -o studyforrest-S_${SLURM_ARRAY_TASK_ID}_R2.html --watch -- --s ${SLURM_ARRAY_TASK_ID} --r 2

#marimo export html 02_StudyForrest-advanced-v2_mo.py -o studyforrest-S_${SLURM_ARRAY_TASK_ID}_R3.html --watch -- --s ${SLURM_ARRAY_TASK_ID} --r 3

marimo export html 02_StudyForrest-advanced-v2_mo.py -o studyforrest-S_${SLURM_ARRAY_TASK_ID}_R4.html --watch -- --s ${SLURM_ARRAY_TASK_ID} --r 4

