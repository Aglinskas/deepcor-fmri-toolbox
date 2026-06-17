#!/bin/bash
#SBATCH --job-name=DeepCor-forrest-V2-simple-jupyter
#SBATCH --output=../Data/DeepCor-Outputs/test-simple-jupyter-V2/slurm_outputs/slurm_outputs_out_%a.txt
#SBATCH --error=../Data/DeepCor-Outputs/test-simple-jupyter-V2/slurm_outputs/slurm_outputs_err_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32gb
#SBATCH --partition=short
#SBATCH --array=0-2


#rm -rf $deepcor_dir$analysis_name # To clear previous ouputs
#sbatch slurm-01-02-deepcor-forrest-CVAE-V2-simple-jupyter.sh


deepcor_dir=../Data/DeepCor-Outputs/
notebook_name=01_StudyForrest-simple-v2
analysis_name=test-simple-jupyter-V2

param_epochs=5
param_repetitions=5


echo $deepcor_dir$analysis_name
mkdir -p $deepcor_dir$analysis_name
mkdir -p $deepcor_dir$analysis_name/slurm_outputs # create this before running the script
mkdir -p $deepcor_dir$analysis_name/logs

outdir=$deepcor_dir$analysis_name/logs

source /home/aglinska/anaconda3/etc/profile.d/conda.sh
conda activate deepcor

echo "HOSTNAME: $HOSTNAME"
echo "CONDA_PREFIX: $CONDA_PREFIX"
which python
which marimo
nvidia-smi


outname=${outdir}/$notebook_name-S${SLURM_ARRAY_TASK_ID}-R1.ipynb
echo $outname
date
papermill $notebook_name.ipynb $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 1 -p analysis_name $analysis_name -p param_epochs $param_epochs -p param_repetitions $param_repetitions

echo $outname
date
outname=${outdir}/$notebook_name-S${SLURM_ARRAY_TASK_ID}-R2.ipynb
papermill $notebook_name.ipynb $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 2 -p analysis_name $analysis_name -p param_epochs $param_epochs -p param_repetitions $param_repetitions


outname=${outdir}/$notebook_name-S${SLURM_ARRAY_TASK_ID}-R3.ipynb
echo $outname
date
papermill $notebook_name.ipynb $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 3 -p analysis_name $analysis_name -p param_epochs $param_epochs -p param_repetitions $param_repetitions

outname=${outdir}/$notebook_name-S${SLURM_ARRAY_TASK_ID}-R4.ipynb
echo $outname
date
papermill $notebook_name.ipynb $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 4 -p analysis_name $analysis_name -p param_epochs $param_epochs -p param_repetitions $param_repetitions

