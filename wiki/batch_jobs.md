# Batch jobs

[← Back to the wiki home](README.md)

DeepCor trains an ensemble per run, so real studies are run as **batch jobs** —
typically one job per subject, fanned out across a cluster with
[SLURM](https://slurm.schedulerd.io/) array jobs. This page covers the three
patterns this repo uses, from simplest to most "notebook-friendly":

1. **Plain Python script** — smallest, fastest, best for production.
2. **Parameterized Jupyter notebook via [papermill](https://papermill.readthedocs.io/)** — runs a notebook per subject and **saves the executed notebook (with outputs)** as a per-subject record.
3. **Parameterized marimo notebook via `marimo export html`** — same idea, exported as a self-contained HTML report.

Working examples live in the
[`tests/`](https://github.com/Aglinskas/deepcor-fmri-toolbox/tree/main/tests)
directory (the `slurm-*.sh` scripts and the `02_StudyForrest-advanced-*`
notebooks).

> **Why notebooks at all?** Approaches 2 and 3 let you keep a rendered,
> per-subject artifact — the training dashboards, printed shapes, and any QC
> plots — frozen alongside the outputs. That's invaluable for spotting a
> subject whose denoising went wrong. Approach 1 is leaner but produces only
> logs + output files.

---

## 1. Plain Python script

The cleanest option: write a small script that takes a subject index, builds a
`DeepCor`, and calls `fit_denoise`. The [high-level API](usage_high_level_api.md)
makes the body trivial.

```python
# run_deepcor.py
import sys
from deepcor import DeepCor

s = int(sys.argv[1])            # subject index (from the SLURM array)
analysis_name = sys.argv[2]

epi       = f"derivatives/sub-{s:02d}_task-rest_desc-preproc_bold.nii.gz"
gm_mask   = f"derivatives/sub-{s:02d}_label-GM_mask.nii.gz"
cf_mask   = f"derivatives/sub-{s:02d}_label-WMCSF_mask.nii.gz"
confounds = f"derivatives/sub-{s:02d}_task-rest_desc-confounds_timeseries.tsv"
output_dir = f"../Data/DeepCor-Outputs/{analysis_name}/sub-{s:02d}"

result = DeepCor(model_version="latest", n_epochs=100, n_repetitions=20).fit_denoise(
    epi, gm_mask, cf_mask, confounds,
    output_dir=output_dir,
    subject_idx=s,
    dashboard="save",      # writes a dashboard PNG per epoch; no display needed
)
print("Denoised:", result.denoised_path)
```

Submit it as a SLURM **array**, one task per subject:

```bash
#!/bin/bash
#SBATCH --job-name=deepcor
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH --array=0-13                      # 14 subjects: indices 0..13
#SBATCH --output=logs/deepcor_%a.out
#SBATCH --error=logs/deepcor_%a.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate deepcor
nvidia-smi

python run_deepcor.py ${SLURM_ARRAY_TASK_ID} my-analysis
```

`${SLURM_ARRAY_TASK_ID}` is the subject index; SLURM launches one task per value
in `--array`.

---

## 2. Parameterized Jupyter notebook (papermill)

This is what the repo's `slurm-01-0*-...-jupyter.sh` scripts do. **papermill**
runs a notebook end-to-end while injecting parameters, and writes a **new,
fully-executed `.ipynb` (with all outputs)** — so you get a rendered record per
subject and run.

**Step 1 — tag a parameters cell** in the notebook. The repo's
`02_StudyForrest-advanced-v2.ipynb` has a cell tagged `parameters`:

```python
# Cell tagged 'parameters' for papermill looping
s = 0                       # subject index
r = 1                       # run / repeat index
param_epochs = 5
param_repetitions = 5
analysis_name = 'test-jupyter'
```

Downstream cells use these to build the config and output path, e.g.:

```python
config.training.n_epochs = param_epochs
config.training.n_repetitions = param_repetitions
output_dir = f"../Data/DeepCor-Outputs/{analysis_name}/DeepCor-Forrest-S{s}-R{r}-cvae_v2"
```

**Step 2 — call papermill from the SLURM script**, overriding the parameters
with `-p`:

```bash
papermill 02_StudyForrest-advanced-v2.ipynb \
  $outdir/02_StudyForrest-advanced-v2-S${SLURM_ARRAY_TASK_ID}-R1.ipynb \
  --autosave-cell-every 5 --progress-bar \
  -p s ${SLURM_ARRAY_TASK_ID} \
  -p r 1 \
  -p analysis_name $analysis_name \
  -p param_epochs $param_epochs \
  -p param_repetitions $param_repetitions
```

Each call writes an executed notebook named for the subject/run. The repo's
scripts loop this four times (`R1`–`R4`) inside each array task, so a single
array (`--array=0-13`) covers 14 subjects × 4 repeats.

---

## 3. Parameterized marimo notebook (HTML export)

The `_mo.py` files are [marimo](https://marimo.io/) notebooks. The
`slurm-01-...-marimo.sh` script runs them headless and exports a self-contained
HTML report per subject/run. marimo notebooks take parameters as **CLI args**
(after `--`), rather than papermill's `-p`:

```bash
marimo export html 02_StudyForrest-advanced-v2_mo.py \
  -o $outdir/studyforrest-S_${SLURM_ARRAY_TASK_ID}_R1.html \
  --watch -- --s ${SLURM_ARRAY_TASK_ID} --r 1
```

This produces a shareable HTML file (dashboards and plots included) for each
subject/run.

---

## Conventions used in the repo's scripts

The example SLURM scripts share a few conventions worth copying:

- **One array task per subject.** `--array=0-13` → `${SLURM_ARRAY_TASK_ID}` is
  the subject index `s`, passed into the script/notebook.
- **An `analysis_name`** groups all outputs under
  `../Data/DeepCor-Outputs/<analysis_name>/`, with `slurm_outputs/` and `logs/`
  subfolders created up front with `mkdir -p`.
- **GPU + memory request:** `--gres=gpu:1`, `--mem=32gb`, a generous
  `--time=48:00:00`. DeepCor trains much faster on a GPU.
- **Environment activation:** `conda activate deepcor`, then `nvidia-smi` /
  `which python` printed for debugging.
- **Small first, then scale.** The example scripts set `param_epochs=5` and
  `param_repetitions=5` for quick test runs — bump these up (e.g. 100 / 20) for
  the real analysis once the pipeline is verified.

> The `../Data/DeepCor-Outputs/...` paths are **relative**, so they resolve
> against the directory the job runs from (the scripts run from `tests/`). Set
> `output_dir` explicitly if you run from elsewhere.

## Outputs

Each run writes its denoised image (plus a preprocessed copy, a CompCor
comparison, and per-repetition signal files) into its `output_dir`. The
[high-level API page](usage_high_level_api.md#what-you-get-back) lists the
`DeepCorResult` fields; a dedicated [outputs page](outputs.md) *(planned)* will
cover file naming in detail.

---

**See also:** [High-level API](usage_high_level_api.md) ·
[Configuration reference](configuration.md) ·
[`tests/` scripts](https://github.com/Aglinskas/deepcor-fmri-toolbox/tree/main/tests)
