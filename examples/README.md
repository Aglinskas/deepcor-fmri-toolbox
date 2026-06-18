# DeepCor examples

Starter scripts for running DeepCor on your own data. **Copy a script, edit the
`PATHS` and `HYPERPARAMETERS` sections at the top, and run it.**

Every script ships with hyperparameters set **low** (`N_EPOCHS=5`,
`N_REPETITIONS=5`) so you can test the full pipeline quickly. For a real
denoising run, bump them up to roughly **100 epochs** and **20 repetitions**
(noted in each file).

| File | What it does |
| --- | --- |
| [`01_quickstart_highlevel.py`](01_quickstart_highlevel.py) | Denoise a **single subject** with the high-level API (recommended starting point). |
| [`02_lowlevel_api.py`](02_lowlevel_api.py) | Build the pipeline by hand with the **low-level API** — v2 (recommended) and v1. |
| [`03_batch_run_subject.py`](03_batch_run_subject.py) | Per-subject runner invoked by the SLURM job below. |
| [`03_slurm_batch_job.sh`](03_slurm_batch_job.sh) | **SLURM array job**: denoise a whole study in parallel (`sbatch 03_slurm_batch_job.sh`). |

## Before you run

Your inputs must be prepared correctly — three images on the **same voxel grid**,
binary and **non-overlapping** masks, and an fMRIPrep confounds TSV. See:

- [Data preparation](../wiki/data_preparation.md)
- [High-level API](../wiki/usage_high_level_api.md)
- [Advanced / low-level API](../wiki/usage_advanced_api.md)
- [Batch jobs](../wiki/batch_jobs.md)

New to DeepCor? Start with the [wiki home](../wiki/README.md).
