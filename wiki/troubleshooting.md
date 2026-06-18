# Troubleshooting

[← Back to the wiki home](README.md)

Common problems and fixes, grouped by **install** and **runtime**. Many runtime
errors come from the inputs — if in doubt, re-check
[data preparation](data_preparation.md) first.

## Installation

### The install downloads gigabytes / takes a long time
Expected. DeepCor depends on **PyTorch** (which bundles CUDA/GPU libraries) and
**ANTsPy**, both large. A fresh install can pull several GB. Let it finish; it's
a one-time cost.

### "No space left on device" during install
The download/build ran out of disk in your temp or environment directory. Free
space, or point pip's cache/temp elsewhere (e.g. `TMPDIR=/path/with/space pip
install -e .`), and install into a fresh environment on a disk with room.

### I only have a CPU, or need a specific CUDA version
The default `torch` wheel targets a recent CUDA. If you're on CPU-only or a
specific CUDA, install the matching PyTorch build **first**, then install
DeepCor:

```bash
# pick the right command from https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### conda vs venv
Either works. See the [README installation section](../README.md#installation).
The key point is to install into a **fresh, isolated environment**, not your
system Python. ANTsPy and PyTorch both come from pip in either case.

### `import deepcor` fails after install
Usually a wrong/contaminated environment. Confirm you're in the env you
installed into (`which python`), and that the install finished without errors.
Reinstall with `pip install -e .` from the repo root if unsure.

## Runtime errors

These are the explicit errors DeepCor raises, and what they mean:

### `ROI and RONI masks overlap`
Your `gm_mask` and `cf_mask` share at least one voxel. They must be **mutually
exclusive**. Re-derive them so no voxel is 1 in both. See
[data preparation](data_preparation.md).

### `model_version 'v2' requires confounds; pass confounds=<path to confounds .tsv>`
The `v2`/`latest` model is confound-aware. Pass a confounds TSV (or array):

```python
DeepCor(model_version="latest").fit_denoise(epi, gm_mask, cf_mask, confounds)
```

…or, if you genuinely have no confounds, use `model_version="v1"` (which doesn't
need them). See [model versions](model_versions.md).

### `Could not find motion columns in confounds file...`
Your confounds TSV doesn't contain the expected motion columns. DeepCor looks
for either `trans_x/y/z, rot_x/y/z` (current fMRIPrep) or `X/Y/Z, RotX/RotY/RotZ`
(older naming). Check the file, or specify columns explicitly via
`DataConfig.confound_columns`. See
[data preparation](data_preparation.md#confounds-which-columns-are-used).

### `Missing columns in confounds file: [...]`
You passed an explicit `confound_columns` list, but some of those names aren't in
the TSV. The error lists the columns that *are* available — copy the exact names.

### `NaNs in motion`
The selected confound columns contain NaNs. fMRIPrep often puts NaNs in the first
row of derivative columns; make sure you're using the plain motion parameters,
or clean/zero-fill the NaNs before passing the file.

### `Unknown model_version '...'`
Valid values are `"v1"`, `"v2"`, `"latest"`, `"cvae"`. Check the spelling.

### `No output_dir available...` / `No per-repetition signals to ensemble...`
You called `denoise()` without first training to disk. Either use `fit_denoise`,
or call `fit(..., output_dir="out/")` before `denoise("out/")` so the
per-repetition signals exist. See [outputs](outputs.md).

### `Unknown optimizer type: ... Use 'adam' or 'adamw'`
`TrainingConfig.optimizer` must be `"adam"` or `"adamw"`.

## Performance & memory

### CUDA out of memory
Lower `batch_size` (e.g. 512 or 256) via the
[config](configuration.md) or the `DeepCor(...)` argument. A smaller batch uses
less GPU memory at some cost in speed.

### Training is very slow
You're probably on CPU. DeepCor is much faster on a GPU — confirm one is being
used. The helper `check_gpu_and_speedup()` reports GPU availability and a
rough CPU-vs-GPU speedup. On a cluster, request a GPU (`--gres=gpu:1`); see
[batch jobs](batch_jobs.md).

### A few ensemble members are missing from the output
By design: `fit` skips (and prints) any repetition that errors, and the average
skips signals containing NaNs. The result is averaged over the members that
succeeded — see [outputs](outputs.md#a-note-on-robustness). If *many* fail, look
at the printed error (often a data/shape problem).

## Shape / space mismatches

If you hit reshape or broadcasting errors during data extraction, your `epi`,
`gm_mask`, and `cf_mask` are almost certainly **not on the same voxel grid**.
DeepCor does not resample — make the masks match the EPI grid first. See
[data preparation](data_preparation.md#hard-requirements-these-will-error-or-give-garbage-if-wrong).

---

Still stuck? Open an issue:
https://github.com/Aglinskas/deepcor-fmri-toolbox/issues

**See also:** [Data preparation](data_preparation.md) ·
[Configuration reference](configuration.md) · [Batch jobs](batch_jobs.md)
