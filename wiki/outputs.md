# Outputs explained

[← Back to the wiki home](README.md)

A DeepCor run writes everything into the `output_dir` you pass to
`fit`/`fit_denoise` (default: `../Data/DeepCor-Outputs`). This page explains
each file: which one is *the* result, which are comparisons, and which are
intermediate artifacts.

The denoised filenames embed the **subject** and **run** labels — `S{s}_R{r}` —
which come from `subject_idx` / `run_idx` (both default to `0`). Set them via
`fit_denoise(..., subject_idx=..., run_idx=...)` or on the
[config](configuration.md).

## What gets written

| File | Written by | What it is |
| --- | --- | --- |
| `denoised_deepcor_S{s}_R{r}_avg.nii.gz` | `denoise` | **The main output** — the denoised functional image, averaged across the ensemble. This is what you analyze. |
| `input_data_S{s}_R{r}.nii.gz` | `denoise` | The **preprocessed (undenoised)** signal in the ROI, as DeepCor saw it. Baseline for "before vs after". |
| `denoised_compcor_S{s}_R{r}.nii.gz` | `denoise` | A **CompCor**-denoised version of the same data (5 components), as an independent comparison/baseline. |
| `signal_rep_{rep}.nii.gz` | `fit` | The denoised signal from **each ensemble member** (`rep` = 0…`n_repetitions`−1). These are averaged to produce the `_avg` file. |
| `model_final_ens{rep}.pt` | `fit` | The trained **model checkpoint** for each ensemble member. |
| `track_rep_{rep}.pickle` | `fit` | The **training-tracking** record (losses/metrics per epoch) for each member. |
| dashboard PNG | `fit` (when `dashboard="save"`/`"jupyter"`) | A rendered training **dashboard** image, named with the subject/run labels. |

> The per-member files (`signal_rep_*`, `model_final_ens*`, `track_rep_*`) are
> only written when you pass an `output_dir` to `fit`. `fit_denoise` always
> does, so a normal run produces all of the above.

## How this maps to `DeepCorResult`

`fit_denoise` / `denoise` return a [`DeepCorResult`](usage_high_level_api.md#what-you-get-back)
whose fields point at the files above:

| `DeepCorResult` field | File / value |
| --- | --- |
| `denoised_path` | `denoised_deepcor_S{s}_R{r}_avg.nii.gz` |
| `preproc_path` | `input_data_S{s}_R{r}.nii.gz` |
| `compcor_path` | `denoised_compcor_S{s}_R{r}.nii.gz` |
| `signal_files` | list of `signal_rep_{rep}.nii.gz` |
| `tracks` | in-memory list of the per-rep tracking dicts (also on disk as `track_rep_*.pickle`) |
| `output_dir` | the directory all of the above live in |

Because `DeepCorResult` stringifies to its `denoised_path`, you can pass the
result straight into anything expecting a path:

```python
import ants
result = DeepCor(model_version="latest").fit_denoise(epi, gm_mask, cf_mask, confounds, output_dir="out/")
img = ants.image_read(result)          # same as ants.image_read(result.denoised_path)
```

## Which file should I use?

- **For your analysis:** `denoised_path` (the ensemble-averaged denoised image).
- **To sanity-check the denoising:** compare `denoised_path` against
  `preproc_path` (before vs after) and `compcor_path` (DeepCor vs a classic
  CompCor baseline).
- **For ensemble diagnostics or re-averaging a subset:** the `signal_files`.
- **To reload a trained model:** the `model_final_ens{rep}.pt` checkpoints (see
  `Trainer.load_checkpoint` in the [advanced API](usage_advanced_api.md)).

## A note on robustness

The ensemble average skips any member whose signal contains NaNs, and `fit`
catches and skips a repetition that errors out (printing a message) rather than
aborting the whole run. So you may occasionally end up with fewer
`signal_rep_*` files than `n_repetitions`; the average is taken over the ones
that succeeded.

---

**See also:** [High-level API](usage_high_level_api.md) ·
[Batch jobs](batch_jobs.md) · [Configuration reference](configuration.md)
