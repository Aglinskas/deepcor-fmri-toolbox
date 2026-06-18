# Data preparation

[← Back to the wiki home](README.md)

This is the page that decides whether DeepCor "just works" for you. DeepCor does
**not** preprocess your data — it assumes you already have a preprocessed
functional run plus a few masks, and denoises from there. Get these inputs right
and everything downstream is a one-liner (see the
[High-level API](usage_high_level_api.md)).

## What DeepCor needs

A single denoising run takes **three spatial inputs** and (for the current
model) **one confounds file**:

| Input | Role | What it is |
| --- | --- | --- |
| `epi` | the data to denoise | A **4D functional (EPI) image**, with time as the last dimension |
| `gm_mask` | **ROI** (region of interest) | A **gray-matter mask** — voxels where the signal of interest lives |
| `cf_mask` | **RONI** (region of *no* interest) | A **noise mask** — typically CSF / non-gray-matter tissue used to estimate structured noise |
| `confounds` | motion regressors | An fMRIPrep **confounds TSV** (required for `v2`/`latest`, ignored for `v1`) |

DeepCor learns what "noise" looks like from the RONI (where there should be no
signal of interest) and uses that to clean the ROI. The two masks therefore
play very different roles — don't swap them.

## Hard requirements (these will error or give garbage if wrong)

1. **Same space, same grid.** `epi`, `gm_mask`, and `cf_mask` must all be in the
   same space and have the **same voxel grid / resolution**. The masks are
   applied to the EPI voxel-for-voxel — there is no resampling step inside
   DeepCor. If your masks are anatomical-resolution, resample them to the EPI
   grid first.
2. **The masks must be binary** (0 = outside, 1 = inside).
3. **The ROI and RONI masks must not overlap.** If any voxel is 1 in both,
   DeepCor raises `'ROI and RONI masks overlap'`. Make them mutually exclusive.
4. **EPI must be 4D with time last.** Voxel intensities are read per-voxel
   across the final (time) axis.
5. **Confounds must contain motion columns** (see below) and have **no NaNs**.

> **Tip:** Voxels with (near-)zero variance over time are dropped automatically,
> so you don't need to mask those out yourself.

## File formats

- **Images** (`epi`, `gm_mask`, `cf_mask`) are read with
  [ANTsPy](https://github.com/ANTsX/ANTsPy). You can pass either:
  - a **path** to a NIfTI file (`.nii` / `.nii.gz`), or
  - an **already-loaded ANTs image** (e.g. from `ants.image_read(...)`).
- **Confounds** are read with pandas as a **tab-separated** file. You can pass
  either a path to the TSV, or a pre-built `(n_confounds, n_timepoints)` NumPy
  array.

## Confounds: which columns are used

DeepCor's confound-aware model uses the **6 rigid-body motion parameters**. The
loader auto-detects the column naming convention in your TSV:

- **Current fMRIPrep:** `trans_x`, `trans_y`, `trans_z`, `rot_x`, `rot_y`, `rot_z`
- **Older naming:** `X`, `Y`, `Z`, `RotX`, `RotY`, `RotZ`

If neither set is present, you'll get a clear error listing the columns that
*are* available. To use a different/explicit set of columns, you can supply them
yourself (see `confound_columns` in the
[configuration reference](configuration.md), *planned*) or pass a pre-built
confounds array.

> `confounds` is **required for `v2`/`latest`** (the confound-aware model) and
> **ignored for `v1`**. If you select `v2` and omit confounds, DeepCor raises an
> error telling you to pass `confounds=<path to confounds .tsv>`.

## Getting these from fMRIPrep

If you preprocess with [fMRIPrep](https://fmriprep.org/), you already have
almost everything:

- **EPI** — the preprocessed BOLD in your chosen space, e.g.
  `*_desc-preproc_bold.nii.gz`.
- **Confounds** — `*_desc-confounds_timeseries.tsv` (contains the `trans_*` /
  `rot_*` columns above).
- **Masks** — derive the gray-matter ROI and the CSF/noise RONI from fMRIPrep's
  tissue probability maps (e.g. `*_label-GM_probseg.nii.gz` and
  `*_label-CSF_probseg.nii.gz`), thresholded to binary and **resampled to the
  BOLD grid**. Ensure the resulting GM and CSF masks do not overlap.

## Optional preprocessing knobs

DeepCor can do a little cleanup on the time axis for you. These are **off by
default** and configured via the [config](configuration.md) (`DataConfig`):

- **Dummy-scan removal** (`n_dummy_scans`, default `0`) — drop the first *N*
  volumes (and the matching confound rows) before training.
- **Frame censoring** (`apply_censoring`, default `False`;
  `censoring_threshold`, default `0.5`) — flag and interpolate high-motion
  frames.

If you'd rather handle dummy scans / censoring in your own pipeline, just leave
these at their defaults.

## Quick sanity checklist

Before you call `fit_denoise`, confirm:

- [ ] `epi`, `gm_mask`, `cf_mask` are in the **same space and resolution**
- [ ] Both masks are **binary** and **do not overlap**
- [ ] `epi` is **4D** with time as the last axis
- [ ] Your confounds TSV has the **motion columns** and **no NaNs** (only needed for `v2`)

---

**Next:** [Run a denoising job with the High-level API →](usage_high_level_api.md)
