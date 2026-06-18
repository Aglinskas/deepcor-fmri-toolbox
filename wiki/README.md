# DeepCor Wiki

Welcome to the DeepCor documentation. DeepCor denoises fMRI data with a
**contrastive variational autoencoder (CVAE)**: it learns to separate the
signal of interest (in gray matter) from structured noise (estimated from
noise regions such as CSF), without needing task labels.

This wiki is aimed at **neuroimagers using DeepCor** — getting your data in,
running a denoising job, and understanding what comes out. For installation and
the full function/class reference, see the
[project README](../README.md).

## Start here

If you just want to denoise a run, read these pages in order:

1. **[Data preparation](data_preparation.md)** — what DeepCor needs as input
   (functional image, masks, confounds), what format they must be in, and how
   to get them from an fMRIPrep derivative.
2. **[High-level API](usage_high_level_api.md)** — the `DeepCorDenoiser` /
   `DeepCor` one-call workflow: required vs optional parameters, with examples.
3. **[Advanced / low-level API](usage_advanced_api.md)** — driving the models,
   datasets, and `Trainer` directly for research and customization.

## All pages

| Page | What it covers |
| --- | --- |
| [Data preparation](data_preparation.md) | Inputs, formats, masks, confounds, fMRIPrep |
| [High-level API](usage_high_level_api.md) | `DeepCorDenoiser` / `DeepCor` usage |
| [Advanced / low-level API](usage_advanced_api.md) | Models, datasets, `Trainer` |
| [What CVAEs are and how DeepCor works](CVAEs.md) | Contrastive VAEs, disentanglement, references |
| [Model versions (v1 vs v2)](model_versions.md) | Differences and when to use each |
| [Configuration reference](configuration.md) | Every config parameter, defaults, tuning |
| [Outputs explained](outputs.md) | The files a run produces and what they mean |
| [Batch jobs](batch_jobs.md) | SLURM scripts, papermill & marimo notebook runs |
| [Troubleshooting](troubleshooting.md) | Install (torch/ANTsPy), CUDA, common errors |
| [Glossary](glossary.md) | ROI, RONI, CVAE, CompCor, confounds, etc. |
| [Backwards compatibility](backwards_compatibility.md) | How model versions are added without breakage |

*Pages marked "planned" are not written yet.*

## A 30-second mental model

DeepCor needs three spatial inputs in the **same space and resolution**:

- a **4D functional (EPI) image** — the run you want to denoise;
- a **gray-matter mask** — the **ROI** (region of interest), where signal lives;
- a **confounds mask** (`cf_mask`) — the **RONI** (region of *no* interest),
  usually a white-matter + CSF mask (aCompCor-style), used to estimate
  structured noise.

The current model (`v2`) also takes **motion confounds** (an fMRIPrep confounds
TSV). It trains a small ensemble of CVAEs, then averages them to produce a
denoised functional image. See [Data preparation](data_preparation.md) for the
exact requirements.

## Citing DeepCor

> Zhu, Y., Aglinskas, A., & Anzellotti, S. (2025). DeepCor: denoising fMRI data
> with contrastive autoencoders. *Nature Methods*, 1–4.
> https://www.nature.com/articles/s41592-025-02967-x
