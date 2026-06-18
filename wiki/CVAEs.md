# What CVAEs are and how DeepCor works

[← Back to the wiki home](README.md)

This page explains the idea behind DeepCor: the **contrastive variational
autoencoder (CVAE)** and how it separates signal from noise in fMRI. It's
conceptual — for code, see the [advanced API](usage_advanced_api.md); for the
inputs these concepts map onto, see [data preparation](data_preparation.md).

## Background: the variational autoencoder (VAE)

A **variational autoencoder** is a neural network that learns to compress data
into a small set of latent variables and reconstruct it back. It has two parts:

- an **encoder** that maps an input (here, a voxel's time series) to a
  distribution over a low-dimensional **latent space**;
- a **decoder** that reconstructs the input from a sample of that latent space.

It's trained to reconstruct its input well **while** keeping the latent space
well-behaved (close to a simple prior). That second objective is the
**KL-divergence** term; weighting it by a factor **β** gives the "β-VAE", where
larger β pushes toward more disentangled, regularized latent representations.

> References: Kingma & Welling (2013), *Auto-Encoding Variational Bayes*,
> https://arxiv.org/abs/1312.6114 · Higgins et al. (2017), *β-VAE*,
> https://openreview.net/forum?id=Sy2fzU9gl

## The contrastive twist (cVAE)

A plain VAE doesn't know which parts of the signal are "interesting." The
**contrastive VAE** solves this by training on **two datasets at once**:

- a **target** set that contains both the signal of interest **and** background
  noise, and
- a **background** set that contains **only** the noise.

The model splits its latent space into **two** parts:

- a **shared / noise latent** (`z`) — meant to capture variation present in
  *both* sets (the noise), and
- a **salient / signal latent** (`s`) — meant to capture variation present
  *only* in the target (the signal of interest).

By contrasting the two, the salient latent is forced to isolate what makes the
target special — the signal — while the shared latent soaks up the common
noise.

> Reference: Abid & Zou (2019), *Contrastive variational autoencoder enhances
> salient features*, https://arxiv.org/abs/1902.04601

## How DeepCor applies this to fMRI

DeepCor maps the contrastive setup onto brain anatomy:

| Contrastive VAE | DeepCor |
| --- | --- |
| **Target** (signal + noise) | voxels in the **gray-matter ROI** (`gm_mask`) |
| **Background** (noise only) | voxels in the **confounds mask** (`cf_mask`, usually WM + CSF) |
| **Shared/noise latent `z`** | structured noise present everywhere |
| **Salient/signal latent `s`** | neural signal of interest, expected only in gray matter |

The confounds mask (white matter + CSF) is treated as a place where there should
be no signal of interest — the same noise region-of-interest idea used by
**aCompCor** (Behzadi et al., 2007, https://pmc.ncbi.nlm.nih.gov/articles/PMC2214855/).
The current model (`v2`) additionally conditions on **motion confounds** and
**spatial coordinates** — see [model versions](model_versions.md).

### Producing the denoised signal

Once trained, denoising is a latent-space operation. To reconstruct a
gray-matter voxel's **denoised** time series, DeepCor:

1. encodes the voxel into both latents (`z` for noise, `s` for signal),
2. **zeros out the noise latent `z`**, and
3. decodes from the signal latent `s` alone.

That signal-only reconstruction is the **foreground** (`forward_fg` in the
code), and it's what gets written as the denoised output. Symmetrically, the
**background** (`forward_bg`) reconstructs from the noise latent with the signal
zeroed, and the full **target** reconstruction is `background + foreground`.

This is why the separation between the two masks matters so much: the cleaner
the contrast between "signal+noise" (ROI) and "noise only" (RONI), the cleaner
the disentanglement.

## Why an ensemble?

Each CVAE is trained from a random initialization, so any single model is a bit
noisy. DeepCor trains several (`n_repetitions`) and **averages** their denoised
outputs, which stabilizes the result. See
[configuration](configuration.md#tuning-guidance) and [outputs](outputs.md).

## The loss, briefly

Training balances several terms, weighted by config fields
([configuration reference](configuration.md)):

- **reconstruction** of the ROI and RONI signals (`scale_MSE_GM`, `scale_MSE_CF`);
- the **KL** regularizer on the latents (`beta`);
- **disentanglement** terms that discourage the signal and noise latents from
  sharing information (`gamma` for total correlation, `delta`, toggled by
  `do_disentangle`).

## Citing DeepCor

> Zhu, Y., Aglinskas, A., & Anzellotti, S. (2025). DeepCor: denoising fMRI data
> with contrastive autoencoders. *Nature Methods*, 1–4.
> https://www.nature.com/articles/s41592-025-02967-x

---

**See also:** [Model versions](model_versions.md) ·
[Data preparation](data_preparation.md) · [Glossary](glossary.md)
