# Model versions (v1 vs v2)

[← Back to the wiki home](README.md)

DeepCor ships two model versions. You pick one with the `model_version`
argument of [`DeepCor`/`DeepCorDenoiser`](usage_high_level_api.md):

```python
DeepCor(model_version="latest")   # → v2 (recommended)
DeepCor(model_version="v2")       # confound-aware CVAE
DeepCor(model_version="v1")       # original cVAE
```

Accepted values: `"v1"`, `"v2"`, and the aliases `"latest"` / `"cvae"` (both
currently resolve to **v2**). An unknown value raises an error listing the valid
options.

## At a glance

| | **v1** (`CVAE_V1`) | **v2** (`CVAE`, = `latest`) |
| --- | --- | --- |
| Confounds | **Not used** | **Required** (motion regressors) |
| Input channels | 1 (signal only) | 4 (signal + 3 spatial coordinates) |
| Latent space | single dim (`latent_dims[0]`) | tuple `(signal_dim, noise_dim)` |
| Disentanglement terms | — | `gamma`, `delta`, `scale_MSE_*`, `do_disentangle` |
| Loss | reconstruction + KL (`beta`) | reconstruction + KL + disentanglement |
| Data loader | `get_obs_noi_list` | `get_obs_noi_list_coords` |
| Use it for | reproducing the original method; runs without confounds | **default / recommended** denoising |

## v1 — the original cVAE

`v1` is the original contrastive VAE (no confound conditioning). It takes just
the ROI and RONI time series — **no confounds file needed** — and uses a single
latent dimension per branch with a plain reconstruction + KL loss. From the
config it reads only `latent_dims[0]` and `beta`; the other `ModelConfig` fields
are ignored.

```python
# v1 needs only the three images — no confounds
DeepCor(model_version="v1").fit_denoise(epi, gm_mask, cf_mask)
```

Reach for v1 when you want to reproduce the original method, when you don't have
usable confounds, or as a comparison against v2.

## v2 — confound-aware CVAE (recommended)

`v2` is the current model and the one `"latest"` points at. Two key differences:

1. **It is confound-aware.** You must pass `confounds` (an fMRIPrep confounds
   TSV or a `(n_conf, nTR)` array). Selecting v2 without confounds raises:
   *"model_version 'v2' requires confounds; pass confounds=<path to confounds .tsv>"*.
2. **It encodes spatial context.** The loader appends each voxel's 3 spatial
   coordinates to its time series, so the model sees 4 input channels instead of
   1. It uses a `(signal_dim, noise_dim)` latent tuple and the full set of
   disentanglement terms (`gamma`, `delta`, the `scale_MSE_*` weights,
   `do_disentangle`) to separate signal from structured noise.

```python
# v2 / latest requires confounds
DeepCor(model_version="latest").fit_denoise(epi, gm_mask, cf_mask, confounds)
```

See the [configuration reference](configuration.md) for the v2-specific loss
weights and their defaults.

## How versions are wired (and adding new ones)

Each version is described by a single entry in a `_VERSION_SPECS` table in
[`deepcor/pipeline.py`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/pipeline.py)
— the data loader, whether confounds are required, the dashboard/track schema,
and the model builder. Everything version-specific lives there, so a future
`v3` is an additive entry rather than edits scattered across the codebase. The
[backwards compatibility](backwards_compatibility.md) page documents this design
and the rules for adding a version without breaking existing notebooks.

---

**See also:** [High-level API](usage_high_level_api.md) ·
[Configuration reference](configuration.md) ·
[Data preparation](data_preparation.md)
