# Configuration reference

[← Back to the wiki home](README.md)

DeepCor's behaviour is controlled by a single [`DeepCorConfig`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/config.py)
object, which bundles three dataclasses:

- **`ModelConfig`** — the network architecture and loss weights
- **`TrainingConfig`** — the optimizer and training loop
- **`DataConfig`** — time-axis preprocessing and output naming

For most runs you don't need to build a config at all — the
[high-level API](usage_high_level_api.md) exposes the common knobs
(`latent_dims`, `n_epochs`, `batch_size`, `learning_rate`, `n_repetitions`)
directly as constructor arguments. Reach for a full config when you want the
finer controls below.

## Two ways to configure

**1. Scalar shortcuts (most common).** Pass the common knobs straight to the
denoiser:

```python
from deepcor import DeepCor

denoiser = DeepCor(
    model_version="latest",
    latent_dims=(8, 8),
    n_epochs=100,
    batch_size=1024,
    learning_rate=0.001,
    n_repetitions=20,
)
```

**2. A full `DeepCorConfig` (for everything else).** Build one, set fields on its
sub-configs, and pass it as `config=`. **If you pass `config`, it takes
precedence and the scalar shortcuts above are ignored.**

```python
from deepcor import DeepCor, DeepCorConfig

config = DeepCorConfig()
config.model.latent_dims = (16, 16)
config.model.beta = 0.05
config.training.n_epochs = 200
config.training.optimizer = "adam"
config.data.n_dummy_scans = 5

denoiser = DeepCor(model_version="latest", config=config)
```

`from deepcor import get_default_config` returns a `DeepCorConfig` with all
defaults if you prefer to start from that.

## `ModelConfig` — architecture & loss

| Field | Default | Meaning |
| --- | --- | --- |
| `latent_dims` | `(8, 8)` | `(signal_dim, noise_dim)` latent sizes. **`v1` uses only the first** entry; `v2` uses both. |
| `hidden_dims` | `None` | Encoder/decoder hidden layer sizes; `None` uses the model's built-in architecture. |
| `beta` | `0.01` | Weight on the KL-divergence term (the "β" in β-VAE). Higher → stronger regularization of the latent space. |
| `gamma` | `0.0` | Weight on the total-correlation term. **`v2` only.** |
| `delta` | `0.0` | Weight on the disentanglement term. **`v2` only.** |
| `scale_MSE_GM` | `1000.0` | Weight on the gray-matter (ROI) reconstruction loss. **`v2` only.** |
| `scale_MSE_CF` | `1000.0` | Weight on the confounds-mask (RONI) reconstruction loss. **`v2` only.** |
| `scale_MSE_FG` | `0.0` | Weight on the foreground (signal) reconstruction loss. **`v2` only.** |
| `do_disentangle` | `True` | Enable the disentanglement machinery. **`v2` only.** |

> **Version note.** The original `v1` model only reads `latent_dims[0]` and
> `beta`. All of the other `ModelConfig` fields (`gamma`, `delta`, the
> `scale_MSE_*` weights, `do_disentangle`) apply to the confound-aware `v2`
> model. See [model versions](model_versions.md).

## `TrainingConfig` — optimizer & loop

| Field | Default | Meaning |
| --- | --- | --- |
| `n_epochs` | `100` | Training epochs per ensemble member. |
| `batch_size` | `1024` | Mini-batch size. |
| `learning_rate` | `0.001` | Optimizer learning rate. |
| `optimizer` | `"adamw"` | `"adam"` or `"adamw"`. |
| `betas` | `(0.9, 0.999)` | Adam/AdamW β parameters. |
| `eps` | `1e-08` | Adam/AdamW epsilon. |
| `max_grad_norm` | `5.0` | Gradient-norm clipping threshold. |
| `n_repetitions` | `20` | Number of models in the ensemble (averaged at the end). |

> `current_epoch` and `current_ensemble` also live on `TrainingConfig`, but
> they are **internal live state** updated by the pipeline during training and
> read by the dashboard. Don't set them yourself.

## `DataConfig` — preprocessing & output naming

| Field | Default | Meaning |
| --- | --- | --- |
| `n_dummy_scans` | `0` | Drop the first *N* volumes (and matching confound rows) before training. **Applied by the high-level pipeline.** |
| `confound_columns` | `None` | Explicit list of confound columns to load; `None` auto-detects the fMRIPrep motion columns. **Applied by the high-level pipeline.** |
| `output_dir` | `None` | Output directory; usually set via the `output_dir` argument of `fit`/`fit_denoise`. |
| `subject_idx` | `0` | Label used to name the dashboard PNG / progress line. |
| `run_idx` | `0` | Label used to name the dashboard PNG / progress line. |
| `apply_censoring` | `False` | *(See note.)* Intended toggle for motion frame-censoring. |
| `censoring_threshold` | `0.5` | *(See note.)* Motion threshold for censoring. |
| `also_nearby_voxels` | `True` | *(See note.)* Whether censoring also affects neighbouring timepoints. |

> **Censoring is not auto-applied by `fit`/`fit_denoise` yet.** The
> `apply_censoring`, `censoring_threshold`, and `also_nearby_voxels` fields are
> declared for future use, but the high-level pipeline currently performs only
> dummy-scan removal on the time axis. If you need frame censoring today, apply
> it yourself with
> [`apply_frame_censoring()` / `censor_and_interpolate()`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/data/preprocessing.py)
> before handing data to the model (see the
> [advanced API](usage_advanced_api.md)).

## Tuning guidance

Start from the defaults — they are the recommended settings. When you do tune:

- **`n_repetitions`** is the main quality/speed dial. The ensemble is averaged,
  so more members give a more stable result at a roughly linear cost in time.
  Drop it (e.g. to 5) for quick tests; raise it for final runs.
- **`n_epochs`** sets how long each member trains.
- **`latent_dims`** controls the capacity of the signal/noise latent spaces.
- **`beta`** trades reconstruction fidelity against latent-space regularization;
  the `scale_MSE_*` weights (v2) balance how much the model cares about
  reconstructing the ROI vs the confounds mask. These interact, so change them
  deliberately and one at a time.

> **Reproducibility tip:** for batch jobs, set `param_epochs` /
> `param_repetitions` low while you debug the pipeline, then scale up for the
> real run. See [batch jobs](batch_jobs.md).

---

**See also:** [High-level API](usage_high_level_api.md) ·
[`deepcor/config.py`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/config.py)
