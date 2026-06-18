# High-level API (`DeepCorDenoiser` / `DeepCor`)

[← Back to the wiki home](README.md)

The high-level API is the recommended way to use DeepCor. It wraps the whole
workflow — load data, train an ensemble of CVAEs, average them, and write the
denoised image — behind a scikit-learn–style object. If you've already prepared
your inputs (see [Data preparation](data_preparation.md)), denoising a run is a
single call.

`DeepCor` is just a friendlier alias for `DeepCorDenoiser`; they are identical.

## The one-liner

```python
from deepcor import DeepCor

denoiser = DeepCor(model_version="latest")

result = denoiser.fit_denoise(
    epi="sub-01_task-rest_desc-preproc_bold.nii.gz",
    gm_mask="sub-01_label-GM_mask.nii.gz",
    cf_mask="sub-01_label-WMCSF_mask.nii.gz",   # confounds mask: white matter + CSF
    confounds="sub-01_task-rest_desc-confounds_timeseries.tsv",
    output_dir="derivatives/deepcor/sub-01",
)

print(result.denoised_path)   # path to the denoised NIfTI
```

`fit_denoise` returns a [`DeepCorResult`](#what-you-get-back). It can also be
used directly as a path string (e.g. `ants.image_read(result)`), because it
points at the denoised file.

## Required vs optional parameters

### Creating the denoiser — `DeepCor(...)`

**Everything is optional** and has a sensible default. The ones you're most
likely to touch:

| Parameter | Default | What it does |
| --- | --- | --- |
| `model_version` | `"latest"` | `"v1"`, `"v2"`, or `"latest"`/`"cvae"` (currently → `v2`). See [model versions](model_versions.md) *(planned)*. |
| `latent_dims` | `(8, 8)` | `(signal_dim, noise_dim)` of the latent space. `v1` uses only the first. |
| `n_epochs` | `100` | Training epochs per ensemble member. |
| `batch_size` | `1024` | Training batch size. |
| `learning_rate` | `0.001` | Optimizer learning rate. |
| `n_repetitions` | `20` | How many models in the ensemble (averaged at the end). More = more stable, but slower. |
| `config` | `None` | A full `DeepCorConfig` (overrides the scalar args above). See [configuration](configuration.md) *(planned)*. |
| `device` | `None` | torch device; defaults to GPU if available, else CPU. |
| `verbose` | `True` | Print device/progress info. |

### Running it — `fit_denoise(...)`

| Parameter | Required? | Default | Notes |
| --- | --- | --- | --- |
| `epi` | **Yes** | — | 4D functional image (path or ANTs image). |
| `gm_mask` | **Yes** | — | Gray-matter ROI mask. |
| `cf_mask` | **Yes** | — | Confounds/RONI mask — usually white-matter + CSF (aCompCor-style). |
| `confounds` | **For `v2`/`latest`** | `None` | fMRIPrep confounds TSV (or `(n_conf, nTR)` array). Required for confound-aware models, ignored for `v1`. |
| `output_dir` | No | `../Data/DeepCor-Outputs` | Where checkpoints, tracks, signal NIfTIs and the denoised output are written (created if missing). |
| `dashboard` | No | `"save"` | `"save"` (write a dashboard PNG each epoch), `"jupyter"` (also display it live in a notebook), or `None` (no dashboard). |
| `subject_idx`, `run_idx` | No | `0`, `0` | Labels used to name the dashboard PNG and the progress line (e.g. `S1R0`). |
| `verbose` | No | `True` | Print a per-epoch progress line. |

So the **minimum call** for the default (confound-aware) model is:

```python
DeepCor().fit_denoise(epi, gm_mask, cf_mask, confounds)
```

and for the original `v1` model (no confounds needed):

```python
DeepCor(model_version="v1").fit_denoise(epi, gm_mask, cf_mask)
```

## `fit` and `denoise` separately

`fit_denoise` is just a convenience wrapper. You can split the two steps — train
once, then write outputs — which is useful if you want to inspect the trained
ensemble first:

```python
denoiser = DeepCor(model_version="latest")
denoiser.fit(epi, gm_mask, cf_mask, confounds, output_dir="out/")   # train
result = denoiser.denoise(output_dir="out/")                        # write outputs
```

- `fit(...)` trains the ensemble and returns the denoiser (`self`).
- `denoise(...)` averages the trained ensemble and writes the denoised image
  (plus a preprocessed copy and a CompCor comparison), returning a
  `DeepCorResult`.

You can also persist the trained models:

```python
denoiser.save("out/models")   # writes one checkpoint per ensemble member
```

## What you get back

`fit_denoise` / `denoise` return a **`DeepCorResult`** with these fields:

| Field | Meaning |
| --- | --- |
| `denoised_path` | Path to the denoised functional image (the main output) |
| `preproc_path` | Path to a preprocessed (undenoised) copy, for comparison |
| `compcor_path` | Path to a CompCor-denoised version, as a baseline comparison |
| `output_dir` | The directory all of the above were written to |
| `signal_files` | The per-repetition denoised signal files that were averaged |
| `tracks` | Per-repetition training-tracking dicts (losses, metrics over epochs) |

A dedicated [Outputs explained](outputs.md) page *(planned)* will go deeper on
the file naming and how to use each one.

## Watching training

By default (`dashboard="save"`) DeepCor writes a dashboard PNG each epoch into
`output_dir`, showing training progress without needing a display. In a Jupyter
notebook, pass `dashboard="jupyter"` to render it live in the cell. Pass
`dashboard=None` to skip it entirely (slightly faster, e.g. for batch jobs).

## Tuning, briefly

- **`n_repetitions`** is the main quality/speed dial — the ensemble is averaged,
  so more repetitions give a more stable denoised result at a linear cost in
  time.
- **`n_epochs`** controls how long each member trains.
- **`latent_dims`** and **`beta`** (a `config`/`ModelConfig` setting) shape the
  signal/noise disentanglement. Defaults are a good starting point; see the
  [configuration reference](configuration.md) *(planned)* before changing them.

## Full configuration

For finer control (optimizer settings, disentanglement weights, dummy-scan
removal, censoring, etc.), build a `DeepCorConfig` and pass it via `config=`:

```python
from deepcor import DeepCor, DeepCorConfig

config = DeepCorConfig()
config.model.latent_dims = (16, 16)
config.training.n_epochs = 200
config.data.n_dummy_scans = 5

denoiser = DeepCor(model_version="latest", config=config)
```

The full list of config fields and their defaults will live on the
[configuration reference](configuration.md) page *(planned)*.

---

**See also:** [Data preparation](data_preparation.md) ·
[project README](../README.md) ·
[`examples/01-quickstart_mo.py`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/examples/01-quickstart_mo.py)
