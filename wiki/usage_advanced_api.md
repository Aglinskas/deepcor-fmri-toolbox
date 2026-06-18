# Advanced / low-level API

[← Back to the wiki home](README.md)

The [high-level API](usage_high_level_api.md) (`DeepCor` / `DeepCorDenoiser`)
covers the standard workflow. Drop down to the low-level API when you want to
**customize the pieces** — swap the training loop, inspect latent codes, build
your own ensemble logic, or integrate DeepCor into a larger pipeline.

> If you just want a denoised image, use the [high-level API](usage_high_level_api.md).
> Everything below is what it does internally.

## The building blocks

A denoising run is assembled from these components:

| Step | What you call | Module |
| --- | --- | --- |
| Load images | `ants.image_read(...)` | (ANTsPy) |
| Load confounds | `get_confounds(path)` | [`data.loaders`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/data/loaders.py) |
| Extract ROI/RONI voxels | `get_obs_noi_list` (v1) / `get_obs_noi_list_coords` (v2) | [`data.preprocessing`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/data/preprocessing.py) |
| Build a dataset | `TrainDataset(obs_list, noi_list)` | [`data.datasets`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/data/datasets.py) |
| Build a model | `get_model(version, ...)` or `CVAE` / `CVAE_V1` | [`models`](https://github.com/Aglinskas/deepcor-fmri-toolbox/tree/main/deepcor/models) |
| Train | `Trainer(model).fit(loader, n_epochs=...)` | [`training.trainer`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/training/trainer.py) |
| Generate + save signal | `save_brain_signals(...)` or `model.generate(x)` | [`training.trainer`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/training/trainer.py) |
| Average an ensemble | `average_signal_ensemble(files, ofn)` | [`analysis.metrics`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/analysis/metrics.py) |
| Array → NIfTI | `array_to_brain(arr, epi, gm, ofn)` | [`data.loaders`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/data/loaders.py) |

## Complete v1 example

`v1` is the simplest to drive by hand because it needs no confounds. This is the
whole pipeline, end to end:

```python
import ants, torch
from deepcor.data import get_obs_noi_list, TrainDataset
from deepcor.models import get_model           # or: from deepcor.models import CVAE_V1
from deepcor.training import Trainer, save_brain_signals

# 1. Load images (same space, same grid; masks binary, non-overlapping)
epi = ants.image_read("bold.nii.gz")
gm  = ants.image_read("gm_mask.nii.gz")        # ROI
cf  = ants.image_read("wmcsf_mask.nii.gz")     # RONI

# 2. Extract ROI / RONI voxel time series
obs_list, noi_list, gm, cf = get_obs_noi_list(epi, gm, cf)
nTR = obs_list.shape[-1]
in_channels = obs_list.shape[-2]               # 1 for v1

# 3. Dataset + loader
dataset = TrainDataset(obs_list, noi_list)
loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True, drop_last=True)

# 4. Model
model = get_model("v1", in_channels=in_channels, in_dim=nTR, latent_dim=8, beta=0.01)

# 5. Train
trainer = Trainer(model, lr=1e-3)
trainer.fit(loader, n_epochs=100)

# 6. Generate the denoised signal and write it to a NIfTI
save_brain_signals(model, dataset, epi, gm, ofn="denoised_signal.nii.gz", kind="FG")
```

`kind="FG"` writes the **foreground** (signal) reconstruction — the denoised
output. `"BG"` (background/noise) and `"TG"` (target) are also available for
inspection.

## v2 (confound-aware) notes

v2 follows the same shape with two differences:

- Use **`get_obs_noi_list_coords`** (it appends spatial coordinates, giving 4
  input channels), and load confounds with **`get_confounds`**.
- The `CVAE` (v2) model takes a **confound tensor** at construction. The
  high-level pipeline tiles the confounds to the batch size before building the
  model; if you assemble v2 by hand you must prepare that tensor yourself.

Because of that confound-tensor wiring, **for v2 we recommend using the
[high-level API](usage_high_level_api.md)** (optionally with a custom
[`DeepCorConfig`](configuration.md)) unless you specifically need to replace the
model or training loop. The canonical v2 assembly lives in `DeepCorDenoiser.fit`
in [`deepcor/pipeline.py`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/pipeline.py)
— read it as the reference implementation.

## Key call signatures

Minimum required arguments are in **bold**; the rest are optional with the
defaults shown.

**`get_obs_noi_list(epi, gm, cf)` / `get_obs_noi_list_coords(epi, gm, cf)`**
- **`epi`**, **`gm`**, **`cf`** — ANTs images. Returns `(obs_list, noi_list, gm, cf)`.

**`TrainDataset(X, Y)`**
- **`X`** = observations (ROI), **`Y`** = noise (RONI). A PyTorch `Dataset`.

**`get_model(version="cvae", **kwargs)`**
- **`version`** — `"v1"`, `"v2"`, `"latest"`/`"cvae"`. Remaining kwargs are
  passed to the model constructor (`in_channels`, `in_dim`, `latent_dim`,
  `beta`, and for v2 `conf`, `gamma`, `delta`, `scale_MSE_*`, `do_disentangle`).

**`Trainer(model, device=None, optimizer_type="adamw", lr=0.001, betas=(0.9, 0.999), eps=1e-08, max_grad_norm=5.0)`**
- **`model`** — the only required argument.

**`Trainer.fit(dataloader, n_epochs=100, callbacks=None, verbose=True)`**
- **`dataloader`** — required. Optional `callbacks` (see
  [`training.callbacks`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/training/callbacks.py):
  `TrackingCallback`, `CheckpointCallback`, `EarlyStoppingCallback`).

**`save_brain_signals(model, train_inputs_coords, epi, gm, ofn, batch_size=512, kind="FG", inv_z_score=True)`**
- **`model`**, **`train_inputs_coords`** (the dataset), **`epi`**, **`gm`**,
  **`ofn`** (output path) required. `kind` ∈ `{"FG", "TG", "BG"}`.

**`average_signal_ensemble(signal_files, ofn)`**
- **`signal_files`** — list of per-repetition NIfTI paths; **`ofn`** — output
  path. NaN-containing files are skipped.

**`array_to_brain(arr, epi, gm, ofn, inv_z_score=True, return_img=False)`**
- Map a voxel array back into the brain volume and write it.

## Building an ensemble by hand

To reproduce the high-level ensemble: loop `n_repetitions` times, re-create and
train a fresh model each time, write each `signal_rep_{i}.nii.gz` with
`save_brain_signals`, then average them:

```python
signal_files = []
for i in range(20):
    model = get_model("v1", in_channels=in_channels, in_dim=nTR, latent_dim=8, beta=0.01)
    Trainer(model, lr=1e-3).fit(loader, n_epochs=100)
    ofn = f"out/signal_rep_{i}.nii.gz"
    save_brain_signals(model, dataset, epi, gm, ofn=ofn, kind="FG")
    signal_files.append(ofn)

from deepcor.analysis import average_signal_ensemble
average_signal_ensemble(signal_files, "out/denoised_avg.nii.gz")
```

This is exactly what `DeepCorDenoiser` automates — see [outputs](outputs.md) for
the file layout it produces.

---

**See also:** [High-level API](usage_high_level_api.md) ·
[Configuration reference](configuration.md) ·
[Model versions](model_versions.md) ·
[`02_StudyForrest-advanced-v2.py`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/tests/02_StudyForrest-advanced-v2.py)
