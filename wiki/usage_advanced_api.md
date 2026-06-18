# Advanced / low-level API

[← Back to the wiki home](README.md)

The [high-level API](usage_high_level_api.md) (`DeepCor` / `DeepCorDenoiser`)
covers the standard workflow. Drop down to the low-level API when you want to
**customize the pieces** — swap the training loop, inspect latent codes, build
your own ensemble logic, or integrate DeepCor into a larger pipeline.

This page builds the full pipeline by hand for the **recommended `v2`
(confound-aware) model**, explaining each step, then shows the simpler `v1`
variant at the end.

> If you just want a denoised image, use the [high-level API](usage_high_level_api.md).
> Everything below is exactly what `DeepCorDenoiser.fit` /`.denoise` do
> internally — the canonical reference implementation lives in
> [`deepcor/pipeline.py`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/pipeline.py).

## The building blocks

A denoising run is assembled from these components:

| Step | What you call | Module |
| --- | --- | --- |
| Load images | `ants.image_read(...)` | (ANTsPy) |
| Load confounds | `get_confounds(path)` | [`data.loaders`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/data/loaders.py) |
| Extract ROI/RONI voxels | `get_obs_noi_list_coords` (v2) / `get_obs_noi_list` (v1) | [`data.preprocessing`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/data/preprocessing.py) |
| Build a dataset | `TrainDataset(obs_list, noi_list)` | [`data.datasets`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/data/datasets.py) |
| Build a model | `get_model(version, ...)` or `CVAE` / `CVAE_V1` | [`models`](https://github.com/Aglinskas/deepcor-fmri-toolbox/tree/main/deepcor/models) |
| Train | `Trainer(model).fit(loader, n_epochs=...)` | [`training.trainer`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/training/trainer.py) |
| Generate + save signal | `save_brain_signals(...)` | [`training.trainer`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/training/trainer.py) |
| Average an ensemble | `average_signal_ensemble(files, ofn)` | [`analysis.metrics`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/analysis/metrics.py) |
| Array → NIfTI | `array_to_brain(arr, epi, gm, ofn)` | [`data.loaders`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/data/loaders.py) |

---

## Complete v2 example (recommended)

This is the whole confound-aware pipeline, end to end — load data, train an
**ensemble** of CVAEs, and average them into a denoised image. Read the
step-by-step notes below the code; one step (building the confound tensor) has a
subtlety that's easy to get wrong.

```python
import ants
import torch
import numpy as np

from deepcor.data import get_obs_noi_list_coords, get_confounds, TrainDataset
from deepcor.models import get_model
from deepcor.training import Trainer, save_brain_signals
from deepcor.analysis import average_signal_ensemble
from deepcor.utils import safe_mkdir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. Load inputs. epi/gm/cf must share the same voxel grid; the masks must be
#    binary and non-overlapping (see the Data preparation page).
# ---------------------------------------------------------------------------
epi = ants.image_read("bold.nii.gz")
gm  = ants.image_read("gm_mask.nii.gz")        # ROI  (gray matter)
cf  = ants.image_read("wmcsf_mask.nii.gz")     # RONI (confounds mask: WM + CSF)

# ---------------------------------------------------------------------------
# 2. Extract the ROI/RONI voxel time series. The *_coords loader appends each
#    voxel's 3 spatial coordinates to its time series, which is what makes this
#    the v2 input (4 channels instead of 1).
# ---------------------------------------------------------------------------
obs_list, noi_list, gm, cf = get_obs_noi_list_coords(epi, gm, cf)
nTR = obs_list.shape[-1]            # number of timepoints
in_channels = obs_list.shape[-2]   # 4 for v2 (signal + x, y, z coordinates)

# ---------------------------------------------------------------------------
# 3. Load the 6 motion confounds as a (n_confounds, nTR) array. The loader
#    auto-detects fMRIPrep's column names (trans_*/rot_* or X/Y/Z/RotX/...).
# ---------------------------------------------------------------------------
conf = get_confounds("confounds.tsv")

# ---------------------------------------------------------------------------
# 4. Dataset + DataLoader. NOTE: drop_last=True is required for v2 (see below).
# ---------------------------------------------------------------------------
batch_size = 1024
dataset = TrainDataset(obs_list, noi_list)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

# ---------------------------------------------------------------------------
# 5. Build the confound tensor. The v2 model stores the confounds internally and
#    its loss indexes them per-sample, so we tile the (n_conf, nTR) array across
#    the batch -> shape (batch_size, n_conf, nTR). This batch dimension MUST match
#    the loader's batch_size (hence drop_last=True, so every batch is full).
# ---------------------------------------------------------------------------
conf_tensor = torch.tensor(np.array([conf for _ in range(batch_size)]))

# ---------------------------------------------------------------------------
# 6. Ensemble loop: train n_repetitions independent models, writing each one's
#    denoised signal to disk, then average them. A fresh model per iteration is
#    what makes it an ensemble.
# ---------------------------------------------------------------------------
n_repetitions = 20
n_epochs = 100
output_dir = "out"
safe_mkdir(output_dir)

signal_files = []
for rep in range(n_repetitions):
    # A new, randomly-initialized v2 model. The disentanglement weights
    # (gamma/delta/scale_MSE_*) match the defaults; tune them via the config.
    model = get_model(
        "v2",
        conf=conf_tensor,
        in_channels=in_channels,
        in_dim=nTR,
        latent_dim=(8, 8),   # (signal_dim, noise_dim)
        beta=0.01,           # KL weight
        gamma=0.0,           # total-correlation weight
        delta=0.0,           # disentanglement weight
        scale_MSE_GM=1e3,    # ROI reconstruction weight
        scale_MSE_CF=1e3,    # RONI reconstruction weight
        scale_MSE_FG=0.0,    # foreground reconstruction weight
        do_disentangle=True,
    ).to(device)

    # Train this ensemble member.
    trainer = Trainer(model, device=device, lr=1e-3)
    trainer.fit(loader, n_epochs=n_epochs)

    # Generate the denoised (foreground) signal and write it to a NIfTI.
    ofn = f"{output_dir}/signal_rep_{rep}.nii.gz"
    save_brain_signals(model, dataset, epi, gm, ofn=ofn, kind="FG")
    signal_files.append(ofn)

# ---------------------------------------------------------------------------
# 7. Average the per-member signals into the final denoised image.
# ---------------------------------------------------------------------------
average_signal_ensemble(signal_files, f"{output_dir}/denoised_avg.nii.gz")
```

### Step-by-step notes

1. **Inputs and grid.** `get_obs_noi_list_coords` reads the EPI through the two
   masks voxel-for-voxel, so they must be on the **same grid**. It also drops
   near-zero-variance voxels for you. If your masks overlap, it raises
   `ROI and RONI masks overlap`.
2. **Why coordinates?** The `*_coords` loader appends each voxel's `(x, y, z)`
   to its time series. That spatial context is the difference between the v2
   input (4 channels) and v1 (1 channel), and it's why you must use this loader
   for v2 — `in_channels` comes straight from the data
   (`obs_list.shape[-2]`).
3. **Confounds.** `get_confounds` returns a `(n_confounds, nTR)` array (6 motion
   parameters by default). To use a custom column set, pass
   `get_confounds(path, columns=[...])`.
4. **`drop_last=True` is not optional here.** See step 5.
5. **The confound tensor — the one tricky bit.** The v2 model keeps the
   confounds *inside* the model (as a buffer) and its loss indexes them with a
   leading batch dimension. So you tile the `(n_conf, nTR)` array to
   `(batch_size, n_conf, nTR)`. Because that batch size is baked in, **every
   training batch must be exactly `batch_size`** — which is why the loader uses
   `drop_last=True`. (You don't move `conf_tensor` to the GPU yourself: it's
   registered as a model buffer, so `model.to(device)` moves it along with the
   weights.)
6. **The ensemble loop.** Each iteration builds a **fresh** model (new random
   init), trains it, and saves its denoised signal. `kind="FG"` writes the
   **foreground** — the reconstruction from the signal latent with the noise
   latent zeroed, i.e. the denoised output. (`"BG"` gives the noise component
   and `"TG"` the full reconstruction, if you want to inspect them.) The
   `Trainer` holds the model, so the loop never has to pass confounds around.
7. **Averaging.** `average_signal_ensemble` reads the per-member NIfTIs and
   averages them, **skipping any file containing NaNs**. The result is your
   denoised image. This is exactly the set of files the high-level API produces
   — see [outputs](outputs.md).

> **Mapping back to the high-level API:** this loop is `DeepCorDenoiser.fit`
> (steps 1–6, written to `output_dir`) followed by `.denoise` (step 7, plus the
> preprocessed and CompCor comparison images). If you only need to change one
> thing (e.g. the loss weights), it's usually less error-prone to pass a custom
> [`DeepCorConfig`](configuration.md) to the high-level API than to reassemble
> the pipeline here.

## Key call signatures

Minimum required arguments are in **bold**; the rest are optional with the
defaults shown.

**`get_obs_noi_list_coords(epi, gm, cf)` / `get_obs_noi_list(epi, gm, cf)`**
- **`epi`**, **`gm`**, **`cf`** — ANTs images. Returns
  `(obs_list, noi_list, gm, cf)`. Use `_coords` for v2, the plain version for v1.

**`get_confounds(confounds_path, columns=None, norm=False)`**
- **`confounds_path`** — TSV path. `columns` overrides auto-detection; `norm` ∈
  `{False, "zscore", "0-1"}`. Returns `(n_confounds, nTR)`.

**`TrainDataset(X, Y)`**
- **`X`** = observations (ROI), **`Y`** = noise (RONI). A PyTorch `Dataset`.

**`get_model(version="cvae", **kwargs)`**
- **`version`** — `"v1"`, `"v2"`, `"latest"`/`"cvae"`. Remaining kwargs go to the
  model constructor. For **v2** (`CVAE`): **`conf`**, **`in_channels`**,
  **`in_dim`**, **`latent_dim`**, plus optional `beta`, `gamma`, `delta`,
  `scale_MSE_GM`, `scale_MSE_CF`, `scale_MSE_FG`, `do_disentangle`.

**`Trainer(model, device=None, optimizer_type="adamw", lr=0.001, betas=(0.9, 0.999), eps=1e-08, max_grad_norm=5.0)`**
- **`model`** — the only required argument.

**`Trainer.fit(dataloader, n_epochs=100, callbacks=None, verbose=True)`**
- **`dataloader`** — required. Optional `callbacks` (see
  [`training.callbacks`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/training/callbacks.py):
  `TrackingCallback`, `CheckpointCallback`, `EarlyStoppingCallback`).
- `Trainer.train_epoch(dataloader)` runs a single epoch if you want to drive the
  loop yourself; `save_checkpoint` / `load_checkpoint` handle model I/O.

**`save_brain_signals(model, train_inputs_coords, epi, gm, ofn, batch_size=512, kind="FG", inv_z_score=True)`**
- **`model`**, **`train_inputs_coords`** (the dataset), **`epi`**, **`gm`**,
  **`ofn`** (output path) required. `kind` ∈ `{"FG", "TG", "BG"}`.

**`average_signal_ensemble(signal_files, ofn)`**
- **`signal_files`** — list of per-repetition NIfTI paths; **`ofn`** — output
  path. NaN-containing files are skipped.

**`array_to_brain(arr, epi, gm, ofn, inv_z_score=True, return_img=False)`**
- Map a voxel array back into the brain volume and write it.

---

## v1 example (no confounds)

`v1` is the original cVAE — no confound conditioning, a single latent dimension,
and a plain reconstruction + KL loss. It's simpler to drive by hand because
there's **no confound tensor** and the data loader doesn't append coordinates
(so `in_channels` is 1). Use it to reproduce the original method, when you have
no usable confounds, or as a comparison against v2.

```python
import ants
import torch
from deepcor.data import get_obs_noi_list, TrainDataset
from deepcor.models import get_model
from deepcor.training import Trainer, save_brain_signals
from deepcor.analysis import average_signal_ensemble

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epi = ants.image_read("bold.nii.gz")
gm  = ants.image_read("gm_mask.nii.gz")        # ROI
cf  = ants.image_read("wmcsf_mask.nii.gz")     # RONI

# Plain loader: no coordinates -> in_channels == 1
obs_list, noi_list, gm, cf = get_obs_noi_list(epi, gm, cf)
nTR = obs_list.shape[-1]
in_channels = obs_list.shape[-2]               # 1 for v1

dataset = TrainDataset(obs_list, noi_list)
loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True, drop_last=True)

signal_files = []
for rep in range(20):
    model = get_model("v1", in_channels=in_channels, in_dim=nTR, latent_dim=8, beta=0.01).to(device)
    Trainer(model, device=device, lr=1e-3).fit(loader, n_epochs=100)
    ofn = f"out/signal_rep_{rep}.nii.gz"
    save_brain_signals(model, dataset, epi, gm, ofn=ofn, kind="FG")
    signal_files.append(ofn)

average_signal_ensemble(signal_files, "out/denoised_avg.nii.gz")
```

The shape mirrors the v2 example, minus confounds: `get_obs_noi_list` instead of
`get_obs_noi_list_coords`, no `conf`/`conf_tensor`, and `get_model("v1", ...)`
with a scalar `latent_dim`. See [model versions](model_versions.md) for the full
comparison.

---

**See also:** [High-level API](usage_high_level_api.md) ·
[Configuration reference](configuration.md) ·
[Model versions](model_versions.md) ·
[`02_StudyForrest-advanced-v2.py`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/tests/02_StudyForrest-advanced-v2.py)
