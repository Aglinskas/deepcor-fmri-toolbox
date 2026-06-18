# DeepCor: Deep Learning-based Denoising for fMRI

DeepCor is a Python package for denoising fMRI data using contrastive variational autoencoders (CVAEs). It provides both high-level and low-level APIs for flexible usage.

## Features

- **Deep learning-based denoising** using Contrastive VAE architecture
- **Disentangled representations** separating signal from noise
- **Ensemble averaging** for reliable denoising
- **High-level API** for easy use (scikit-learn style)
- **Low-level API** for research and customization
- **Comprehensive analysis tools** for correlation and contrast analyses
- **Visualization utilities** for monitoring training

## Installation

DeepCor depends on PyTorch and ANTsPy, so we recommend installing it into a
fresh, isolated environment. Python 3.10–3.12 is recommended.

### Option A — conda (recommended)

```bash
conda create -n deepcor python=3.11
conda activate deepcor

git clone https://github.com/Aglinskas/deepcor-fmri-toolbox.git
cd deepcor-fmri-toolbox
pip install -e .
```

### Option B — venv (no conda needed)

```bash
python3 -m venv deepcor-env
source deepcor-env/bin/activate     # Windows: deepcor-env\Scripts\activate

git clone https://github.com/Aglinskas/deepcor-fmri-toolbox.git
cd deepcor-fmri-toolbox
pip install -e .
```

> **Note:** This is a large install (several GB) because PyTorch ships with
> GPU/CUDA libraries. On a Linux machine with an NVIDIA GPU it works out of the
> box. If you only have a CPU, or need a specific CUDA version, install the
> matching PyTorch build first from https://pytorch.org/get-started/locally/
> *before* running `pip install -e .`.

### From PyPI (coming soon)

```bash
pip install deepcor
```

## Quick Start

### High-Level API (Recommended for most users)

```python
from deepcor import DeepCorDenoiser

# Initialize denoiser
denoiser = DeepCorDenoiser(
    # model_version:
    # - 'v1': original CVAE implementation
    # - 'v2': current CVAE implementation
    # - 'latest': alias for the current recommended model (currently v2)
    model_version='latest',
    latent_dims=(8, 8),
    n_epochs=100,
    batch_size=1024,
    n_repetitions=20
)

# Denoise your fMRI data
result = denoiser.fit_denoise(
    epi='data.nii.gz',
    gm_mask='gm_mask.nii',
    cf_mask='cf_mask.nii',
    confounds='confounds.tsv',
    output_dir='output/',
    verbose=True
)

# `result` is a DeepCorResult; the denoised file path is on `result.denoised_path`
print(result.denoised_path)
```

### Low-Level API (For researchers)

```python
from deepcor.models import CVAE
from deepcor.data import TrainDataset, get_obs_noi_list_coords
from deepcor.training import Trainer
import torch

# Prepare data
obs_list, noi_list, gm, cf = get_obs_noi_list_coords(epi, gm, cf)
dataset = TrainDataset(obs_list, noi_list)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512)

# Create model
model = CVAE(
    conf=confounds,
    in_channels=4,
    in_dim=n_timepoints,
    latent_dim=(8, 8),
    beta=0.01
)

# Train
trainer = Trainer(model, lr=1e-3)
trainer.fit(dataloader, n_epochs=100)

# Generate denoised data
denoised = model.generate(data)
```

## Package Structure

```
deepcor/
├── pyproject.toml
├── README.md
├── deepcor/
│   ├── __init__.py
│   ├── models/              # Model architectures
│   ├── data/                # Data loading and preprocessing
│   ├── training/            # Training utilities
│   ├── analysis/            # Analysis tools
│   ├── visualization/       # Visualization tools
│   ├── utils/               # Utilities
│   ├── pipeline.py          # High-level API
│   └── config.py            # Configuration management
├── examples/                # Example scripts and notebooks
└── tests/                   # Test scripts and notebooks
```

The sections below break down each module and list the functions/classes it
exposes, with a brief note on what each one does. Click any file name to view
its source on GitHub.

### [`deepcor/pipeline.py`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/pipeline.py) — high-level API

- **`DeepCorDenoiser`** — scikit-learn-style high-level denoiser; the main entry point for most users.
  - `.fit()` — train the ensemble of models on an EPI run (without writing output).
  - `.denoise()` — average the trained ensemble and write the denoised output (plus a preprocessed copy and a CompCor comparison).
  - `.fit_denoise()` — convenience wrapper that runs `.fit()` then `.denoise()`.
  - `.save()` — save the trained ensemble of models to disk.
- **`DeepCorResult`** — dataclass holding the artifacts of a run (`denoised_path`, `preproc_path`, `compcor_path`, `output_dir`, `signal_files`, `tracks`); usable directly as a path string.
- **`DeepCor`** — friendlier alias for `DeepCorDenoiser`.

### [`deepcor/config.py`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/config.py) — configuration

- **`ModelConfig`** — dataclass for model architecture settings (latent dims, beta, etc.).
- **`TrainingConfig`** — dataclass for training settings (epochs, batch size, learning rate).
- **`DataConfig`** — dataclass for data-processing settings.
- **`DeepCorConfig`** — top-level config bundling the three configs above.
- `get_default_config()` — return a `DeepCorConfig` populated with defaults.

### [`deepcor/models/`](https://github.com/Aglinskas/deepcor-fmri-toolbox/tree/main/deepcor/models) — model architectures

[**`cvae.py`**](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/models/cvae.py) — current (v2) confound-aware Contrastive VAE.
- **`CVAE`** — Contrastive VAE for fMRI denoising with disentangled signal/noise latent spaces.
  - `.encode_z()` / `.encode_s()` — encode input to the signal / noise latent space.
  - `.encode()` — encode input to both latent spaces.
  - `.decode()` — decode latent codes back to a time series.
  - `.forward_tg()` / `.forward_fg()` / `.forward_bg()` — forward pass for target (ROI), foreground (signal), and background (noise) branches.
  - `.forward()` — standard forward pass.
  - `.ncc()` — normalized cross-correlation loss term.
  - `.loss_function()` — VAE loss with disentanglement terms.
  - `.generate()` — produce denoised output from input.
- **`GradientReversalFunction`** / **`GradientReversalLayer`** — gradient reversal used for adversarial confound removal.
- `compute_in()`, `compute_in_size()`, `compute_out_size()`, `compute_padding()` — helpers that compute conv/deconv sizes and padding for the architecture.

[**`cvae_v1.py`**](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/models/cvae_v1.py) — original CVAE without confound conditioning.
- **`CVAE_V1`** — original (v1) cVAE model.
  - `.encode_z()` / `.encode_s()` — encode to signal / noise latent space.
  - `.decode()` — decode concatenated latent codes to a time series.
  - `.reparameterize()` — reparameterization trick for sampling.
  - `.forward_tg()` / `.forward_bg()` / `.forward_fg()` — target / background / foreground forward passes.
  - `.loss_function()` — reconstruction + KL loss.
  - `.sample()` — sample from the latent prior and decode.
  - `.generate()` — produce denoised output (foreground branch).

[**`base.py`**](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/models/base.py)
- **`BaseModel`** — abstract base class defining the model interface (`encode`, `decode`, `forward`, `loss_function`, `generate`, `reparameterize`).

[**`registry.py`**](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/models/registry.py)
- `get_model()` — instantiate a model by version string (`'v1'`, `'v2'`, `'latest'`).
- `list_models()` — list available model versions.

### [`deepcor/data/`](https://github.com/Aglinskas/deepcor-fmri-toolbox/tree/main/deepcor/data) — data loading and preprocessing

[**`loaders.py`**](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/data/loaders.py)
- `get_confounds()` — load motion confounds from an fMRIPrep TSV file.
- `plot_timeseries()` — plot ROI and RONI timeseries.
- `array_to_brain()` — convert a voxel array back to a brain volume (NIfTI).
- `load_pickle()` / `save_pickle()` — pickle I/O helpers.

[**`preprocessing.py`**](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/data/preprocessing.py)
- `get_roi_and_roni()` — build ROI (gray matter) and RONI (non-gray-matter) masks.
- `get_obs_noi_list_coords()` — extract observation/noise voxel lists with coordinates.
- `get_obs_noi_list()` — extract observation/noise voxel lists.
- `apply_dummy()` — drop dummy scans from EPI and confounds.
- `censor_and_interpolate()` — censor and interpolate bad timepoints.
- `apply_frame_censoring()` — apply motion-based frame censoring to the data.
- `remove_std0()` — drop voxels with zero standard deviation.
- `regress_from_data()` — regress a set of nuisance regressors out of the data.

[**`datasets.py`**](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/data/datasets.py)
- **`TrainDataset`** — PyTorch `Dataset` pairing observation (ROI) and noise (RONI) samples.

### [`deepcor/training/`](https://github.com/Aglinskas/deepcor-fmri-toolbox/tree/main/deepcor/training) — training utilities

[**`trainer.py`**](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/training/trainer.py)
- **`Trainer`** — training loop, optimization, and checkpointing.
  - `.train_epoch()` — run a single training epoch.
  - `.fit()` — train the model for N epochs.
  - `.save_checkpoint()` / `.load_checkpoint()` — checkpoint I/O.
- `save_model()` — legacy checkpoint saver kept for backward compatibility.
- `save_brain_signals()` — generate and save denoised brain signals from a trained model.

[**`callbacks.py`**](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/training/callbacks.py)
- **`TrackingCallback`** — record training metrics into a tracking dict.
- **`CheckpointCallback`** — periodically save checkpoints during training.
- **`EarlyStoppingCallback`** — stop training when loss stops improving.

### [`deepcor/analysis/`](https://github.com/Aglinskas/deepcor-fmri-toolbox/tree/main/deepcor/analysis) — analysis tools

[**`contrasts.py`**](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/analysis/contrasts.py)
- `get_design_matrix()` — build a first-level GLM design matrix from an events file.
- `get_contrast_val()` — compute per-voxel contrast values.
- `calc_contrast_map()` — compute a whole-brain contrast map.
- `run_contrast_analysis_from_spec()` — run a contrast analysis from a spec dictionary.

[**`correlations.py`**](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/analysis/correlations.py)
- `correlate_columns()` — Pearson correlation between matching columns of two matrices.
- `calc_corr_map()` — compute a correlation map between an image and a target.
- `run_correlation_analysis_from_spec()` — run a correlation analysis from a spec dictionary.

[**`metrics.py`**](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/analysis/metrics.py)
- `calc_mse()` — variance explained (R²) between two arrays.
- `calc_and_save_compcor()` — compute and save CompCor-denoised data for comparison.
- `average_signal_ensemble()` — average multiple denoised signal files into an ensemble.
- `correlation()` — correlation between two vectors.

### [`deepcor/visualization/`](https://github.com/Aglinskas/deepcor-fmri-toolbox/tree/main/deepcor/visualization) — visualization tools

[**`dashboard.py`**](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/visualization/dashboard.py)
- `get_varexp()` — compute variance explained between model input and output.
- `init_track()` — initialize a tracking dictionary for a model version.
- `update_track()` — update the tracking dict with current training state/metrics.
- `save_track()` — save a tracking dictionary to file.
- `format_progress_title()` — build a progress/title string from the current track.
- `show_dahsboard_v1_marimo()` / `show_dahsboard_v2_marimo()` — render the V1 / V2 training dashboard as a matplotlib figure (for marimo notebooks).
- `show_dahsboard_v1_jupyter()` / `show_dahsboard_v2_jupyter()` — same dashboards for Jupyter notebooks.
- `show_dahsboard_marimo()` / `show_dahsboard_jupyter()` — dispatch to the right dashboard for the model version in the track (latest by default).

[**`plots.py`**](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/visualization/plots.py)
- `plot_timeseries()` — plot ROI/RONI timeseries.

### [`deepcor/utils/`](https://github.com/Aglinskas/deepcor-fmri-toolbox/tree/main/deepcor/utils) — utilities

[**`io.py`**](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/utils/io.py)
- `safe_mkdir()` — create a directory if it doesn't already exist.

[**`helpers.py`**](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/deepcor/utils/helpers.py)
- `check_gpu_and_speedup()` — check GPU availability and benchmark speedup vs CPU.

## Key Components

### Models

- **CVAE**: Contrastive Variational Autoencoder with disentangled latent spaces
  - `s`: Signal-related latent variables
  - `z`: Noise-related latent variables
- **Model Registry**: Access to different model versions

### Data Processing

- **ROI/RONI extraction**: Separate signal and noise regions
- **Frame censoring**: Handle motion artifacts
- **Coordinate encoding**: Spatial information preservation
- **Data augmentation**: Improve model robustness

### Training

- **Trainer class**: Handles training loop, optimization, and checkpointing
- **Callbacks**: Monitoring, early stopping, checkpointing
- **Ensemble training**: Multiple repetitions for robust results

### Analysis

- **Correlation analysis**: Correlate denoised data with task regressors
- **Contrast analysis**: GLM-based contrast analysis
- **CompCor comparison**: Automatic comparison with CompCor denoising
- **Metrics**: Variance explained, correlation, etc.

## Configuration

DeepCor uses dataclasses for configuration management:

```python
from deepcor import DeepCorConfig, ModelConfig, TrainingConfig

config = DeepCorConfig()
config.model.latent_dims = (16, 16)
config.training.n_epochs = 200
config.training.batch_size = 2048

denoiser = DeepCorDenoiser(config=config)
```

## Examples

See the [`examples/`](https://github.com/Aglinskas/deepcor-fmri-toolbox/tree/main/examples) directory for:
- [`01-quickstart_mo.py`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/examples/01-quickstart_mo.py): simple usage example (marimo notebook)
- [`advanced_usage.py`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/examples/advanced_usage.py): full pipeline with analysis
- [`02-Advanced-Usage.ipynb`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/examples/02-Advanced-Usage.ipynb) and [`03-StudyForrest.ipynb`](https://github.com/Aglinskas/deepcor-fmri-toolbox/blob/main/examples/03-StudyForrest.ipynb): worked Jupyter notebook examples

## Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- ANTsPy >= 0.3.0
- nilearn >= 0.8.0
- scikit-learn >= 0.24.0
- numpy, pandas, matplotlib, tqdm, IPython

## Citation

If you find DeepCor useful, please cite:

```
Zhu, Y., Aglinskas, A., & Anzellotti, S. (2025). DeepCor: denoising fMRI data with contrastive autoencoders. Nature Methods, 1-4.
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are very welcome! Fork, debug, experiment and submit a pull request!

## Support

For issues and questions:
- GitHub Issues: https://github.com/Aglinskas/deepcor-fmri-toolbox/issues
- Documentation: https://deepcor.readthedocs.io (coming soon)

## Acknowledgments

Original CVAE implementation: Abid & Zou (2019) https://arxiv.org/abs/1902.04601
Original DeepCor publication Zhu, Aglinskas & Anzellotti (2025) https://www.nature.com/articles/s41592-025-02967-x
