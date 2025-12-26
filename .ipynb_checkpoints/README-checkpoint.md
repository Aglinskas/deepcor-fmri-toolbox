# DeepCor: Deep Learning-based Denoising for fMRI

DeepCor is a Python package for denoising fMRI data using conditional variational autoencoders (cVAE). It provides both high-level and low-level APIs for flexible usage.

## Features

- **Deep learning-based denoising** using conditional VAE architecture
- **Disentangled representations** separating signal from noise
- **Ensemble averaging** for robust denoising
- **High-level API** for easy use (scikit-learn style)
- **Low-level API** for research and customization
- **Comprehensive analysis tools** for correlation and contrast analyses
- **Visualization utilities** for monitoring training

## Installation

### From PyPI (coming soon)

```bash
pip install deepcor
```

### From source

```bash
git clone https://github.com/yourusername/deepcor-fmri-toolbox.git
cd deepcor-fmri-toolbox
pip install -e .
```

## Quick Start

### High-Level API (Recommended for most users)

```python
from deepcor import DeepCorDenoiser

# Initialize denoiser
denoiser = DeepCorDenoiser(
    model_version='cvae',
    latent_dims=(8, 8),
    n_epochs=100,
    batch_size=1024,
    n_repetitions=20
)

# Denoise your fMRI data
output_path = denoiser.fit_denoise(
    epi_path='data.nii.gz',
    gm_mask_path='gm_mask.nii',
    cf_mask_path='cf_mask.nii',
    confounds_path='confounds.tsv',
    output_dir='output/',
    verbose=True
)
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
├── setup.py / pyproject.toml
├── README.md
├── requirements.txt
├── deepcor/
│   ├── __init__.py
│   ├── models/              # Model architectures
│   │   ├── __init__.py
│   │   ├── cvae.py          # cVAE model
│   │   ├── base.py          # Base model class
│   │   └── registry.py      # Model registry
│   ├── data/                # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── loaders.py       # Data loading utilities
│   │   ├── preprocessing.py # Preprocessing functions
│   │   └── datasets.py      # PyTorch datasets
│   ├── training/            # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py       # Training loop
│   │   └── callbacks.py     # Training callbacks
│   ├── analysis/            # Analysis tools
│   │   ├── __init__.py
│   │   ├── contrasts.py     # Contrast analysis
│   │   ├── correlations.py  # Correlation analysis
│   │   └── metrics.py       # Evaluation metrics
│   ├── visualization/       # Visualization tools
│   │   ├── __init__.py
│   │   └── dashboard.py     # Training dashboard
│   ├── utils/               # Utilities
│   │   ├── __init__.py
│   │   ├── io.py            # I/O utilities
│   │   └── helpers.py       # Helper functions
│   ├── pipeline.py          # High-level API
│   └── config.py            # Configuration management
├── examples/
│   ├── quickstart.py
│   └── advanced_usage.py
└── tests/
    └── ...
```

## Key Components

### Models

- **CVAE**: Conditional Variational Autoencoder with disentangled latent spaces
  - `z`: Signal-related latent variables
  - `s`: Noise-related latent variables
- **Model Registry**: Easy access to different model versions

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

See the `examples/` directory for:
- `quickstart.py`: Simple usage example
- `advanced_usage.py`: Full pipeline with analysis

## Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- ANTsPy >= 0.3.0
- nilearn >= 0.8.0
- scikit-learn >= 0.24.0
- numpy, pandas, scipy, matplotlib, seaborn, tqdm

## Citation

If you use DeepCor in your research, please cite:

```
[Citation information to be added]
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/deepcor-fmri-toolbox/issues
- Documentation: https://deepcor.readthedocs.io (coming soon)

## Acknowledgments

[Acknowledgments to be added] 
