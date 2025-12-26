"""
DeepCor: Deep Learning-based Denoising for fMRI

A Python package for denoising fMRI data using conditional variational autoencoders.
"""

__version__ = '0.1.0'
__author__ = 'DeepCor Development Team'

# High-level API
from .pipeline import DeepCorDenoiser
from .config import DeepCorConfig, ModelConfig, TrainingConfig, DataConfig

# Models
from .models import CVAE, cVAE, get_model, list_models

# Training
from .training import Trainer, save_model, save_brain_signals

# Data
from .data import (
    TrainDataset,
    get_roi_and_roni,
    get_obs_noi_list_coords,
    apply_dummy,
    censor_and_interpolate,
    apply_frame_censoring,
)

# Analysis
from .analysis import (
    run_correlation_analysis_from_spec,
    run_contrast_analysis_from_spec,
    calc_and_save_compcor,
    average_signal_ensemble,
    get_design_matrix,
)

# Visualization
from .visualization import (
    init_track,
    update_track,
    show_dashboard,
    save_track,
)

# Utilities
from .utils import safe_mkdir, check_gpu_and_speedup

__all__ = [
    # Version
    '__version__',
    '__author__',
    # High-level API
    'DeepCorDenoiser',
    'DeepCorConfig',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    # Models
    'CVAE',
    'cVAE',
    'get_model',
    'list_models',
    # Training
    'Trainer',
    'save_model',
    'save_brain_signals',
    # Data
    'TrainDataset',
    'get_roi_and_roni',
    'get_obs_noi_list_coords',
    'apply_dummy',
    'censor_and_interpolate',
    'apply_frame_censoring',
    # Analysis
    'run_correlation_analysis_from_spec',
    'run_contrast_analysis_from_spec',
    'calc_and_save_compcor',
    'average_signal_ensemble',
    'get_design_matrix',
    # Visualization
    'init_track',
    'update_track',
    'show_dashboard',
    'save_track',
    # Utilities
    'safe_mkdir',
    'check_gpu_and_speedup',
]
