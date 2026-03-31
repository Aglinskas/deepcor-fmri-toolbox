# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.21.1",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full", auto_download=["ipynb", "html"])


@app.cell
def _():
    import os
    import numpy as np
    import pandas as pd
    import torch
    import ants
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    return (os,)


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    # Import DeepCor configuration classes
    from deepcor.config import (
        DeepCorConfig,
        ModelConfig,
        TrainingConfig,
        DataConfig,
    )

    # Import models
    from deepcor.models import CVAE, cVAE, get_model, list_models

    # Import training utilities
    from deepcor.training import Trainer, save_model, save_brain_signals

    # Import data utilities
    from deepcor.data import (
        TrainDataset,
        get_roi_and_roni,
        get_obs_noi_list_coords,
        apply_dummy,
        censor_and_interpolate,
        apply_frame_censoring,
        plot_timeseries,
        array_to_brain,
        load_pickle,
        save_pickle,
    )

    # Import analysis utilities
    from deepcor.analysis import (
        run_correlation_analysis_from_spec,
        run_contrast_analysis_from_spec,
        calc_and_save_compcor,
        average_signal_ensemble,
        get_design_matrix,
    )

    # Import visualization utilities
    from deepcor.visualization import (
        init_track,
        update_track,
        show_dashboard,
        save_track,
    )

    # Import general utilities
    from deepcor.utils import safe_mkdir, check_gpu_and_speedup

    # Import high-level API
    from deepcor.pipeline import DeepCorDenoiser

    return (
        DataConfig,
        DeepCorConfig,
        ModelConfig,
        TrainingConfig,
        check_gpu_and_speedup,
    )


@app.cell
def _(check_gpu_and_speedup):
    check_gpu_and_speedup()
    return


@app.cell
def _(DataConfig, DeepCorConfig, ModelConfig, TrainingConfig):
    print("=" * 80)
    print("EXAMPLE 2: Configuration Classes")
    print("=" * 80)

    # Example 2a: Using individual config classes
    print("\n--- Individual Configuration Classes ---")

    # ModelConfig: Configure the model architecture
    model_config = ModelConfig(
        latent_dims=(8, 8),  # (shared dim, specific dim)
        beta=0.01,           # KLD loss weight
        gamma=0.0,           # TC loss weight
        delta=0.0,           # RONI zero constraint weight
        scale_MSE_GM=1e3,    # Gray matter reconstruction loss scale
        scale_MSE_CF=1e3,    # Non-gray matter reconstruction loss scale
        scale_MSE_FG=0.0,    # Foreground reconstruction loss scale
        do_disentangle=True  # Enable disentanglement
    )
    print(f"ModelConfig: latent_dims={model_config.latent_dims}, beta={model_config.beta}")

    # TrainingConfig: Configure training parameters
    training_config = TrainingConfig(
        n_epochs=100,
        batch_size=1024,
        learning_rate=0.001,
        optimizer='adamw',
        betas=(0.9, 0.999),
        eps=1e-08,
        max_grad_norm=5.0,
        n_repetitions=20  # Number of ensemble repetitions
    )
    print(f"TrainingConfig: n_epochs={training_config.n_epochs}, batch_size={training_config.batch_size}")

    # DataConfig: Configure data preprocessing
    data_config = DataConfig(
        n_dummy_scans=0,
        apply_censoring=False,
        censoring_threshold=0.5,
        confound_columns=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
    )

    # Create a complete configuration
    config = DeepCorConfig(
        model=model_config,
        training=training_config,
        data=data_config
    )
    return


@app.cell
def _(os):
    s = 0
    r = 1
    sub = 'sub-01'
    indir = '../Data/fmriprep-forrest'
    analysis_name = 'deepcor-advanced-example'

    # Input file paths
    epi_fn = os.path.join(
        indir,
        f'{sub}/ses-localizer/func/{sub}_ses-localizer_task-objectcategories_run-{r}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
    )
    cf_fn = os.path.join(indir, 'mask_roni.nii')  # Non-gray matter mask
    gm_fn = os.path.join(indir, 'mask_roi.nii')   # Gray matter mask
    conf_fn = os.path.join(indir,f'{sub}/ses-localizer/func/{sub}_ses-localizer_task-objectcategories_run-{r}_bold_confounds.tsv')

    print(f"Subject: {sub}")
    print(f"Run: {r}")
    print(f"EPI file: {epi_fn}")
    print(f"Gray matter mask: {gm_fn}")
    print(f"Non-gray matter mask: {cf_fn}")
    print(f"Confounds file: {conf_fn}")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
