"""
Advanced Usage Example for DeepCor fMRI Denoising Toolbox

This example demonstrates comprehensive usage of the deepcor package including:
- Configuration classes (DeepCorConfig, ModelConfig, TrainingConfig, DataConfig)
- Data loading and preprocessing
- Model initialization and training
- Post-training analyses (correlations and contrasts)
- Visualization and tracking
- Both high-level and low-level API usage
"""

import os
import numpy as np
import pandas as pd
import torch
import ants
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import DeepCor configuration classes
from deepcor import (
    DeepCorConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
)

# Import models
from deepcor import CVAE, cVAE, get_model, list_models

# Import training utilities
from deepcor import Trainer, save_model, save_brain_signals

# Import data utilities
from deepcor import (
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
from deepcor import (
    run_correlation_analysis_from_spec,
    run_contrast_analysis_from_spec,
    calc_and_save_compcor,
    average_signal_ensemble,
    get_design_matrix,
)

# Import visualization utilities
from deepcor import (
    init_track,
    update_track,
    show_dashboard,
    save_track,
)

# Import general utilities
from deepcor import safe_mkdir, check_gpu_and_speedup

# Import high-level API
from deepcor import DeepCorDenoiser


# ============================================================================
# EXAMPLE 1: GPU Check and Setup
# ============================================================================
print("=" * 80)
print("EXAMPLE 1: GPU Check and Setup")
print("=" * 80)

# Check GPU availability and speedup
result = check_gpu_and_speedup()
print(f"GPU available: {result['gpu_available']}")
if result['gpu_available']:
    print(f"GPU name: {result['gpu_name']}")
    print(f"Average CPU time per op: {result['cpu_time']:.6f} s")
    print(f"Average GPU time per op: {result['gpu_time']:.6f} s")
    print(f"Speedup (CPU/GPU): {result['speedup']:.2f}x")
else:
    print(f"Average CPU time per op: {result['cpu_time']:.6f} s")

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}\n")


# ============================================================================
# EXAMPLE 2: Configuration Setup
# ============================================================================
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
print(f"DataConfig: n_dummy_scans={data_config.n_dummy_scans}, confounds={data_config.confound_columns}")

# Example 2b: Using the complete DeepCorConfig
print("\n--- Complete DeepCorConfig ---")

# Create a complete configuration
config = DeepCorConfig(
    model=model_config,
    training=training_config,
    data=data_config
)
print(f"Complete config created with:")
print(f"  - Model: latent_dims={config.model.latent_dims}")
print(f"  - Training: {config.training.n_epochs} epochs, batch_size={config.training.batch_size}")
print(f"  - Data: {config.data.n_dummy_scans} dummy scans to remove\n")


# ============================================================================
# EXAMPLE 3: Data Loading and File Paths
# ============================================================================
print("=" * 80)
print("EXAMPLE 3: Data Loading and Preprocessing")
print("=" * 80)

# Define file paths (customize these for your data)
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
conf_fn = os.path.join(
    indir,
    f'{sub}/ses-localizer/func/{sub}_ses-localizer_task-objectcategories_run-{r}_bold_confounds.tsv'
)

print(f"Subject: {sub}")
print(f"Run: {r}")
print(f"EPI file: {epi_fn}")
print(f"Gray matter mask: {gm_fn}")
print(f"Non-gray matter mask: {cf_fn}")
print(f"Confounds file: {conf_fn}")

# Load data (example - will fail if files don't exist)
# Uncomment when you have actual data:
# epi = ants.image_read(epi_fn)
# gm = ants.image_read(gm_fn)
# cf = ants.image_read(cf_fn)
# df_conf = pd.read_csv(conf_fn, delimiter='\t')

# For demonstration, we'll show the preprocessing steps:
print("\n--- Data Preprocessing Steps ---")
print("1. Load EPI, gray matter mask, and non-gray matter mask using ANTs")
print("2. Load confounds from fMRIPrep TSV file")
print("3. Apply dummy scan removal using apply_dummy()")
print("4. Extract and normalize confound regressors")
print("5. Create observation and noise coordinate lists using get_obs_noi_list_coords()")
print("6. Validate data (check for NaNs and zero-std voxels)")


# ============================================================================
# EXAMPLE 4: Output Directory Setup
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 4: Output Directory Setup")
print("=" * 80)

# Create output directory
ofdir_root = '../Data/DeepCor-Outputs/'
ofdir = os.path.join(ofdir_root, analysis_name)
safe_mkdir(ofdir)
print(f"Output directory: {ofdir}")
print(f"Directory created successfully using safe_mkdir()\n")


# ============================================================================
# EXAMPLE 5: Post-Training Analysis Configuration
# ============================================================================
print("=" * 80)
print("EXAMPLE 5: Post-Training Analysis Configuration")
print("=" * 80)

# Configure correlation and contrast analyses
run_post_analyses = True
if run_post_analyses:
    print("Configuring post-training analyses...")

    # Example: Create design matrix from events file
    print("\n--- Design Matrix ---")
    events_fn = f'../Data/Events/{sub}_ses-localizer_task-objectcategories_run-{r}_events.tsv'
    print(f"Events file: {events_fn}")
    print("Use get_design_matrix(epi, events_fn) to create design matrix")

    # Correlation analyses
    print("\n--- Correlation Analyses ---")
    correlation_analyses = []

    # Example 1: Correlate with face regressor
    correlation_analyses.append({
        'corr_target': None,  # Would be X1['face'].values from design matrix
        'filename': os.path.join(ofdir, f'corr2face_S{s}_R{r}.nii.gz'),
        'plot': True,
        'ROI': f'../Data/Misc/rFFA_final_mask_{sub}_bin.nii.gz'
    })
    print("Added correlation analysis for face regressor")

    # Example 2: Correlate with place regressor (average of house and scene)
    correlation_analyses.append({
        'corr_target': None,  # Would be X1[['house','scene']].values.mean(axis=1)
        'filename': os.path.join(ofdir, f'corr2place_S{s}_R{r}.nii.gz'),
        'plot': True,
        'ROI': f'../Data/Misc/rPPA_final_mask_{sub}_bin.nii.gz'
    })
    print("Added correlation analysis for place regressor")

    # Contrast analyses
    print("\n--- Contrast Analyses ---")
    contrast_analyses = []

    # Example 1: Face contrast
    contrast_analyses.append({
        'contrast_vec': [-1, 5, -1, -1, -1, -1, 0, 0, 0, 0],
        'design_matrix': None,  # Would be X1
        'filename': os.path.join(ofdir, f'contrast_face_S{s}_R{r}.nii.gz'),
        'plot': True,
        'ROI': f'../Data/Misc/rFFA_final_mask_{sub}_bin.nii.gz'
    })
    print("Added contrast analysis for faces")

    # Example 2: Place contrast
    contrast_analyses.append({
        'contrast_vec': [-1, -1, 2, -1, 2, -1, 0, 0, 0, 0],
        'design_matrix': None,  # Would be X1
        'filename': os.path.join(ofdir, f'contrast_place_S{s}_R{r}.nii.gz'),
        'plot': True,
        'ROI': f'../Data/Misc/rPPA_final_mask_{sub}_bin.nii.gz'
    })
    print("Added contrast analysis for places")
else:
    correlation_analyses = []
    contrast_analyses = []

print(f"\nTotal correlation analyses: {len(correlation_analyses)}")
print(f"Total contrast analyses: {len(contrast_analyses)}\n")


# ============================================================================
# EXAMPLE 6: High-Level API Usage
# ============================================================================
print("=" * 80)
print("EXAMPLE 6: High-Level API - DeepCorDenoiser")
print("=" * 80)

# Example 6a: Using DeepCorDenoiser with individual parameters
print("\n--- Method 1: Individual Parameters ---")
denoiser1 = DeepCorDenoiser(
    model_version='cvae',
    latent_dims=(8, 8),
    n_epochs=100,
    batch_size=1024,
    learning_rate=0.001,
    n_repetitions=20,
    device=device,
    verbose=True
)
print("Created DeepCorDenoiser with individual parameters")

# Example 6b: Using DeepCorDenoiser with DeepCorConfig
print("\n--- Method 2: Using DeepCorConfig ---")
denoiser2 = DeepCorDenoiser(
    config=config,
    device=device,
    verbose=True
)
print("Created DeepCorDenoiser with DeepCorConfig object")

# Example usage (commented out - requires actual data):
# output_path = denoiser2.fit_denoise(
#     epi_path=epi_fn,
#     gm_mask_path=gm_fn,
#     cf_mask_path=cf_fn,
#     confounds_path=conf_fn,
#     output_dir=ofdir,
#     verbose=True
# )
print("\nTo use: denoiser.fit_denoise(epi_path, gm_mask_path, cf_mask_path, confounds_path, output_dir)\n")


# ============================================================================
# EXAMPLE 7: Low-Level API - Manual Training Loop
# ============================================================================
print("=" * 80)
print("EXAMPLE 7: Low-Level API - Manual Model Training")
print("=" * 80)

print("\n--- Step-by-step manual training process ---")

# Simulated parameters (in real usage, these come from loaded data)
nTR = 288  # Number of TRs
t_r = 2.0  # Repetition time
batch_size = config.training.batch_size

print(f"Number of TRs: {nTR}")
print(f"Repetition Time: {t_r} s")
print(f"Batch size: {batch_size}")

# Step 1: Data preprocessing (demonstrated, not executed)
print("\n1. Data Preprocessing:")
print("   epi, df_conf = apply_dummy(epi, df_conf, n_dummy_scans)")
print("   conf = prepare_confounds(df_conf, confound_columns)")
print("   obs_list_coords, noi_list_coords, gm, cf = get_obs_noi_list_coords(epi, gm, cf)")

# Step 2: Create dataset and dataloader
print("\n2. Create Dataset and DataLoader:")
print("   train_dataset = TrainDataset(obs_list_coords, noi_list_coords)")
print("   train_loader = torch.utils.data.DataLoader(")
print("       train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)")

# Step 3: Initialize model
print("\n3. Initialize Model:")
print(f"   model = CVAE(")
print(f"       conf_batch,")
print(f"       in_channels=4,")
print(f"       in_dim={nTR},")
print(f"       latent_dim={config.model.latent_dims},")
print(f"       beta={config.model.beta},")
print(f"       gamma={config.model.gamma},")
print(f"       delta={config.model.delta},")
print(f"       scale_MSE_GM={config.model.scale_MSE_GM},")
print(f"       scale_MSE_CF={config.model.scale_MSE_CF},")
print(f"       scale_MSE_FG={config.model.scale_MSE_FG},")
print(f"       do_disentangle={config.model.do_disentangle}")
print(f"   )")

# Step 4: Initialize trainer
print("\n4. Initialize Trainer:")
print(f"   trainer = Trainer(")
print(f"       model,")
print(f"       device={device},")
print(f"       optimizer_type='{config.training.optimizer}',")
print(f"       lr={config.training.learning_rate},")
print(f"       betas={config.training.betas},")
print(f"       eps={config.training.eps},")
print(f"       max_grad_norm={config.training.max_grad_norm}")
print(f"   )")

# Step 5: Train the model
print("\n5. Train the Model:")
print(f"   trainer.fit(train_loader, n_epochs={config.training.n_epochs}, verbose=True)")

# Step 6: Save outputs
print("\n6. Save Model and Brain Signals:")
print("   trainer.save_checkpoint('model.pt', epoch, loss)")
print("   save_brain_signals(model, train_dataset, epi, gm, 'denoised.nii.gz', kind='FG')")


# ============================================================================
# EXAMPLE 8: Model Registry and Available Models
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 8: Model Registry and Available Models")
print("=" * 80)

# List available models
print("\nAvailable models in registry:")
available_models = list_models()
for model_name in available_models:
    print(f"  - {model_name}")

# Get a specific model
print("\nGetting model from registry:")
print("  model_class = get_model('cvae')")
print("  This returns the CVAE class that can be instantiated")


# ============================================================================
# EXAMPLE 9: Visualization and Tracking
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 9: Visualization and Tracking")
print("=" * 80)

# Define tracking keys
keys = [
    'l', 'kld_loss', 'recons_loss_roi', 'recons_loss_roni',
    'loss_recon_conf_s', 'loss_recon_conf_z', 'ncc_loss_tg',
    'ncc_loss_bg', 'ncc_loss_conf_s', 'ncc_loss_conf_z',
    'smoothness_loss', 'recons_loss_fg', 'batch_varexp',
    'tg_mu_z', 'tg_log_var_z', 'tg_mu_s', 'tg_log_var_s',
    'tg_z', 'tg_s', 'bg_log_var_z', 'bg_mu_z',
    'tg_log_var_z_mean', 'bg_log_var_z_mean', 'tg_log_var_s_mean',
    'tg_mu_z_std', 'bg_mu_z_std', 'tg_mu_s_std', 'batch_signal',
    'batch_noise', 'batch_in', 'batch_out', 'batch_varexp',
    'confounds_pred_z', 'confounds_pred_s',
]

print("Initialize tracking dictionary:")
print(f"  track = init_track(keys)")
print(f"  Tracking {len(keys)} different metrics")

print("\nDuring training loop:")
print("  track = update_track(track, model, inputs_gm, inputs_cf)")
print("  Updates tracking dictionary with current metrics")

print("\nVisualization:")
print("  show_dashboard(track, single_fig=True)")
print("  Displays comprehensive dashboard of training metrics")

print("\nSave tracking data:")
print("  save_track('track.pickle', track)")
print("  load_pickle('track.pickle')")


# ============================================================================
# EXAMPLE 10: Ensemble Training and Analysis
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 10: Ensemble Training and Post-Processing")
print("=" * 80)

# Ensemble training loop
print(f"\nTraining {config.training.n_repetitions} repetitions for ensemble:")
print("for rep in range(n_repetitions):")
print("    # Initialize new model")
print("    model = CVAE(...)")
print("    trainer = Trainer(model, ...)")
print("    ")
print("    # Train")
print("    trainer.fit(train_loader, n_epochs=n_epochs)")
print("    ")
print("    # Save outputs")
print("    trainer.save_checkpoint(f'model_rep_{rep}.pt', epoch, loss)")
print("    save_brain_signals(model, dataset, epi, gm, f'signal_rep_{rep}.nii.gz', kind='FG')")

# Ensemble averaging
print("\nEnsemble Averaging:")
print("  signal_files = ['signal_rep_0.nii.gz', 'signal_rep_1.nii.gz', ...]")
print("  averaged = average_signal_ensemble(signal_files, 'signal_avg.nii.gz')")
print("  Averages all repetitions to create robust denoised output")

# Comparison outputs
print("\nCreate Comparison Outputs:")
print("  # Original preprocessed data")
print("  array_to_brain(obs_list_coords[:, 0, :], epi, gm, 'preproc.nii.gz', inv_z_score=True)")
print("  ")
print("  # CompCor denoising (baseline)")
print("  calc_and_save_compcor(epi, gm, cf, 'compcor.nii.gz', n_components=5)")


# ============================================================================
# EXAMPLE 11: Post-Training Analyses
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 11: Running Post-Training Analyses")
print("=" * 80)

print("\n--- Correlation Analyses ---")
print("Run correlation analyses to identify voxels correlated with task regressors:")
print("for analysis_spec in correlation_analyses:")
print("    run_correlation_analysis_from_spec(")
print("        analysis_spec, epi, compcor, denoised, gm)")
print("\nThis generates correlation maps for each specified regressor")

print("\n--- Contrast Analyses ---")
print("Run GLM-based contrast analyses:")
print("for analysis_spec in contrast_analyses:")
print("    run_contrast_analysis_from_spec(")
print("        analysis_spec, epi, compcor, denoised, gm)")
print("\nThis generates statistical contrast maps (e.g., faces > objects)")


# ============================================================================
# EXAMPLE 12: Complete Pipeline Example
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 12: Complete Pipeline Pseudocode")
print("=" * 80)

complete_pipeline = """
# 1. Setup and Configuration
config = DeepCorConfig(
    model=ModelConfig(latent_dims=(8, 8), beta=0.01),
    training=TrainingConfig(n_epochs=100, batch_size=1024, n_repetitions=20),
    data=DataConfig(n_dummy_scans=0, confound_columns=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ'])
)

# 2. Check GPU
result = check_gpu_and_speedup()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 3. Load Data
epi = ants.image_read('epi.nii.gz')
gm = ants.image_read('gm_mask.nii.gz')
cf = ants.image_read('cf_mask.nii.gz')
df_conf = pd.read_csv('confounds.tsv', delimiter='\\t')

# 4. Preprocess
epi, df_conf = apply_dummy(epi, df_conf, config.data.n_dummy_scans)
conf = prepare_confounds(df_conf, config.data.confound_columns)
obs_list, noi_list, gm, cf = get_obs_noi_list_coords(epi, gm, cf)

# 5. Visualize Data
plot_timeseries(epi, gm, cf)

# 6. Option A: High-Level API
denoiser = DeepCorDenoiser(config=config, device=device)
output = denoiser.fit_denoise(epi_path, gm_path, cf_path, conf_path, output_dir)

# 6. Option B: Low-Level API with Manual Control
for rep in range(config.training.n_repetitions):
    # Create dataset
    dataset = TrainDataset(obs_list, noi_list)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size)

    # Initialize model
    model = CVAE(conf_batch, 4, nTR, config.model.latent_dims, **config.model.__dict__)

    # Initialize trainer
    trainer = Trainer(model, device=device, lr=config.training.learning_rate)

    # Train
    track = init_track(keys)
    trainer.fit(loader, n_epochs=config.training.n_epochs)

    # Save
    save_brain_signals(model, dataset, epi, gm, f'signal_rep_{rep}.nii.gz', kind='FG')
    show_dashboard(track)

# 7. Ensemble and Compare
averaged = average_signal_ensemble(signal_files, 'denoised_avg.nii.gz')
calc_and_save_compcor(epi, gm, cf, 'compcor.nii.gz', n_components=5)

# 8. Post-Training Analyses
X1 = get_design_matrix(epi, events_file)
run_correlation_analysis_from_spec(correlation_spec, epi, compcor, averaged, gm)
run_contrast_analysis_from_spec(contrast_spec, epi, compcor, averaged, gm)
"""

print(complete_pipeline)


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

summary = """
This advanced example demonstrates the complete DeepCor API:

Key Components:
1. Configuration Classes:
   - DeepCorConfig: Complete configuration
   - ModelConfig: Model architecture settings
   - TrainingConfig: Training hyperparameters
   - DataConfig: Data preprocessing settings

2. High-Level API:
   - DeepCorDenoiser: One-line denoising with fit_denoise()

3. Low-Level API:
   - CVAE/cVAE models for custom workflows
   - Trainer class for flexible training
   - Manual control over all steps

4. Data Processing:
   - apply_dummy(): Remove dummy scans
   - get_obs_noi_list_coords(): Extract training data
   - TrainDataset: PyTorch dataset
   - plot_timeseries(): Visualize data

5. Training:
   - Trainer.fit(): Train model
   - save_model(): Save checkpoints
   - save_brain_signals(): Export denoised data

6. Analysis:
   - get_design_matrix(): Create GLM design
   - run_correlation_analysis_from_spec(): Correlation maps
   - run_contrast_analysis_from_spec(): Contrast maps
   - average_signal_ensemble(): Ensemble averaging
   - calc_and_save_compcor(): CompCor baseline

7. Visualization:
   - init_track(): Initialize tracking
   - update_track(): Update metrics
   - show_dashboard(): Display dashboard
   - save_track(): Save metrics

8. Utilities:
   - check_gpu_and_speedup(): GPU diagnostics
   - safe_mkdir(): Directory creation
   - list_models(): Available models
   - get_model(): Model registry

For more information, see the documentation and quickstart.py example.
"""

print(summary)

print("\n" + "=" * 80)
print("Advanced example complete!")
print("=" * 80)
