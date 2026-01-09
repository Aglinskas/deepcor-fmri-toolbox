"""
QuickStart Example for DeepCor

This script demonstrates the simplest way to use DeepCor
for fMRI denoising using the high-level API.
"""

from deepcor import DeepCorDenoiser

# Initialize denoiser with default settings
denoiser = DeepCorDenoiser(
    model_version='cvae',
    latent_dims=(8, 8),
    n_epochs=100,
    batch_size=1024,
    n_repetitions=20
)

# Denoise your fMRI data
output_path = denoiser.fit_denoise(
    epi_path='path/to/epi.nii.gz',
    gm_mask_path='path/to/gm_mask.nii',
    cf_mask_path='path/to/cf_mask.nii',
    confounds_path='path/to/confounds.tsv',
    output_dir='path/to/output/directory',
    verbose=True
)

print(f"Denoising complete! Output saved to: {output_path}")

# The output directory will contain:
# - denoised_deepcor.nii.gz: Ensemble-averaged DeepCor denoised data
# - denoised_compcor.nii.gz: CompCor denoised data for comparison
# - preproc.nii.gz: Preprocessed (non-denoised) data
# - signal_rep_*.nii.gz: Individual repetition outputs
# - model_rep_*.pt: Trained model checkpoints
