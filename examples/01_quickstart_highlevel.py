"""
DeepCor quickstart — denoise a SINGLE subject with the high-level API.

This is the simplest way to use DeepCor. It trains an ensemble of contrastive
VAEs on one functional run and writes a denoised image, using the recommended
("latest") model.

HOW TO USE
----------
1. Edit the PATHS section below to point at your own data.
2. (Optional) Adjust the HYPERPARAMETERS section.
3. Run:  python 01_quickstart_highlevel.py

See the wiki for details:
  - Data preparation:  wiki/data_preparation.md
  - High-level API:    wiki/usage_high_level_api.md
  - Outputs:           wiki/outputs.md
"""

import os
from deepcor import DeepCor

# ===========================================================================
# PATHS  — EDIT THESE to point at your own data.
# ===========================================================================
# All three images must be in the SAME space and on the SAME voxel grid.
# The masks must be binary and must NOT overlap (see wiki/data_preparation.md).
EPI_PATH       = "/path/to/sub-01_task-rest_desc-preproc_bold.nii.gz"   # 4D functional run
GM_MASK_PATH   = "/path/to/sub-01_label-GM_mask.nii.gz"                 # ROI:  gray matter
CF_MASK_PATH   = "/path/to/sub-01_label-WMCSF_mask.nii.gz"             # RONI: white matter + CSF (the "confounds mask")
CONFOUNDS_PATH = "/path/to/sub-01_task-rest_desc-confounds_timeseries.tsv"  # fMRIPrep confounds TSV

# Where outputs are written (created if missing).
OUTPUT_DIR = "./deepcor-output/sub-01"

# ===========================================================================
# HYPERPARAMETERS
# ===========================================================================
# These are set LOW so you can do a quick test run end-to-end in a few minutes.
# For a REAL denoising run, use roughly:
#     N_EPOCHS = 100, N_REPETITIONS = 20
N_EPOCHS      = 5    # training epochs per ensemble member
N_REPETITIONS = 5    # number of models in the ensemble (their outputs are averaged)

# ===========================================================================
# Run DeepCor
# ===========================================================================
def main():
    # Fail early with a clear message if a path is wrong.
    for label, path in [
        ("EPI", EPI_PATH),
        ("GM mask", GM_MASK_PATH),
        ("CF mask", CF_MASK_PATH),
        ("confounds", CONFOUNDS_PATH),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}\nEdit the PATHS section of this script.")

    # Build the denoiser. "latest" selects the recommended confound-aware model
    # (currently v2). It needs the confounds file; if you have no confounds you
    # can use model_version="v1" and omit `confounds` below.
    denoiser = DeepCor(
        model_version="latest",
        n_epochs=N_EPOCHS,
        n_repetitions=N_REPETITIONS,
    )

    # Train the ensemble and write the denoised output in one call.
    result = denoiser.fit_denoise(
        epi=EPI_PATH,
        gm_mask=GM_MASK_PATH,
        cf_mask=CF_MASK_PATH,
        confounds=CONFOUNDS_PATH,
        output_dir=OUTPUT_DIR,
        subject_idx=1,        # label used in output filenames / dashboard
        dashboard="save",     # writes a training-dashboard PNG; use None to skip
        verbose=True,
    )

    # `result` is a DeepCorResult. The main output is the denoised image:
    print("\nDone.")
    print("Denoised image :", result.denoised_path)
    print("Preproc copy   :", result.preproc_path)   # undenoised, for comparison
    print("CompCor compare:", result.compcor_path)    # CompCor baseline
    print("Output folder  :", result.output_dir)


if __name__ == "__main__":
    main()
