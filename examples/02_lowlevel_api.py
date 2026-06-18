"""
DeepCor low-level API — build the denoising pipeline by hand.

Use this when you want to customize the pieces (training loop, model, ensemble
logic) rather than call the high-level DeepCor() object. This script shows the
recommended v2 (confound-aware) workflow first, then the simpler v1 variant.

Everything here is exactly what the high-level API does internally. If you only
need a denoised image, prefer 01_quickstart_highlevel.py.

HOW TO USE
----------
1. Edit the PATHS section.
2. (Optional) Adjust the HYPERPARAMETERS section.
3. Run:  python 02_lowlevel_api.py        # runs the v2 workflow
   (call run_v1() instead if you want the no-confounds model)

See wiki/usage_advanced_api.md for a fully-annotated walkthrough.
"""

import os
import numpy as np
import ants
import torch

from deepcor.data import get_obs_noi_list_coords, get_obs_noi_list, get_confounds, TrainDataset
from deepcor.models import get_model
from deepcor.training import Trainer, save_brain_signals
from deepcor.analysis import average_signal_ensemble
from deepcor.utils import safe_mkdir

# ===========================================================================
# PATHS  — EDIT THESE.  (Same-grid images; binary, non-overlapping masks.)
# ===========================================================================
EPI_PATH       = "/path/to/sub-01_task-rest_desc-preproc_bold.nii.gz"
GM_MASK_PATH   = "/path/to/sub-01_label-GM_mask.nii.gz"          # ROI:  gray matter
CF_MASK_PATH   = "/path/to/sub-01_label-WMCSF_mask.nii.gz"      # RONI: white matter + CSF
CONFOUNDS_PATH = "/path/to/sub-01_task-rest_desc-confounds_timeseries.tsv"  # only needed for v2
OUTPUT_DIR     = "./deepcor-output/sub-01-lowlevel"

# ===========================================================================
# HYPERPARAMETERS
# ===========================================================================
# Set LOW for a quick test. For a REAL run use roughly N_EPOCHS=100, N_REPETITIONS=20.
N_EPOCHS      = 5
N_REPETITIONS = 5
BATCH_SIZE    = 1024
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# v2 (recommended): confound-aware model
# ===========================================================================
def run_v2():
    safe_mkdir(OUTPUT_DIR)

    # 1. Load inputs.
    epi = ants.image_read(EPI_PATH)
    gm = ants.image_read(GM_MASK_PATH)     # ROI
    cf = ants.image_read(CF_MASK_PATH)     # RONI (confounds mask)

    # 2. Extract ROI/RONI time series WITH spatial coordinates (the v2 input).
    #    in_channels comes straight from the data: 4 for v2 (signal + x, y, z).
    obs_list, noi_list, gm, cf = get_obs_noi_list_coords(epi, gm, cf)
    nTR = obs_list.shape[-1]
    in_channels = obs_list.shape[-2]

    # 3. Load the 6 motion confounds -> (n_confounds, nTR).
    conf = get_confounds(CONFOUNDS_PATH)

    # 4. Dataset + loader. drop_last=True is REQUIRED for v2 (see step 5).
    dataset = TrainDataset(obs_list, noi_list)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # 5. The v2 model stores the confounds internally and indexes them per-sample,
    #    so tile the (n_conf, nTR) array across the batch -> (BATCH_SIZE, n_conf, nTR).
    #    Because this batch size is baked in, every training batch must be full,
    #    which is why the loader uses drop_last=True. (No .to(device) needed here:
    #    it becomes a model buffer and moves with model.to(device).)
    conf_tensor = torch.tensor(np.array([conf for _ in range(BATCH_SIZE)]))

    # 6. Ensemble loop: a fresh model each repetition, trained then saved.
    signal_files = []
    for rep in range(N_REPETITIONS):
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
        ).to(DEVICE)

        print(f"[v2] training ensemble member {rep + 1}/{N_REPETITIONS} ...")
        Trainer(model, device=DEVICE, lr=LEARNING_RATE).fit(loader, n_epochs=N_EPOCHS)

        # kind="FG" = foreground = the denoised signal (noise latent zeroed).
        ofn = os.path.join(OUTPUT_DIR, f"signal_rep_{rep}.nii.gz")
        save_brain_signals(model, dataset, epi, gm, ofn=ofn, kind="FG")
        signal_files.append(ofn)

    # 7. Average the ensemble into the final denoised image.
    denoised_path = os.path.join(OUTPUT_DIR, "denoised_avg.nii.gz")
    average_signal_ensemble(signal_files, denoised_path)
    print("\nDone. Denoised image:", denoised_path)


# ===========================================================================
# v1 (original cVAE): no confounds, single latent dim, 1 input channel
# ===========================================================================
def run_v1():
    safe_mkdir(OUTPUT_DIR)

    epi = ants.image_read(EPI_PATH)
    gm = ants.image_read(GM_MASK_PATH)
    cf = ants.image_read(CF_MASK_PATH)

    # Plain loader: no coordinates -> in_channels == 1. No confounds needed.
    obs_list, noi_list, gm, cf = get_obs_noi_list(epi, gm, cf)
    nTR = obs_list.shape[-1]
    in_channels = obs_list.shape[-2]

    dataset = TrainDataset(obs_list, noi_list)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    signal_files = []
    for rep in range(N_REPETITIONS):
        model = get_model(
            "v1", in_channels=in_channels, in_dim=nTR, latent_dim=8, beta=0.01
        ).to(DEVICE)

        print(f"[v1] training ensemble member {rep + 1}/{N_REPETITIONS} ...")
        Trainer(model, device=DEVICE, lr=LEARNING_RATE).fit(loader, n_epochs=N_EPOCHS)

        ofn = os.path.join(OUTPUT_DIR, f"signal_rep_{rep}.nii.gz")
        save_brain_signals(model, dataset, epi, gm, ofn=ofn, kind="FG")
        signal_files.append(ofn)

    denoised_path = os.path.join(OUTPUT_DIR, "denoised_avg.nii.gz")
    average_signal_ensemble(signal_files, denoised_path)
    print("\nDone. Denoised image:", denoised_path)


if __name__ == "__main__":
    run_v2()      # recommended model
    # run_v1()    # uncomment to use the original, no-confounds model instead
