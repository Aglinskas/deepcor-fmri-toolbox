"""Data loading utilities for fMRI data."""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ants


def plot_timeseries(epi, gm, cf):
    """
    Plot ROI and RONI timeseries.

    Args:
        epi: EPI image
        gm: Gray matter mask
        cf: Non-gray matter mask
    """
    plt.figure(figsize=(20, 5))
    plt.plot(epi.numpy()[gm.numpy() == 1].mean(axis=0))
    plt.title('EPI ROI timeseries')
    plt.ylabel('BOLD')
    plt.xlabel('timepoints')

    plt.figure(figsize=(20, 5))
    plt.plot(epi.numpy()[cf.numpy() == 1].mean(axis=0))
    plt.title('EPI RONI timeseries')
    plt.ylabel('BOLD')
    plt.xlabel('timepoints')


def array_to_brain(arr, epi, gm, ofn, inv_z_score=True, return_img=False):
    """
    Convert voxel array back to brain volume.

    Args:
        arr: Array of voxel timeseries
        epi: Original EPI image for shape reference
        gm: Gray matter mask
        ofn: Output filename
        inv_z_score: Whether to invert z-scoring
        return_img: Whether to return the image object

    Returns:
        ANTs image if return_img=True, else None
    """
    nTR = epi.shape[-1]
    epi_flat = epi.numpy().reshape(-1, nTR)
    gm_flat = gm.numpy().flatten()

    if inv_z_score:
        epi_mean = epi_flat[gm_flat == 1, :].mean(axis=1)
        epi_std = epi_flat[gm_flat == 1, :].std(axis=1)
        arr = (arr * epi_std[:, np.newaxis] + epi_mean[:, np.newaxis])

    assert arr.shape[0] == int(gm.numpy().sum()), \
        f'shape mismatch: arr {arr.shape[0]}, GM {int(gm.numpy().sum())}'

    brain_signals_arr = np.zeros(epi_flat.shape)
    brain_signals_arr[gm_flat == 1, :] = arr
    brain_signals_arr = brain_signals_arr.reshape(epi.shape)
    brain_signals_arr = epi.new_image_like(brain_signals_arr)
    brain_signals_arr.to_filename(ofn)

    if return_img:
        return brain_signals_arr


def load_pickle(fn):
    """
    Load pickle file.

    Args:
        fn: Filename

    Returns:
        Loaded dictionary
    """
    if os.path.exists(fn):
        with open(fn, 'rb') as file:
            loaded_dict = pickle.load(file)
        return loaded_dict
    else:
        raise FileNotFoundError(f"File not found: {fn}")


def save_pickle(fn, data):
    """
    Save data to pickle file.

    Args:
        fn: Filename
        data: Data to save
    """
    with open(fn, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
