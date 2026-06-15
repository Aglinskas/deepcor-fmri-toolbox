"""Data loading utilities for fMRI data."""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_confounds(confounds_path, columns=None, norm=False):
    """Load motion confounds from an fMRIPrep TSV file.

    Args:
        confounds_path: Path to the confounds TSV file.
        columns: Optional explicit list of columns to load.
        norm: False, 'zscore', or '0-1'.

    Returns:
        Numpy array with shape (n_confounds, n_timepoints).
    """
    df_conf = pd.read_csv(confounds_path, delimiter='\t')

    use_cols_cur = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    use_cols_old = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']

    if columns is not None:
        use_cols = list(columns)
        missing = [col for col in use_cols if col not in df_conf.columns]
        if missing:
            raise ValueError(
                f"Missing columns in confounds file: {missing}. "
                f"Available columns are: {list(df_conf.columns)}"
            )
    elif all(col in df_conf.columns for col in use_cols_cur):
        use_cols = use_cols_cur
    elif all(col in df_conf.columns for col in use_cols_old):
        use_cols = use_cols_old
    else:
        raise ValueError(
            "Could not find motion columns in confounds file. "
            f"Expected either {use_cols_cur} or {use_cols_old}. "
            f"Available columns are: {list(df_conf.columns)}"
        )

    conf = df_conf.loc[:, use_cols].to_numpy(dtype=float).T

    assert np.isnan(conf).sum() == 0, 'NaNs in motion'

    norm_axis = -1
    if norm is False:
        pass
    elif norm == 'zscore':
        mean = conf.mean(axis=norm_axis, keepdims=True)
        std = conf.std(axis=norm_axis, keepdims=True)
        std[std == 0] = 1
        conf = (conf - mean) / std
    elif norm == '0-1':
        min_val = conf.min(axis=norm_axis, keepdims=True)
        max_val = conf.max(axis=norm_axis, keepdims=True)
        denom = max_val - min_val
        denom[denom == 0] = 1
        conf = (conf - min_val) / denom
    else:
        raise ValueError("norm must be False, 'zscore', or '0-1'")

    return conf


def plot_timeseries(epi, gm, cf):
    """
    Plot ROI and RONI timeseries.

    Args:
        epi: EPI image
        gm: Gray matter mask
        cf: Non-gray matter mask
    """
    plt.figure(figsize=(15, 5))
    plt.plot(epi.numpy()[gm.numpy() == 1].mean(axis=0))
    plt.title('EPI ROI timeseries')
    plt.ylabel('BOLD')
    plt.xlabel('timepoints')

    plt.figure(figsize=(15, 5))
    plt.plot(epi.numpy()[cf.numpy() == 1].mean(axis=0))
    plt.title('EPI RONI timeseries')
    plt.ylabel('BOLD')
    plt.xlabel('timepoints')
    plt.show()


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
