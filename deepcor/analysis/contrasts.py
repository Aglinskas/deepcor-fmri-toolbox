"""Contrast analysis utilities."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from nilearn.glm.first_level import make_first_level_design_matrix


def get_design_matrix(epi, events_fn):
    """
    Create first-level design matrix from events file.

    Args:
        epi: EPI image (for extracting TR and number of timepoints)
        events_fn: Path to events TSV file

    Returns:
        Design matrix (pandas DataFrame)
    """
    events = pd.read_csv(events_fn, delimiter='\t')

    t_r = np.round(epi.spacing[-1], 2)
    nTR = epi.shape[-1]
    n_scans = nTR
    frame_times = (np.arange(n_scans) * t_r)
    X1 = make_first_level_design_matrix(
        frame_times,
        events,
        drift_model="polynomial",
        drift_order=3,
        hrf_model="SPM"
    )
    return X1


def get_contrast_val(Y, contrast_vec, X1):
    """
    Compute contrast values for voxels.

    Args:
        Y: Voxel timeseries array (n_voxels, n_timepoints)
        contrast_vec: Contrast vector
        X1: Design matrix

    Returns:
        Contrast values for each voxel
    """
    # Z-score values
    Y = (Y - Y.mean(axis=1)[:, np.newaxis]) / Y.std(axis=1)[:, np.newaxis]
    Y = Y.transpose()
    X = X1.values

    # Compute beta coefficients
    beta = np.linalg.inv(X.T @ X1) @ X1.T @ Y
    beta = beta.T
    beta = beta.values

    # Apply contrast
    contrast_values = beta @ contrast_vec

    return contrast_values


def calc_contrast_map(im, mask, contrast_vec, X1):
    """
    Calculate contrast map.

    Args:
        im: Input image (ANTs image)
        mask: Binary mask (ANTs image)
        contrast_vec: Contrast vector
        X1: Design matrix

    Returns:
        Contrast map (ANTs image)
    """
    assert len(contrast_vec) == X1.shape[1], \
        f'bad contrast shape. numel {len(contrast_vec)}: expected {X1.shape[1]}'
    assert sum(contrast_vec) == 0, \
        f'contrast does not sum to zero: sum={sum(contrast_vec)}'

    im_flat = im.numpy().reshape(-1, im.shape[-1])
    mask_flat = mask.numpy().flatten() == 1
    n = mask_flat.sum()

    con_vals = get_contrast_val(im_flat[mask_flat, :], contrast_vec, X1)

    res_flat = np.zeros(mask_flat.shape)
    res_flat[mask_flat] = con_vals

    temp_3d = im.slice_image(axis=-1, idx=0)
    res = res_flat.reshape(temp_3d.shape)

    res_nii = temp_3d.new_image_like(res)

    return res_nii


def run_contrast_analysis_from_spec(
    analysis_spec,
    epi,
    compcor,
    signals_averaged,
    gm
):
    """
    Run contrast analysis from specification dictionary.

    Args:
        analysis_spec: Dictionary with analysis specification
        epi: Original EPI image
        compcor: CompCor denoised image
        signals_averaged: DeepCor denoised image
        gm: Gray matter mask

    Required keys in analysis_spec:
        - contrast_vec: Contrast vector
        - design_matrix: Design matrix (pandas DataFrame)
        - filename: Output filename

    Optional keys:
        - plot: Whether to plot results (default: False)
        - ROI: ROI mask for plotting (required if plot=True)
    """
    import ants

    assert 'contrast_vec' in analysis_spec, \
        "contrast analysis spec has no key: contrast_vec"
    assert 'filename' in analysis_spec, \
        "filename needs to be specified for a file to be saved"

    contrast_vec = analysis_spec['contrast_vec']
    X1 = analysis_spec['design_matrix']

    assert len(contrast_vec) == X1.shape[1], \
        f"mismatching number of conditions: Design matrix: {X1.shape[1]}, " \
        f"contrast length: {len(contrast_vec)}"

    do_plot = analysis_spec.get('plot', False)
    if do_plot:
        assert 'ROI' in analysis_spec, 'plot requested, but ROI not specified'
        roi_fn = analysis_spec['ROI']
        assert os.path.exists(roi_fn), f'ROI does not exist: {roi_fn}'

    con_map_preproc = calc_contrast_map(epi, gm, contrast_vec, X1)
    con_map_compcor = calc_contrast_map(compcor, gm, contrast_vec, X1)
    con_map_signal = calc_contrast_map(signals_averaged, gm, contrast_vec, X1)

    ofn = analysis_spec['filename']
    file_ext = ''.join(pathlib.Path(ofn).suffixes)

    con_map_preproc.to_filename(ofn.replace(file_ext, '_preproc' + file_ext))
    con_map_compcor.to_filename(ofn.replace(file_ext, '_compcor' + file_ext))
    con_map_signal.to_filename(ofn.replace(file_ext, '_deepcor' + file_ext))

    print('saved as: {}'.format(ofn.replace(file_ext, '_preproc' + file_ext)))
    print('saved as: {}'.format(ofn.replace(file_ext, '_compcor' + file_ext)))
    print('saved as: {}'.format(ofn.replace(file_ext, '_deepcor' + file_ext)))

    if do_plot:
        roi = ants.image_read(roi_fn)
        mask = roi.numpy() == 1

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        val1 = con_map_preproc.numpy()[mask]
        val2 = con_map_compcor.numpy()[mask]
        val3 = con_map_signal.numpy()[mask]

        ys = [val1.mean(), val2.mean(), val3.mean()]
        xs = [0, 1, 2]
        plt.bar(xs, ys)
        plt.xticks(
            xs,
            labels=[
                f'preproc\n{ys[0]:.2f}',
                f'compcor\n{ys[1]:.2f}',
                f'signal\n{ys[2]:.2f}'
            ]
        )
        plt.title(ofn.split('/')[-1] + '\n' + roi_fn.split('/')[-1])
