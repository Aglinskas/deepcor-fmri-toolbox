"""Correlation analysis utilities."""

import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib


def correlate_columns(arr1, arr2):
    """
    Compute Pearson correlation between corresponding columns of two matrices.

    Args:
        arr1: First matrix of shape (n, m)
        arr2: Second matrix of shape (n, m)

    Returns:
        1D array of correlations for each column (size m)
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    # Center the arrays
    arr1_centered = arr1 - np.mean(arr1, axis=0)
    arr2_centered = arr2 - np.mean(arr2, axis=0)

    # Compute numerator (covariance)
    numerator = np.sum(arr1_centered * arr2_centered, axis=0)

    # Compute denominator (product of standard deviations)
    denominator = np.sqrt(
        np.sum(arr1_centered ** 2, axis=0) *
        np.sum(arr2_centered ** 2, axis=0)
    )

    # Compute correlation
    correlation = numerator / denominator

    return correlation


def calc_corr_map(im, mask, corr_target):
    """
    Calculate correlation map between image and target.

    Args:
        im: Input image (ANTs image)
        mask: Binary mask (ANTs image)
        corr_target: Target timeseries to correlate with

    Returns:
        Correlation map (ANTs image)
    """
    mask_flat = mask.numpy().flatten() == 1
    n = mask_flat.sum()
    im_flat = im.numpy().reshape(-1, im.shape[-1])

    r_vals = correlate_columns(
        im_flat[mask_flat, :].transpose(),
        np.array([corr_target for _ in range(n)]).transpose()
    )

    res_flat = np.zeros(mask_flat.shape)
    res_flat[mask_flat] = r_vals

    temp_3d = im.slice_image(axis=-1, idx=0)
    res = res_flat.reshape(temp_3d.shape)

    res_nii = temp_3d.new_image_like(res)

    return res_nii


def run_correlation_analysis_from_spec(
    analysis_spec,
    epi,
    compcor,
    signals_averaged,
    gm
):
    """
    Run correlation analysis from specification dictionary.

    Args:
        analysis_spec: Dictionary with analysis specification
        epi: Original EPI image
        compcor: CompCor denoised image
        signals_averaged: DeepCor denoised image
        gm: Gray matter mask

    Required keys in analysis_spec:
        - corr_target: Target timeseries to correlate with
        - filename: Output filename

    Optional keys:
        - plot: Whether to plot results (default: False)
        - ROI: ROI mask for plotting (required if plot=True)
    """
    import ants

    assert 'corr_target' in analysis_spec, \
        "correlation analysis spec has no key: corr_target"
    assert 'filename' in analysis_spec, \
        "filename needs to be specified for a file to be saved"

    corr_target = analysis_spec['corr_target']

    assert len(corr_target) == epi.shape[-1], \
        f"mismatching durations: epi sequence: {epi.shape[-1]}, " \
        f"corr_target: {len(corr_target)}"

    do_plot = analysis_spec.get('plot', False)
    if do_plot:
        assert 'ROI' in analysis_spec, 'plot requested, but ROI not specified'
        roi_fn = analysis_spec['ROI']
        assert os.path.exists(roi_fn), f'ROI does not exist: {roi_fn}'

    corr_map_preproc = calc_corr_map(epi, gm, corr_target)
    corr_map_compcor = calc_corr_map(compcor, gm, corr_target)
    corr_map_signal = calc_corr_map(signals_averaged, gm, corr_target)

    ofn = analysis_spec['filename']
    file_ext = ''.join(pathlib.Path(ofn).suffixes)

    corr_map_preproc.to_filename(ofn.replace(file_ext, '_preproc' + file_ext))
    corr_map_compcor.to_filename(ofn.replace(file_ext, '_compcor' + file_ext))
    corr_map_signal.to_filename(ofn.replace(file_ext, '_deepcor' + file_ext))

    print('saved as: {}'.format(ofn.replace(file_ext, '_preproc' + file_ext)))
    print('saved as: {}'.format(ofn.replace(file_ext, '_compcor' + file_ext)))
    print('saved as: {}'.format(ofn.replace(file_ext, '_deepcor' + file_ext)))

    if do_plot:
        roi = ants.image_read(roi_fn)
        mask = roi.numpy() == 1

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        val1 = corr_map_preproc.numpy()[mask]
        val2 = corr_map_compcor.numpy()[mask]
        val3 = corr_map_signal.numpy()[mask]

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
