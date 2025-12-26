"""Evaluation metrics for fMRI denoising."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn import linear_model


def calc_mse(y_true, y_pred, axis=0, clip=True):
    """
    Calculate variance explained (R^2) between two arrays.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        axis: Axis along which to compute mean
        clip: Whether to clip negative values to 0

    Returns:
        Variance explained (R^2)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=axis)) ** 2)
    varexp = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    if clip and varexp < 0:
        varexp = 0

    return varexp


def calc_and_save_compcor(
    epi,
    gm,
    cf,
    ofn,
    n_components=5,
    return_img=False,
    do_center=True
):
    """
    Calculate and save CompCor denoised data.

    Args:
        epi: EPI image
        gm: Gray matter mask
        cf: Non-gray matter mask
        ofn: Output filename
        n_components: Number of PCA components
        return_img: Whether to return the image
        do_center: Whether to center/z-score the data

    Returns:
        Denoised image if return_img=True, else None
    """
    from ..data.loaders import array_to_brain

    nTR = epi.shape[-1]
    epi_flat = epi.numpy().reshape(-1, nTR)

    if do_center:
        std0 = epi_flat.std(axis=-1) < 1e-3
        epi_flat[~std0, :] = (
            (epi_flat[~std0, :] - epi_flat[~std0, :].mean(axis=-1)[:, np.newaxis]) /
            epi_flat[~std0, :].std(axis=-1)[:, np.newaxis]
        )
        epi_flat[std0, :] = 0

    gm_flat = gm.numpy().flatten()
    cf_flat = cf.numpy().flatten()

    epi_cf = epi_flat[cf_flat == 1, :].transpose()
    epi_gm = epi_flat[gm_flat == 1, :].transpose()

    conf_pcs = PCA(n_components=n_components).fit_transform(epi_cf)
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(conf_pcs, epi_gm)

    compcor = epi_gm - lin_reg.predict(conf_pcs)
    compcor = compcor.transpose()

    n_std0 = (compcor.std(axis=1) < 1e-3).sum()
    if n_std0 > 0:
        print(f'n_std0:{n_std0}')

    if return_img:
        img = array_to_brain(compcor, epi, gm, ofn, inv_z_score=False, return_img=True)
        return img
    else:
        array_to_brain(compcor, epi, gm, ofn, inv_z_score=False, return_img=False)


def average_signal_ensemble(signal_files, ofn):
    """
    Average multiple denoised signal files into an ensemble.

    Args:
        signal_files: List of signal file paths
        ofn: Output filename

    Returns:
        Averaged signal image
    """
    import ants
    from tqdm import tqdm

    c = 0
    im = ants.image_read(signal_files[0])
    signal_avg = np.zeros(im.shape)
    for signal_file in tqdm(signal_files):
        im = ants.image_read(signal_file)
        arr = im.numpy()
        if np.isnan(arr).sum() == 0:
            signal_avg += arr
            c += 1
    signal_avg = signal_avg / c
    print(f'signals averaged: {c}')
    im = im.new_image_like(signal_avg)
    im.to_filename(ofn)
    return im


def correlation(x, y):
    """
    Compute correlation between two vectors.

    Args:
        x: First vector
        y: Second vector

    Returns:
        Pearson correlation coefficient
    """
    x_mean = np.repeat(x.mean(), x.shape, axis=0)
    y_mean = np.repeat(y.mean(), y.shape, axis=0)
    cov = (x - x_mean) * (y - y_mean)
    r = cov.sum() / (x.std() * y.std() * x.shape[0])
    return r
