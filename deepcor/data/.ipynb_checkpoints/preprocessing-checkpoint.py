"""Data preprocessing utilities for fMRI data."""

import numpy as np
import warnings
import ants


def get_roi_and_roni(epi, anat, anat_gm, anat_wm, anat_csf, do_plot=True):
    """
    Create ROI (gray matter) and RONI (non-gray matter) masks.

    Args:
        epi: EPI image (ANTs image)
        anat: Anatomical image
        anat_gm: Gray matter probability map
        anat_wm: White matter probability map
        anat_csf: CSF probability map
        do_plot: Whether to plot the masks

    Returns:
        Tuple of (gm_mask, cf_mask) as ANTs images
    """
    gm_values = anat_gm.numpy()
    gm_mask = (gm_values > 0.5)
    wm_values = anat_wm.numpy()
    csf_values = anat_csf.numpy()
    cf_values = wm_values + csf_values
    cf_mask = (cf_values > 0.5)

    # Remove overlap between masks
    diff = gm_mask & cf_mask
    gm_mask_c = gm_mask ^ diff
    cf_mask_c = cf_mask ^ diff

    # Remove voxels with low standard deviation
    gm_mask_c = gm_mask_c * (epi.std(axis=-1) > 1e-3)
    cf_mask_c = cf_mask_c * (epi.std(axis=-1) > 1e-3)

    gm = anat_gm.new_image_like(gm_mask_c * 1.0)
    cf = anat_gm.new_image_like(cf_mask_c * 1.0)

    if do_plot:
        epi_mean = epi.numpy().mean(axis=-1)
        epi_mean_nii = gm.new_image_like(epi_mean)

        epi_mean_nii.plot_ortho(
            flat=True, xyz_lines=False, orient_labels=False,
            figsize=2, overlay_alpha=.3
        )
        anat.plot_ortho(
            flat=True, xyz_lines=False, orient_labels=False,
            figsize=2, overlay_alpha=.3
        )

        epi_mean_nii.plot_ortho(
            gm * 1.0 + cf * 2.0, flat=True, xyz_lines=False,
            orient_labels=False, figsize=2, overlay_alpha=.3,
            overlay_cmap='jet', title='red=CF, green=GM',
            textfontcolor='black'
        )
        anat.plot_ortho(
            gm * 1.0 + cf * 2.0, flat=True, xyz_lines=False,
            orient_labels=False, figsize=2, overlay_alpha=.3,
            overlay_cmap='jet'
        )

    return gm, cf


def get_obs_noi_list_coords(epi, gm, cf):
    """
    Extract observation and noise voxel lists with coordinates.

    Args:
        epi: EPI image
        gm: Gray matter mask
        cf: Non-gray matter mask

    Returns:
        Tuple of (obs_list_coords, noi_list_coords, gm, cf)
    """
    nTR = epi.shape[-1]
    epi_flat = epi.numpy().reshape(-1, nTR)
    gm_flat = gm.numpy().flatten()
    cf_flat = cf.numpy().flatten()

    # Drop STD0 voxels from mask
    std1 = epi_flat.std(axis=-1) > 1e-3
    gm_flat = gm_flat * std1
    cf_flat = cf_flat * std1

    func_gm = epi_flat[gm_flat == 1, :].copy()
    func_cf = epi_flat[cf_flat == 1, :].copy()

    assert max(np.unique(cf_flat + gm_flat)) != 2, \
        'ROI and RONI masks overlap'

    obs_list = func_gm
    noi_list = func_cf

    # Create 3D coordinate grids
    x_coords, y_coords, z_coords = np.meshgrid(
        np.arange(gm.shape[0]),
        np.arange(gm.shape[1]),
        np.arange(gm.shape[2]),
        indexing="ij"
    )
    x_coords_flat = x_coords.flatten()
    y_coords_flat = y_coords.flatten()
    z_coords_flat = z_coords.flatten()

    gm_x_coords = x_coords_flat[gm_flat.astype(bool)]
    gm_y_coords = y_coords_flat[gm_flat.astype(bool)]
    gm_z_coords = z_coords_flat[gm_flat.astype(bool)]
    gm_coords = np.stack((gm_x_coords, gm_y_coords, gm_z_coords), axis=-1)

    cf_x_coords = x_coords_flat[cf_flat.astype(bool)]
    cf_y_coords = y_coords_flat[cf_flat.astype(bool)]
    cf_z_coords = z_coords_flat[cf_flat.astype(bool)]
    cf_coords = np.stack((cf_x_coords, cf_y_coords, cf_z_coords), axis=-1)

    obs_list_coords = np.concatenate([
        obs_list[:, :, np.newaxis],
        np.stack([gm_coords for _ in range(nTR)], axis=1)
    ], axis=-1)
    noi_list_coords = np.concatenate([
        noi_list[:, :, np.newaxis],
        np.stack([cf_coords for _ in range(nTR)], axis=1)
    ], axis=-1)
    obs_list_coords = np.swapaxes(obs_list_coords, 1, 2)
    noi_list_coords = np.swapaxes(noi_list_coords, 1, 2)

    # Z-score
    obs_list_coords[:, 0, :] = (
        (obs_list_coords[:, 0, :] - obs_list_coords[:, 0, :].mean(axis=1)[:, np.newaxis]) /
        obs_list_coords[:, 0, :].std(axis=1)[:, np.newaxis]
    )
    noi_list_coords[:, 0, :] = (
        (noi_list_coords[:, 0, :] - noi_list_coords[:, 0, :].mean(axis=1)[:, np.newaxis]) /
        noi_list_coords[:, 0, :].std(axis=1)[:, np.newaxis]
    )

    print(f'obs_list_coords.shape: {obs_list_coords.shape}')
    print(f'noi_list_coords.shape: {noi_list_coords.shape}')

    # Upsample if needed
    if obs_list_coords.shape[0] > noi_list_coords.shape[0]:
        print('upsampling noi_list_coords')
        n_pad = obs_list_coords.shape[0] - noi_list_coords.shape[0]
        pad_idx = np.random.randint(
            low=0,
            high=noi_list_coords.shape[0],
            size=n_pad
        )
        noi_list_coords = np.concatenate([
            noi_list_coords,
            np.array([noi_list_coords[i, :, :] for i in pad_idx])
        ])
        print(f'obs_list_coords.shape: {obs_list_coords.shape}')
        print(f'noi_list_coords.shape: {noi_list_coords.shape}')

    gm = gm.new_image_like(gm_flat.reshape(gm.shape))
    cf = cf.new_image_like(cf_flat.reshape(gm.shape))

    return obs_list_coords, noi_list_coords, gm, cf



def get_obs_noi_list(epi, gm, cf):
    """
    Extract observation and noise voxel lists with coordinates.

    Args:
        epi: EPI image
        gm: Gray matter mask
        cf: Non-gray matter mask

    Returns:
        Tuple of (obs_list_coords, noi_list_coords, gm, cf)
    """
    nTR = epi.shape[-1]
    epi_flat = epi.numpy().reshape(-1, nTR)
    gm_flat = gm.numpy().flatten()
    cf_flat = cf.numpy().flatten()

    # Drop STD0 voxels from mask
    std1 = epi_flat.std(axis=-1) > 1e-3
    gm_flat = gm_flat * std1
    cf_flat = cf_flat * std1

    func_gm = epi_flat[gm_flat == 1, :].copy()
    func_cf = epi_flat[cf_flat == 1, :].copy()

    assert max(np.unique(cf_flat + gm_flat)) != 2, \
        'ROI and RONI masks overlap'

    obs_list = func_gm
    noi_list = func_cf

    # Z-score
    obs_list = (obs_list - obs_list.mean(axis=1)[:, np.newaxis]) / obs_list.std(axis=1)[:, np.newaxis]
    noi_list = (noi_list - noi_list.mean(axis=1)[:, np.newaxis]) / noi_list.std(axis=1)[:, np.newaxis]

    print(f'obs_list.shape: {obs_list.shape}')
    print(f'noi_list.shape: {noi_list.shape}')

    # Upsample if needed
    if obs_list.shape[0] > noi_list.shape[0]:
        print('upsampling noi_list_coords')
        n_pad = obs_list.shape[0] - noi_list.shape[0]
        pad_idx = np.random.randint(
            low=0,
            high=noi_list.shape[0],
            size=n_pad
        )
        noi_list = np.concatenate([noi_list,np.array([noi_list[i, :] for i in pad_idx]) ])

    print(f'obs_list.shape: {obs_list.shape}')
    print(f'noi_list.shape: {noi_list.shape}')

    gm = gm.new_image_like(gm_flat.reshape(gm.shape))
    cf = cf.new_image_like(cf_flat.reshape(gm.shape))

    return obs_list, noi_list, gm, cf


def apply_dummy(epi, df_conf, ndummy):
    """
    Apply dummy scan removal to EPI and confounds.

    Args:
        epi: EPI image
        df_conf: Confounds dataframe
        ndummy: Number of dummy scans to remove

    Returns:
        Tuple of (epi, df_conf) with dummy scans removed
    """
    if ndummy > 0:
        epi_arr = epi.numpy()
        epi_arr[:, :, :, 0:ndummy] = epi_arr[:, :, :, ndummy::].mean(axis=-1)[:, :, :, np.newaxis]
        epi = epi.new_image_like(epi_arr)

        df_conf.iloc[:ndummy, :] = 0

    return epi, df_conf


def censor_and_interpolate(arr_flat, idx_censor, do3=True):
    """
    Censor and interpolate bad timepoints.

    Given a data matrix and a censor mask, (1) optionally expand the mask by one
    timepoint on either side of every True, and (2) replace each censored column
    with the average of its nearest uncensored neighbors.

    Args:
        arr_flat: Data array of shape (n_voxels, n_timepoints)
        idx_censor: Boolean array indicating frames to censor
        do3: Whether to expand censoring to neighboring timepoints

    Returns:
        Tuple of (idx_censor3, arr_flat_corrected)
    """
    arr_flat = np.asarray(arr_flat)
    idx_censor = np.asarray(idx_censor, dtype=bool)
    n_voxels, T = arr_flat.shape

    if idx_censor.shape[0] != T:
        raise ValueError(
            f"Length of idx_censor ({idx_censor.shape[0]}) "
            f"must match number of timepoints ({T})."
        )

    # Expand the censor mask by one before/after
    prev_neighbor = np.concatenate([[False], idx_censor[:-1]])
    next_neighbor = np.concatenate([idx_censor[1:], [False]])
    idx_censor3 = idx_censor | prev_neighbor | next_neighbor

    if not do3:
        idx_censor3 = idx_censor

    # Build lookup tables for nearest uncensored before/after each t
    prev_good = np.full(T, -1, dtype=int)
    last = -1
    for i in range(T):
        if not idx_censor3[i]:
            last = i
        prev_good[i] = last

    next_good = np.full(T, -1, dtype=int)
    nxt = -1
    for i in range(T - 1, -1, -1):
        if not idx_censor3[i]:
            nxt = i
        next_good[i] = nxt

    # Copy data and fill in censored columns
    arr_flat_corrected = arr_flat.copy()
    bad_idxs = np.where(idx_censor3)[0]
    for t in bad_idxs:
        pg = prev_good[t]
        ng = next_good[t]
        if pg >= 0 and ng >= 0:
            arr_flat_corrected[:, t] = (arr_flat[:, pg] + arr_flat[:, ng]) / 2
        elif pg >= 0:
            arr_flat_corrected[:, t] = arr_flat[:, pg]
        elif ng >= 0:
            arr_flat_corrected[:, t] = arr_flat[:, ng]
        else:
            warnings.warn(
                f"No uncensored neighbors found for timepoint {t}; "
                "leaving original values in place."
            )

    return idx_censor3, arr_flat_corrected


def apply_frame_censoring(im, df_conf, idx_censor, also_nearby_voxels):
    """
    Apply frame censoring to fMRI data and confounds.

    Args:
        im: fMRI image
        df_conf: Confounds dataframe
        idx_censor: Boolean array indicating frames to censor
        also_nearby_voxels: Whether to also censor neighboring timepoints

    Returns:
        Tuple of (im_corrected, df_conf_corrected)
    """
    arr_flat = im.numpy().reshape(-1, im.shape[-1])
    idx_censor3, arr_flat_corrected = censor_and_interpolate(
        arr_flat, idx_censor, do3=also_nearby_voxels
    )

    l = len(idx_censor3)
    n_censored = idx_censor3.sum()
    perc_censored = n_censored / l * 100

    print(f'Censored {perc_censored:.2f}% of voxels {n_censored}/{l}')

    if perc_censored > 40:
        warnings.warn(
            'High number of censored frames: consider lowering the threshold '
            'or removing the subject from analyses'
        )

    im_corrected = im.new_image_like(arr_flat_corrected.reshape(im.shape))

    idx_censor3, conf_corrected = censor_and_interpolate(
        df_conf.values.transpose(), idx_censor, do3=also_nearby_voxels
    )
    df_conf_corrected = df_conf.copy()
    df_conf_corrected.iloc[:, :] = conf_corrected.transpose()

    return im_corrected, df_conf_corrected


def remove_std0(arr):
    """
    Remove voxels with zero standard deviation.

    Args:
        arr: Array of voxel timeseries

    Returns:
        Array with std0 voxels removed
    """
    std0 = np.argwhere(np.std(arr, axis=1) == 0.0)
    arr_o = np.delete(arr, std0, axis=0)
    return arr_o
