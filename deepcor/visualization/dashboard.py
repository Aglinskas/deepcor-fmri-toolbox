"""Visualization and dashboard utilities."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from IPython import display


def init_track(keys):
    """
    Initialize tracking dictionary.

    Args:
        keys: List of keys to track

    Returns:
        Dictionary with initialized tracking lists
    """
    track = {}
    for key in keys:
        track[key] = []

    track['T_start'] = datetime.now()
    return track


def update_track(track, model, inputs_gm, inputs_cf):
    """
    Update tracking dictionary with current batch statistics.

    Args:
        track: Tracking dictionary
        model: Model to evaluate
        inputs_gm: Gray matter inputs
        inputs_cf: Non-gray matter inputs

    Returns:
        Updated tracking dictionary
    """
    import torch
    from ..analysis.metrics import calc_mse

    model.eval()

    [outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s,
     tg_log_var_s, tg_z, tg_s] = model.forward_tg(inputs_gm)
    [outputs_cf, inputs_cf, bg_mu_z, bg_log_var_z] = model.forward_bg(inputs_cf)
    loss = model.loss_function(
        outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z,
        tg_mu_s, tg_log_var_s, tg_z, tg_s,
        outputs_cf, inputs_cf, bg_mu_z, bg_log_var_z
    )

    track['l'].append(loss['loss'].detach().cpu().numpy())
    track['kld_loss'].append(loss['kld_loss'].detach().cpu().numpy())
    track['recons_loss_roi'].append(loss['recons_loss_roi'].detach().cpu().numpy())
    track['recons_loss_roni'].append(loss['recons_loss_roni'].detach().cpu().numpy())
    track['loss_recon_conf_s'].append(loss['loss_recon_conf_s'].detach().cpu().numpy())
    track['loss_recon_conf_z'].append(loss['loss_recon_conf_z'].detach().cpu().numpy())
    track['ncc_loss_tg'].append(loss['ncc_loss_tg'].detach().cpu().numpy())
    track['ncc_loss_bg'].append(loss['ncc_loss_bg'].detach().cpu().numpy())
    track['ncc_loss_conf_s'].append(loss['ncc_loss_conf_s'].detach().cpu().numpy())
    track['ncc_loss_conf_z'].append(loss['ncc_loss_conf_z'].detach().cpu().numpy())
    track['smoothness_loss'].append(loss['smoothness_loss'].detach().cpu().numpy())
    track['recons_loss_fg'].append(loss['recons_loss_fg'].detach().cpu().numpy())

    track['tg_log_var_z'] = tg_log_var_z.detach().cpu().numpy().flatten()
    track['bg_log_var_z'] = bg_log_var_z.detach().cpu().numpy().flatten()
    track['tg_log_var_s'] = tg_log_var_s.detach().cpu().numpy().flatten()

    track['tg_mu_z'] = tg_mu_z.detach().cpu().numpy().flatten()
    track['bg_mu_z'] = bg_mu_z.detach().cpu().numpy().flatten()
    track['tg_mu_s'] = tg_mu_s.detach().cpu().numpy().flatten()

    track['tg_log_var_z_mean'].append(tg_log_var_z.detach().cpu().numpy().mean())
    track['bg_log_var_z_mean'].append(bg_log_var_z.detach().cpu().numpy().mean())
    track['tg_log_var_s_mean'].append(tg_log_var_s.detach().cpu().numpy().mean())

    track['tg_mu_z_std'].append(tg_mu_z.detach().cpu().numpy().std())
    track['bg_mu_z_std'].append(bg_mu_z.detach().cpu().numpy().std())
    track['tg_mu_s_std'].append(tg_mu_s.detach().cpu().numpy().std())

    batch_signal = model.forward_fg(inputs_gm)[0].detach().cpu().numpy()
    batch_noise = model.forward_bg(inputs_gm)[0].detach().cpu().numpy()

    batch_in = inputs_gm.detach().cpu().numpy()
    batch_out = outputs_gm.detach().cpu().numpy()

    cf_signal = model.forward_fg(inputs_cf)[0].detach().cpu().numpy()
    cf_noise = model.forward_bg(inputs_cf)[0].detach().cpu().numpy()

    batch_varexp = calc_mse(batch_in[:, 0, :], batch_out[:, 0, :])

    confounds_pred_z = model.decoder_confounds_z(
        torch.unsqueeze(tg_z, 2)
    ).detach().cpu().numpy()
    confounds_pred_s = model.decoder_confounds_s(
        torch.unsqueeze(tg_s, 2)
    ).detach().cpu().numpy()

    track['batch_signal'] = batch_signal[0, 0, :]
    track['batch_noise'] = batch_noise[0, 0, :]
    track['batch_in'] = batch_in[0, 0, :]
    track['batch_out'] = batch_out[0, 0, :]
    track['batch_coords_in'] = batch_in[0, 1::, :]
    track['batch_coords_out'] = batch_out[0, 1::, :]

    track['batch_varexp'].append(batch_varexp)
    track['confounds_pred_z'] = confounds_pred_z
    track['confounds_pred_s'] = confounds_pred_s

    track['cf_input'] = inputs_cf.detach().cpu().numpy()[0, 0, :]
    track['cf_signal'] = cf_signal[0, 0, :]
    track['cf_noise'] = cf_noise[0, 0, :]

    model.train()

    return track


def show_dashboard(track, single_fig=True):
    """
    Display training dashboard with loss curves and diagnostics.

    Args:
        track: Tracking dictionary
        single_fig: Whether to create a single figure
    """
    nrows = 5
    ncols = 8
    sp = 0

    if single_fig:
        plt.close()
        sys.stdout.flush()
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.figure(figsize=(5 * ncols, 5 * nrows))

    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.plot(track['l'])
    plt.title('total loss: {:.2f}'.format(track['l'][-1]))

    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.plot(track['batch_varexp'])
    plt.title('batch_varexp: {:.2f}'.format(track['batch_varexp'][-1]))

    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.plot(track['recons_loss_roi'])
    plt.title('recons_loss_roi: {:.2f}'.format(track['recons_loss_roi'][-1]))

    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.plot(track['recons_loss_roni'])
    plt.title('recons_loss_roni: {:.2f}'.format(track['recons_loss_roni'][-1]))

    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.plot(track['kld_loss'])
    plt.title('kld_loss: {:.2f}'.format(track['kld_loss'][-1]))

    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.plot(track['recons_loss_fg'])
    plt.title('recons_loss_fg: {:.2f}'.format(track['recons_loss_fg'][-1]))

    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.plot(track['ncc_loss_tg'])
    plt.title('ncc_loss_tg: {:.2f}'.format(track['ncc_loss_tg'][-1]))

    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.plot(track['ncc_loss_bg'])
    plt.title('ncc_loss_bg: {:.2f}'.format(track['ncc_loss_bg'][-1]))

    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.plot(track['smoothness_loss'])
    plt.title('smoothness_loss: {:.2f}'.format(track['smoothness_loss'][-1]))

    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.plot(track['loss_recon_conf_z'])
    plt.title('loss_recon_conf_z: {:.2f}'.format(track['loss_recon_conf_z'][-1]))

    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.plot(track['ncc_loss_conf_z'])
    plt.title('ncc_loss_conf_z: {:.2f}'.format(track['ncc_loss_conf_z'][-1]))

    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.plot(track['loss_recon_conf_s'])
    plt.title('loss_recon_conf_s: {:.2f}'.format(track['loss_recon_conf_s'][-1]))

    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.plot(track['ncc_loss_conf_s'])
    plt.title('ncc_loss_conf_s: {:.2f}'.format(track['ncc_loss_conf_s'][-1]))

    # Confound predictions
    if 'conf' in track:
        sp += 1
        plt.subplot(nrows, ncols, sp)
        plt.plot(track['conf'][0, :])
        plt.plot(track['confounds_pred_z'][0, 0, :])
        plt.title('conf from z')

        sp += 1
        plt.subplot(nrows, ncols, sp)
        plt.plot(track['conf'][0, :])
        plt.plot(track['confounds_pred_s'][0, 0, :])
        plt.title('conf from s')
    else:
        sp += 2

    # Coordinate tracking
    sp += 1
    plt.subplot(nrows, ncols, sp)
    idx = 0
    plt.plot(track['batch_coords_in'][idx, :])
    plt.plot(track['batch_coords_out'][idx, :])
    plt.title('batch coords')

    # Latent variable distributions
    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.hist(track['tg_mu_z'], alpha=.3, color='b')
    plt.hist(track['bg_mu_z'], alpha=.3, color='r')
    plt.hist(track['tg_mu_s'], alpha=.3, color='g')
    plt.legend(['tg_mu_z', 'bg_mu_z', 'tg_mu_s'])

    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.hist(track['tg_log_var_z'], alpha=.3, color='b')
    plt.hist(track['bg_log_var_z'], alpha=.3, color='r')
    plt.hist(track['tg_log_var_s'], alpha=.3, color='g')
    plt.legend(['tg_log_var_z', 'bg_log_var_z', 'tg_log_var_s'])

    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.plot(track['tg_mu_z_std'], 'b')
    plt.plot(track['bg_mu_z_std'], 'r')
    plt.plot(track['tg_mu_s_std'], 'g')
    plt.legend(['tg_mu_z_std', 'bg_mu_z_std', 'tg_mu_s_std'])
    plt.title('mu over time')

    sp += 1
    plt.subplot(nrows, ncols, sp)
    plt.plot(track['tg_log_var_z_mean'], 'b')
    plt.plot(track['bg_log_var_z_mean'], 'r')
    plt.plot(track['tg_log_var_s_mean'], 'g')
    plt.legend(['tg_log_var_z_mean', 'bg_log_var_z_mean', 'tg_log_var_s_mean'])
    plt.title('logvar over time')

    # Timecourse plots
    sp += 1
    row, col = divmod(sp - 1, ncols)
    plt.subplot2grid((nrows, ncols), (row, col), colspan=2)
    sp += 1

    plt.plot(track['batch_in'])
    plt.plot(track['batch_out'])
    plt.plot(track['batch_signal'], 'r-')
    plt.plot(track['batch_noise'], 'g-')
    plt.title('batch timecourse (single voxel)')

    sp += 1
    row, col = divmod(sp - 1, ncols)
    plt.subplot2grid((nrows, ncols), (row, col), colspan=2)
    sp += 1

    plt.plot(track['cf_input'])
    plt.plot(track['cf_signal'], 'g-')
    plt.plot(track['cf_noise'], 'r-')
    plt.title('CF batch voxel')

    elapsed = datetime.now() - track['T_start']
    if single_fig:
        if all(k in track for k in ['sub', 'r', 'rep', 'epoch']):
            plt.suptitle(
                '{sub}-R{r}-rep-{rep} E:{epoch} T:{elapsed}'.format(
                    sub=track['sub'], r=track['r'],
                    rep=track['rep'], epoch=track['epoch'],
                    elapsed=elapsed
                ),
                y=.91, fontsize=20
            )
        if 'ofdir' in track:
            ofn = os.path.join(
                track['ofdir'],
                'dashboard_S{s}_R{r}_rep_{rep}.jpg'.format(
                    s=track.get('s', 0),
                    r=track.get('r', 0),
                    rep=track.get('rep', 0)
                )
            )
            plt.savefig(ofn)
        plt.show()


def save_track(track_ofn, track):
    """
    Save tracking dictionary to file.

    Args:
        track_ofn: Output filename
        track: Tracking dictionary
    """
    import pickle
    with open(track_ofn, 'wb') as handle:
        pickle.dump(track, handle, protocol=pickle.HIGHEST_PROTOCOL)
