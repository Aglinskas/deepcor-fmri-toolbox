# """Visualization and dashboard utilities."""
import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from deepcor.analysis import correlate_columns

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_varexp(mat_input,mat_output):
    # Calculates, variance explained
    # if varexp very negative, truncates at -1 
    SS_total = ((mat_input[:,0,:]-mat_input[:,0,:].mean(axis=-1)[:,np.newaxis])**2).sum()
    SS_model = ((mat_input[:,0,:]-mat_output[:,0,:])**2).sum()
    varexp = 1-(SS_model/SS_total)
    if varexp<-1:
        varexp = -1

    return varexp

def update_track(track,train_loader,model,config):
    model_version = track['model_version']

    if model_version=='V1':
        with torch.no_grad():
            sample_batch = next(iter(train_loader))
            batch_gm = sample_batch[0].to(device).float()
            batch_cf = sample_batch[1].to(device).float()
            [batch_gm_recon, batch_gm_input, batch_gm_mu_z, batch_gm_log_var_z, batch_gm_mu_s, batch_gm_log_var_s, batch_gm_z, batch_gm_s] = model.forward_tg(batch_gm)
            [batch_cf_recon, batch_cf_input, batch_cf_mu_z, batch_cf_log_var_z, batch_cf_mu_s, batch_cf_log_var_s, batch_cf_z, batch_cf_s] = model.forward_tg(batch_cf)
        
            loss = model.loss_function(batch_gm_recon,batch_gm_input,batch_gm_mu_z,batch_gm_log_var_z,
                                       batch_gm_mu_s,batch_gm_log_var_s,batch_gm_z,batch_gm_s,
                                       batch_cf_recon,batch_cf_input,batch_cf_mu_z,batch_cf_log_var_z)
        
            
            [batch_gm_recon, batch_gm_input, batch_gm_mu_z, 
             batch_gm_log_var_z, batch_gm_mu_s, batch_gm_log_var_s, 
             batch_gm_z, batch_gm_s] = [val.detach().cpu().numpy() for val in [batch_gm_recon, batch_gm_input, batch_gm_mu_z, 
                                                                               batch_gm_log_var_z, batch_gm_mu_s, batch_gm_log_var_s, 
                                                                               batch_gm_z, batch_gm_s]]
        
            [batch_cf_recon, batch_cf_input, batch_cf_mu_z, 
             batch_cf_log_var_z, batch_cf_mu_s, batch_cf_log_var_s, 
             batch_cf_z, batch_cf_s] = [val.detach().cpu().numpy() for val in [batch_cf_recon, batch_cf_input, batch_cf_mu_z, 
                                                                               batch_cf_log_var_z, batch_cf_mu_s, batch_cf_log_var_s, 
                                                                               batch_cf_z, batch_cf_s] ]
            
            batch_gm_recon_fg = model.forward_fg(batch_gm)[0].detach().cpu().numpy()
            batch_cf_recon_fg = model.forward_fg(batch_cf)[0].detach().cpu().numpy()
            batch_gm_recon_bg = model.forward_bg(batch_gm)[0].detach().cpu().numpy()
        
        varexp_gm = get_varexp(batch_gm_input,batch_gm_recon)
        varexp_cf = get_varexp(batch_cf_input,batch_cf_recon)
        in_out_corr_gm = correlate_columns(np.squeeze(batch_gm_input).transpose(),np.squeeze(batch_gm_recon).transpose()).mean()
        in_out_corr_cf = correlate_columns(np.squeeze(batch_cf_input).transpose(),np.squeeze(batch_cf_recon).transpose()).mean()
        
        varexp_gm_fg = get_varexp(batch_gm_input,batch_gm_recon_fg)
        varexp_cf_fg = get_varexp(batch_cf_input,batch_cf_recon_fg)
        in_out_corr_gm_fg = correlate_columns(np.squeeze(batch_gm_input).transpose(),np.squeeze(batch_gm_recon_fg).transpose()).mean()
        in_out_corr_cf_fg = correlate_columns(np.squeeze(batch_cf_input).transpose(),np.squeeze(batch_cf_recon_fg).transpose()).mean()

        loss_val = loss['loss'].detach().cpu().numpy()
        Reconstruction_Loss = loss['Reconstruction_Loss'].detach().cpu().numpy()
        kld_loss = loss['kld_loss'].detach().cpu().numpy()
        
        # Lists to be appended 
        track['loss'].append(loss_val)
        track['Reconstruction_Loss'].append(Reconstruction_Loss)
        track['kld_loss'].append(kld_loss)
        track['varexp_gm'].append(varexp_gm)
        track['varexp_cf'].append(varexp_cf)
        track['in_out_corr_gm'].append(in_out_corr_gm)
        track['in_out_corr_cf'].append(in_out_corr_cf)
        track['varexp_gm_fg'].append(varexp_gm_fg)
        track['varexp_cf_fg'].append(varexp_cf_fg)
        track['in_out_corr_gm_fg'].append(in_out_corr_gm_fg)
        track['in_out_corr_cf_fg'].append(in_out_corr_cf_fg)

        # Arrays to be updated
        track['batch_gm_input'] = batch_gm_input
        track['batch_gm_recon'] = batch_gm_recon
        track['batch_gm_recon_fg'] = batch_gm_recon_fg
        track['batch_cf_recon'] = batch_cf_recon
        track['batch_cf_input'] = batch_cf_input
        track['batch_cf_recon_fg'] = batch_cf_recon_fg
        track['batch_gm_mu_z'] = batch_gm_mu_z
        track['batch_cf_mu_z'] = batch_cf_mu_z
        track['batch_gm_mu_s'] = batch_gm_mu_s
        track['batch_cf_mu_s'] = batch_cf_mu_s

        # Progress metadata for the dashboard title (started/elapsed/ETA across
        # all ensembles). Mirrors the V2 branch so show_dahsboard_v1_marimo can
        # render the same title.
        track['current_ensemble'] = config.training.current_ensemble
        track['current_epoch'] = config.training.current_epoch
        track['n_ensembles'] = config.training.n_repetitions
        track['n_epochs'] = config.training.n_epochs
        track['output_dir'] = config.data.output_dir
        track['subject_idx'] = config.data.subject_idx
        track['run_idx'] = config.data.run_idx

        #
       
      
      
    
        # output['loss'] = loss['loss'].detach().cpu().numpy()
        # output['Reconstruction_Loss'] = loss['Reconstruction_Loss'].detach().cpu().numpy()
        # output['kld_loss'] = loss['kld_loss'].detach().cpu().numpy()

    elif model_version == 'V2':
        with torch.no_grad():
            sample_batch = next(iter(train_loader))
            inputs_gm = sample_batch[0].to(device).float()
            inputs_cf = sample_batch[1].to(device).float()

            outputs = model.forward_tg(inputs_gm)
            (outputs_gm, _, tg_mu_z, tg_log_var_z,
             tg_mu_s, tg_log_var_s, tg_z, tg_s) = outputs

            outputs_bg = model.forward_bg(inputs_cf)
            outputs_cf, _, bg_mu_z, bg_log_var_z = outputs_bg

            loss = model.loss_function(
                outputs_gm, inputs_gm,
                tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s, tg_z, tg_s,
                outputs_cf, inputs_cf, bg_mu_z, bg_log_var_z
            )

            batch_signal = model.forward_fg(inputs_gm)[0].detach().cpu().numpy()
            batch_noise = model.forward_bg(inputs_gm)[0].detach().cpu().numpy()

            noise_in = inputs_cf.detach().cpu().numpy()
            noise_recon = model.forward_bg(inputs_cf)[0].detach().cpu().numpy()
            noise_signal = model.forward_fg(inputs_cf)[0].detach().cpu().numpy()

            batch_in = inputs_gm.detach().cpu().numpy()
            batch_out = outputs_gm.detach().cpu().numpy()

        # Variance explained on the reconstructed BOLD channel (channel 0)
        batch_SST = ((batch_in[:, 0, :] - batch_in[:, 0, :].mean(axis=0)) ** 2).sum()
        batch_SSM = ((batch_in[:, 0, :] - batch_out[:, 0, :]) ** 2).sum()
        batch_varexp = float((1 - batch_SSM / batch_SST).round(2))

        # Mean input vs mean reconstruction correlation (channel 0)
        c_io = float(np.corrcoef(
            batch_in[:, 0, :].mean(axis=0),
            batch_out[:, 0, :].mean(axis=0)
        )[0, 1])

        # Scalar loss curves (logging only -> detach)
        track['l'].append(float(loss['loss'].detach().cpu().numpy()))
        track['kld_loss'].append(float(loss['kld_loss'].detach().cpu().numpy()))
        track['recons_loss_roi'].append(float(loss['recons_loss_roi'].detach().cpu().numpy()))
        track['recons_loss_roni'].append(float(loss['recons_loss_roni'].detach().cpu().numpy()))
        track['loss_recon_conf_s'].append(float(loss['loss_recon_conf_s'].detach().cpu().numpy()))
        track['loss_recon_conf_z'].append(float(loss['loss_recon_conf_z'].detach().cpu().numpy()))
        track['ncc_loss_tg'].append(float(loss['ncc_loss_tg'].detach().cpu().numpy()))
        track['ncc_loss_bg'].append(float(loss['ncc_loss_bg'].detach().cpu().numpy()))
        track['ncc_loss_conf_s'].append(float(loss['ncc_loss_conf_s'].detach().cpu().numpy()))
        track['ncc_loss_conf_z'].append(float(loss['ncc_loss_conf_z'].detach().cpu().numpy()))
        track['smoothness_loss'].append(float(loss['smoothness_loss'].detach().cpu().numpy()))
        track['recons_loss_fg'].append(float(loss['recons_loss_fg'].detach().cpu().numpy()))

        # Reconstruction quality metrics
        track['varexp'].append(batch_varexp)
        track['batch_varexp'].append(batch_varexp)
        track['ffa_io'].append(c_io)

        # First-unit latent traces (one value per epoch)
        track['tg_mu_z'].append(float(tg_mu_z.detach().cpu().numpy().std()))
        track['tg_mu_s'].append(float(tg_mu_s.detach().cpu().numpy().std()))
        track['bg_mu_z'].append(float(bg_mu_z.detach().cpu().numpy().std()))
        track['tg_log_var_s'].append(float(tg_log_var_s.detach().cpu().numpy().mean()))
        track['tg_log_var_z'].append(float(tg_log_var_z.detach().cpu().numpy().mean()))
        track['bg_log_var_z'].append(float(bg_log_var_z.detach().cpu().numpy().mean()))
        track['tg_z'] = tg_z.detach().cpu().numpy()
        track['tg_s'] = tg_s.detach().cpu().numpy()
        
        

        # Batch arrays for the timeseries panels (overwritten each epoch)
        track['batch_in'] = batch_in
        track['batch_out'] = batch_out
        track['batch_signal'] = batch_signal
        track['batch_noise'] = batch_noise

        track['noise_in'] = noise_in
        track['noise_recon'] = noise_recon
        track['noise_signal'] = noise_signal

        track['current_ensemble'] = config.training.current_ensemble
        track['current_epoch'] = config.training.current_epoch
        track['n_ensembles'] = config.training.n_repetitions
        track['n_epochs'] = config.training.n_epochs
      
        track['output_dir'] = config.data.output_dir
        track['subject_idx'] = config.data.subject_idx
        track['run_idx'] = config.data.run_idx

    return track

# Track-schema version -> tracked keys. Mirrors registry.py's dict style so
# adding a new schema is a small, local, additive change. The schema version
# labels the dashboard/track layout, NOT the model class (see
# backwards_compatibility.md section 4).
_TRACK_KEYS = {
    'V1': [
        'loss',
        'Reconstruction_Loss',
        'kld_loss',
        'varexp_gm',
        'varexp_cf',
        'in_out_corr_gm',
        'in_out_corr_cf',
        'varexp_gm_fg',
        'varexp_cf_fg',
        'in_out_corr_gm_fg',
        'in_out_corr_cf_fg',
    ],
    'V2': [
        'l',
        'kld_loss',
        'recons_loss_roi',
        'recons_loss_roni',
        'loss_recon_conf_s',
        'loss_recon_conf_z',
        'ncc_loss_tg',
        'ncc_loss_bg',
        'ncc_loss_conf_s',
        'ncc_loss_conf_z',
        'smoothness_loss',
        'recons_loss_fg',
        'varexp',
        'batch_varexp',
        'ffa_io',
        'tg_mu_z',
        'tg_log_var_z',
        'tg_mu_s',
        'tg_log_var_s',
        'tg_z',
        'tg_s',
        'bg_log_var_z',
        'bg_mu_z',
    ],
}
LATEST_TRACK_VERSION = 'V2'


def init_track(model_version=None, keys=None):
    """
    Initialize tracking dictionary.

    Args:
        model_version: which track schema to use ('V1', 'V2', ...). Defaults to
            the latest schema. Keys are chosen accordingly.
        keys: Explicit list of keys to track (overrides the schema table).

    Returns:
        Dictionary with initialized tracking lists
    """
    from datetime import datetime

    version = model_version or LATEST_TRACK_VERSION
    if keys is None:
        if version not in _TRACK_KEYS:
            raise ValueError(
                f"Unknown track version {version!r}; "
                f"available: {sorted(_TRACK_KEYS)}"
            )
        keys = _TRACK_KEYS[version]

    track = {}
    for key in keys:
        track[key] = []

    track['model_version'] = version
    track['T_start'] = datetime.now()
    return track



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


def format_progress_title(track):
    """Build the 'S{sub}R{run} ... started ... elapsed ... ETA ...' string.

    Shared by the dashboard suptitle and the high-level pipeline's per-epoch
    progress line so both report identical subject/run/started/elapsed/ETA
    information. Reads the per-run metadata populated by update_track.
    """
    from datetime import datetime

    current_ensemble = track['current_ensemble']
    current_epoch = track['current_epoch']
    n_repetitions = track['n_ensembles']
    n_epochs = track['n_epochs']
    subject_idx = track.get('subject_idx')
    run_idx = track.get('run_idx')

    T_display_start = track.get('T_overall_start', track.get('T_start'))
    tnow = datetime.now()
    telapsed = tnow - T_display_start

    tnow_formatted = str(T_display_start)[0:19]
    telapsed_formatted = str(telapsed)[0:9]

    epochs_done = current_epoch + 1
    total_epochs_done = current_ensemble * n_epochs + epochs_done
    per_epoch = telapsed / total_epochs_done
    remaining_epochs = (
        (n_epochs - epochs_done)
        + (n_repetitions - 1 - current_ensemble) * n_epochs
    )
    eta = per_epoch * remaining_epochs
    eta_formatted = str(eta)[0:9]

    return (
        f'S{subject_idx}R{run_idx} Training started at: {tnow_formatted}, '
        f'elapsed: {telapsed_formatted} Ens:{current_ensemble+1}/{n_repetitions}, '
        f'E:{current_epoch+1}/{n_epochs} ETA:{eta_formatted}'
    )

def show_dahsboard_v1_marimo(track, fig=None, save_fig=True):
    """Render the V1 (original CVAE) training dashboard.

    Pass fig=<existing fig> to reuse a figure object across iterations (avoids
    accumulating figures in memory). If save_fig is True the figure is written
    to output_dir/dashboard_S{subject_idx}_R{run_idx}_rep_{current_ensemble}.png
    using the progress metadata populated by update_track's V1 branch.
    """
    from datetime import datetime

    ncols = 4
    nrows = 4
    if fig is None:
        fig = plt.figure(figsize=(5*ncols,5*nrows))
    else:
        plt.figure(fig.number)
    sp = 0
    
    model_version = track["model_version"]
    loss = track["loss"]
    Reconstruction_Loss = track["Reconstruction_Loss"]
    kld_loss = track["kld_loss"]
    varexp_gm = track["varexp_gm"]
    varexp_cf = track["varexp_cf"]
    in_out_corr_gm = track["in_out_corr_gm"]
    in_out_corr_cf = track["in_out_corr_cf"]
    varexp_gm_fg = track["varexp_gm_fg"]
    varexp_cf_fg = track["varexp_cf_fg"]
    in_out_corr_gm_fg = track["in_out_corr_gm_fg"]
    in_out_corr_cf_fg = track["in_out_corr_cf_fg"]
    T_start = track["T_start"]
    batch_gm_input = track["batch_gm_input"]
    batch_gm_recon = track["batch_gm_recon"]
    batch_gm_recon_fg = track["batch_gm_recon_fg"]
    batch_cf_recon = track["batch_cf_recon"]
    batch_cf_input = track["batch_cf_input"]
    batch_cf_recon_fg = track["batch_cf_recon_fg"]
    batch_gm_mu_z = track["batch_gm_mu_z"]
    batch_cf_mu_z = track["batch_cf_mu_z"]
    batch_gm_mu_s = track["batch_gm_mu_s"]
    batch_cf_mu_s = track["batch_cf_mu_s"]
    
    
    sp+=1;plt.subplot(nrows,ncols,sp)
    plt.plot(loss)
    plt.title(f'loss: {loss[-1]:.2f}')

    sp+=1;plt.subplot(nrows,ncols,sp)
    if len(loss)>10:
        plt.plot(loss[-10::])
        plt.title(f'loss (last 10): {loss[-10]:.2f} vs. {loss[-1]:.2f}')
    else:
        plt.plot(loss)
        plt.title(f'loss (last 10): {loss[0]:.2f} vs. {loss[-1]:.2f}')

    sp+=1;plt.subplot(nrows,ncols,sp)
    plt.plot(Reconstruction_Loss)
    plt.title(f'Reconstruction_Loss: {Reconstruction_Loss[-1]:.2f}')

    sp+=1;plt.subplot(nrows,ncols,sp)
    plt.plot(kld_loss)
    plt.title(f'kld_loss: {kld_loss[-1]:.2f}')

    sp+=1;plt.subplot(nrows,ncols,sp)
    plt.plot(varexp_gm)
    plt.title(f'varexp_gm: {varexp_gm[-1]:.2f}')

    sp+=1;plt.subplot(nrows,ncols,sp)
    plt.plot(varexp_gm_fg)
    plt.title(f'varexp_gm_fg: {varexp_gm_fg[-1]:.2f}')

    sp+=1;plt.subplot(nrows,ncols,sp)
    plt.plot(in_out_corr_gm)
    plt.title(f'in_out_corr_gm: {in_out_corr_gm[-1]:.2f}')

    sp+=1;plt.subplot(nrows,ncols,sp)
    plt.plot(varexp_cf)
    plt.title(f'varexp_cf: {varexp_cf[-1]:.2f}')

    sp+=1;plt.subplot(nrows,ncols,sp)
    plt.plot(varexp_cf_fg)
    plt.title(f'varexp_cf_fg: {varexp_cf_fg[-1]:.2f}')

    sp+=1;plt.subplot(nrows,ncols,sp)
    plt.plot(in_out_corr_cf)
    plt.title(f'in_out_corr_cf: {in_out_corr_cf[-1]:.2f}')
    
    sp+=1;plt.subplot(nrows,ncols,sp)
    plt.plot(batch_gm_input[0,0,:],alpha=.5)
    plt.plot(batch_gm_recon[0,0,:],alpha=.5)
    plt.legend(['input','recon'])
    plt.title(f'GM voxel in/out: varexp={varexp_gm[-1]:.2f}, corr={in_out_corr_gm[-1]:.2f}')
    
    
    sp+=1;plt.subplot(nrows,ncols,sp)
    plt.plot(batch_gm_input[0,0,:],alpha=.5)
    plt.plot(batch_gm_recon_fg[0,0,:],alpha=.5)
    plt.legend(['input','recon'])
    plt.title(f'GM voxel denoised: varexp={varexp_cf_fg[-1]:.2f}, corr={in_out_corr_cf_fg[-1]:.2f}')
    
    sp+=1;plt.subplot(nrows,ncols,sp)
    plt.plot(batch_cf_input[0,0,:],alpha=.5)
    plt.plot(batch_cf_recon[0,0,:],alpha=.5)
    plt.legend(['input','recon'])
    plt.title(f'CF voxel in/out: varexp={varexp_cf[-1]:.2f}, corr={in_out_corr_cf[-1]:.2f}')
    
    
    sp+=1;plt.subplot(nrows,ncols,sp)
    plt.plot(batch_cf_input[0,0,:],alpha=.5)
    plt.plot(batch_cf_recon_fg[0,0,:],alpha=.5)
    plt.legend(['input','recon'])
    plt.title(f'CF voxel denoised: varexp={varexp_cf_fg[-1]:.2f}, corr={in_out_corr_cf_fg[-1]:.2f}')
    
    sp+=1;plt.subplot(nrows,ncols,sp)
    plt.hist(batch_gm_mu_z.flatten(),alpha=.5)
    plt.hist(batch_cf_mu_z.flatten(),alpha=.5)
    plt.legend(['GM Z','CF Z'])
    plt.title('Z features')
    
    sp+=1;plt.subplot(nrows,ncols,sp)
    plt.hist(batch_gm_mu_s.flatten(),alpha=.5)
    plt.hist(batch_cf_mu_s.flatten(),alpha=.5)
    plt.legend(['GM S','CF S'])
    plt.title('S features')
    
    # Progress title (started / elapsed / ETA across all ensembles), matching
    # show_dahsboard_v2_marimo. Metadata is populated by update_track's V1 branch.
    ttl = format_progress_title(track)
    plt.suptitle(ttl, y=.92)

    if save_fig:
        png_path = os.path.join(
            track['output_dir'],
            f"dashboard_S{track.get('subject_idx')}_R{track.get('run_idx')}"
            f"_rep_{track['current_ensemble']}.png"
        )
        fig.savefig(png_path, dpi=100, bbox_inches='tight')

    #plt.show()

    return fig


def show_dahsboard_v1_jupyter(track, fig=None, save_fig=True, display=True):
    """Render the V1 dashboard with the (matplotlib) Jupyter backend.

    This is pure matplotlib — it never imports marimo. When ``display`` is True
    it additionally clears the cell output and shows the figure in-place
    (IPython, lazily imported), matching mo.output.replace_at_index. Set
    ``display=False`` to only render and save the figure, which needs neither
    marimo nor IPython installed.

    Pass fig=this_fig to reuse the same figure object across iterations. The
    caller is responsible for closing the figure when done.
    """
    fig = show_dahsboard_v1_marimo(track, fig=fig, save_fig=save_fig)

    if display:
        import sys
        from IPython import display as ipython_display

        sys.stdout.flush()
        ipython_display.clear_output(wait=True)
        ipython_display.display(fig)

    return fig


def show_dahsboard_v2_marimo(track, fig=None, save_fig=True):
    """Render the V2 (confound-aware CVAE) training dashboard.

    Mirrors show_dahsboard_v1_marimo but plots the V2 track schema: the
    individual loss components, reconstruction quality metrics, in/out and
    denoised timeseries, and the first-unit latent traces.
    """
    from datetime import datetime

    ncols = 4
    nrows = 5
    if fig is None:
        fig = plt.figure(figsize=(5 * ncols, 5 * nrows))
    else:
        plt.figure(fig.number)
    #fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    #axes = axes.flatten()
    sp = 0

    T_start = track['T_start']

    # Loss curves
    loss_curves = [
        ('l', 'loss'),
        ('kld_loss', 'kld_loss'),
        ('recons_loss_roi', 'recons_loss_roi'),
        ('recons_loss_roni', 'recons_loss_roni'),
        ('loss_recon_conf_s', 'loss_recon_conf_s'),
        ('loss_recon_conf_z', 'loss_recon_conf_z'),
        ('ncc_loss_tg', 'ncc_loss_tg'),
        ('ncc_loss_bg', 'ncc_loss_bg'),
        ('ncc_loss_conf_s', 'ncc_loss_conf_s'),
        ('ncc_loss_conf_z', 'ncc_loss_conf_z'),
        ('smoothness_loss', 'smoothness_loss'),
        ('recons_loss_fg', 'recons_loss_fg'),
    ]
    for key, title in loss_curves:
        vals = track[key]
        sp += 1; plt.subplot(nrows, ncols, sp)
        plt.plot(vals)
        if len(vals):
            plt.title(f'{title}: {vals[-1]:.2f}')
        else:
            plt.title(title)

    # Reconstruction quality metrics
    varexp = track['varexp']
    ffa_io = track['ffa_io']

    sp += 1; plt.subplot(nrows, ncols, sp)
    plt.plot(varexp)
    plt.title(f'varexp: {varexp[-1]:.2f}' if varexp else 'varexp')

    sp += 1; plt.subplot(nrows, ncols, sp)
    plt.plot(ffa_io)
    plt.title(f'in/out corr: {ffa_io[-1]:.2f}' if ffa_io else 'in/out corr')

    # Timeseries panels (channel 0 = BOLD)
    batch_in = track.get('batch_in')
    batch_out = track.get('batch_out')
    batch_signal = track.get('batch_signal')
    batch_noise = track.get('batch_noise')

    noise_in = track.get('noise_in')
    noise_recon = track.get('noise_recon')
    noise_signal = track.get('noise_signal')

    if batch_in is not None and batch_out is not None:
        sp += 1; plt.subplot(nrows, ncols, sp)
        plt.plot(batch_in[0, 0, :], alpha=.5)
        plt.plot(batch_out[0, 0, :], alpha=.5)
        plt.legend(['input', 'recon'])
        plt.title(f'GM voxel in/out: varexp={varexp[-1]:.2f}' if varexp else 'GM voxel in/out')

    if batch_in is not None and batch_signal is not None:
        sp += 1; plt.subplot(nrows, ncols, sp)
        plt.plot(batch_in[0, 0, :], alpha=.5)
        plt.plot(batch_signal[0, 0, :],'g-',alpha=.5)
        plt.legend(['input', 'signal (fg)'])
        plt.title('GM voxel denoised (foreground)')

    if batch_in is not None and batch_noise is not None:
        sp += 1; plt.subplot(nrows, ncols, sp)
        plt.plot(noise_in[0, 0, :], alpha=.5)  
        plt.plot(noise_recon[0, 0, :], alpha=.5)
        plt.plot(noise_signal[0, 0, :], alpha=.5)
        plt.legend(['input', 'noise (bg)', 'signal'])
        plt.title('CF voxel')

    # First-unit latent traces over epochs
    sp += 1; plt.subplot(nrows, ncols, sp)
    plt.plot(track['tg_mu_z'])
    plt.plot(track['tg_mu_s'])
    plt.plot(track['bg_mu_z'])
    plt.legend(['tg_mu_z', 'tg_mu_s', 'bg_mu_z'])
    plt.title('variance in mu')

    sp += 1; plt.subplot(nrows, ncols, sp)
    plt.hist(track['tg_z'].flatten(),alpha=.3)
    plt.hist(track['tg_s'].flatten(),alpha=.3)
    plt.legend(['tg_z','tg_s'])
    plt.title('latents hist')

    sp += 1; plt.subplot(nrows, ncols, sp)
    plt.plot(track['tg_log_var_z'])
    plt.plot(track['bg_log_var_z'])
    plt.plot(track['tg_log_var_s'])
    plt.legend(['tg_log_var_z', 'bg_log_var_z', 'tg_log_var_s'])
    plt.title('avg of log_var')


    ttl = format_progress_title(track)
    plt.suptitle(ttl, y=.90)

    if save_fig:
        png_path = os.path.join(
            track['output_dir'],
            f"dashboard_S{track.get('subject_idx')}_R{track.get('run_idx')}"
            f"_rep_{track['current_ensemble']}.png"
        )
        fig.savefig(png_path, dpi=100, bbox_inches='tight')

    return fig


def show_dahsboard_v2_jupyter(track, fig=None, save_fig=True, display=True):
    """Render the V2 dashboard with the (matplotlib) Jupyter backend.

    This is pure matplotlib — it never imports marimo. When ``display`` is True
    it additionally clears the cell output and shows the figure in-place
    (IPython, lazily imported), matching mo.output.replace_at_index. Set
    ``display=False`` to only render and save the figure, which needs neither
    marimo nor IPython installed.

    Pass fig=this_fig to reuse the same figure object across iterations. The
    caller is responsible for closing the figure when done.
    """
    fig = show_dahsboard_v2_marimo(track, fig=fig, save_fig=save_fig)

    if display:
        import sys
        from IPython import display as ipython_display

        sys.stdout.flush()
        ipython_display.clear_output(wait=True)
        ipython_display.display(fig)

    return fig


# Track-schema version -> dashboard renderer. Add new renderers here and
# repoint LATEST_DASHBOARD_VERSION when a new schema becomes the default.
_DASHBOARD_RENDERERS = {
    'V1': show_dahsboard_v1_marimo,
    'V2': show_dahsboard_v2_marimo,
}
LATEST_DASHBOARD_VERSION = 'V2'


def show_dahsboard_marimo(track, fig=None, save_fig=True):
    """Render the dashboard for track['model_version'] (latest by default).

    Pass fig=<existing fig> to reuse a figure object across epochs; save_fig
    controls whether the figure is written to output_dir.
    """
    version = track.get('model_version', LATEST_DASHBOARD_VERSION)
    renderer = _DASHBOARD_RENDERERS.get(version)
    if renderer is None:
        raise ValueError(
            f"No dashboard renderer for {version!r}; "
            f"available: {sorted(_DASHBOARD_RENDERERS)}"
        )
    return renderer(track, fig=fig, save_fig=save_fig)


# Track-schema version -> Jupyter dashboard renderer. The Jupyter renderers
# display the figure in-place (IPython clear_output + display) and save it,
# without requiring marimo. Add new renderers here alongside _DASHBOARD_RENDERERS.
_DASHBOARD_RENDERERS_JUPYTER = {
    'V1': show_dahsboard_v1_jupyter,
    'V2': show_dahsboard_v2_jupyter,
}


def show_dahsboard_jupyter(track, fig=None, save_fig=True, display=True):
    """Render the Jupyter dashboard for track['model_version'] (latest by default).

    Marimo-free counterpart of show_dahsboard_marimo: dispatches to the
    version-appropriate ``show_dahsboard_v{N}_jupyter`` renderer (pure
    matplotlib). When ``save_fig`` it writes the figure to output_dir; when
    ``display`` it also shows it in-place via IPython. With ``display=False``
    it only renders + saves and needs neither marimo nor IPython installed.
    Pass fig=<existing fig> to reuse a figure object across epochs.
    """
    version = track.get('model_version', LATEST_DASHBOARD_VERSION)
    renderer = _DASHBOARD_RENDERERS_JUPYTER.get(version)
    if renderer is None:
        raise ValueError(
            f"No Jupyter dashboard renderer for {version!r}; "
            f"available: {sorted(_DASHBOARD_RENDERERS_JUPYTER)}"
        )
    return renderer(track, fig=fig, save_fig=save_fig, display=display)

