# """Visualization and dashboard utilities."""
from deepcor.analysis import correlate_columns

def get_varexp(mat_input,mat_output):
    # Calculates, variance explained
    # if varexp very negative, truncates at -1 
    SS_total = ((mat_input[:,0,:]-mat_input[:,0,:].mean(axis=-1)[:,np.newaxis])**2).sum()
    SS_model = ((mat_input[:,0,:]-mat_output[:,0,:])**2).sum()
    varexp = 1-(SS_model/SS_total)
    if varexp<-1:
        varexp = -1

    return varexp

def update_track(track,train_loader,model):
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
    
        # output['loss'] = loss['loss'].detach().cpu().numpy()
        # output['Reconstruction_Loss'] = loss['Reconstruction_Loss'].detach().cpu().numpy()
        # output['kld_loss'] = loss['kld_loss'].detach().cpu().numpy()

    

    return track

def init_track(model_version='V1',keys=None):
    from datetime import datetime
    """
    Initialize tracking dictionary.

    Args:
        model_version: which CVAE version is used. Keys are chosen accordingly
        keys: List of keys to track

    Returns:
        Dictionary with initialized tracking lists
    """
    if model_version=='V1':
        keys = [
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
        ]
    else:
        pass
        
    track = {}
    for key in keys:
        track[key] = []

    track['model_version'] = model_version
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

def show_dahsboard_v1_marimo(track):
    from datetime import datetime
    
    ncols = 4
    nrows = 4
    fig = plt.figure(figsize=(5*ncols,5*nrows))
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
    
    tnow = datetime.now()
    telapsed = tnow-T_start
    plt.suptitle(f'Training started at: {T_start}, elapsed: {telapsed}',y=.92)
    
    #plt.show()

    return fig


  
