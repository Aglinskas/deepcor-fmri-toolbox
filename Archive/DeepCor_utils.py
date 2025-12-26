import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal
from itertools import combinations_with_replacement
from numpy import savetxt
import math
from numpy import random
import sklearn.preprocessing  
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn import linear_model
from torch.autograd import Function

import ants # ANTSpy in the toolbox for manipulating MRI files 
from tqdm import tqdm # Easy progress bars
import seaborn as sns

import sys
from IPython import display
import pickle

from nilearn.glm.first_level import make_first_level_design_matrix


print(f'numpy version: {np.__version__}')
print(f'sklearn version: {sklearn.__version__}')
print(f'torch version: {torch.__version__}')
print(f'AntsPy version: {ants.__version__}')

def correlation(x,y):
  x_mean = np.repeat(x.mean(),x.shape,axis=0)
  y_mean = np.repeat(y.mean(),y.shape,axis=0)
  cov = (x-x_mean)*(y-y_mean)
  r = cov.sum()/(x.std()*y.std()*x.shape[0])
  return r

def remove_std0(arr):
    std0 = np.argwhere(np.std(arr, axis=1) == 0.0)
    arr_o = np.delete(arr,std0 ,axis=0) 
    return arr_o


class TrainDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, Y):
    self.obs = X
    self.noi = Y

  def __len__(self):
    return min(self.obs.shape[0],self.noi.shape[0])

  def __getitem__(self, index):
    observation = self.obs[index]
    noise = self.noi[index]
    s = 2*random.beta(4,4,1)
    noise_aug = s*noise
    return observation, noise_aug


def safe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

        
def plot_timeseries(epi,gm,cf):
    plt.figure(figsize=(20,5))
    plt.plot(epi.numpy()[gm.numpy()==1].mean(axis=0))
    plt.title('EPI ROI timeseries')
    plt.ylabel('BOLD')
    plt.xlabel('timepoints')

    plt.figure(figsize=(20,5))
    plt.plot(epi.numpy()[cf.numpy()==1].mean(axis=0))
    plt.title('EPI RONI timeseries')
    plt.ylabel('BOLD')
    plt.xlabel('timepoints')
    
    
def get_obs_noi_list_coords(epi,gm,cf):
    
    
    nTR = epi.shape[-1]
    epi_flat = epi.numpy().reshape(-1,nTR)
    gm_flat = gm.numpy().flatten()
    cf_flat = cf.numpy().flatten()

    # Drop STD0 voxels from mask
    std1 = epi_flat.std(axis=-1)>1e-3
    gm_flat=gm_flat*std1
    cf_flat=cf_flat*std1

    func_gm = epi_flat[gm_flat==1,:].copy() # Data that will be used as the ROI data
    func_cf = epi_flat[cf_flat==1,:].copy() # Data that will be used as the RONI data

    assert max(np.unique(cf_flat+gm_flat))!=2, 'overlap' # Assert that voxels in the ROI are NOT in the RONI and vice versa

    obs_list = func_gm
    noi_list = func_cf

    # Create 3D coordinate grids
    x_coords, y_coords, z_coords = np.meshgrid(
    np.arange(gm.shape[0]),  # x-coordinates
    np.arange(gm.shape[1]),  # y-coordinates
    np.arange(gm.shape[2]),  # z-coordinates
    indexing="ij"  # "ij" for matrix-style indexing
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

    obs_list_coords = np.concatenate([obs_list[:,:,np.newaxis],np.stack([gm_coords for _ in range(nTR)],axis=1)],axis=-1)
    noi_list_coords = np.concatenate([noi_list[:,:,np.newaxis],np.stack([cf_coords for _ in range(nTR)],axis=1)],axis=-1)
    obs_list_coords = np.swapaxes(obs_list_coords,1,2)
    noi_list_coords = np.swapaxes(noi_list_coords,1,2)


    # Z score 
    obs_list_coords[:,0,:] = (obs_list_coords[:,0,:]-obs_list_coords[:,0,:].mean(axis=1)[:,np.newaxis])/obs_list_coords[:,0,:].std(axis=1)[:,np.newaxis]
    noi_list_coords[:,0,:] = (noi_list_coords[:,0,:]-noi_list_coords[:,0,:].mean(axis=1)[:,np.newaxis])/noi_list_coords[:,0,:].std(axis=1)[:,np.newaxis]

    # std0_gm = obs_list_coords[:,0,:].std(axis=-1)<1e-3
    # std0_cf = noi_list_coords[:,0,:].std(axis=-1)<1e-3

    # print(f'std0 GM {std0_gm.sum()}')
    # print(f'std0 CF {std0_cf.sum()}')

    # print('{}/{}'.format(obs_list_coords.shape[0],noi_list_coords.shape[0]))
    # obs_list_coords = obs_list_coords[~std0_gm,:,:]
    # noi_list_coords = noi_list_coords[~std0_cf,:,:]
    # print('{}/{}'.format(obs_list_coords.shape[0],noi_list_coords.shape[0]))

    print(f'obs_list_coords.shape: {obs_list_coords.shape}')
    print(f'noi_list_coords.shape: {noi_list_coords.shape}')
    # Upsample
    if obs_list_coords.shape[0]>noi_list_coords.shape[0]:
        print('upsampling noi_list_coords')
        n_pad = obs_list_coords.shape[0]-noi_list_coords.shape[0]
        pad_idx = np.random.randint(low=0,high=noi_list_coords.shape[0],size=n_pad)
        noi_list_coords = np.concatenate([noi_list_coords,np.array([noi_list_coords[i,:,:] for i in pad_idx])])
        print(f'obs_list_coords.shape: {obs_list_coords.shape}')
        print(f'noi_list_coords.shape: {noi_list_coords.shape}')
    elif noi_list_coords.shape[0]>obs_list_coords.shape[0]:
        pass
        #raise Exception(f'CF mask too small: {noi_list_coords.shape[0]}, needs at least: {obs_list_coords.shape[0]+1}')


    gm = gm.new_image_like(gm_flat.reshape(gm.shape))
    cf = cf.new_image_like(cf_flat.reshape(gm.shape))

    return obs_list_coords,noi_list_coords,gm,cf


def apply_dummy(epi,df_conf,ndummy):
    if ndummy>0:
        epi_arr = epi.numpy()
        epi_arr[:,:,:,0:ndummy]=epi_arr[:,:,:,ndummy::].mean(axis=-1)[:,:,:,np.newaxis]
        epi = epi.new_image_like(epi_arr)

        #df_conf = pd.read_csv(conf_fn,delimiter='\t')
        df_conf.iloc[:ndummy, :] = 0
        
    return epi,df_conf


def init_track(keys):
    track = {}
    for key in keys:
        track[key] = []
        
    track['T_start'] = datetime.now()
    return track


def update_track(track,model,inputs_gm,inputs_cf):
    
    model.eval()
    
    [outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s,tg_z,tg_s] = model.forward_tg(inputs_gm)
    [outputs_cf, inputs_cf, bg_mu_z, bg_log_var_z] = model.forward_bg(inputs_cf)
    loss = model.loss_function(outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s,tg_z,tg_s, outputs_cf, inputs_cf, bg_mu_z, bg_log_var_z)
    
    track['l'].append( loss['loss'].detach().cpu().numpy() )
    track['kld_loss'].append( loss['kld_loss'].detach().cpu().numpy() )
    track['recons_loss_roi'].append( loss['recons_loss_roi'].detach().cpu().numpy() )
    track['recons_loss_roni'].append( loss['recons_loss_roni'].detach().cpu().numpy() )
    track['loss_recon_conf_s'].append( loss['loss_recon_conf_s'].detach().cpu().numpy() )
    track['loss_recon_conf_z'].append( loss['loss_recon_conf_z'].detach().cpu().numpy() )
    track['ncc_loss_tg'].append( loss['ncc_loss_tg'].detach().cpu().numpy() )
    track['ncc_loss_bg'].append( loss['ncc_loss_bg'].detach().cpu().numpy() )
    track['ncc_loss_conf_s'].append( loss['ncc_loss_conf_s'].detach().cpu().numpy() )
    track['ncc_loss_conf_z'].append( loss['ncc_loss_conf_z'].detach().cpu().numpy() )
    track['smoothness_loss'].append( loss['smoothness_loss'].detach().cpu().numpy() )
    track['recons_loss_fg'].append( loss['recons_loss_fg'].detach().cpu().numpy() )

    track['tg_log_var_z'] = tg_log_var_z.detach().cpu().numpy().flatten()
    track['bg_log_var_z'] = bg_log_var_z.detach().cpu().numpy().flatten()
    track['tg_log_var_s'] = tg_log_var_s.detach().cpu().numpy().flatten()

    track['tg_mu_z'] = tg_mu_z.detach().cpu().numpy().flatten()
    track['bg_mu_z'] = bg_mu_z.detach().cpu().numpy().flatten()
    track['tg_mu_s'] = tg_mu_s.detach().cpu().numpy().flatten()

    track['tg_log_var_z_mean'].append(tg_log_var_z.detach().cpu().numpy().mean())
    track['bg_log_var_z_mean'].append(bg_log_var_z.detach().cpu().numpy().mean())
    track['tg_log_var_s_mean'].append(tg_log_var_s.detach().cpu().numpy().mean())

    track['tg_mu_z'] = tg_mu_z.detach().cpu().numpy().flatten()
    track['bg_mu_z'] = bg_mu_z.detach().cpu().numpy().flatten()
    track['tg_mu_s'] = tg_mu_s.detach().cpu().numpy().flatten()

    track['tg_mu_z_std'].append(tg_mu_z.detach().cpu().numpy().std())
    track['bg_mu_z_std'].append(bg_mu_z.detach().cpu().numpy().std())
    track['tg_mu_s_std'].append(tg_mu_s.detach().cpu().numpy().std())

    batch_signal = model.forward_fg(inputs_gm)[0].detach().cpu().numpy()
    batch_out = model.forward_tg
    batch_noise = model.forward_bg(inputs_gm)[0].detach().cpu().numpy()

    batch_in = inputs_gm.detach().cpu().numpy()
    batch_out = outputs_gm.detach().cpu().numpy()
    
    cf_signal = model.forward_fg(inputs_cf)[0].detach().cpu().numpy()
    cf_noise = model.forward_bg(inputs_cf)[0].detach().cpu().numpy()

    batch_varexp = calc_mse(batch_in[:,0,:],batch_out[:,0,:])

    confounds_pred_z = model.decoder_confounds_z(torch.unsqueeze(tg_z,2)).detach().cpu().numpy()
    confounds_pred_s = model.decoder_confounds_s(torch.unsqueeze(tg_s,2)).detach().cpu().numpy()

    track['batch_signal'] = batch_signal[0,0,:]
    track['batch_noise'] = batch_noise[0,0,:]
    track['batch_in'] = batch_in[0,0,:]
    track['batch_out'] = batch_out[0,0,:]
    track['batch_coords_in'] = batch_in[0,1::,:]
    track['batch_coords_out'] = batch_out[0,1::,:]
    
    track['batch_varexp'].append(batch_varexp)
    track['confounds_pred_z'] = confounds_pred_z
    track['confounds_pred_s'] = confounds_pred_s

    track['cf_input'] = inputs_cf.detach().cpu().numpy()[0,0,:]
    track['cf_signal'] = cf_signal[0,0,:]
    track['cf_noise'] = cf_noise[0,0,:]
    
    #track['conf'] = conf
    model.train()
    
    return track

def show_bashboard(track,single_fig=True):

    nrows=5
    ncols=8
    sp=0
    
    if single_fig==True:
        plt.close()
        sys.stdout.flush()
        display.clear_output(wait=True);
        display.display(plt.gcf());
        plt.figure(figsize=(5*ncols,5*nrows))
    
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(track['l']);plt.title('total loss: {:.2f}'.format(track['l'][-1]))
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(track['batch_varexp']);plt.title('batch_varexp: {:.2f}'.format(track['batch_varexp'][-1]))    
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(track['recons_loss_roi']);plt.title('recons_loss_roi: {:.2f}'.format(track['recons_loss_roi'][-1]))
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(track['recons_loss_roni']);plt.title('recons_loss_roni: {:.2f}'.format(track['recons_loss_roni'][-1]))
    
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(track['kld_loss']);plt.title('kld_loss: {:.2f}'.format(track['kld_loss'][-1]))
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(track['recons_loss_fg']);plt.title('recons_loss_fg: {:.2f}'.format(track['recons_loss_fg'][-1]))
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(track['ncc_loss_tg']);plt.title('ncc_loss_tg: {:.2f}'.format(track['ncc_loss_tg'][-1]))
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(track['ncc_loss_bg']);plt.title('ncc_loss_bg: {:.2f}'.format(track['ncc_loss_bg'][-1]))
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(track['smoothness_loss']);plt.title('smoothness_loss: {:.2f}'.format(track['smoothness_loss'][-1]))
    
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(track['loss_recon_conf_z']);plt.title('loss_recon_conf_z: {:.2f}'.format(track['loss_recon_conf_z'][-1]))
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(track['ncc_loss_conf_z']);plt.title('ncc_loss_conf_z: {:.2f}'.format(track['ncc_loss_conf_z'][-1]))
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(track['loss_recon_conf_s']);plt.title('loss_recon_conf_s: {:.2f}'.format(track['loss_recon_conf_s'][-1]))
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(track['ncc_loss_conf_s']);plt.title('ncc_loss_conf_s: {:.2f}'.format(track['ncc_loss_conf_s'][-1]))
     
    idx = 3
    sp+=1;plt.subplot(nrows,ncols,sp);
    plt.plot(track['conf'][0,:])
    plt.plot(track['confounds_pred_z'][0,0,:])
    plt.title('conf from z')
    
    sp+=1;plt.subplot(nrows,ncols,sp);
    plt.plot(track['conf'][0,:])
    plt.plot(track['confounds_pred_s'][0,0,:])
    plt.title('conf from s')
    
    sp+=1;plt.subplot(nrows,ncols,sp);
    idx = 0
    plt.plot(track['batch_coords_in'][idx,:])
    plt.plot(track['batch_coords_out'][idx,:])
    plt.title('batch coords')

    sp+=1;plt.subplot(nrows,ncols,sp);
    plt.hist(track['tg_mu_z'],alpha=.3,color='b')
    plt.hist(track['bg_mu_z'],alpha=.3,color='r')
    plt.hist(track['tg_mu_s'],alpha=.3,color='g')
    plt.legend(['tg_mu_z','bg_mu_z','tg_mu_s'])

    sp+=1;plt.subplot(nrows,ncols,sp);
    plt.hist(track['tg_log_var_z'],alpha=.3,color='b')
    plt.hist(track['bg_log_var_z'],alpha=.3,color='r')
    plt.hist(track['tg_log_var_s'],alpha=.3,color='g')
    plt.legend(['bg_log_var_z','bg_log_var_z','tg_log_var_s'])

    sp+=1;plt.subplot(nrows,ncols,sp);
    plt.plot(track['tg_mu_z_std'],'b')
    plt.plot(track['bg_mu_z_std'],'r')
    plt.plot(track['tg_mu_s_std'],'g')
    plt.legend(['tg_mu_z_std','bg_mu_z_std','tg_mu_s_std'])
    plt.title('mu over time')

    sp+=1;plt.subplot(nrows,ncols,sp);
    plt.plot(track['tg_log_var_z_mean'],'b')
    plt.plot(track['bg_log_var_z_mean'],'r')
    plt.plot(track['tg_log_var_s_mean'],'g')
    plt.legend(['tg_log_var_z_mean','bg_log_var_z_mean','tg_log_var_s_mean'])
    plt.title('logvar over time')
    
    
    #sp+=1;plt.subplot(nrows,ncols,sp);
    sp+=1
    row, col = divmod(sp-1, ncols)
    plt.subplot2grid((nrows, ncols), (row, col), colspan=2)
    sp+=1
    
    plt.plot(track['batch_in'])
    plt.plot(track['batch_out'])
    plt.plot(track['batch_signal'],'r-')
    plt.plot(track['batch_noise'],'g-')
    plt.title('batch timecourse (single voxel)')

    sp+=1
    row, col = divmod(sp-1, ncols)
    plt.subplot2grid((nrows, ncols), (row, col), colspan=2)
    sp+=1
    
    plt.plot(track['cf_input'])
    plt.plot(track['cf_signal'],'g-')
    plt.plot(track['cf_noise'],'r-')
    plt.title('CF batch voxel')

    elapsed = datetime.now()-track['T_start']    
    if single_fig==True:
        #plt.suptitle(f'{sub}-R{r}-rep-{rep} E:{epoch} T:{elapsed}',y=.91,fontsize=20)
        #plt.suptitle(f'{track['sub']}-R{track['r']}-rep-{track['rep']} E:{track['epoch']} T:{elapsed}',y=.91,fontsize=20)
        #plt.suptitle('{track['sub']}-R{track['r']}-rep-{track['rep']} E:{track['epoch']} T:{elapsed}',y=.91,fontsize=20)
        plt.suptitle('{sub}-R{r}-rep-{rep} E:{epoch} T:{elapsed}'.format(sub=track['sub'],r=track['r'],rep=track['rep'],epoch=track['epoch'],elapsed=elapsed),y=.91,fontsize=20)
        #plt.savefig(os.path.join(track['ofdir'],f'dashboard_S{track['s']}_R{track['r']}_rep_{track['rep']}.jpg'))
        plt.savefig(os.path.join(track['ofdir'],'dashboard_S{s}_R{r}_rep_{rep}.jpg'.format(s=track['s'],r=track['r'],rep=track['rep'])))
        plt.show()
        
def calc_mse(y_true, y_pred,axis=0,clip=True):
    """
    Calculate variance explained (R^2) between two arrays.
    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
    Returns:
        float: Variance explained (R^2).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true,axis=axis)) ** 2)
    varexp = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    if all((clip==True,varexp<0)): # if it's less than 0 - set it to 0
        varexp=0
    
    return varexp


def save_brain_signals(model, train_inputs_coords,epi, gm,ofn,batch_size=512,kind='FG',inv_z_score=True):
    """
    Runs the model in eval mode to generate foreground signals for all GM voxels,
    reconstructs the 4D brain signal array, and saves it as a NIfTI file.

    Args:
        model: Trained model with a .forward_fg() method.
        train_inputs_coords: PyTorch Dataset for DataLoader.
        batch_size: Batch size for DataLoader.
        device: Torch device ('cuda' or 'cpu').
        epi: ANTsImage, used for shape and new_image_like.
        gm_flat: 1D numpy array, GM mask flattened.
        epi_flat: 2D numpy array, epi image flattened to (voxels, time).
        ofdir: Output directory.
        s: Subject index or ID.
        r: Run index or ID.
        rep: Repetition index or ID.
        tqdm: tqdm function for progress bar.
    """
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    nTR = epi.shape[-1]
    epi_flat = epi.numpy().reshape(-1,nTR)
    gm_flat = gm.numpy().flatten()
    
    train_in_coords = torch.utils.data.DataLoader(train_inputs_coords, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    dataloader_iter_in = iter(train_in_coords)
    brain_signals = []
    for _ in range(len(train_in_coords)):
        inputs_gm, inputs_cf = next(dataloader_iter_in)
        inputs_gm = inputs_gm.float().to(device)
        if kind=='FG':
            fg_signal = model.forward_fg(inputs_gm)[0].detach().cpu().numpy()[:, 0, :]
        elif kind=='TG':
            fg_signal = model.forward_tg(inputs_gm)[0].detach().cpu().numpy()[:, 0, :]
        elif kind=='BG':    
            fg_signal = model.forward_bg(inputs_gm)[0].detach().cpu().numpy()[:, 0, :]
        else:
            raise Exception(f'{kind}: not implemented')
            
        brain_signals.append(fg_signal)

    # Reconstruct the full brain array
    brain_signals_arr = np.zeros(epi_flat.shape)
    assert np.vstack(brain_signals).shape[0]==gm_flat.sum(), 'mismatch in voxel sizes: {}/{}'.format(np.vstack(brain_signals).shape[0],gm_flat.sum())

    brain_signals = np.vstack(brain_signals)
    
    if inv_z_score==True:
        epi_mean = epi_flat[gm_flat==1,:].mean(axis=1)
        epi_std = epi_flat[gm_flat==1,:].std(axis=1)
        brain_signals = (brain_signals*epi_std[:,np.newaxis]+epi_mean[:,np.newaxis]) # Invert z-scoring
        
    valid_voxels = gm_flat==1
    brain_signals_arr[valid_voxels, :] = brain_signals
    brain_signals_arr = brain_signals_arr.reshape(epi.shape)
    brain_signals_img = epi.new_image_like(brain_signals_arr)
    brain_signals_img.to_filename(ofn)
    
def save_track(track_ofn,track):
    import pickle
    with open(track_ofn, 'wb') as handle:
        pickle.dump(track, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def save_model(model_ofn,model,optimizer,epoch,loss):
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss, 
    }, model_ofn)
    
    
def get_roi_and_roni(epi,anat,anat_gm,anat_wm,anat_csf,do_plot=True):

    gm_values = anat_gm.numpy()
    gm_mask = (gm_values>0.5)
    wm_values = anat_wm.numpy()
    csf_values = anat_csf.numpy()
    cf_values = wm_values+csf_values
    cf_mask = (cf_values>0.5)

    diff = gm_mask & cf_mask
    gm_mask_c = gm_mask ^ diff
    cf_mask_c = cf_mask ^ diff

    gm_mask_c = gm_mask_c * epi.std(axis=-1)>1e-3
    cf_mask_c = cf_mask_c * epi.std(axis=-1)>1e-3
    
    gm = anat_gm.new_image_like(gm_mask_c*1.0)
    cf = anat_gm.new_image_like(cf_mask_c*1.0)
    
    if do_plot==True:
        epi_mean = epi.numpy().mean(axis=-1)
        epi_mean_nii = gm.new_image_like(epi_mean)

        epi_mean_nii.plot_ortho(flat=True,xyz_lines=False,orient_labels=False,figsize=2,overlay_alpha=.3)
        anat.plot_ortho(flat=True,xyz_lines=False,orient_labels=False,figsize=2,overlay_alpha=.3)

        epi_mean_nii.plot_ortho(gm*1.0+cf*2.0,flat=True,xyz_lines=False,orient_labels=False,figsize=2,overlay_alpha=.3,overlay_cmap='jet',title=f'red=CF, green=GM',textfontcolor='black')
        anat.plot_ortho(gm*1.0+cf*2.0,flat=True,xyz_lines=False,orient_labels=False,figsize=2,overlay_alpha=.3,overlay_cmap='jet')
        
    return gm,cf



def censor_and_interpolate(arr_flat: np.ndarray,idx_censor: np.ndarray,do3=True):
    import numpy as np
    import warnings
    """
    Given a data matrix and a censor mask, (1) expand the mask by one timepoint
    on either side of every True, and (2) replace each censored column with the
    average of its nearest uncensored neighbors.

    Parameters
    ----------
    arr_flat : np.ndarray, shape (n_voxels, n_timepoints)
        The data to be corrected.
    idx_censor : array_like of bool, shape (n_timepoints,)
        True where frames should be censored.

    Returns
    -------
    idx_censor3 : np.ndarray of bool, shape (n_timepoints,)
        The expanded censor mask (original True plus one neighbor before and after).
    arr_flat_corrected : np.ndarray, same shape as arr_flat
        A copy of arr_flat in which every column where idx_censor3 is True has
        been replaced by the mean of its nearest uncensored column before and after.
        If only one neighbor exists (e.g. at the edges), that neighbor is used alone.
    """
    arr_flat = np.asarray(arr_flat)
    idx_censor = np.asarray(idx_censor, dtype=bool)
    n_voxels, T = arr_flat.shape

    if idx_censor.shape[0] != T:
        raise ValueError(f"Length of idx_censor ({idx_censor.shape[0]}) "
                         f"must match number of timepoints ({T}).")

    # Step 1: expand the censor mask by one before/after
    prev_neighbor = np.concatenate([[False], idx_censor[:-1]])
    next_neighbor = np.concatenate([idx_censor[1:], [False]])
    idx_censor3 = idx_censor | prev_neighbor | next_neighbor
    
    if do3==False:
        idx_censor3=idx_censor

    # Step 2: build lookup tables for nearest uncensored before/after each t
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
            warnings.warn(f"No uncensored neighbors found for timepoint {t}; "
                          "leaving original values in place.")

    return idx_censor3, arr_flat_corrected



def apply_frame_censoring(im,df_conf,idx_censor,also_nearby_voxels):
    
    #im = epi
    #df_conf = df_conf
    #idx_censor = df_conf['FramewiseDisplacement'].values > .5
    arr_flat = im.numpy().reshape(-1,im.shape[-1])
    idx_censor3, arr_flat_corrected = censor_and_interpolate(arr_flat,idx_censor,do3=also_nearby_voxels)

    l = len(idx_censor3)
    n_censored = idx_censor3.sum()
    perc_censored = n_censored/l*100

    print(f'Censored {perc_censored:.2f}% of voxels {n_censored}/{l}')

    if perc_censored>40:
        import warnings
        warnings.warn('High number of censored frames: consider lowering the treshold or removing the subject from analyses')

    im_corrected = im.new_image_like(arr_flat_corrected.reshape(im.shape))

    idx_censor3,conf_corrected = censor_and_interpolate(df_conf.values.transpose(),idx_censor,do3=also_nearby_voxels)
    df_conf_corrected = df_conf.copy()
    df_conf_corrected.iloc[:,:]=conf_corrected.transpose()

    return im_corrected,df_conf_corrected



def array_to_brain(arr,epi,gm,ofn,inv_z_score=True,return_img=False):
    nTR = epi.shape[-1]
    epi_flat = epi.numpy().reshape(-1,nTR)
    gm_flat = gm.numpy().flatten()

    if inv_z_score==True:
        epi_mean = epi_flat[gm_flat==1,:].mean(axis=1)
        epi_std = epi_flat[gm_flat==1,:].std(axis=1)
        arr = (arr*epi_std[:,np.newaxis]+epi_mean[:,np.newaxis]) # Invert z-scoring

    assert arr.shape[0]==int(gm.numpy().sum()), f'shape mismatch: arr {arr.shape[0]}, GM {int(gm.numpy().sum())}'

    brain_signals_arr = np.zeros(epi_flat.shape)
    brain_signals_arr[gm_flat==1,:] = arr
    brain_signals_arr = brain_signals_arr.reshape(epi.shape)
    brain_signals_arr = epi.new_image_like(brain_signals_arr)
    brain_signals_arr.to_filename(ofn)
    
    if return_img==True:
        return brain_signals_arr
    
    
    
def load_pickle(fn):
    if os.path.exists(fn):
        with open(fn, 'rb') as file:
            loaded_dict = pickle.load(file)
    return loaded_dict



def average_signal_ensemble(signal_files,ofn):
    c = 0
    im = ants.image_read(signal_files[0])
    signal_avg = np.zeros(im.shape)
    for signal_file in tqdm(signal_files):
        im = ants.image_read(signal_file)
        arr = im.numpy()
        #if all([np.isnan(arr).sum()==0,np.max(arr)<1e3,np.max(arr)>1e-3]): # QA: only average if there arent NaNs in the ouputand the model hadn't collapsed or exploded
        if all([np.isnan(arr).sum()==0]): # QA: only average if there arent NaNs in the ouput
            signal_avg+=arr
            c+=1
    signal_avg = signal_avg/c # average
    #signal_avg = signal_avg*epi.numpy().std(axis=-1)[:,:,:,np.newaxis]+epi.numpy().mean(axis=-1)[:,:,:,np.newaxis] # inv Scale
    print(f'signals averaged: {c}')
    im = im.new_image_like(signal_avg)
    im.to_filename(ofn)
    return im



def calc_and_save_compcor(epi,gm,cf,ofn,n_components=5,return_img=False,do_center=True):
    from sklearn.decomposition import PCA
    from sklearn import linear_model

    n_components=5

    nTR = epi.shape[-1]
    epi_flat = epi.numpy().reshape(-1,nTR)
    
    if do_center==True:
        std0 = epi_flat.std(axis=-1)<1e-3
        epi_flat[~std0,:] = (epi_flat[~std0,:]-epi_flat[~std0,:].mean(axis=-1)[:,np.newaxis])/epi_flat[~std0,:].std(axis=-1)[:,np.newaxis] # Z-score the data
        epi_flat[std0,:]=0

    gm_flat = gm.numpy().flatten()
    cf_flat = cf.numpy().flatten()

    epi_cf = epi_flat[cf_flat==1,:].transpose()
    epi_gm = epi_flat[gm_flat==1,:].transpose()

    conf_pcs = PCA(n_components=n_components).fit_transform(epi_cf)
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(conf_pcs,epi_gm);

    compcor = epi_gm-lin_reg.predict(conf_pcs)
    compcor = compcor.transpose()

    n_std0 = (compcor.std(axis=1)<1e-3).sum()
    if n_std0>0:
        print(f'n_std0:{n_std0}')
        
    if return_img==False:
        array_to_brain(compcor,epi,gm,ofn,inv_z_score=False,return_img=False)
    else:
        img = array_to_brain(compcor,epi,gm,ofn,inv_z_score=False,return_img=True)
        return img 
    
    
    
def get_design_matrix(epi,events_fn):
    from nilearn.glm.first_level import make_first_level_design_matrix

    events = pd.read_csv(events_fn,delimiter='\t')

    t_r = np.round(epi.spacing[-1],2)
    nTR = epi.shape[-1]
    n_scans = nTR
    frame_times = (np.arange(n_scans) * t_r)
    X1 = make_first_level_design_matrix(frame_times,events,drift_model="polynomial",drift_order=3,hrf_model="SPM") 
    return X1



def correlate_columns(arr1, arr2):
    """
    Computes the Pearson correlation between corresponding columns of two matrices.
    
    Parameters:
    arr1 (np.ndarray): First matrix of shape (370, 1000)
    arr2 (np.ndarray): Second matrix of shape (370, 1000)
    
    Returns:
    np.ndarray: 1D array of correlations for each column (size 1000)
    """
    # Ensure input arrays are numpy arrays
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    
    # Subtract the mean of each column (normalize)
    arr1_centered = arr1 - np.mean(arr1, axis=0)
    arr2_centered = arr2 - np.mean(arr2, axis=0)
    
    # Compute the numerator (covariance)
    numerator = np.sum(arr1_centered * arr2_centered, axis=0)
    
    # Compute the denominator (product of standard deviations)
    denominator = np.sqrt(np.sum(arr1_centered**2, axis=0) * np.sum(arr2_centered**2, axis=0))
    
    # Compute the Pearson correlation for each column
    correlation = numerator / denominator
    
    return correlation



def calc_corr_map(im,mask,corr_target):

    mask_flat = mask.numpy().flatten()==1
    n = mask_flat.sum()
    im_flat = im.numpy().reshape(-1,im.shape[-1])

    r_vals = correlate_columns(im_flat[mask_flat,:].transpose(),np.array([corr_target for _ in range(n)]).transpose())

    res_flat = np.zeros(mask_flat.shape)
    res_flat[mask_flat]=r_vals
    
    temp_3d = im.slice_image(axis=-1,idx=0)
    res = res_flat.reshape(temp_3d.shape)
    
    res_nii = temp_3d.new_image_like(res)
    
    return res_nii



def get_contrast_val(Y,contrast_vec):
    
    Y = (Y-Y.mean(axis=1)[:,np.newaxis])/Y.std(axis=1)[:,np.newaxis] # Z score values
    Y = Y.transpose()
    X = X1.values
    beta = np.linalg.inv(X.T @ X1) @ X1.T @ Y
    beta = beta.T
    beta = beta.values
    
    contrast_values = beta @ contrast_vec

    return contrast_values

def calc_contrast_map(im,mask,contrast_vec):

    assert len(contrast_vec)==X1.shape[1], f'bad contrast shape. numel {len(contrast_vec)}: expected {X1.shape[1]}'
    assert sum(contrast_vec)==0, f'contrast does not sum to zero: sum={sum(contrast_vec)}'

    im_flat = im.numpy().reshape(-1,im.shape[-1])
    mask_flat = mask.numpy().flatten()==1
    n = mask_flat.sum()

    con_vals = get_contrast_val(im_flat[mask_flat,:],contrast_vec)

    res_flat = np.zeros(mask_flat.shape)
    res_flat[mask_flat]=con_vals

    temp_3d = im.slice_image(axis=-1,idx=0)
    res = res_flat.reshape(temp_3d.shape)

    res_nii = temp_3d.new_image_like(res)
    
    return res_nii



def run_correlation_analysis_from_spec(analysis_spec,epi,compcor,signals_averaged,gm):
    import pathlib

    assert 'corr_target' in analysis_spec, "correlation analysis spec has no key: corr_target"
    assert 'filename' in analysis_spec, "filename needs to be specified for a file to be saved"

    corr_target = analysis_spec['corr_target']

    assert len(corr_target)==epi.shape[-1], f"mismatching durations: epi sequence: {epi.shape[-1]}, corr_target: {len(corr_target)}"

    do_plot = analysis_spec.get('plot',False)
    if do_plot==True:
        assert 'ROI' in analysis_spec, 'plot requested, but ROI not specified'
        roi_fn = analysis_spec['ROI']
        assert os.path.exists(roi_fn), f'ROI does not exist: {roi_fn}'


    corr_map_preproc = calc_corr_map(epi,gm,corr_target)
    corr_map_compcor = calc_corr_map(compcor,gm,corr_target)
    corr_map_signal = calc_corr_map(signals_averaged,gm,corr_target)

    ofn = analysis_spec['filename']
    file_ext = ''.join(pathlib.Path(ofn).suffixes)

    corr_map_preproc.to_filename(ofn.replace(file_ext,'_preproc'+file_ext))
    corr_map_compcor.to_filename(ofn.replace(file_ext,'_compcor'+file_ext))
    corr_map_signal.to_filename(ofn.replace(file_ext,'_deepcor'+file_ext))

    print('saved as: {}'.format(ofn.replace(file_ext,'_preproc'+file_ext)))
    print('saved as: {}'.format(ofn.replace(file_ext,'_compcor'+file_ext)))
    print('saved as: {}'.format(ofn.replace(file_ext,'_deepcor'+file_ext)))

    if do_plot==True:
        roi = ants.image_read(roi_fn)
        mask = roi.numpy()==1

        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        val1=corr_map_preproc.numpy()[mask]
        val2=corr_map_compcor.numpy()[mask]
        val3=corr_map_signal.numpy()[mask]

        ys = [val1.mean(),val2.mean(),val3.mean()];
        xs = [0,1,2];
        plt.bar(xs,ys);
        plt.xticks(xs,labels=[f'preproc\n{ys[0]:.2f}',f'compcor\n{ys[1]:.2f}',f'signal\n{ys[2]:.2f}']);
        plt.title(ofn.split('/')[-1]+'\n'+roi_fn.split('/')[-1])
        
        
        
def run_contrast_analysis_from_spec(analysis_spec,epi,compcor,signals_averaged,gm):
    import pathlib

    assert 'contrast_vec' in analysis_spec, "contrast analysis spec has no key: contrast_vec"
    assert 'design_matrix' in analysis_spec, "contrast analysis spec has no key: design_matrix"
    assert 'filename' in analysis_spec, "filename needs to be specified for a file to be saved"

    contrast_vec = analysis_spec['contrast_vec']
    
    

    #assert len(corr_target)==epi.shape[-1], f"mismatching durations: epi sequence: {epi.shape[-1]}, corr_target: {len(corr_target)}"
    assert len(contrast_vec)==epi.shape[-1], f"mismatching durations: epi sequence: {epi.shape[-1]}, corr_target: {len(corr_target)}"

    do_plot = analysis_spec.get('plot',False)
    if do_plot==True:
        assert 'ROI' in analysis_spec, 'plot requested, but ROI not specified'
        roi_fn = analysis_spec['ROI']
        assert os.path.exists(roi_fn), f'ROI does not exist: {roi_fn}'


    con_map_preproc = calc_contrast_map(epi,gm,contrast_vec)
    con_map_compcor = calc_contrast_map(compcor,gm,contrast_vec)
    con_map_signal = calc_contrast_map(signals_averaged,gm,contrast_vec)

    ofn = analysis_spec['filename']
    file_ext = ''.join(pathlib.Path(ofn).suffixes)

    con_map_preproc.to_filename(ofn.replace(file_ext,'_preproc'+file_ext))
    con_map_compcor.to_filename(ofn.replace(file_ext,'_compcor'+file_ext))
    con_map_signal.to_filename(ofn.replace(file_ext,'_deepcor'+file_ext))

    print('saved as: {}'.format(ofn.replace(file_ext,'_preproc'+file_ext)))
    print('saved as: {}'.format(ofn.replace(file_ext,'_compcor'+file_ext)))
    print('saved as: {}'.format(ofn.replace(file_ext,'_deepcor'+file_ext)))

    if do_plot==True:
        roi = ants.image_read(roi_fn)
        mask = roi.numpy()==1

        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        val1=con_map_preproc.numpy()[mask]
        val2=con_map_compcor.numpy()[mask]
        val3=con_map_signal.numpy()[mask]

        ys = [val1.mean(),val2.mean(),val3.mean()];
        xs = [0,1,2];
        plt.bar(xs,ys);
        plt.xticks(xs,labels=[f'preproc\n{ys[0]:.2f}',f'compcor\n{ys[1]:.2f}',f'signal\n{ys[2]:.2f}']);
        plt.title(ofn.split('/')[-1]+'\n'+roi_fn.split('/')[-1])
        
        
        
        
def get_design_matrix(epi,events_fn):
    from nilearn.glm.first_level import make_first_level_design_matrix

    events = pd.read_csv(events_fn,delimiter='\t')

    t_r = np.round(epi.spacing[-1],2)
    nTR = epi.shape[-1]
    n_scans = nTR
    frame_times = (np.arange(n_scans) * t_r)
    X1 = make_first_level_design_matrix(frame_times,events,drift_model="polynomial",drift_order=3,hrf_model="SPM") 
    return X1



def correlate_columns(arr1, arr2):
    """
    Computes the Pearson correlation between corresponding columns of two matrices.
    
    Parameters:
    arr1 (np.ndarray): First matrix of shape (370, 1000)
    arr2 (np.ndarray): Second matrix of shape (370, 1000)
    
    Returns:
    np.ndarray: 1D array of correlations for each column (size 1000)
    """
    # Ensure input arrays are numpy arrays
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    
    # Subtract the mean of each column (normalize)
    arr1_centered = arr1 - np.mean(arr1, axis=0)
    arr2_centered = arr2 - np.mean(arr2, axis=0)
    
    # Compute the numerator (covariance)
    numerator = np.sum(arr1_centered * arr2_centered, axis=0)
    
    # Compute the denominator (product of standard deviations)
    denominator = np.sqrt(np.sum(arr1_centered**2, axis=0) * np.sum(arr2_centered**2, axis=0))
    
    # Compute the Pearson correlation for each column
    correlation = numerator / denominator
    
    return correlation


def calc_corr_map(im,mask,corr_target):

    mask_flat = mask.numpy().flatten()==1
    n = mask_flat.sum()
    im_flat = im.numpy().reshape(-1,im.shape[-1])

    r_vals = correlate_columns(im_flat[mask_flat,:].transpose(),np.array([corr_target for _ in range(n)]).transpose())

    res_flat = np.zeros(mask_flat.shape)
    res_flat[mask_flat]=r_vals
    
    temp_3d = im.slice_image(axis=-1,idx=0)
    res = res_flat.reshape(temp_3d.shape)
    
    res_nii = temp_3d.new_image_like(res)
    
    return res_nii


def get_contrast_val(Y,contrast_vec,X1):
    
    Y = (Y-Y.mean(axis=1)[:,np.newaxis])/Y.std(axis=1)[:,np.newaxis] # Z score values
    Y = Y.transpose()
    X = X1.values
    beta = np.linalg.inv(X.T @ X1) @ X1.T @ Y
    beta = beta.T
    beta = beta.values
    
    contrast_values = beta @ contrast_vec

    return contrast_values

def calc_contrast_map(im,mask,contrast_vec,X1):

    assert len(contrast_vec)==X1.shape[1], f'bad contrast shape. numel {len(contrast_vec)}: expected {X1.shape[1]}'
    assert sum(contrast_vec)==0, f'contrast does not sum to zero: sum={sum(contrast_vec)}'

    im_flat = im.numpy().reshape(-1,im.shape[-1])
    mask_flat = mask.numpy().flatten()==1
    n = mask_flat.sum()

    con_vals = get_contrast_val(im_flat[mask_flat,:],contrast_vec,X1)

    res_flat = np.zeros(mask_flat.shape)
    res_flat[mask_flat]=con_vals

    temp_3d = im.slice_image(axis=-1,idx=0)
    res = res_flat.reshape(temp_3d.shape)

    res_nii = temp_3d.new_image_like(res)
    
    return res_nii


def run_correlation_analysis_from_spec(analysis_spec,epi,compcor,signals_averaged,gm):
    import pathlib

    assert 'corr_target' in analysis_spec, "correlation analysis spec has no key: corr_target"
    assert 'filename' in analysis_spec, "filename needs to be specified for a file to be saved"

    corr_target = analysis_spec['corr_target']

    assert len(corr_target)==epi.shape[-1], f"mismatching durations: epi sequence: {epi.shape[-1]}, corr_target: {len(corr_target)}"

    do_plot = analysis_spec.get('plot',False)
    if do_plot==True:
        assert 'ROI' in analysis_spec, 'plot requested, but ROI not specified'
        roi_fn = analysis_spec['ROI']
        assert os.path.exists(roi_fn), f'ROI does not exist: {roi_fn}'


    corr_map_preproc = calc_corr_map(epi,gm,corr_target)
    corr_map_compcor = calc_corr_map(compcor,gm,corr_target)
    corr_map_signal = calc_corr_map(signals_averaged,gm,corr_target)

    ofn = analysis_spec['filename']
    file_ext = ''.join(pathlib.Path(ofn).suffixes)

    corr_map_preproc.to_filename(ofn.replace(file_ext,'_preproc'+file_ext))
    corr_map_compcor.to_filename(ofn.replace(file_ext,'_compcor'+file_ext))
    corr_map_signal.to_filename(ofn.replace(file_ext,'_deepcor'+file_ext))

    print('saved as: {}'.format(ofn.replace(file_ext,'_preproc'+file_ext)))
    print('saved as: {}'.format(ofn.replace(file_ext,'_compcor'+file_ext)))
    print('saved as: {}'.format(ofn.replace(file_ext,'_deepcor'+file_ext)))

    if do_plot==True:
        roi = ants.image_read(roi_fn)
        mask = roi.numpy()==1

        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        val1=corr_map_preproc.numpy()[mask]
        val2=corr_map_compcor.numpy()[mask]
        val3=corr_map_signal.numpy()[mask]

        ys = [val1.mean(),val2.mean(),val3.mean()];
        xs = [0,1,2];
        plt.bar(xs,ys);
        plt.xticks(xs,labels=[f'preproc\n{ys[0]:.2f}',f'compcor\n{ys[1]:.2f}',f'signal\n{ys[2]:.2f}']);
        plt.title(ofn.split('/')[-1]+'\n'+roi_fn.split('/')[-1])
        
        
def run_contrast_analysis_from_spec(analysis_spec,epi,compcor,signals_averaged,gm):
    import pathlib

    assert 'contrast_vec' in analysis_spec, "contrast analysis spec has no key: contrast_vec"
    assert 'filename' in analysis_spec, "filename needs to be specified for a file to be saved"

    contrast_vec = analysis_spec['contrast_vec']
    X1 = analysis_spec['design_matrix']
    
    assert len(contrast_vec)==X1.shape[1], f"mismatching number of conditions: Design matrix: {X1.shape[1]}, contrast length: {len(contrast_vec)}"

    do_plot = analysis_spec.get('plot',False)
    if do_plot==True:
        assert 'ROI' in analysis_spec, 'plot requested, but ROI not specified'
        roi_fn = analysis_spec['ROI']
        assert os.path.exists(roi_fn), f'ROI does not exist: {roi_fn}'


    con_map_preproc = calc_contrast_map(epi,gm,contrast_vec,X1)
    con_map_compcor = calc_contrast_map(compcor,gm,contrast_vec,X1)
    con_map_signal = calc_contrast_map(signals_averaged,gm,contrast_vec,X1)

    ofn = analysis_spec['filename']
    file_ext = ''.join(pathlib.Path(ofn).suffixes)

    con_map_preproc.to_filename(ofn.replace(file_ext,'_preproc'+file_ext))
    con_map_compcor.to_filename(ofn.replace(file_ext,'_compcor'+file_ext))
    con_map_signal.to_filename(ofn.replace(file_ext,'_deepcor'+file_ext))

    print('saved as: {}'.format(ofn.replace(file_ext,'_preproc'+file_ext)))
    print('saved as: {}'.format(ofn.replace(file_ext,'_compcor'+file_ext)))
    print('saved as: {}'.format(ofn.replace(file_ext,'_deepcor'+file_ext)))

    if do_plot==True:
        roi = ants.image_read(roi_fn)
        mask = roi.numpy()==1

        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        val1=con_map_preproc.numpy()[mask]
        val2=con_map_compcor.numpy()[mask]
        val3=con_map_signal.numpy()[mask]

        ys = [val1.mean(),val2.mean(),val3.mean()];
        xs = [0,1,2];
        plt.bar(xs,ys);
        plt.xticks(xs,labels=[f'preproc\n{ys[0]:.2f}',f'compcor\n{ys[1]:.2f}',f'signal\n{ys[2]:.2f}']);
        plt.title(ofn.split('/')[-1]+'\n'+roi_fn.split('/')[-1])


def check_gpu_and_speedup(tensor_size=(1000, 1000), n_iter=100):
    """
    Checks if a GPU is available, gets its name, and calculates the speedup of a simple operation
    (matrix multiplication) on GPU vs CPU.

    Args:
        tensor_size (tuple): Size of the random tensors to multiply.
        n_iter (int): Number of iterations to average timing.

    Returns:
        dict: {
            'gpu_available': bool,
            'gpu_name': str or None,
            'cpu_time': float,
            'gpu_time': float or None,
            'speedup': float or None
        }
    """
    import torch
    import time

    # Generate random data
    a_cpu = torch.randn(tensor_size)
    b_cpu = torch.randn(tensor_size)

    # Time on CPU
    start_cpu = time.time()
    for _ in range(n_iter):
        c_cpu = torch.mm(a_cpu, b_cpu)
    end_cpu = time.time()
    cpu_time = (end_cpu - start_cpu) / n_iter

    gpu_available = torch.cuda.is_available()
    gpu_time = None
    speedup = None
    gpu_name = None

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        a_gpu = a_cpu.to('cuda')
        b_gpu = b_cpu.to('cuda')
        # Warm up GPU
        for _ in range(5):
            _ = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        # Time on GPU
        start_gpu = time.time()
        for _ in range(n_iter):
            c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        end_gpu = time.time()
        gpu_time = (end_gpu - start_gpu) / n_iter
        speedup = cpu_time / gpu_time if gpu_time > 0 else None

    return {
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': speedup
            }