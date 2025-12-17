"""### Get Files"""

import os
import numpy as np
import pandas as pd
import ants

import importlib

import DeepCor_utils
importlib.reload(DeepCor_utils)

import DeepCor_models
importlib.reload(DeepCor_models)

from DeepCor_utils import *
from DeepCor_models import *

### GPU Checks
result = check_gpu_and_speedup()
print(f"GPU available: {result['gpu_available']}")
if result['gpu_available']:
    print(f"GPU name: {result['gpu_name']}")
print(f"Average CPU time per op: {result['cpu_time']:.6f} s")
if result['gpu_available']:
    print(f"Average GPU time per op: {result['gpu_time']:.6f} s")
    print(f"Speedup (CPU/GPU): {result['speedup']:.2f}x")

check_gpu_and_speedup()

s = 0
r = 1
sub = 'sub-01'
indir = '../Data/fmriprep-forrest'
analysis_name = 'forrest-colab-test-v1'

epi_fn = os.path.join(indir,'{sub}/ses-localizer/func/{sub}_ses-localizer_task-objectcategories_run-{r}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')
cf_fn = os.path.join(indir,'mask_roni.nii') # non-gray matter mask
gm_fn = os.path.join(indir,'mask_roi.nii') # Gray matter mask
conf_fn = os.path.join(indir,f'{sub}/ses-localizer/func/{sub}_ses-localizer_task-objectcategories_run-{r}_bold_confounds.tsv') # fMRIprep confounds file

epi = ants.image_read(epi_fn.format(sub=sub,r=r))
gm = ants.image_read(gm_fn)
cf = ants.image_read(cf_fn)
df_conf = pd.read_csv(conf_fn,delimiter='\t')

## EPI parameters: Check that these are correct
nTR = epi.shape[-1]
t_r = round(epi.spacing[-1],2)
ndummy = 0 # How many dummy scans to discard

print(f'Number of scans: {nTR}')
print(f'Repetition Time: {t_r}')
print(f'Dummy scans to discard: {ndummy}')

# Directory where to save the outputs
ofdir_root = '../Data/DeepCor-Outputs/'
ofdir = os.path.join(ofdir_root,analysis_name)
safe_mkdir(ofdir)
print(ofdir)

"""## Post-training analyses"""

run_post_analyses = True # Whether to run contrast and correlation analyses

if run_post_analyses:
  events_fn = os.path.join(f'../Data/Events/{sub}_ses-localizer_task-objectcategories_run-{r}_events.tsv')
  X1 = get_design_matrix(epi,events_fn)
  X1

# If no post-training analyses needed, leave these empty
contrast_analyses = [] # Runs a voxelvise GLM and contrast analyses based on spec below
correlation_analyses = [] # Correlates each voxel with a specific target regressor

if run_post_analyses==True:
  correlation_analyses.append(
      {'corr_target' : X1['face'].values, # Correlate each voxel with this regressor
      'filename' : os.path.join(ofdir,f'corr2face_S{s}_R{r}.nii.gz'), # Output filename
      'plot' : True, # Automatically plot? If so specify a ROI
      'ROI' : f'../Data/Misc/rFFA_final_mask_{sub}_bin.nii.gz'}) # ROI for plotting (Can be None)


  correlation_analyses.append(
      {'corr_target' : X1[['house','scene']].values.mean(axis=1),
      'filename' : os.path.join(ofdir,f'corr2place_S{s}_R{r}.nii.gz'),
      'plot' : True,
      'ROI' : f'../Data/Misc/rPPA_final_mask_{sub}_bin.nii.gz'})

  contrast_analyses.append(
      {'contrast_vec' : [-1,5,-1,-1,-1,-1,0,0,0,0], # Contrast Vector spec
      'design_matrix' : X1,
      'filename' : os.path.join(ofdir,f'contrast_face_{s}_R{r}.nii.gz'),
      'plot' : True,
      'ROI' : f'../Data/Misc/rFFA_final_mask_{sub}_bin.nii.gz'})

  contrast_analyses.append(
      {'contrast_vec' : [-1,-1,2,-1,2,-1,0,0,0,0],
      'design_matrix' : X1,
      'filename' : os.path.join(ofdir,f'contrast_place_S{s}_R{r}.nii.gz'),
      'plot' : True,
      'ROI' : f'../Data/Misc/rPPA_final_mask_{sub}_bin.nii.gz'})





# Model Hyperparameters
hyperparams_cvae = {}
hyperparams_cvae['nrep'] = 20
hyperparams_cvae['epoch_num'] = 100
hyperparams_cvae['batch_size'] = 1024
hyperparams_cvae['latent_dim'] = (8,8) # Shared, Specific
hyperparams_cvae['beta'] = 0.01
hyperparams_cvae['gamma'] = 0 # TC scaling
hyperparams_cvae['delta'] = 0 # "Denoised RONI should be zero" scaling
hyperparams_cvae['scale_MSE_GM'] = 1e3 # Scale ROI loss
hyperparams_cvae['scale_MSE_CF'] = 1e3 # Scale RONI loss
hyperparams_cvae['scale_MSE_FG'] = 0 # "Denoised should be similar to input" scaling
# Optimizer
hyperparams_cvae['lr'] = 0.001
hyperparams_cvae['betas'] = (0.9, 0.999)
hyperparams_cvae['eps'] = 1e-08

"""# END of user-specified parameters"""

epi,df_conf = apply_dummy(epi,df_conf,ndummy)
#gm,cf = get_roi_and_roni(anat_gm,anat_wm,anat_csf,do_plot=True) # If using individual masks

#use_cols = ['trans_x','trans_y','trans_z', 'rot_x','rot_y','rot_z']
use_cols = ['X','Y','Z','RotX','RotY','RotZ']
assert np.isnan(df_conf.loc[:,use_cols].values).sum()==0,'NaNs in motion'
conf = df_conf.loc[:,use_cols].values.transpose()
conf[0:3,:] = (conf[0:3,:]-conf[0:3,:].min()) / (conf[0:3,:].max()-conf[0:3,:].min())
conf[3:,:] = (conf[3:,:]-conf[3:,:].min()) / (conf[3:,:].max()-conf[3:,:].min())
print(df_conf.shape)
df_conf.head()





plot_timeseries(epi,gm,cf)

# The keys differ between different fMRIprep versions.

#conf_keys = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
conf_keys = ['X','Y','Z','RotX','RotY','RotZ']
plt.figure(figsize=(25,10))
nrows = 2
ncols = 3
sp=0
for key in conf_keys:
    sp=sp+1;plt.subplot(nrows,ncols,sp)
    plt.plot(df_conf[key].values)
    plt.title(key)

plt.figure(figsize=(25,5))
#conf_keys = ['framewise_displacement']
conf_keys = ['FramewiseDisplacement']
plt.plot(df_conf[conf_keys].values)
thresh=.2
vec = (df_conf[conf_keys].values>thresh)*1.0*df_conf[conf_keys].values
vec[vec==0]=np.nan
plt.plot(vec,'r*',markersize=15)
plt.title(conf_keys)

plt.figure(figsize=(25,5))
#conf_keys = ['a_comp_cor_01','a_comp_cor_02',  'a_comp_cor_03',  'a_comp_cor_04','a_comp_cor_05']
conf_keys = ['aCompCor01','aCompCor02',  'aCompCor03',  'aCompCor04','aCompCor05']
plt.plot(df_conf[conf_keys].values)
plt.title(conf_keys)





# build the arrays which will be used to train
obs_list_coords,noi_list_coords,gm,cf = get_obs_noi_list_coords(epi,gm,cf)





# Run some sanity check, for flat voxels (not good for training) and NaNs
assert np.isnan(obs_list_coords).sum()==0, 'NaNs in obs_list_coords'
assert np.isnan(noi_list_coords).sum()==0, 'NaNs in noi_list_coords'
assert (obs_list_coords[:,0,:].std(axis=-1)<1e-3).sum()==0,'Std0 in obs_list_coords'
assert (noi_list_coords[:,0,:].std(axis=-1)<1e-3).sum()==0,'Std0 in noi_list_coords'



import importlib

import DeepCor_utils
importlib.reload(DeepCor_utils)

import DeepCor_models

importlib.reload(DeepCor_models)

from DeepCor_utils import *
from DeepCor_models import *



import traceback

nrep = hyperparams_cvae.get('nrep',20)
epoch_num = hyperparams_cvae.get('epoch_num',100)
batch_size = hyperparams_cvae.get('batch_size',512)
latent_dim = hyperparams_cvae.get('latent_dim',(8,8))
beta = hyperparams_cvae.get('beta',0.01)
gamma = hyperparams_cvae.get('gamma',0)
delta = hyperparams_cvae.get('delta',0)
scale_MSE_GM = hyperparams_cvae.get('scale_MSE_GM',1e3)
scale_MSE_CF = hyperparams_cvae.get('scale_MSE_CF',1e3)
scale_MSE_FG = hyperparams_cvae.get('scale_MSE_FG',0) # "Denoised should be similar to input" scaling
lr = hyperparams_cvae.get('lr',0.001)

print(f'nrep: {nrep}')
print(f'epoch_num: {epoch_num}')
print(f'batch_size: {batch_size}')
print(f'latent_dim: {latent_dim}')
print(f'beta: {beta}')
print(f'gamma: {gamma}')
print(f'delta: {delta}')
print(f'scale_MSE_GM: {scale_MSE_GM}')
print(f'scale_MSE_CF: {scale_MSE_CF}')
print(f'scale_MSE_FG: {scale_MSE_FG}')
print(f'lr: {lr}')

train_inputs_coords = TrainDataset(obs_list_coords,noi_list_coords)
train_in_coords = torch.utils.data.DataLoader(train_inputs_coords, batch_size=batch_size,shuffle=True, num_workers=1,drop_last=True)

global device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device is {device}')
Tensor = TypeVar('torch.tensor')
conf_batch = torch.tensor(np.array([conf for _ in range(batch_size)])).to(device)

keys = ['l', 'kld_loss', 'recons_loss_roi', 'recons_loss_roni',
       'loss_recon_conf_s', 'loss_recon_conf_z', 'ncc_loss_tg',
       'ncc_loss_bg', 'ncc_loss_conf_s', 'ncc_loss_conf_z',
       'smoothness_loss', 'recons_loss_fg',  'batch_varexp',
        'tg_mu_z', 'tg_log_var_z', 'tg_mu_s', 'tg_log_var_s',
        'tg_z', 'tg_s', 'bg_log_var_z', 'bg_mu_z',
        'tg_log_var_z_mean','bg_log_var_z_mean','tg_log_var_s_mean',
        'tg_mu_z_std','bg_mu_z_std','tg_mu_s_std','batch_signal',
        'batch_noise','batch_in','batch_out','batch_varexp',
        'confounds_pred_z','confounds_pred_s',]


track = init_track(keys)
errors = []

for rep in tqdm(range(nrep)):
    try:
        track = init_track(keys)
        track['sub'] = sub
        track['s'] = s
        track['r'] = r
        track['ofdir'] = ofdir
        track['conf'] = conf
        track['rep'] = rep

        track_ofn = os.path.join(ofdir,f'track_S{s}_R{r}_rep_{rep}.pickle')
        model_ofn = os.path.join(ofdir,f'model_S{s}_R{r}_rep_{rep}.pickle')

        model = cVAE(conf_batch,4,nTR,latent_dim, beta=beta, gamma=gamma,delta=delta,scale_MSE_GM=scale_MSE_GM,scale_MSE_CF=scale_MSE_CF,scale_MSE_FG=scale_MSE_FG,freq_exp=0,freq_scale=0,do_disentangle=True)
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)

        for epoch in range(epoch_num):  # loop over the dataset multiple times
            model.train()
            dataloader_iter_in = iter(train_in_coords)
            track['epoch'] = epoch
            for i in range(len(train_in_coords)):
                optimizer.zero_grad()
                inputs_gm,inputs_cf = next(dataloader_iter_in)

                inputs_gm = inputs_gm.float().to(device)
                inputs_cf = inputs_cf.float().to(device)

                [outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s,tg_z,tg_s] = model.forward_tg(inputs_gm)
                [outputs_cf, inputs_cf, bg_mu_z, bg_log_var_z] = model.forward_bg(inputs_cf)

                loss = model.loss_function(outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s,tg_z,tg_s, outputs_cf, inputs_cf, bg_mu_z, bg_log_var_z)

                if np.isnan(loss['loss'].detach().cpu().numpy()): raise ValueError(f'{rep}|{epoch}|{i}: loss is NaN') # If loss is NaNs, abort training and move on to the next repetition

                loss['loss'].backward() # Do a backward pass
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Prevent gradient explosion
                optimizer.step()

            track = update_track(track,model,inputs_gm,inputs_cf)

            if np.mod(epoch,10)==0:
              show_bashboard(track,single_fig=True)

        save_model(model_ofn,model,optimizer,epoch,loss)
        save_track(track_ofn,track)
        save_brain_signals(model,train_inputs_coords,epi,gm,ofn=os.path.join(ofdir,f'signal_S{s}_R{r}_rep_{rep}.nii.gz'),batch_size=512,kind='FG') # Save Denoised fMRI data
        #save_brain_signals(model,train_inputs_coords,epi,gm,ofn=os.path.join(ofdir,f'recon_S{s}_R{r}_rep_{rep}.nii.gz'),batch_size=512,kind='TG') # Optional to save reconstructions
        #save_brain_signals(model,train_inputs_coords,epi,gm,ofn=os.path.join(ofdir,f'noise_S{s}_R{r}_rep_{rep}.nii.gz'),batch_size=512,kind='BG') # Optional to save noise estimates

    except:
        errors.append(f'{rep}|{epoch}|{i}: loss is NaN')
        traceback.print_exc()
print('done training')
print(errors)



# Once training is finished
signal_files = [os.path.join(ofdir,f) for f in os.listdir(ofdir) if all((f.startswith(f'signal_S{s}_R{r}_rep_'),f.endswith('.nii.gz')))]
track_files = [os.path.join(ofdir,f) for f in os.listdir(ofdir) if all((f.startswith(f'track_S{s}_R{r}_rep_'),f.endswith('.pickle')))]
signal_files.sort()
track_files.sort()
print('Ensemble of {} repetitions'.format(len(signal_files)))

# Ensembling: Average the individual denoised files
signals_averaged = average_signal_ensemble(signal_files,os.path.join(ofdir,f'signal_S{s}_R{r}_avg.nii.gz'))

# Save a copy of the Nodenoised
array_to_brain(obs_list_coords[:,0,:],epi,gm,os.path.join(ofdir,f'preproc_S{s}_R{r}.nii.gz'),inv_z_score=True,return_img=False)

# Save the CompCor version
compcor = calc_and_save_compcor(epi,gm,cf,os.path.join(ofdir,f'compcor_S{s}_R{r}.nii.gz'),n_components=5,return_img=True)


for analysis_spec in correlation_analyses:
    run_correlation_analysis_from_spec(analysis_spec,epi,compcor,signals_averaged,gm)

for analysis_spec in contrast_analyses:
    run_contrast_analysis_from_spec(analysis_spec,epi,compcor,signals_averaged,gm)


tracks = [load_pickle(track_file) for track_file in track_files]
import warnings
warnings.filterwarnings("ignore")
plt.figure(figsize=(5*9,5*5))
for track in tracks:
    try:
        show_bashboard(track,single_fig=False)
    except:
        print('bad track')



