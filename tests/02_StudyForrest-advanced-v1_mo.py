# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.21.1",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full", auto_download=["ipynb", "html"])


@app.cell
def _():
    import os
    import numpy as np
    import pandas as pd
    import torch
    import ants
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    return ants, os, pd, plt, torch


@app.cell
def _(os):
    import marimo as mo
    os.chdir(mo.notebook_dir()) #Jupyterlab-like, change path to where the notebook is, all paths relative to this
    return (mo,)


@app.cell
def _():
    import deepcor

    return (deepcor,)


@app.cell
def _():
    #deepcor.utils.check_gpu_and_speedup(tensor_size=(1000,1000), n_iter=100)
    return


@app.cell
def _():
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()
    return


@app.cell
def _(deepcor):
    # ModelConfig: Configure the model architecture
    model_config = deepcor.config.ModelConfig(
        latent_dims=(8, 8),  # (shared dim, specific dim)
        beta=0.01,           # KLD loss weight
        gamma=0.0,           # TC loss weight
        delta=0.0,           # RONI zero constraint weight
        scale_MSE_GM=1e3,    # Gray matter reconstruction loss scale
        scale_MSE_CF=1e3,    # Non-gray matter reconstruction loss scale
        scale_MSE_FG=0.0,    # Foreground reconstruction loss scale
        do_disentangle=True  # Enable disentanglement
    )


    # TrainingConfig: Configure training parameters
    training_config = deepcor.config.TrainingConfig(
        n_epochs=100,
        batch_size=256,
        learning_rate=0.001,
        optimizer='adamw',
        betas=(0.9, 0.999),
        eps=1e-08,
        max_grad_norm=5.0,
        n_repetitions=20  # Number of ensemble repetitions
    )


    # DataConfig: Configure data preprocessing
    data_config = deepcor.config.DataConfig(
        n_dummy_scans=0,
        apply_censoring=False,
        censoring_threshold=0.5,
        confound_columns=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
    )

    # Create a complete configuration
    config = deepcor.config.DeepCorConfig(
        model=model_config,
        training=training_config,
        data=data_config
    )


    print(model_config);print('\n')
    print(training_config);print('\n')
    print(data_config);print('\n')
    print(config);print('\n')
    return (config,)


@app.cell
def _(os):
    # Define Data Paths
    # Cell Tagged parameters for papermill looping

    bids_path = '../Data/fMRI-Data/studyforrest-fmriprep/'

    subs = [sub for sub in os.listdir(os.path.join(bids_path)) if sub.startswith('sub-')]
    subs.sort()

    session = 'ses-localizer'
    task = 'objectcategories'
    space = 'MNI152NLin2009cAsym'

    s = 0
    r = 1
    analysis_name = 'test-advanced'
    return analysis_name, bids_path, r, s, session, space, subs, task


@app.cell
def _(analysis_name, r, s, subs):
    sub_id = subs[s]
    run = str(r)

    print(sub_id)
    print(run)
    print(analysis_name)
    return run, sub_id


@app.cell
def _(analysis_name, bids_path, os, run, session, space, sub_id, task):
    base = os.path.join(bids_path,sub_id,session)

    # EPI
    epi_path = os.path.join(base,'func',f'{sub_id}_{session}_task-{task}_run-{run}_bold_space-{space}_preproc.nii.gz')

    # Confounds
    confounds_path = os.path.join(base,'func',f'{sub_id}_{session}_task-{task}_run-{run}_bold_confounds.tsv')

    gm_mask_path = os.path.join(bids_path,'mask_roi.nii')
    cf_mask_path = os.path.join(bids_path,'mask_roni.nii')

    assert os.path.exists(epi_path), 'epi_path does not exist'
    assert os.path.exists(confounds_path), 'confounds_path does not exist'
    assert os.path.exists(gm_mask_path), 'gm_mask_path does not exist'
    assert os.path.exists(cf_mask_path), 'cf_mask_path does not exist'

    os.makedirs(os.path.join('../Data/DeepCor-Outputs',analysis_name), exist_ok=True)
    output_dir = os.path.join('../Data/DeepCor-Outputs',analysis_name,f'DeepCor-Forrest-{sub_id}-{task}-{run}-cvae_v1')

    print("EPI:", epi_path)
    print("Confounds:", confounds_path)
    print("output_dir:", output_dir)
    return cf_mask_path, confounds_path, epi_path, gm_mask_path, output_dir


@app.cell
def _(ants, cf_mask_path, confounds_path, deepcor, epi_path, gm_mask_path, pd):
    epi = ants.image_read(epi_path)
    df_conf = pd.read_csv(confounds_path, sep='\t') # Use tab separator
    gm = ants.image_read(gm_mask_path)
    cf = ants.image_read(cf_mask_path)

    # Apply Dummy Scans
    do_dummy = False
    if do_dummy:
        ndummy = 8
        epi, df_conf = deepcor.data.apply_dummy(epi, df_conf, ndummy=ndummy)

    do_frame_censoring = False
    if do_frame_censoring:
        idx_censor = df_conf['FramewiseDisplacement'].values>0.01
        epi, df_conf = deepcor.data.apply_frame_censoring(epi,df_conf,idx_censor,also_nearby_voxels=True)
    return cf, df_conf, epi, gm


@app.cell
def _(cf, deepcor, epi, gm):
    deepcor.data.plot_timeseries(epi,gm,cf)
    return


@app.cell
def _(cf, deepcor, epi, gm):
    obs_list, noi_list, gm2, cf2 = deepcor.data.get_obs_noi_list(epi, gm, cf)
    return noi_list, obs_list


@app.cell
def _():
    return


@app.cell
def _(df_conf):
    df_conf[['X','Y','Z','RotX','RotY','RotZ']]
    return


@app.cell
def _(config, deepcor, noi_list, obs_list, torch):
    # Create DataLoader 
    train_dataset = deepcor.data.TrainDataset(obs_list, noi_list)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True, 
        drop_last=True
    )
    return (train_loader,)


@app.cell
def _(deepcor, obs_list, torch):
    in_dim = obs_list.shape[-1]
    # Initialize Model
    # We pass the Prepared confounds tensor here
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = deepcor.models.CVAE_V1(in_channels=1,in_dim=in_dim,latent_dim=8)
    model = model.to(device)
    print("Model initialized and moved to device")
    return device, model


@app.cell
def _(config):
    config.training.n_epochs = 10
    return


@app.cell
def _(
    config,
    deepcor,
    device,
    fig,
    mo,
    model,
    os,
    output_dir,
    plt,
    train_loader,
):
    # Initialize Trainer
    trainer = deepcor.training.Trainer(
        model,
        device=device,
        optimizer_type=config.training.optimizer,
        lr=config.training.learning_rate,
        betas=config.training.betas,
        eps=config.training.eps,
        max_grad_norm=config.training.max_grad_norm
    )
    print("Trainer initialized")

    # Train the model
    print(f"Starting training for {config.training.n_epochs} epochs...")
    # We use a simple tracking dict for visualization

    track = deepcor.visualization.init_track('V1')

    # Training loop
    loss_history = []
    for epoch in range(config.training.n_epochs):
        # Train one epoch
        avg_loss = trainer.train_epoch(train_loader)
        loss_history.append(avg_loss)


        deepcor.visualization.update_track(track,train_loader,model)
        # Update tracking (using a batch from loader for viz)
        # with torch.no_grad():
        #     # Get a sample batch
        #     sample_batch = next(iter(train_loader))
        #     inputs_gm = sample_batch[0].to(device)
        #     inputs_cf = sample_batch[1].to(device)

        #     # Update track
        #     track = deepcor.visualization.update_track(track, model, inputs_gm, inputs_cf)




        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config.training.n_epochs}, Loss: {avg_loss:.4f}")

        deepcor.visualization.show_dahsboard_v1_marimo(track)
        mo.output.replace(fig)   # replace previous plot with the new one
        plt.close(fig)           # avoid duplicate/static matplotlib display

    # Save outputs
    print("Saving model and results...")
    # Save checkpoint
    trainer.save_checkpoint(
        os.path.join(output_dir, 'model_final.pt'), 
        config.training.n_epochs, 
        avg_loss
    )

    # Save track
    deepcor.visualization.save_track(os.path.join(output_dir, 'track.pickle'), track)
    return


@app.cell
def _(deepcor):
    deepcor.visualization.save_track
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
