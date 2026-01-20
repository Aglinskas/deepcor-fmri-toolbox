"""High-level API for DeepCor fMRI denoising."""

import os
import numpy as np
import pandas as pd
import torch
import ants
from tqdm import tqdm

from .models import get_model
from .data import TrainDataset, get_obs_noi_list, get_obs_noi_list_coords, apply_dummy
from .training import Trainer, save_brain_signals
from .analysis import average_signal_ensemble, calc_and_save_compcor
from .config import DeepCorConfig, ModelConfig, TrainingConfig
from .utils import safe_mkdir


class DeepCorDenoiser:
    """
    High-level API for DeepCor fMRI denoising.

    This class provides a scikit-learn style interface for easy denoising
    of fMRI data.
    """

    def __init__(
        self,
        model_version='cvae',
        latent_dims=(8, 8),
        n_epochs=100,
        batch_size=1024,
        learning_rate=0.001,
        n_repetitions=20,
        config=None,
        device=None,
        verbose=True):
        """
        Initialize DeepCor denoiser.

        Args:
            model_version: Model version to use
            latent_dims: Tuple of (signal_dim, noise_dim)
            n_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            n_repetitions: Number of training repetitions for ensembling
            config: Optional DeepCorConfig object (overrides other args)
            device: PyTorch device (defaults to GPU if available)
        """
        if config is None:
            self.config = DeepCorConfig()
            self.config.model.latent_dims = latent_dims
            self.config.training.n_epochs = n_epochs
            self.config.training.batch_size = batch_size
            self.config.training.learning_rate = learning_rate
            self.config.training.n_repetitions = n_repetitions
        else:
            self.config = config

        self.model_version = model_version
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.trainers = []

        if verbose:
            print(f'device is {self.device}')

    def fit_denoise(
        self,
        epi_path,
        gm_mask_path,
        cf_mask_path,
        confounds_path,
        output_dir,
        verbose=True
    ):
        """
        Fit model and generate denoised output.

        Args:
            epi_path: Path to EPI image
            gm_mask_path: Path to gray matter mask
            cf_mask_path: Path to non-gray matter mask
            confounds_path: Path to confounds TSV file
            output_dir: Directory to save outputs
            verbose: Whether to show progress

        Returns:
            Path to denoised image
        """
        # Create output directory
        safe_mkdir(output_dir)

        # Load data
        if verbose:
            print("Loading data...")
        epi = ants.image_read(epi_path)
        gm = ants.image_read(gm_mask_path)
        cf = ants.image_read(cf_mask_path)
        df_conf = pd.read_csv(confounds_path, delimiter='\t')

        # Apply dummy scan removal
        epi, df_conf = apply_dummy(
            epi, df_conf, self.config.data.n_dummy_scans
        )

        # Prepare confounds
        use_cols = self.config.data.confound_columns
        conf = df_conf.loc[:, use_cols].values.transpose()
        conf[0:3, :] = (conf[0:3, :] - conf[0:3, :].min()) / (
            conf[0:3, :].max() - conf[0:3, :].min()
        )
        conf[3:, :] = (conf[3:, :] - conf[3:, :].min()) / (
            conf[3:, :].max() - conf[3:, :].min()
        )

        # Get observation and noise lists
        if verbose:
            print("Preparing training data...")
        model_version = str(self.model_version).lower()
        if model_version == "v1":
            # CVAE v1: old-style voxel lists without coordinate channels
            obs_list, noi_list, gm, cf = get_obs_noi_list(epi, gm, cf)

            # TrainDataset expects (N, C, T) tensors/arrays; v1 is (N, T)
            obs_list = obs_list[:, np.newaxis, :]  # (N, 1, T)
            noi_list = noi_list[:, np.newaxis, :]  # (N, 1, T)
            train_dataset = TrainDataset(obs_list, noi_list)
        else:
            # CVAE v2 / latest: coordinate-augmented lists (4 channels)
            obs_list_coords, noi_list_coords, gm, cf = get_obs_noi_list_coords(
                epi, gm, cf
            )
            train_dataset = TrainDataset(obs_list_coords, noi_list_coords)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True
        )

        # Prepare confounds for model
        nTR = epi.shape[-1]
        conf_batch = torch.tensor(
            np.array([conf for _ in range(self.config.training.batch_size)])
        ).to(self.device)

        # Train multiple repetitions
        signal_files = []
        iterator = tqdm(range(self.config.training.n_repetitions)) if verbose \
            else range(self.config.training.n_repetitions)

        for rep in iterator:
            if verbose:
                iterator.set_description(f"Training repetition {rep + 1}")

            try:
                # Initialize model (supports v1/v2/latest via registry)
                if model_version == "v1":
                    latent_dim_v1 = self.config.model.latent_dims
                    if isinstance(latent_dim_v1, (tuple, list)):
                        latent_dim_v1 = int(latent_dim_v1[0])
                    model = get_model(
                        "v1",
                        in_channels=1,
                        in_dim=nTR,
                        latent_dim=int(latent_dim_v1),
                        hidden_dims=self.config.model.hidden_dims,
                    )
                else:
                    model = get_model(
                        str(self.model_version).lower(),
                        conf=conf_batch,
                        in_channels=4,
                        in_dim=nTR,
                        latent_dim=self.config.model.latent_dims,
                        hidden_dims=self.config.model.hidden_dims,
                        beta=self.config.model.beta,
                        gamma=self.config.model.gamma,
                        delta=self.config.model.delta,
                        scale_MSE_GM=self.config.model.scale_MSE_GM,
                        scale_MSE_CF=self.config.model.scale_MSE_CF,
                        scale_MSE_FG=self.config.model.scale_MSE_FG,
                        do_disentangle=self.config.model.do_disentangle,
                    )

                # Initialize trainer
                trainer = Trainer(
                    model,
                    device=self.device,
                    optimizer_type=self.config.training.optimizer,
                    lr=self.config.training.learning_rate,
                    betas=self.config.training.betas,
                    eps=self.config.training.eps,
                    max_grad_norm=self.config.training.max_grad_norm
                )

                # Train
                trainer.fit(
                    train_loader,
                    n_epochs=self.config.training.n_epochs,
                    verbose=False
                )

                # Save model and outputs
                model_path = os.path.join(output_dir, f'model_rep_{rep}.pt')
                trainer.save_checkpoint(model_path, self.config.training.n_epochs, 0)

                signal_path = os.path.join(output_dir, f'signal_rep_{rep}.nii.gz')
                save_brain_signals(
                    model, train_dataset, epi, gm,
                    signal_path,
                    batch_size=512,
                    kind='FG'
                )
                signal_files.append(signal_path)

                self.models.append(model)
                self.trainers.append(trainer)

            except Exception as e:
                if verbose:
                    print(f"Error in repetition {rep}: {e}")
                continue

        # Ensemble averaging
        if verbose:
            print("Averaging ensemble predictions...")
        output_path = os.path.join(output_dir, 'denoised_deepcor.nii.gz')
        average_signal_ensemble(signal_files, output_path)

        # Save preprocessing versions
        if verbose:
            print("Saving comparison outputs...")
        from .data.loaders import array_to_brain
        if model_version == "v1":
            # v1 path: we built obs_list as (N, 1, T)
            preproc_arr = obs_list[:, 0, :]
        else:
            # v2/latest path: we built obs_list_coords as (N, 4, T)
            preproc_arr = obs_list_coords[:, 0, :]

        array_to_brain(
            preproc_arr,
            epi,
            gm,
            os.path.join(output_dir, "preproc.nii.gz"),
            inv_z_score=True,
        )

        calc_and_save_compcor(
            epi, gm, cf,
            os.path.join(output_dir, 'denoised_compcor.nii.gz'),
            n_components=5,
            return_img=False
        )

        if verbose:
            print(f"Denoising complete! Output saved to: {output_path}")

        return output_path

    def save(self, path):
        """Save trained models."""
        for i, trainer in enumerate(self.trainers):
            model_path = os.path.join(path, f'model_{i}.pt')
            trainer.save_checkpoint(model_path, 0, 0)

    def load(self, path):
        """Load trained models."""
        model_files = [f for f in os.listdir(path) if f.startswith('model_') and f.endswith('.pt')]
        for model_file in model_files:
            # Create new model and trainer
            # This is a placeholder - would need proper initialization
            pass
