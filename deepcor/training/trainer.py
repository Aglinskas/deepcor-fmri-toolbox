"""Training utilities for DeepCor models."""

import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import traceback


class Trainer:
    """
    Trainer class for DeepCor models.

    Handles the training loop, optimization, and checkpointing.
    """

    def __init__(
        self,
        model,
        device=None,
        optimizer_type='adamw',
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        max_grad_norm=5.0
    ):
        """
        Initialize trainer.

        Args:
            model: DeepCor model to train
            device: PyTorch device (defaults to GPU if available)
            optimizer_type: Type of optimizer ('adam' or 'adamw')
            lr: Learning rate
            betas: Adam/AdamW beta parameters
            eps: Adam/AdamW epsilon
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.device = device or torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        self.max_grad_norm = max_grad_norm

        # Initialize optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps
            )
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps
            )
        else:
            raise ValueError(
                f"Unknown optimizer type: {optimizer_type}. "
                "Use 'adam' or 'adamw'."
            )

    def train_epoch(self, dataloader):
        """
        Train for one epoch.

        Args:
            dataloader: PyTorch DataLoader

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        epoch_losses = []

        for inputs_gm, inputs_cf in dataloader:
            inputs_gm = inputs_gm.float().to(self.device)
            inputs_cf = inputs_cf.float().to(self.device)

            self.optimizer.zero_grad()

            # Forward passes
            outputs = self.model.forward_tg(inputs_gm)
            outputs_gm, _, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s, tg_z, tg_s = outputs

            outputs_bg = self.model.forward_bg(inputs_cf)
            outputs_cf, _, bg_mu_z, bg_log_var_z = outputs_bg

            # Compute loss
            loss = self.model.loss_function(
                outputs_gm, inputs_gm,
                tg_mu_z, tg_log_var_z,
                tg_mu_s, tg_log_var_s,
                tg_z, tg_s,
                outputs_cf, inputs_cf,
                bg_mu_z, bg_log_var_z
            )

            # Check for NaN
            if torch.isnan(loss['loss']):
                raise ValueError('Loss is NaN')

            # Backward pass
            loss['loss'].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_grad_norm
            )
            self.optimizer.step()

            epoch_losses.append(loss['loss'].item())

        return np.mean(epoch_losses)

    def fit(
        self,
        dataloader,
        n_epochs=100,
        callbacks=None,
        verbose=True
    ):
        """
        Train the model.

        Args:
            dataloader: PyTorch DataLoader
            n_epochs: Number of epochs to train
            callbacks: List of callback functions to call after each epoch
            verbose: Whether to show progress bar

        Returns:
            Training history dictionary
        """
        history = {'loss': []}
        iterator = tqdm(range(n_epochs)) if verbose else range(n_epochs)

        for epoch in iterator:
            try:
                avg_loss = self.train_epoch(dataloader)
                history['loss'].append(avg_loss)

                if verbose:
                    iterator.set_description(f"Loss: {avg_loss:.4f}")

                # Execute callbacks
                if callbacks is not None:
                    for callback in callbacks:
                        callback(epoch, self.model, history)

            except Exception as e:
                print(f"Error in epoch {epoch}: {e}")
                traceback.print_exc()
                break

        return history

    def save_checkpoint(self, filepath, epoch, loss):
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            loss: Current loss
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
        }, filepath)

    def load_checkpoint(self, filepath):
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Dictionary with epoch and loss information
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return {
            'epoch': checkpoint['epoch'],
            'loss': checkpoint['loss']
        }


def save_model(model_ofn, model, optimizer, epoch, loss):
    """
    Save model checkpoint (legacy function for backward compatibility).

    Args:
        model_ofn: Output filename
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }, model_ofn)


def save_brain_signals(
    model,
    train_inputs_coords,
    epi,
    gm,
    ofn,
    batch_size=512,
    kind='FG',
    inv_z_score=True
):
    """
    Generate and save brain signals from trained model.

    Args:
        model: Trained model
        train_inputs_coords: PyTorch Dataset
        epi: Original EPI image
        gm: Gray matter mask
        ofn: Output filename
        batch_size: Batch size for inference
        kind: Signal type ('FG', 'TG', or 'BG')
        inv_z_score: Whether to invert z-scoring
    """
    import ants
    import numpy as np

    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    nTR = epi.shape[-1]
    epi_flat = epi.numpy().reshape(-1, nTR)
    gm_flat = gm.numpy().flatten()

    train_in_coords = torch.utils.data.DataLoader(
        train_inputs_coords,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )

    brain_signals = []
    for inputs_gm, inputs_cf in train_in_coords:
        inputs_gm = inputs_gm.float().to(device)
        if kind == 'FG':
            fg_signal = model.forward_fg(inputs_gm)[0].detach().cpu().numpy()[:, 0, :]
        elif kind == 'TG':
            fg_signal = model.forward_tg(inputs_gm)[0].detach().cpu().numpy()[:, 0, :]
        elif kind == 'BG':
            fg_signal = model.forward_bg(inputs_gm)[0].detach().cpu().numpy()[:, 0, :]
        else:
            raise ValueError(f'{kind}: not implemented')

        brain_signals.append(fg_signal)

    # Reconstruct the full brain array
    brain_signals_arr = np.zeros(epi_flat.shape)
    brain_signals = np.vstack(brain_signals)

    assert brain_signals.shape[0] == gm_flat.sum(), \
        f'mismatch in voxel sizes: {brain_signals.shape[0]}/{gm_flat.sum()}'

    if inv_z_score:
        epi_mean = epi_flat[gm_flat == 1, :].mean(axis=1)
        epi_std = epi_flat[gm_flat == 1, :].std(axis=1)
        brain_signals = (
            brain_signals * epi_std[:, np.newaxis] +
            epi_mean[:, np.newaxis]
        )

    valid_voxels = gm_flat == 1
    brain_signals_arr[valid_voxels, :] = brain_signals
    brain_signals_arr = brain_signals_arr.reshape(epi.shape)
    brain_signals_img = epi.new_image_like(brain_signals_arr)
    brain_signals_img.to_filename(ofn)
