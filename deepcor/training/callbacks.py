"""Callback functions for training monitoring."""

import os
from datetime import datetime


class TrackingCallback:
    """Callback for tracking training metrics."""

    def __init__(self, track_dict, update_interval=10):
        """
        Initialize tracking callback.

        Args:
            track_dict: Dictionary to store tracking data
            update_interval: Interval (in epochs) to update tracking
        """
        self.track = track_dict
        self.update_interval = update_interval
        self.track['T_start'] = datetime.now()

    def __call__(self, epoch, model, history):
        """
        Called after each epoch.

        Args:
            epoch: Current epoch number
            model: Current model
            history: Training history dictionary
        """
        if epoch % self.update_interval == 0:
            self.track['epoch'] = epoch
            self.track['l'].append(history['loss'][-1])


class CheckpointCallback:
    """Callback for saving model checkpoints."""

    def __init__(self, checkpoint_dir, save_interval=10):
        """
        Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_interval: Interval (in epochs) to save checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        os.makedirs(checkpoint_dir, exist_ok=True)

    def __call__(self, epoch, model, history):
        """
        Called after each epoch.

        Args:
            epoch: Current epoch number
            model: Current model
            history: Training history dictionary
        """
        if (epoch + 1) % self.save_interval == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            import torch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': history['loss'][-1],
            }, checkpoint_path)


class EarlyStoppingCallback:
    """Callback for early stopping based on loss."""

    def __init__(self, patience=10, min_delta=1e-4):
        """
        Initialize early stopping callback.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, epoch, model, history):
        """
        Called after each epoch.

        Args:
            epoch: Current epoch number
            model: Current model
            history: Training history dictionary

        Raises:
            StopIteration: If early stopping criterion is met
        """
        current_loss = history['loss'][-1]

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                raise StopIteration
