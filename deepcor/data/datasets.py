"""Dataset classes for fMRI data."""

import torch
import numpy as np
from numpy import random


class TrainDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for training fMRI denoising models.

    This dataset provides pairs of observations (ROI) and noise (RONI) samples
    with data augmentation.
    """

    def __init__(self, X, Y):
        """
        Initialize dataset.

        Args:
            X: Observation data (ROI voxels)
            Y: Noise data (RONI voxels)
        """
        self.obs = X
        self.noi = Y

    def __len__(self):
        """Return the minimum size of observations and noise."""
        return min(self.obs.shape[0], self.noi.shape[0])

    def __getitem__(self, index):
        """
        Get a training sample.

        Args:
            index: Sample index

        Returns:
            Tuple of (observation, augmented_noise)
        """
        observation = self.obs[index]
        noise = self.noi[index]

        # Data augmentation: scale noise with random beta distribution
        s = 2 * random.beta(4, 4, 1)
        noise_aug = s * noise

        return observation, noise_aug
