"""Base model class for DeepCor models."""

import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import List, Tuple
from torch import Tensor


class BaseModel(nn.Module, ABC):
    """Abstract base class for DeepCor models."""

    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def encode(self, input: Tensor) -> List[Tensor]:
        """Encode input to latent space."""
        pass

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """Decode latent codes to output space."""
        pass

    @abstractmethod
    def forward(self, input: Tensor) -> List[Tensor]:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def loss_function(self, *args, **kwargs) -> dict:
        """Calculate loss function."""
        pass

    @abstractmethod
    def generate(self, x: Tensor) -> Tensor:
        """Generate denoised output."""
        pass

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu: Mean of the latent Gaussian [B x D]
            logvar: Log variance of the latent Gaussian [B x D]

        Returns:
            Sampled latent codes [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
