"""Conditional Variational Autoencoder (cVAE) model for fMRI denoising."""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch.autograd import Function
from typing import List, Tuple

from .base import BaseModel


class GradientReversalFunction(Function):
    """Gradient Reversal Layer function for adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_
        grad_input = grad_output.neg() * lambda_
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer module."""

    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


def compute_in(x):
    """Helper function for padding computation."""
    return (x - 3) / 2 + 1


def compute_in_size(x):
    """Compute input size after multiple convolutions."""
    for i in range(4):
        x = compute_in(x)
    return x


def compute_out_size(x):
    """Compute output size after deconvolutions."""
    return ((((x * 2 + 1) * 2 + 1) * 2 + 1) * 2 + 1)


def compute_padding(x):
    """
    Compute padding values for encoder and decoder.

    Args:
        x: Input temporal dimension

    Returns:
        Tuple of (padding string, final size, padding output string)
    """
    rounding = np.ceil(compute_in_size(x)) - compute_in_size(x)
    y = ((((rounding * 2) * 2) * 2) * 2)
    pad = bin(int(y)).replace('0b', '')
    if len(pad) < 4:
        for i in range(4 - len(pad)):
            pad = '0' + pad
    final_size = compute_in_size(x + y)
    pad_out = bin(int(compute_out_size(final_size) - x)).replace('0b', '')
    if len(pad_out) < 4:
        for i in range(4 - len(pad_out)):
            pad_out = '0' + pad_out
    return pad, final_size, pad_out


class CVAE(BaseModel):
    """
    Conditional Variational Autoencoder for fMRI denoising.

    This model learns to separate signal from noise in fMRI data using
    a disentangled latent representation with two components:
    - z: Signal-related latent variables
    - s: Noise-related latent variables
    """

    def __init__(
        self,
        conf: Tensor,
        in_channels: int,
        in_dim: int,
        latent_dim: Tuple[int, int],
        hidden_dims: List[int] = None,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 1.0,
        scale_MSE_GM: float = 1.0,
        scale_MSE_CF: float = 1.0,
        scale_MSE_FG: float = 1.0,
        do_disentangle: bool = True,
        freq_exp: float = 1.0,
        freq_scale: float = 1.0
    ):
        """
        Initialize cVAE model.

        Args:
            conf: Confound variables tensor
            in_channels: Number of input channels
            in_dim: Input temporal dimension
            latent_dim: Tuple of (signal_dim, noise_dim) for latent space
            hidden_dims: List of hidden dimensions for encoder/decoder
            beta: KL divergence loss weight
            gamma: Total correlation loss weight
            delta: Disentanglement loss weight
            scale_MSE_GM: Gray matter reconstruction loss scale
            scale_MSE_CF: Non-gray matter reconstruction loss scale
            scale_MSE_FG: Foreground reconstruction loss scale
            do_disentangle: Whether to apply disentanglement
            freq_exp: Frequency exponent parameter
            freq_scale: Frequency scale parameter
        """
        super(CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.latent_dim_z = self.latent_dim[0]
        self.latent_dim_s = self.latent_dim[1]
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.scale_MSE_GM = scale_MSE_GM
        self.scale_MSE_CF = scale_MSE_CF
        self.do_disentangle = do_disentangle
        self.freq_exp = freq_exp
        self.freq_scale = freq_scale
        self.scale_MSE_FG = scale_MSE_FG
        self.confounds = conf.float()
        self.grl = GradientReversalLayer(lambda_=1.0)

        nTR = in_dim

        # Confound decoder for z
        self.decoder_confounds_z = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=latent_dim[0],
                out_channels=128,
                kernel_size=nTR,
                stride=1,
                bias=False
            ),
            nn.Conv1d(128, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(16, 6, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

        # Confound decoder for s
        self.decoder_confounds_s = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=latent_dim[1],
                out_channels=128,
                kernel_size=nTR,
                stride=1,
                bias=False
            ),
            nn.Conv1d(128, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(16, 6, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

        # Set default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 256]

        self.pad, self.final_size, self.pad_out = compute_padding(self.in_dim)

        # Build encoder for z (signal)
        modules_z = []
        for i in range(len(hidden_dims)):
            h_dim = hidden_dims[i]
            modules_z.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=int(self.pad[-i - 1])
                    ),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder_z = nn.Sequential(*modules_z)
        self.fc_mu_z = nn.Linear(
            hidden_dims[-1] * int(self.final_size),
            self.latent_dim_z
        )
        self.fc_var_z = nn.Linear(
            hidden_dims[-1] * int(self.final_size),
            self.latent_dim_z
        )

        # Build encoder for s (noise)
        modules_s = []
        in_channels = self.in_channels
        for i in range(len(hidden_dims)):
            h_dim = hidden_dims[i]
            modules_s.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=int(self.pad[-i - 1])
                    ),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder_s = nn.Sequential(*modules_s)
        self.fc_mu_s = nn.Linear(
            hidden_dims[-1] * int(self.final_size),
            self.latent_dim_s
        )
        self.fc_var_s = nn.Linear(
            hidden_dims[-1] * int(self.final_size),
            self.latent_dim_s
        )

        # Build decoder
        modules = []
        self.decoder_input = nn.Linear(
            self.latent_dim_s + self.latent_dim_z,
            hidden_dims[-1] * int(self.final_size)
        )

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=int(self.pad_out[-4 + i]),
                        output_padding=int(self.pad_out[-4 + i])
                    ),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=int(self.pad_out[-1]),
                output_padding=int(self.pad_out[-1])
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                hidden_dims[-1],
                out_channels=self.in_channels,
                kernel_size=3,
                padding=1
            )
        )

    def encode_z(self, input: Tensor) -> List[Tensor]:
        """
        Encode input to signal latent space.

        Args:
            input: Input tensor [N x C x H x W]

        Returns:
            List of [mu, log_var] for signal latent codes
        """
        result = self.encoder_z(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu_z(result)
        log_var = self.fc_var_z(result)
        return [mu, log_var]

    def encode_s(self, input: Tensor) -> List[Tensor]:
        """
        Encode input to noise latent space.

        Args:
            input: Input tensor [N x C x H x W]

        Returns:
            List of [mu, log_var] for noise latent codes
        """
        result = self.encoder_s(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu_s(result)
        log_var = self.fc_var_s(result)
        return [mu, log_var]

    def encode(self, input: Tensor) -> List[Tensor]:
        """Encode input to both signal and noise latent spaces."""
        mu_z, log_var_z = self.encode_z(input)
        mu_s, log_var_s = self.encode_s(input)
        return [mu_z, log_var_z, mu_s, log_var_s]

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent codes to output space.

        Args:
            z: Concatenated latent codes [B x D]

        Returns:
            Reconstructed output [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 256, int(self.final_size))
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward_tg(self, input: Tensor) -> List[Tensor]:
        """Forward pass for target (ROI) data."""
        tg_mu_z, tg_log_var_z = self.encode_z(input)
        tg_mu_s, tg_log_var_s = self.encode_s(input)
        tg_z = self.reparameterize(tg_mu_z, tg_log_var_z)
        tg_s = self.reparameterize(tg_mu_s, tg_log_var_s)
        output = self.forward_bg(input)[0] + self.forward_fg(input)[0]
        return [output, input, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s, tg_z, tg_s]

    def forward_fg(self, input: Tensor) -> List[Tensor]:
        """Forward pass for foreground (signal) extraction."""
        tg_mu_s, tg_log_var_s = self.encode_s(input)
        tg_s = self.reparameterize(tg_mu_s, tg_log_var_s)
        zeros = torch.zeros(tg_s.shape[0], self.latent_dim_z)
        zeros = zeros.to(self.device)
        output = self.decode(torch.cat((zeros, tg_s), 1))
        return [output, input, tg_mu_s, tg_log_var_s]

    def forward_bg(self, input: Tensor) -> List[Tensor]:
        """Forward pass for background (noise) extraction."""
        bg_mu_z, bg_log_var_z = self.encode_z(input)
        bg_z = self.reparameterize(bg_mu_z, bg_log_var_z)
        zeros = torch.zeros(bg_z.shape[0], self.latent_dim_s)
        zeros = zeros.to(self.device)
        output = self.decode(torch.cat((bg_z, zeros), 1))
        return [output, input, bg_mu_z, bg_log_var_z]

    def forward(self, input: Tensor) -> List[Tensor]:
        """Standard forward pass."""
        return self.forward_tg(input)

    def ncc(self, x, y, eps=1e-8):
        """
        Compute Normalized Cross-Correlation loss.

        Args:
            x: First tensor
            y: Second tensor
            eps: Small value for numerical stability

        Returns:
            NCC loss (1 - NCC to minimize)
        """
        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)

        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_std = x.std(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        ncc = (x - x_mean) * (y - y_mean) / (x_std * y_std + eps)
        ncc = ncc.mean(dim=1)

        return 1 - ncc

    def loss_function(self, *args) -> dict:
        """
        Compute VAE loss function with disentanglement.

        Args:
            args: List of tensors from forward passes

        Returns:
            Dictionary of loss components
        """
        beta = self.beta
        gamma = self.gamma
        delta = self.delta

        recons_tg = args[0]
        input_tg = args[1]
        tg_mu_z = args[2]
        tg_log_var_z = args[3]
        tg_mu_s = args[4]
        tg_log_var_s = args[5]
        tg_z = args[6]
        tg_s = args[7]
        recons_bg = args[8]
        input_bg = args[9]
        bg_mu_z = args[10]
        bg_log_var_z = args[11]

        recons_fg = self.forward_fg(input_tg)[0]

        # Reconstruction losses
        recons_loss_roi = F.mse_loss(
            recons_tg[:, 0, :],
            input_tg[:, 0, :]
        ) * self.scale_MSE_GM
        recons_loss_roni = F.mse_loss(
            recons_bg[:, 0, :],
            input_bg[:, 0, :]
        ) * self.scale_MSE_CF
        recons_loss = recons_loss_roi + recons_loss_roni

        # Confound prediction
        confounds_pred_z = self.decoder_confounds_z(torch.unsqueeze(tg_z, 2))
        confounds_pred_s = self.decoder_confounds_s(torch.unsqueeze(tg_s, 2))

        loss_recon_conf_s = self.grl(
            F.mse_loss(confounds_pred_s, self.confounds)
        ) * 1e2
        loss_recon_conf_z = F.mse_loss(
            confounds_pred_z,
            self.confounds
        ) * 1e2

        # NCC losses
        ncc_loss_tg = self.ncc(input_tg, recons_tg).mean() * 1
        ncc_loss_bg = self.ncc(input_bg, recons_bg).mean() * 1

        ncc_loss_conf_s = 0
        for i in range(self.confounds.shape[1]):
            ncc_loss_conf_s += self.ncc(
                self.confounds[:, i, :],
                confounds_pred_s[:, i, :]
            ).mean() * 1e1
        ncc_loss_conf_s = self.grl(ncc_loss_conf_s)

        ncc_loss_conf_z = 0
        for i in range(self.confounds.shape[1]):
            ncc_loss_conf_z += self.ncc(
                self.confounds[:, i, :],
                confounds_pred_z[:, i, :]
            ).mean() * 1e1

        # Foreground-background orthogonality
        recond_bg = self.forward_bg(input_tg)[0]
        fg_bg_ncc = self.ncc(recond_bg[:, 0, :], recons_fg[:, 0, :]).mean()
        recons_loss_fg = F.mse_loss(
            torch.zeros_like(fg_bg_ncc),
            1 - fg_bg_ncc
        ) * 1e4

        recons_loss += F.mse_loss(recons_fg[:, 0, :], input_tg[:, 0, :]) * .0001
        recons_loss += F.mse_loss(recons_bg[:, 0, :], input_bg[:, 0, :]) * .00001

        # Smoothness loss
        smoothness_loss = torch.mean(
            (recons_fg[:, 0, :][:, 1:] - recons_fg[:, 0, :][:, :-1]) ** 2
        )
        smoothness_loss += torch.mean(
            (recond_bg[:, 0, :][:, 1:] - recond_bg[:, 0, :][:, :-1]) ** 2
        )
        smoothness_loss *= .01

        # KL divergence
        kld_loss = torch.mean(
            -0.5 * torch.sum(
                1 + tg_log_var_z - tg_mu_z ** 2 - tg_log_var_z.exp(),
                dim=1
            ),
            dim=0
        )
        kld_loss += torch.mean(
            -0.5 * torch.sum(
                1 + tg_log_var_s - tg_mu_s ** 2 - tg_log_var_s.exp(),
                dim=1
            ),
            dim=0
        )
        kld_loss += torch.mean(
            -0.5 * torch.sum(
                1 + bg_log_var_z - bg_mu_z ** 2 - bg_log_var_z.exp(),
                dim=1
            ),
            dim=0
        )
        kld_loss = kld_loss / 3
        kld_loss = kld_loss * beta

        # Total loss
        loss = torch.sum(
            recons_loss + kld_loss + recons_loss_fg + ncc_loss_tg +
            ncc_loss_bg + loss_recon_conf_s + loss_recon_conf_z +
            recons_loss_fg + smoothness_loss + ncc_loss_conf_z +
            ncc_loss_conf_s
        )

        return {
            'loss': loss,
            'kld_loss': kld_loss,
            'recons_loss_roi': recons_loss_roi,
            'recons_loss_roni': recons_loss_roni,
            'loss_recon_conf_s': loss_recon_conf_s,
            'loss_recon_conf_z': loss_recon_conf_z,
            'ncc_loss_tg': ncc_loss_tg,
            'ncc_loss_bg': ncc_loss_bg,
            'ncc_loss_conf_s': ncc_loss_conf_s,
            'ncc_loss_conf_z': ncc_loss_conf_z,
            'smoothness_loss': smoothness_loss,
            'recons_loss_fg': recons_loss_fg
        }

    def generate(self, x: Tensor) -> Tensor:
        """
        Generate denoised output from input.

        Args:
            x: Input tensor

        Returns:
            Denoised tensor (foreground signal only)
        """
        return self.forward_fg(x)[0]

    def _compute_log_density_gaussian(self, z, mu, log_var):
        """Compute log density of a Gaussian for each sample in the batch."""
        normalization = -0.5 * (math.log(2 * math.pi) + log_var)
        log_prob = normalization - 0.5 * ((z - mu) ** 2 / log_var.exp())
        return log_prob.sum(dim=1)


# Alias for backward compatibility
cVAE = CVAE
