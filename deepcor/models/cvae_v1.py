"""Original Conditional Variational Autoencoder (cVAE) model (v1).

This module implements the original cVAE architecture used in the
DeepCor paper, adapted to live inside the ``deepcor.models`` package.

The implementation follows the historical `cvae_v1.py` but is cleaned
up for style and integrated with the rest of the toolbox.
"""

from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cvae import compute_padding


class CVAE_V1(nn.Module):
    """Original cVAE model (version 1) without confound conditioning."""

    def __init__(
        self,
        in_channels: int,
        in_dim: int,
        latent_dim: int,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.in_dim = in_dim

        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 256]

        self.pad, self.final_size, self.pad_out = compute_padding(self.in_dim)

        # ---------------------------------------------------------------------
        # Encoder (z branch)
        # ---------------------------------------------------------------------
        modules_z: List[nn.Module] = []
        in_ch = in_channels
        for i, h_dim in enumerate(hidden_dims):
            modules_z.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_ch,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=int(self.pad[-i - 1]),
                    ),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_ch = h_dim

        self.encoder_z = nn.Sequential(*modules_z)
        self.fc_mu_z = nn.Linear(hidden_dims[-1] * int(self.final_size), latent_dim)
        self.fc_var_z = nn.Linear(hidden_dims[-1] * int(self.final_size), latent_dim)

        # ---------------------------------------------------------------------
        # Encoder (s branch)
        # ---------------------------------------------------------------------
        modules_s: List[nn.Module] = []
        in_ch = self.in_channels
        for i, h_dim in enumerate(hidden_dims):
            modules_s.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_ch,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=int(self.pad[-i - 1]),
                    ),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_ch = h_dim

        self.encoder_s = nn.Sequential(*modules_s)
        self.fc_mu_s = nn.Linear(hidden_dims[-1] * int(self.final_size), latent_dim)
        self.fc_var_s = nn.Linear(hidden_dims[-1] * int(self.final_size), latent_dim)

        # ---------------------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------------------
        self.decoder_input = nn.Linear(
            2 * latent_dim, hidden_dims[-1] * int(self.final_size)
        )

        hidden_dims_dec = list(hidden_dims)
        hidden_dims_dec.reverse()

        modules: List[nn.Module] = []
        for i in range(len(hidden_dims_dec) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        hidden_dims_dec[i],
                        hidden_dims_dec[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=int(self.pad_out[-4 + i]),
                        output_padding=int(self.pad_out[-4 + i]),
                    ),
                    nn.BatchNorm1d(hidden_dims_dec[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(
                hidden_dims_dec[-1],
                hidden_dims_dec[-1],
                kernel_size=3,
                stride=2,
                padding=int(self.pad_out[-1]),
                output_padding=int(self.pad_out[-1]),
            ),
            nn.BatchNorm1d(hidden_dims_dec[-1]),
            nn.LeakyReLU(),
            nn.Conv1d(
                hidden_dims_dec[-1],
                out_channels=1,
                kernel_size=3,
                padding=1,
            ),
        )

    # -------------------------------------------------------------------------
    # Core VAE operations
    # -------------------------------------------------------------------------
    def encode_z(self, input: Tensor) -> List[Tensor]:
        """Encode input to the z (signal) latent space."""
        result = self.encoder_z(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu_z(result)
        log_var = self.fc_var_z(result)
        return [mu, log_var]

    def encode_s(self, input: Tensor) -> List[Tensor]:
        """Encode input to the s (noise) latent space."""
        result = self.encoder_s(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu_s(result)
        log_var = self.fc_var_s(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """Decode concatenated latent codes back to time series."""
        result = self.decoder_input(z)
        result = result.view(-1, 256, int(self.final_size))
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick to sample from N(mu, var) using N(0, 1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    # -------------------------------------------------------------------------
    # Forward passes
    # -------------------------------------------------------------------------
    def forward_tg(self, input: Tensor) -> List[Tensor]:
        """Forward pass for target (ROI) data."""
        tg_mu_z, tg_log_var_z = self.encode_z(input)
        tg_mu_s, tg_log_var_s = self.encode_s(input)
        tg_z = self.reparameterize(tg_mu_z, tg_log_var_z)
        tg_s = self.reparameterize(tg_mu_s, tg_log_var_s)
        output = self.decode(torch.cat((tg_z, tg_s), dim=1))
        return [output, input, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s, tg_z, tg_s]

    def forward_bg(self, input: Tensor) -> List[Tensor]:
        """Forward pass for background (noise) data."""
        # NOTE: The toolbox training loop (`deepcor.training.Trainer`) expects
        # background latents to come from the "z" encoder, mirroring CVAE v2.
        bg_mu_z, bg_log_var_z = self.encode_z(input)
        bg_z = self.reparameterize(bg_mu_z, bg_log_var_z)
        zeros = torch.zeros_like(bg_z)
        output = self.decode(torch.cat((bg_z, zeros), dim=1))
        return [output, input, bg_mu_z, bg_log_var_z]

    def forward_fg(self, input: Tensor) -> List[Tensor]:
        """Forward pass for foreground (signal) data."""
        fg_mu_z, fg_log_var_z = self.encode_z(input)
        fg_z = self.reparameterize(fg_mu_z, fg_log_var_z)
        zeros = torch.zeros_like(fg_z)
        output = self.decode(torch.cat((fg_z, zeros), dim=1))
        return [output, input, fg_mu_z, fg_log_var_z]

    # -------------------------------------------------------------------------
    # Loss & sampling
    # -------------------------------------------------------------------------
    def loss_function(self, *args) -> dict:
        """Standard VAE reconstruction + KL loss."""
        beta = 1e-5

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

        del tg_z, tg_s  # Unused, kept for API parity

        recons_loss = F.mse_loss(recons_tg, input_tg)
        recons_loss = recons_loss + F.mse_loss(recons_bg, input_bg)

        kld_loss = 1 + tg_log_var_z - tg_mu_z**2 - tg_log_var_z.exp()
        kld_loss = kld_loss + 1 + tg_log_var_s - tg_mu_s**2 - tg_log_var_s.exp()
        kld_loss = kld_loss + 1 + bg_log_var_z - bg_mu_z**2 - bg_log_var_z.exp()
        kld_loss = torch.mean(-0.5 * torch.sum(kld_loss, dim=1), dim=0)

        loss = torch.mean(recons_loss + beta * kld_loss)
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": kld_loss.detach(),
        }

    def sample(self, num_samples: int, current_device: int) -> Tensor:
        """Sample from the latent prior and decode."""
        z = torch.randn(num_samples, self.latent_dim, device=current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor) -> Tensor:
        """Generate denoised output for input x (foreground branch)."""
        return self.forward_fg(x)[0]


# Backwards-compatible alias using the original class name
cVAE_V1 = CVAE_V1

