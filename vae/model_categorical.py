"""Categorical VAE for Task 2c.

Pixel values are discretised into k = 4 equal-width bins:

    bin 0: [0.00, 0.25)   midpoint 0.125
    bin 1: [0.25, 0.50)   midpoint 0.375
    bin 2: [0.50, 0.75)   midpoint 0.625
    bin 3: [0.75, 1.00]   midpoint 0.875

The decoder produces k un-normalised logits per pixel position; log_softmax
over the k-channel dimension gives log Cat(X_d | pi_d(z)).

We chose k = 4 because MNIST pixels are concentrated near 0 and 1 with a
roughly bimodal distribution; four bins capture this bimodality while keeping
the output head small (4 channels instead of, e.g., 256).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .model import ConvEncoder, GaussianParams

NUM_BINS: int = 4  # k in the assignment


def discretize(x: Tensor, k: int = NUM_BINS) -> Tensor:
    """Map continuous pixel values in [0, 1] to integer bin indices in [0, k-1]."""
    return (x * k).long().clamp(0, k - 1)


def bin_midpoints(k: int, device: torch.device) -> Tensor:
    """Return the midpoint value of each bin as a float tensor of shape (k,)."""
    return (torch.arange(k, device=device).float() + 0.5) / k


class CatDecoder(nn.Module):
    """Maps a latent code z to per-pixel logits over k categories.

    The upsample path is identical to ConvDecoder from Task 1; only the output
    head changes: instead of two 1-channel heads we produce one k-channel head
    whose output is fed through log_softmax (in the ELBO, not here).
    """

    def __init__(self, latent_dim: int, k: int = NUM_BINS) -> None:
        super().__init__()
        self.k = k
        self.project = nn.Linear(latent_dim, 128 * 4 * 4)
        self.upsample = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),  # 4 -> 7
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),   # 14 -> 28
            nn.ReLU(inplace=True),
        )
        # k output channels: one logit per bin per pixel
        self.head_logits = nn.Conv2d(32, k, kernel_size=1)

    def forward(self, z: Tensor) -> Tensor:
        """Return raw logits of shape (B, k, H, W)."""
        h = self.project(z).view(-1, 128, 4, 4)
        h = self.upsample(h)
        return self.head_logits(h)


class CatVAE(nn.Module):
    """VAE with a Categorical conditional p(X | z) (Task 2c)."""

    def __init__(self, latent_dim: int = 16, k: int = NUM_BINS) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.k = k
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = CatDecoder(latent_dim, k)

    @staticmethod
    def reparameterise(q: GaussianParams) -> Tensor:
        eps = torch.randn_like(q.mean)
        return q.mean + q.std * eps

    def encode(self, x: Tensor) -> GaussianParams:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Return logits of shape (B, k, H, W)."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[GaussianParams, Tensor, Tensor]:
        q = self.encode(x)
        z = self.reparameterise(q)
        logits = self.decode(z)
        return q, z, logits
