"""Beta VAE for Task 2b.

The decoder now outputs the parameters alpha(z) and beta(z) of a product of
independent Beta distributions:

    p(X | z) = prod_{d=1}^{D} Beta(X_d | alpha_d(z), beta_d(z)).

Both alpha and beta must be strictly positive; we enforce this with a softplus
activation.  Because Beta(x) is only defined for x in (0, 1) we clamp pixel
values away from the boundary in the log-likelihood (see vae/elbo.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import ConvEncoder, GaussianParams


@dataclass
class BetaParams:
    """Parameters of a product of Beta distributions (one per pixel)."""

    alpha: Tensor  # shape (B, 1, H, W), all > 0
    beta: Tensor   # shape (B, 1, H, W), all > 0


class BetaDecoder(nn.Module):
    """Maps a latent code z to the parameters (alpha, beta) of p(X | z).

    Architecture mirrors ConvDecoder from Task 1; the only change is that the
    two output heads produce alpha and beta instead of mean and log-variance.
    Softplus ensures both parameters are strictly positive.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
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
        self.head_alpha = nn.Conv2d(32, 1, kernel_size=1)
        self.head_beta = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, z: Tensor) -> BetaParams:
        h = self.project(z).view(-1, 128, 4, 4)
        h = self.upsample(h)
        # softplus maps R -> (0, inf); add a tiny floor for numerical safety.
        alpha = F.softplus(self.head_alpha(h)) + 1e-6
        beta = F.softplus(self.head_beta(h)) + 1e-6
        return BetaParams(alpha=alpha, beta=beta)


class BetaVAE(nn.Module):
    """VAE with a Beta conditional p(X | z) (Task 2b)."""

    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = BetaDecoder(latent_dim)

    @staticmethod
    def reparameterise(q: GaussianParams) -> Tensor:
        eps = torch.randn_like(q.mean)
        return q.mean + q.std * eps

    def encode(self, x: Tensor) -> GaussianParams:
        return self.encoder(x)

    def decode(self, z: Tensor) -> BetaParams:
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[GaussianParams, Tensor, BetaParams]:
        q = self.encode(x)
        z = self.reparameterise(q)
        px = self.decode(z)
        return q, z, px
