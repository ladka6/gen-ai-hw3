"""Bernoulli VAE for Task 2d.

The model now treats p(B | z) as a product of independent Bernoulli
distributions over binary pixel variables:

    p(B | z) = prod_{d=1}^{D} Bern(B_d | pi_d(z)).

Because the MNIST data are grayscale, we interpret each pixel value x_d in
[0, 1] as the probability that pixel d should be "on" under a distribution
p(B | x) = prod_d Bern(B_d | x_d).  Training then maximises

    -H(p(B | x), p(B | z)) - KL(q(Z | x) || p(Z))

where the first term is the *negative cross-entropy*

    -H(p(B | x), p(B | z)) = sum_d [ x_d log pi_d(z) + (1 - x_d) log(1 - pi_d(z)) ].

The decoder outputs raw logits (pre-sigmoid); the sigmoid is applied inside
the ELBO computation using `binary_cross_entropy_with_logits` for numerical
stability.  For image visualisation we use pi_d = sigmoid(logits) directly
(the posterior mean E[B_d | z] = pi_d), which produces sharper images than
sampling b ~ Bern(pi_d).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .model import ConvEncoder, GaussianParams


class BernoulliDecoder(nn.Module):
    """Maps a latent code z to per-pixel Bernoulli logits.

    Same upsample path as ConvDecoder; the single output head produces raw
    logits (no activation) so that `binary_cross_entropy_with_logits` can be
    applied in a numerically stable way.
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
        self.head_logits = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, z: Tensor) -> Tensor:
        """Return raw logits of shape (B, 1, H, W)."""
        h = self.project(z).view(-1, 128, 4, 4)
        h = self.upsample(h)
        return self.head_logits(h)


class BernoulliVAE(nn.Module):
    """VAE with a Bernoulli conditional p(B | z) (Task 2d)."""

    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = BernoulliDecoder(latent_dim)

    @staticmethod
    def reparameterise(q: GaussianParams) -> Tensor:
        eps = torch.randn_like(q.mean)
        return q.mean + q.std * eps

    def encode(self, x: Tensor) -> GaussianParams:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Return raw logits of shape (B, 1, H, W)."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[GaussianParams, Tensor, Tensor]:
        q = self.encode(x)
        z = self.reparameterise(q)
        logits = self.decode(z)
        return q, z, logits
