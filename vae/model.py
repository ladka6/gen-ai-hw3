"""The (de)convolutional Gaussian VAE for Task 1.

The model has three parts:

* an **encoder** ``q(Z | x) = N(mu_enc(x), diag(sigma_enc^2(x)))`` built from
  convolutions, mapping a 28x28 image to a ``K``-dimensional latent space;
* the **reparameterisation trick** to draw ``z ~ q(Z | x)`` differentiably;
* a **decoder** that, evaluated at ``z``, outputs the parameters ``mu(z)`` and
  ``sigma^2(z)`` of the conditional ``p(X | z)``, which for Task 1 is a product
  of independent Gaussians (one per pixel).

The decoder produces, per pixel ``d``, a mean ``mu_d(z)`` and a (log-)variance
``sigma_d^2(z)`` as required by

    p(X | z) = prod_{d=1}^{D} N(X_d | mu_d(z), sigma_d^2(z)).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

# Bounds for the predicted log-variances. Clamping keeps the Gaussian
# log-likelihood numerically stable (it contains 1 / sigma^2 and log sigma^2).
_MIN_LOGVAR = -6.0
_MAX_LOGVAR = 2.0


@dataclass
class GaussianParams:
    """Parameters of a diagonal Gaussian, used for both q(Z|x) and p(X|z)."""

    mean: Tensor
    logvar: Tensor

    @property
    def std(self) -> Tensor:
        return torch.exp(0.5 * self.logvar)


class ConvEncoder(nn.Module):
    """Maps an image to the parameters of ``q(Z | x)``.

    Three stride-2 convolutions reduce 28x28 -> 14x14 -> 7x7 -> 4x4, after
    which a linear layer outputs the ``K`` means and ``K`` log-variances.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 7 -> 4
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x: Tensor) -> GaussianParams:
        h = self.flatten(self.features(x))
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h).clamp(_MIN_LOGVAR, _MAX_LOGVAR)
        return GaussianParams(mean=mean, logvar=logvar)


class ConvDecoder(nn.Module):
    """Maps a latent code ``z`` to the parameters of ``p(X | z)``.

    A linear layer reshapes ``z`` into a 128x4x4 feature map, which transposed
    convolutions upsample back to 28x28. Two separate output heads produce the
    per-pixel mean and log-variance of the Gaussian likelihood.
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
        # Separate 1x1 convolutions for the mean and the log-variance heads.
        self.head_mean = nn.Conv2d(32, 1, kernel_size=1)
        self.head_logvar = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, z: Tensor) -> GaussianParams:
        h = self.project(z).view(-1, 128, 4, 4)
        h = self.upsample(h)
        # The data lies in [0, 1], so we squash the predicted mean with a
        # sigmoid; the log-variance is left unconstrained but clamped.
        mean = torch.sigmoid(self.head_mean(h))
        logvar = self.head_logvar(h).clamp(_MIN_LOGVAR, _MAX_LOGVAR)
        return GaussianParams(mean=mean, logvar=logvar)


class GaussianVAE(nn.Module):
    """The full VAE with a Gaussian conditional ``p(X | z)`` (Task 1)."""

    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    @staticmethod
    def reparameterise(q: GaussianParams) -> Tensor:
        """Sample ``z ~ q(Z | x)`` via ``z = mu + sigma * eps``, eps ~ N(0, I)."""
        eps = torch.randn_like(q.mean)
        return q.mean + q.std * eps

    def encode(self, x: Tensor) -> GaussianParams:
        return self.encoder(x)

    def decode(self, z: Tensor) -> GaussianParams:
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[GaussianParams, Tensor, GaussianParams]:
        """Run a full encode -> sample -> decode pass.

        Returns the approximate posterior ``q(Z | x)``, the sampled latent
        ``z``, and the likelihood parameters ``p(X | z)``.
        """
        q = self.encode(x)
        z = self.reparameterise(q)
        px = self.decode(z)
        return q, z, px
