"""The ELBO objective for the Gaussian VAE.

For a single data-point ``x`` the ELBO is

    ELBO(x) = E_{q(Z|x)}[ log p(x | z) ] - KL( q(Z | x) || p(Z) ),

with a standard-normal prior ``p(Z) = N(0, I)``. We maximise the ELBO, i.e. we
minimise ``-ELBO``.

**Conditional log-likelihood (answer to Task 1b).** Because ``p(X | z)`` is a
product of independent Gaussians, one per feature ``d = 1 ... D``,

    log p(x | z) = sum_{d=1}^{D} log N(x_d | mu_d(z), sigma_d^2(z))
                 = sum_{d=1}^{D} -1/2 [ log(2*pi)
                                        + log sigma_d^2(z)
                                        + (x_d - mu_d(z))^2 / sigma_d^2(z) ].

**KL term.** For a diagonal-Gaussian posterior ``q(Z|x) = N(mu, diag(sigma^2))``
and standard-normal prior the KL has the closed form

    KL = 1/2 * sum_{k=1}^{K} [ sigma_k^2 + mu_k^2 - 1 - log sigma_k^2 ].

The expectation over ``q(Z|x)`` is approximated with a single Monte-Carlo
sample obtained through the reparameterisation trick (see :mod:`vae.model`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .model import GaussianParams

_LOG_2PI = math.log(2.0 * math.pi)


def gaussian_log_likelihood(x: Tensor, px: GaussianParams) -> Tensor:
    """Return ``log p(x | z)`` per data-point, summed over the ``D`` features."""
    var = torch.exp(px.logvar)
    log_prob = -0.5 * (_LOG_2PI + px.logvar + (x - px.mean) ** 2 / var)
    # Sum over all feature dimensions (channels x height x width); keep batch.
    return log_prob.flatten(start_dim=1).sum(dim=1)


def kl_to_standard_normal(q: GaussianParams) -> Tensor:
    """Return ``KL(q(Z|x) || N(0, I))`` per data-point, summed over ``K``."""
    var = torch.exp(q.logvar)
    kl = 0.5 * (var + q.mean**2 - 1.0 - q.logvar)
    return kl.sum(dim=1)


@dataclass
class ElboTerms:
    """The ELBO and its two summands, each averaged over a batch."""

    elbo: Tensor             # E_q[log p(x|z)] - KL, averaged over the batch
    log_likelihood: Tensor   # reconstruction term, averaged over the batch
    kl: Tensor               # KL term, averaged over the batch


def elbo(x: Tensor, q: GaussianParams, px: GaussianParams) -> ElboTerms:
    """Compute the (batch-averaged) ELBO from a single forward pass.

    ``q`` and ``px`` are the encoder/decoder outputs; ``px`` is evaluated at a
    reparameterised sample ``z ~ q(Z|x)``, so ``log p(x|z)`` is the 1-sample
    Monte-Carlo estimate of ``E_{q(Z|x)}[log p(x|z)]``.
    """
    log_likelihood = gaussian_log_likelihood(x, px)
    kl = kl_to_standard_normal(q)
    elbo_per_point = log_likelihood - kl
    return ElboTerms(
        elbo=elbo_per_point.mean(),
        log_likelihood=log_likelihood.mean(),
        kl=kl.mean(),
    )
