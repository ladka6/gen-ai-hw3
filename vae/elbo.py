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
import torch.nn.functional as F
from torch import Tensor

from .model import GaussianParams
from .model_beta import BetaParams

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


# ---------------------------------------------------------------------------
# Task 2b -- Beta log-likelihood and ELBO
# ---------------------------------------------------------------------------

def beta_log_likelihood(x: Tensor, px: BetaParams, eps: float = 1e-6) -> Tensor:
    """Return log p(x | z) per data-point, summed over D features.

    log Beta(x_d | alpha_d, beta_d)
        = (alpha_d - 1) log x_d + (beta_d - 1) log(1 - x_d)
          - lgamma(alpha_d) - lgamma(beta_d) + lgamma(alpha_d + beta_d)

    Pixel values are clamped to (eps, 1-eps) to avoid log(0).
    """
    x_safe = x.clamp(eps, 1.0 - eps)
    alpha, beta = px.alpha, px.beta
    log_prob = ((alpha - 1.0) * torch.log(x_safe)
                + (beta - 1.0) * torch.log(1.0 - x_safe))
    log_beta_fn = (torch.lgamma(alpha) + torch.lgamma(beta)
                   - torch.lgamma(alpha + beta))
    return (log_prob - log_beta_fn).flatten(start_dim=1).sum(dim=1)


def elbo_beta(x: Tensor, q: GaussianParams, px: BetaParams) -> ElboTerms:
    """Batch-averaged ELBO for the Beta VAE (Task 2b)."""
    log_likelihood = beta_log_likelihood(x, px)
    kl = kl_to_standard_normal(q)
    elbo_per_point = log_likelihood - kl
    return ElboTerms(
        elbo=elbo_per_point.mean(),
        log_likelihood=log_likelihood.mean(),
        kl=kl.mean(),
    )


# ---------------------------------------------------------------------------
# Task 2c -- Categorical log-likelihood and ELBO
# ---------------------------------------------------------------------------

def cat_log_likelihood(x: Tensor, logits: Tensor, k: int) -> Tensor:
    """Return log p(x | z) per data-point, summed over D features.

    Each pixel is assigned to one of k equal-width bins; the log-likelihood is
    the log-softmax probability of the correct bin.

    Args:
        x:      Continuous pixel values in [0, 1], shape (B, 1, H, W).
        logits: Raw decoder logits, shape (B, k, H, W).
        k:      Number of bins.
    """
    x_bins = (x * k).long().clamp(0, k - 1).squeeze(1)  # (B, H, W)
    log_probs = F.log_softmax(logits, dim=1)              # (B, k, H, W)
    # nll_loss expects (B, k, H, W) log-probs and (B, H, W) targets.
    nll = F.nll_loss(log_probs, x_bins, reduction="none")  # (B, H, W)
    return -nll.flatten(start_dim=1).sum(dim=1)            # (B,)


def elbo_cat(x: Tensor, q: GaussianParams, logits: Tensor, k: int) -> ElboTerms:
    """Batch-averaged ELBO for the Categorical VAE (Task 2c)."""
    log_likelihood = cat_log_likelihood(x, logits, k)
    kl = kl_to_standard_normal(q)
    elbo_per_point = log_likelihood - kl
    return ElboTerms(
        elbo=elbo_per_point.mean(),
        log_likelihood=log_likelihood.mean(),
        kl=kl.mean(),
    )


# ---------------------------------------------------------------------------
# Task 2d -- Bernoulli cross-entropy and ELBO
# ---------------------------------------------------------------------------

def bernoulli_neg_cross_entropy(x: Tensor, logits: Tensor) -> Tensor:
    """Return -H(p(B|x), p(B|z)) per data-point, summed over D features.

    -H(p(B|x), p(B|z))
        = sum_d [ x_d log pi_d(z) + (1 - x_d) log(1 - pi_d(z)) ]
        = -BCE_with_logits(logits, x)  (summed over pixels)

    Using `binary_cross_entropy_with_logits` for numerical stability.
    """
    neg_ce = -F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    return neg_ce.flatten(start_dim=1).sum(dim=1)


def elbo_bernoulli(x: Tensor, q: GaussianParams, logits: Tensor) -> ElboTerms:
    """Batch-averaged modified ELBO for the Bernoulli VAE (Task 2d).

    The objective is  -H(p(B|x), p(B|z)) - KL(q(Z|x) || p(Z)),
    where the first term plays the role of the reconstruction term.
    """
    log_likelihood = bernoulli_neg_cross_entropy(x, logits)
    kl = kl_to_standard_normal(q)
    elbo_per_point = log_likelihood - kl
    return ElboTerms(
        elbo=elbo_per_point.mean(),
        log_likelihood=log_likelihood.mean(),
        kl=kl.mean(),
    )
