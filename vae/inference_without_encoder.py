"""Task 4: optimise a latent posterior without using the encoder.

The trained decoder is kept fixed.  For each test image we introduce a free
diagonal Gaussian q(Z | theta) = N(mu, diag(exp(logvar))) and optimise theta
directly with a one-sample Monte Carlo estimate of the ELBO.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset

from .elbo import kl_to_standard_normal
from .model import GaussianParams
from .model_bernoulli import BernoulliVAE


@dataclass
class OptimisedPosterior:
    q: GaussianParams
    best_elbo: float
    steps: int


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bernoulli_reconstruction_term(x: Tensor, logits: Tensor, mask: Tensor | None) -> Tensor:
    neg_ce = -F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    if mask is not None:
        neg_ce = neg_ce * mask
    return neg_ce.flatten(start_dim=1).sum(dim=1)


def _sample_q(mean: Tensor, logvar: Tensor) -> Tensor:
    eps = torch.randn_like(mean)
    return mean + torch.exp(0.5 * logvar) * eps


def optimise_posterior_for_image(
    model: BernoulliVAE,
    x: Tensor,
    *,
    mask: Tensor | None = None,
    steps: int = 800,
    learning_rate: float = 5e-2,
    patience: int = 120,
    min_delta: float = 1e-3,
) -> OptimisedPosterior:
    """Optimise theta for one image and return q(Z | theta*).

    Args:
        model: Trained Bernoulli VAE. Only its decoder is used.
        x: Image tensor of shape (1, 1, 28, 28).
        mask: Optional observed-pixel mask. For Task 4b this is one on the left
            half and zero on the right half, so the likelihood ignores missing
            pixels.
    """
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    device = x.device
    mean = nn.Parameter(torch.zeros(1, model.latent_dim, device=device))
    logvar = nn.Parameter(torch.zeros(1, model.latent_dim, device=device))
    optimiser = torch.optim.Adam([mean, logvar], lr=learning_rate)

    best_elbo = -float("inf")
    best_mean = mean.detach().clone()
    best_logvar = logvar.detach().clone()
    stale_steps = 0

    for step in range(1, steps + 1):
        optimiser.zero_grad()
        z = _sample_q(mean, logvar)
        logits = model.decode(z)
        q = GaussianParams(mean=mean, logvar=logvar)
        reconstruction = _bernoulli_reconstruction_term(x, logits, mask)
        kl = kl_to_standard_normal(q)
        elbo = (reconstruction - kl).mean()
        loss = -elbo
        loss.backward()
        optimiser.step()

        current_elbo = elbo.item()
        if current_elbo > best_elbo + min_delta:
            best_elbo = current_elbo
            best_mean = mean.detach().clone()
            best_logvar = logvar.detach().clone()
            stale_steps = 0
        else:
            stale_steps += 1
            if stale_steps >= patience:
                break

    return OptimisedPosterior(
        q=GaussianParams(mean=best_mean, logvar=best_logvar),
        best_elbo=best_elbo,
        steps=step,
    )


@torch.no_grad()
def decode_mean(model: BernoulliVAE, z: Tensor) -> Tensor:
    """Display image E[B | z] = sigmoid(logits)."""
    return torch.sigmoid(model.decode(z))


@torch.no_grad()
def encoder_reconstruction(model: BernoulliVAE, x: Tensor) -> Tensor:
    q = model.encode(x)
    z = model.reparameterise(q)
    return decode_mean(model, z)


def _draw_grid(images: list[Tensor], titles: list[str], nrow: int, ncol: int) -> plt.Figure:
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.1 * nrow))
    for row in range(nrow):
        for col in range(ncol):
            ax = axes[row, col] if nrow > 1 else axes[col]
            image = images[row * ncol + col].squeeze(0).squeeze(0).detach().cpu().numpy()
            ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    return fig


def reconstruction_without_encoder_grid(
    model: BernoulliVAE,
    test_set: Dataset,
    device: torch.device,
    path: str | Path,
    *,
    num_images: int = 8,
    steps: int = 800,
    learning_rate: float = 5e-2,
    patience: int = 120,
) -> list[OptimisedPosterior]:
    """Task 4a figure: original, optimised-theta reconstruction, encoder reconstruction."""
    rows: list[Tensor] = []
    posteriors: list[OptimisedPosterior] = []
    for idx in range(num_images):
        x = test_set[idx][0].unsqueeze(0).to(device)
        posterior = optimise_posterior_for_image(
            model,
            x,
            steps=steps,
            learning_rate=learning_rate,
            patience=patience,
        )
        posteriors.append(posterior)
        z = _sample_q(posterior.q.mean, posterior.q.logvar)
        optimised_recon = decode_mean(model, z)
        encoded_recon = encoder_reconstruction(model, x)
        rows.extend([x.cpu(), optimised_recon.cpu(), encoded_recon.cpu()])

    fig = _draw_grid(
        rows,
        ["original", "optimised q", "encoder q"],
        nrow=num_images,
        ncol=3,
    )
    _save(fig, path)
    return posteriors


def completion_without_encoder_grid(
    model: BernoulliVAE,
    test_set: Dataset,
    device: torch.device,
    path: str | Path,
    *,
    num_images: int = 8,
    steps: int = 800,
    learning_rate: float = 5e-2,
    patience: int = 120,
) -> list[OptimisedPosterior]:
    """Task 4b figure: original, observed left half, completed image."""
    left_mask = torch.zeros(1, 1, 28, 28, device=device)
    left_mask[:, :, :, :14] = 1.0

    rows: list[Tensor] = []
    posteriors: list[OptimisedPosterior] = []
    for idx in range(num_images):
        x = test_set[idx][0].unsqueeze(0).to(device)
        observed = x * left_mask
        posterior = optimise_posterior_for_image(
            model,
            x,
            mask=left_mask,
            steps=steps,
            learning_rate=learning_rate,
            patience=patience,
        )
        posteriors.append(posterior)
        z = _sample_q(posterior.q.mean, posterior.q.logvar)
        decoded = decode_mean(model, z)
        completed = observed + decoded * (1.0 - left_mask)
        rows.extend([x.cpu(), observed.cpu(), completed.cpu()])

    fig = _draw_grid(
        rows,
        ["original", "left half", "completion"],
        nrow=num_images,
        ncol=3,
    )
    _save(fig, path)
    return posteriors
