"""Figures for the report: the ELBO curve, reconstructions and prior samples.

These cover the deliverables of Task 1b (ELBO vs. epochs) and Task 1d
(reconstructions of test images, and images sampled from the prior).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .model import GaussianParams, GaussianVAE
from .training import TrainHistory


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sample_image(px: GaussianParams) -> Tensor:
    """Draw ``x' ~ p(X | z)`` from the Gaussian likelihood.

    The sample ``mu + sigma * eps`` is clamped to ``[0, 1]`` purely for display,
    since pixel intensities live in that range.
    """
    eps = torch.randn_like(px.mean)
    return (px.mean + px.std * eps).clamp(0.0, 1.0)


def plot_elbo_curve(history: TrainHistory, path: str | Path) -> None:
    """Plot how the training and validation ELBO evolve over the epochs."""
    epochs = range(1, len(history.train_elbo) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history.train_elbo, label="train (50.000 images)", marker="o", ms=3)
    ax.plot(epochs, history.val_elbo, label="validation (10.000 images)", marker="s", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("ELBO (per image)")
    ax.set_title("ELBO over training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def _grid_from_tensor(images: Tensor, nrow: int, ncol: int, title: str) -> plt.Figure:
    """Render ``nrow * ncol`` single-channel images into a tidy grid."""
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx].squeeze(0).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(title)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig


@torch.no_grad()
def reconstruction_grid(
    model: GaussianVAE,
    test_set: Dataset,
    device: torch.device,
    path: str | Path,
    num_images: int = 32,
) -> None:
    """Reconstruct the first ``num_images`` test images and save an 8x8 grid.

    The grid interleaves blocks of originals and reconstructions row by row, so
    every reconstruction sits directly *below* its original image. With 32
    originals + 32 reconstructions this fills the requested 8x8 (64-image) grid.

    Reconstruction follows Task 1d: encode ``x`` to ``q(Z|x)``, draw ``z``,
    decode to ``p(X|z)``, and sample ``x' ~ p(X|z)``.
    """
    model.eval()
    originals = torch.stack([test_set[i][0] for i in range(num_images)]).to(device)

    q = model.encode(originals)
    z = model.reparameterise(q)
    px = model.decode(z)
    reconstructions = sample_image(px)

    cols = 8
    block_rows = num_images // cols  # 4 blocks of originals + 4 of reconstructions
    grid = torch.empty(2 * block_rows, cols, 1, originals.size(2), originals.size(3))
    for b in range(block_rows):
        grid[2 * b] = originals[b * cols : (b + 1) * cols]
        grid[2 * b + 1] = reconstructions[b * cols : (b + 1) * cols]
    grid = grid.reshape(-1, 1, originals.size(2), originals.size(3))

    fig = _grid_from_tensor(
        grid, nrow=2 * block_rows, ncol=cols,
        title="Test images (rows 1,3,5,7) and their reconstructions (rows 2,4,6,8)",
    )
    _save(fig, path)


@torch.no_grad()
def prior_sample_grid(
    model: GaussianVAE,
    device: torch.device,
    path: str | Path,
    nrow: int = 8,
    ncol: int = 8,
) -> None:
    """Generate brand-new images by sampling ``z ~ p(Z)`` and ``x' ~ p(X|z)``."""
    model.eval()
    num = nrow * ncol
    z = torch.randn(num, model.latent_dim, device=device)  # z ~ p(Z) = N(0, I)
    px = model.decode(z)
    samples = sample_image(px)
    fig = _grid_from_tensor(samples, nrow, ncol, title="Samples from the prior p(Z)")
    _save(fig, path)
