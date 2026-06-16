"""Figures for Task 3 -- investigating the structure of the latent space.

All three sub-tasks operate on the *trained encoder* of our preferred model
(the Bernoulli VAE of Task 2d -- see the report for the motivation):

* :func:`latent_scatter_2d`  -- Task 3a: a 2-D latent space ``Z = R^2`` is
  visualised directly by plotting the encoder means ``mu(x_i)`` of the first
  1000 test images, colour-coded by digit class.
* :func:`latent_scatter_pca` -- Task 3b: a ``K``-dimensional latent space
  (``K >= 10``) is visualised by projecting the encoder means onto their first
  two principal components.
* :func:`interpolation_grid` -- Task 3c: linear interpolation between the
  latent codes of two test images with different class labels, decoded back to
  image space.

The encoder of every VAE in this package exposes ``encode(x) -> GaussianParams``
whose ``.mean`` is ``mu(x)``; the decoder of the Bernoulli model returns raw
logits, so the posterior-mean image is ``sigmoid(logits)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

# A consistent, perceptually distinct colour for each of the ten digits.
_DIGIT_CMAP = "tab10"
_NUM_CLASSES = 10


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def encode_means(
    model: nn.Module,
    test_set: Dataset,
    device: torch.device,
    num_points: int = 1000,
) -> tuple[Tensor, Tensor]:
    """Return the encoder means ``mu(x_i)`` and labels for the first ``num_points``.

    Evaluates the *trained encoder* on the first ``num_points`` test images and
    stacks the posterior means into a matrix ``U`` with ``num_points`` rows and
    ``K`` columns (Task 3a/3b). No gradients are tracked.
    """
    model.eval()
    images = torch.stack([test_set[i][0] for i in range(num_points)]).to(device)
    labels = torch.tensor([test_set[i][1] for i in range(num_points)])
    means = model.encode(images).mean  # (num_points, K)
    return means.cpu(), labels


def _scatter_by_class(ax: plt.Axes, coords: Tensor, labels: Tensor) -> None:
    """Scatter 2-D ``coords`` with a distinct colour per digit class."""
    cmap = plt.get_cmap(_DIGIT_CMAP, _NUM_CLASSES)
    for digit in range(_NUM_CLASSES):
        mask = labels == digit
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=10,
            color=cmap(digit),
            label=str(digit),
            alpha=0.7,
            edgecolors="none",
        )
    ax.legend(title="digit", markerscale=2, fontsize=8, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)


def latent_scatter_2d(
    model: nn.Module,
    test_set: Dataset,
    device: torch.device,
    path: str | Path,
    num_points: int = 1000,
) -> None:
    """Task 3a: scatter the 2-D encoder means ``mu(x_i)``, colour-coded by class."""
    means, labels = encode_means(model, test_set, device, num_points)
    if means.size(1) != 2:
        raise ValueError(
            f"latent_scatter_2d expects a 2-D latent space, got K={means.size(1)}."
        )

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter_by_class(ax, means, labels)
    ax.set_xlabel(r"$\mu_1(x)$")
    ax.set_ylabel(r"$\mu_2(x)$")
    ax.set_title(f"2-D latent means of the first {num_points} test images")
    _save(fig, path)


def latent_scatter_pca(
    model: nn.Module,
    test_set: Dataset,
    device: torch.device,
    path: str | Path,
    num_points: int = 1000,
) -> None:
    """Task 3b: PCA-project the ``K``-D encoder means to 2-D and scatter them.

    The first two principal components of the means matrix ``U`` (one row per
    test image) are computed with scikit-learn; the projected points are
    colour-coded by digit class exactly as in Task 3a.
    """
    from sklearn.decomposition import PCA

    means, labels = encode_means(model, test_set, device, num_points)
    pca = PCA(n_components=2)
    projected = torch.from_numpy(pca.fit_transform(means.numpy()))
    var = pca.explained_variance_ratio_ * 100.0

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter_by_class(ax, projected, labels)
    ax.set_xlabel(f"PC 1 ({var[0]:.1f}% variance)")
    ax.set_ylabel(f"PC 2 ({var[1]:.1f}% variance)")
    ax.set_title(
        f"PCA of the {means.size(1)}-D latent means "
        f"(first {num_points} test images)"
    )
    _save(fig, path)


def _first_pair_with_different_labels(
    test_set: Dataset, start: int
) -> tuple[int, int]:
    """Return indices ``(i, j)``, ``j > i >= start``, of two differently-labelled images."""
    i = start
    label_i = test_set[i][1]
    j = i + 1
    while test_set[j][1] == label_i:
        j += 1
    return i, j


@torch.no_grad()
def interpolation_grid(
    model: nn.Module,
    test_set: Dataset,
    device: torch.device,
    path: str | Path,
    decode_mean: Callable[[nn.Module, Tensor], Tensor],
    num_rows: int = 8,
    num_interp: int = 8,
) -> None:
    """Task 3c: latent-space interpolation between pairs of test images.

    For each row we pick two test images ``x`` and ``x'`` with *different* class
    labels, encode them to samples ``z ~ q(Z|x)`` and ``z' ~ q(Z|x')``, and
    linearly interpolate ``z_lambda = lambda * z + (1 - lambda) * z'`` for
    ``num_interp`` values of ``lambda`` on a uniform partition of ``[0, 1]``.
    Each interpolated code is decoded to its posterior-mean image.

    The grid layout per row is::

        [ x | x_{lambda=1} ... x_{lambda=0} | x' ]

    i.e. the leftmost cell is the original image ``x``, the rightmost is ``x'``,
    and the ``num_interp`` cells in between sweep from the reconstruction of
    ``x`` (``lambda = 1``) to that of ``x'`` (``lambda = 0``).

    ``decode_mean(model, z)`` maps a batch of latent codes to posterior-mean
    images in ``[0, 1]`` (for the Bernoulli model this is ``sigmoid(logits)``).
    """
    model.eval()
    # lambda = 1 -> z (left, the x side); lambda = 0 -> z' (right, the x' side).
    lambdas = torch.linspace(1.0, 0.0, num_interp, device=device)

    cols = num_interp + 2  # original x + interpolations + original x'
    h, w = test_set[0][0].shape[1], test_set[0][0].shape[2]
    fig, axes = plt.subplots(num_rows, cols, figsize=(cols, num_rows))

    cursor = 0
    for row in range(num_rows):
        i, j = _first_pair_with_different_labels(test_set, cursor)
        cursor = j + 1

        x = test_set[i][0].unsqueeze(0).to(device)
        x_prime = test_set[j][0].unsqueeze(0).to(device)

        z = model.reparameterise(model.encode(x))          # z  ~ q(Z|x)
        z_prime = model.reparameterise(model.encode(x_prime))  # z' ~ q(Z|x')

        # Stack all interpolated codes and decode them in a single batch.
        z_lambda = lambdas.view(-1, 1) * z + (1.0 - lambdas).view(-1, 1) * z_prime
        interp_images = decode_mean(model, z_lambda)  # (num_interp, 1, H, W)

        row_images = [x.squeeze(0), *interp_images, x_prime.squeeze(0)]
        for col, img in enumerate(row_images):
            ax = axes[row, col]
            ax.imshow(img.squeeze(0).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

    # Column annotations on the top row only.
    axes[0, 0].set_title(r"$x$", fontsize=9)
    axes[0, -1].set_title(r"$x'$", fontsize=9)
    fig.suptitle("Latent-space interpolations between digit pairs")
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    _save(fig, path)
