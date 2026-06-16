"""Task 3b -- visualise a K-dimensional latent space with PCA.

We reuse our preferred model with a ``K``-dimensional latent space (``K = 16``,
which satisfies ``K >= 10``): the Bernoulli VAE already trained to completion in
Task 2d. We evaluate the trained encoder on the first 1000 test images, store
their latent means ``mu(x_i)`` in a matrix ``U`` (1000 rows, K columns), run PCA
on ``U``, and scatter the rows projected onto the first two principal
components, colour-coded by digit class.

Run with::

    uv run task3b.py
    uv run task3b.py --checkpoint checkpoints/task2d_bernoulli.pt --latent-dim 16
"""

from __future__ import annotations

import argparse

import torch

from vae.data import load_mnist
from vae.latent import latent_scatter_pca
from vae.model_bernoulli import BernoulliVAE


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 3b: PCA of the K-D latent means.")
    parser.add_argument("--latent-dim", type=int, default=16, help="K >= 10")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-points", type=int, default=1000)
    parser.add_argument("--checkpoint", default="checkpoints/task2d_bernoulli.pt")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def load_bernoulli(checkpoint_path: str, latent_dim: int, device: torch.device) -> BernoulliVAE:
    """Rebuild a trained Bernoulli VAE from a Task 2d checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = BernoulliVAE(latent_dim=latent_dim)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def main() -> None:
    args = parse_args()
    if args.latent_dim < 10:
        raise ValueError(f"Task 3b requires K >= 10, got K={args.latent_dim}.")

    device = select_device()
    print(f"Using device: {device}")

    data = load_mnist(batch_size=args.batch_size)
    model = load_bernoulli(args.checkpoint, args.latent_dim, device)
    print(f"Loaded Bernoulli VAE (K={args.latent_dim}) from {args.checkpoint}")

    latent_scatter_pca(
        model, data.test_set, device,
        f"{args.output_dir}/task3b_latent_pca.png",
        num_points=args.num_points,
    )
    print(f"Saved figure to {args.output_dir}/")


if __name__ == "__main__":
    main()
