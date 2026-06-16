"""Task 2b -- train the Beta VAE on MNIST and produce report figures.

Run with::

    uv run task2b.py
    uv run task2b.py --max-epochs 50 --latent-dim 16
"""

from __future__ import annotations

import argparse
import functools

import torch

from vae.data import load_mnist
from vae.elbo import elbo_beta
from vae.model_beta import BetaVAE
from vae.training import train_generic
from vae.visualize import (
    plot_elbo_curve,
    prior_sample_grid_beta,
    reconstruction_grid_beta,
)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Task 2b Beta VAE.")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/task2b_beta.pt")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = select_device()
    print(f"Using device: {device}")

    data = load_mnist(batch_size=args.batch_size)
    print(
        f"Loaded MNIST: {len(data.train_set)} train, "
        f"{len(data.val_set)} val, {len(data.test_set)} test images."
    )

    model = BetaVAE(latent_dim=args.latent_dim)
    history = train_generic(
        model,
        elbo_beta,
        data,
        device,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        checkpoint_path=args.checkpoint,
    )

    history.save(f"{args.output_dir}/task2b_history.json")
    plot_elbo_curve(history, f"{args.output_dir}/task2b_elbo.png")
    reconstruction_grid_beta(model, data.test_set, device, f"{args.output_dir}/task2b_reconstructions.png")
    prior_sample_grid_beta(model, device, f"{args.output_dir}/task2b_prior_samples.png")
    print(f"Saved history and figures to {args.output_dir}/")


if __name__ == "__main__":
    main()
