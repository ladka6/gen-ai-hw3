"""Task 2d -- train the Bernoulli VAE on MNIST and produce report figures.

Run with::

    uv run task2d.py
    uv run task2d.py --max-epochs 50 --latent-dim 16
"""

from __future__ import annotations

import argparse

import torch

from vae.data import load_mnist
from vae.elbo import elbo_bernoulli
from vae.model_bernoulli import BernoulliVAE
from vae.training import train_generic
from vae.visualize import (
    plot_elbo_curve,
    prior_sample_grid_bernoulli,
    reconstruction_grid_bernoulli,
)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Task 2d Bernoulli VAE.")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/task2d_bernoulli.pt")
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

    model = BernoulliVAE(latent_dim=args.latent_dim)
    history = train_generic(
        model,
        elbo_bernoulli,
        data,
        device,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        checkpoint_path=args.checkpoint,
    )

    history.save(f"{args.output_dir}/task2d_history.json")
    plot_elbo_curve(history, f"{args.output_dir}/task2d_elbo.png")
    reconstruction_grid_bernoulli(model, data.test_set, device, f"{args.output_dir}/task2d_reconstructions.png")
    prior_sample_grid_bernoulli(model, device, f"{args.output_dir}/task2d_prior_samples.png")
    print(f"Saved history and figures to {args.output_dir}/")


if __name__ == "__main__":
    main()
