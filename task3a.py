"""Task 3a -- visualise a 2-D latent space.

We retrain our preferred model (the Bernoulli VAE of Task 2d) with a
2-dimensional latent space ``Z = R^2`` to completion, using the same objective
and early-stopping criterion as before. We then evaluate the trained encoder on
the first 1000 test images and scatter their latent means ``mu(x_i)``,
colour-coded by digit class.

Run with::

    uv run task3a.py
    uv run task3a.py --max-epochs 50
"""

from __future__ import annotations

import argparse

import torch

from vae.data import load_mnist
from vae.elbo import elbo_bernoulli
from vae.latent import latent_scatter_2d
from vae.model_bernoulli import BernoulliVAE
from vae.training import train_generic
from vae.visualize import plot_elbo_curve


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Task 3a 2-D Bernoulli VAE.")
    parser.add_argument("--latent-dim", type=int, default=2, help="must be 2 for Task 3a")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=1000)
    parser.add_argument("--checkpoint", default="checkpoints/task3a_bernoulli_2d.pt")
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

    history.save(f"{args.output_dir}/task3a_history.json")
    plot_elbo_curve(history, f"{args.output_dir}/task3a_elbo.png")
    latent_scatter_2d(
        model, data.test_set, device,
        f"{args.output_dir}/task3a_latent_scatter.png",
        num_points=args.num_points,
    )
    print(f"Saved history and figures to {args.output_dir}/")


if __name__ == "__main__":
    main()
