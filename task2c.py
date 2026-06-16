"""Task 2c -- train the Categorical VAE on MNIST and produce report figures.

Run with::

    uv run task2c.py
    uv run task2c.py --max-epochs 50 --latent-dim 16 --num-bins 4
"""

from __future__ import annotations

import argparse
import functools

import torch

from vae.data import load_mnist
from vae.elbo import elbo_cat
from vae.model_categorical import CatVAE, NUM_BINS
from vae.training import train_generic
from vae.visualize import (
    plot_elbo_curve,
    prior_sample_grid_cat,
    reconstruction_grid_cat,
)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Task 2c Categorical VAE.")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--num-bins", type=int, default=NUM_BINS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/task2c_cat.pt")
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

    k = args.num_bins
    model = CatVAE(latent_dim=args.latent_dim, k=k)

    # Bind k so the generic loop can call elbo_cat(x, q, logits) with one call.
    elbo_fn = functools.partial(elbo_cat, k=k)

    history = train_generic(
        model,
        elbo_fn,
        data,
        device,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        checkpoint_path=args.checkpoint,
    )

    history.save(f"{args.output_dir}/task2c_history.json")
    plot_elbo_curve(history, f"{args.output_dir}/task2c_elbo.png")
    reconstruction_grid_cat(model, data.test_set, device, f"{args.output_dir}/task2c_reconstructions.png")
    prior_sample_grid_cat(model, device, f"{args.output_dir}/task2c_prior_samples.png")
    print(f"Saved history and figures to {args.output_dir}/")


if __name__ == "__main__":
    main()
