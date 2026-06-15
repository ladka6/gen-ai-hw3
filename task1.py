"""Task 1 -- train the Gaussian VAE on MNIST and produce all report figures.

Run with::

    uv run task1.py                 # sensible defaults
    uv run task1.py --max-epochs 50 --latent-dim 16

The script loads MNIST (Task 1), trains with Adam while maximising the ELBO and
early-stopping on the validation ELBO (Tasks 1b, 1c), then writes the figures
needed for the report (Tasks 1b and 1d) into the ``outputs/`` directory.
"""

from __future__ import annotations

import argparse

import torch

from vae.data import load_mnist
from vae.model import GaussianVAE
from vae.training import train
from vae.visualize import plot_elbo_curve, prior_sample_grid, reconstruction_grid


def select_device() -> torch.device:
    """Pick CUDA, then Apple MPS, then CPU -- whatever is available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Task 1 Gaussian VAE.")
    parser.add_argument("--latent-dim", type=int, default=16, help="latent size K (>1)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5, help="early-stopping patience")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/task1_gaussian.pt")
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

    model = GaussianVAE(latent_dim=args.latent_dim)
    history = train(
        model,
        data,
        device,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        checkpoint_path=args.checkpoint,
    )

    # Persist the ELBO curves (numbers) and all figures for the report.
    history.save(f"{args.output_dir}/task1_history.json")
    plot_elbo_curve(history, f"{args.output_dir}/task1_elbo.png")
    reconstruction_grid(model, data.test_set, device, f"{args.output_dir}/task1_reconstructions.png")
    prior_sample_grid(model, device, f"{args.output_dir}/task1_prior_samples.png")
    print(f"Saved history and figures to {args.output_dir}/")


if __name__ == "__main__":
    main()
