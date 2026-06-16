"""Task 3c -- interpolation in the latent space.

We reuse our preferred model trained to completion (the Bernoulli VAE of
Task 2d, ``K = 16``). For each row of the figure we take two test images ``x``
and ``x'`` with different class labels, encode them to latent samples ``z`` and
``z'``, linearly interpolate ``z_lambda = lambda * z + (1 - lambda) * z'`` for a
uniform partition of ``[0, 1]``, and decode each code back to an image.

Run with::

    uv run task3c.py
    uv run task3c.py --num-rows 8 --num-interp 8
"""

from __future__ import annotations

import argparse

import torch

from vae.data import load_mnist
from vae.latent import interpolation_grid
from vae.model_bernoulli import BernoulliVAE


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 3c: latent-space interpolation.")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-rows", type=int, default=8, help="number of digit pairs")
    parser.add_argument("--num-interp", type=int, default=8, help="interpolation steps (k > 1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/task2d_bernoulli.pt")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def load_bernoulli(checkpoint_path: str, latent_dim: int, device: torch.device) -> BernoulliVAE:
    """Rebuild a trained Bernoulli VAE from a Task 2d checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = BernoulliVAE(latent_dim=latent_dim)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def decode_mean(model: BernoulliVAE, z: torch.Tensor) -> torch.Tensor:
    """Posterior-mean image E[B | z] = sigmoid(logits) for the Bernoulli decoder."""
    return torch.sigmoid(model.decode(z))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = select_device()
    print(f"Using device: {device}")

    data = load_mnist(batch_size=args.batch_size)
    model = load_bernoulli(args.checkpoint, args.latent_dim, device)
    print(f"Loaded Bernoulli VAE (K={args.latent_dim}) from {args.checkpoint}")

    interpolation_grid(
        model, data.test_set, device,
        f"{args.output_dir}/task3c_interpolation.png",
        decode_mean=decode_mean,
        num_rows=args.num_rows,
        num_interp=args.num_interp,
    )
    print(f"Saved figure to {args.output_dir}/")


if __name__ == "__main__":
    main()
