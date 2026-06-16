"""Task 4 -- inference without the encoder.

Run with::

    uv run task4.py
    uv run task4.py --num-images 8 --steps 800
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vae.data import load_mnist
from vae.inference_without_encoder import (
    completion_without_encoder_grid,
    reconstruction_without_encoder_grid,
)
from vae.model_bernoulli import BernoulliVAE


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 4: optimise q(Z|theta) without the encoder.")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=5e-2)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/task2d_bernoulli.pt")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def load_bernoulli(checkpoint_path: str, latent_dim: int, device: torch.device) -> BernoulliVAE:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = BernoulliVAE(latent_dim=latent_dim)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def _summarise(posteriors) -> list[dict[str, float | int]]:
    return [
        {
            "index": idx,
            "best_elbo": posterior.best_elbo,
            "steps": posterior.steps,
        }
        for idx, posterior in enumerate(posteriors)
    ]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = select_device()
    print(f"Using device: {device}")

    data = load_mnist(batch_size=args.batch_size)
    model = load_bernoulli(args.checkpoint, args.latent_dim, device)
    print(f"Loaded Bernoulli VAE (K={args.latent_dim}) from {args.checkpoint}")

    output_dir = Path(args.output_dir)
    task4a = reconstruction_without_encoder_grid(
        model,
        data.test_set,
        device,
        output_dir / "task4a_reconstruction_without_encoder.png",
        num_images=args.num_images,
        steps=args.steps,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )
    task4b = completion_without_encoder_grid(
        model,
        data.test_set,
        device,
        output_dir / "task4b_completion_without_encoder.png",
        num_images=args.num_images,
        steps=args.steps,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )

    summary = {
        "task4a": _summarise(task4a),
        "task4b": _summarise(task4b),
        "settings": {
            "num_images": args.num_images,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "checkpoint": args.checkpoint,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task4_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved Task 4 figures and summary to {output_dir}/")


if __name__ == "__main__":
    main()
