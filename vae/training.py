"""Training loop, evaluation, early stopping and checkpointing.

Task 1b asks for a graph of the ELBO over the 50.000 training images and on the
10.000 validation images as a function of the epoch. We therefore evaluate the
ELBO on *both* splits after every epoch and record it in :class:`TrainHistory`.

Task 1c asks for a stopping criterion. We use **early stopping on the
validation ELBO**: training halts when the validation ELBO has not improved for
``patience`` consecutive epochs. This is a standard, well-motivated criterion --
it stops as soon as the model stops generalising better, which both avoids
overfitting and saves compute. The parameters that achieved the best validation
ELBO are saved to disk and restored at the end (so we never have to re-train).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .data import MnistData
from .elbo import elbo
from .model import GaussianVAE


@dataclass
class TrainHistory:
    """Per-epoch ELBO curves, ready to be plotted for the report."""

    train_elbo: list[float] = field(default_factory=list)
    val_elbo: list[float] = field(default_factory=list)
    train_log_likelihood: list[float] = field(default_factory=list)
    train_kl: list[float] = field(default_factory=list)
    val_log_likelihood: list[float] = field(default_factory=list)
    val_kl: list[float] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))


@dataclass
class EvalResult:
    elbo: float
    log_likelihood: float
    kl: float


@torch.no_grad()
def evaluate(model: GaussianVAE, loader: DataLoader, device: torch.device) -> EvalResult:
    """Average ELBO (and its terms) over a whole dataset split.

    Wrapped in ``torch.no_grad()`` so that evaluating on the validation data
    never tracks gradients and can therefore not leak into training, exactly as
    the assignment stresses for the validation ELBO.
    """
    model.eval()
    total_elbo = total_ll = total_kl = 0.0
    total_count = 0
    for x, _ in loader:
        x = x.to(device)
        q, _, px = model(x)
        terms = elbo(x, q, px)
        batch = x.size(0)
        total_elbo += terms.elbo.item() * batch
        total_ll += terms.log_likelihood.item() * batch
        total_kl += terms.kl.item() * batch
        total_count += batch
    return EvalResult(
        elbo=total_elbo / total_count,
        log_likelihood=total_ll / total_count,
        kl=total_kl / total_count,
    )


def save_checkpoint(model: GaussianVAE, path: str | Path) -> None:
    """Persist the model so training does not have to be repeated."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"latent_dim": model.latent_dim, "state_dict": model.state_dict()}, path
    )


def load_checkpoint(path: str | Path, device: torch.device) -> GaussianVAE:
    """Rebuild a :class:`GaussianVAE` from a checkpoint and move it to ``device``."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = GaussianVAE(latent_dim=checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def train(
    model: GaussianVAE,
    data: MnistData,
    device: torch.device,
    *,
    learning_rate: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 5,
    checkpoint_path: str | Path = "checkpoints/task1_gaussian.pt",
) -> TrainHistory:
    """Maximise the ELBO with Adam, using early stopping on the validation ELBO.

    Returns the recorded :class:`TrainHistory`; the best model (by validation
    ELBO) is both saved to ``checkpoint_path`` and loaded back into ``model``.
    """
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = TrainHistory()

    best_val_elbo = -float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, _ in data.train_loader:
            x = x.to(device)
            optimiser.zero_grad()
            q, _, px = model(x)
            # Maximise the ELBO == minimise its negative.
            loss = -elbo(x, q, px).elbo
            loss.backward()
            optimiser.step()

        # End-of-epoch ELBO on both splits (no gradient tracking).
        train_eval = evaluate(model, data.train_loader, device)
        val_eval = evaluate(model, data.val_loader, device)

        history.train_elbo.append(train_eval.elbo)
        history.val_elbo.append(val_eval.elbo)
        history.train_log_likelihood.append(train_eval.log_likelihood)
        history.train_kl.append(train_eval.kl)
        history.val_log_likelihood.append(val_eval.log_likelihood)
        history.val_kl.append(val_eval.kl)

        print(
            f"epoch {epoch:3d} | "
            f"train ELBO {train_eval.elbo:9.3f} | "
            f"val ELBO {val_eval.elbo:9.3f}"
        )

        # Early stopping: keep the best model, stop when it stalls.
        if val_eval.elbo > best_val_elbo:
            best_val_elbo = val_eval.elbo
            epochs_without_improvement = 0
            save_checkpoint(model, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}: no validation improvement "
                    f"for {patience} epochs (best val ELBO {best_val_elbo:.3f})."
                )
                break

    # Restore the best parameters before returning.
    best_model = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(best_model.state_dict())
    return history
