"""Training and evaluation loops."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .losses import negative_elbo
from .model import VariationalAutoencoder


def run_epoch(
    model: VariationalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    dynamic_binarization: bool = False,
) -> dict[str, float]:
    """Run one train or evaluation epoch and return per-image metrics."""

    training = optimizer is not None
    model.train(training)
    totals = {"nelbo": 0.0, "reconstruction": 0.0, "kl": 0.0}
    count = 0

    for images, _ in loader:
        x = images.view(images.size(0), -1).to(device)
        if training and dynamic_binarization:
            x = torch.bernoulli(x)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits, mu, logvar = model(x)
            loss, reconstruction, kl = negative_elbo(logits, x, mu, logvar)
            if training:
                (loss / x.size(0)).backward()
                optimizer.step()

        count += x.size(0)
        totals["nelbo"] += loss.item()
        totals["reconstruction"] += reconstruction.item()
        totals["kl"] += kl.item()

    return {name: value / count for name, value in totals.items()}


def fit(
    model: VariationalAutoencoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainConfig,
) -> list[dict[str, float]]:
    """Train for config.epochs and return the complete metric history."""

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        train = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            dynamic_binarization=config.dynamic_binarization,
        )
        test = run_epoch(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_nelbo": train["nelbo"],
            "train_reconstruction": train["reconstruction"],
            "train_kl": train["kl"],
            "test_nelbo": test["nelbo"],
            "test_reconstruction": test["reconstruction"],
            "test_kl": test["kl"],
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_nelbo={train['nelbo']:.3f} "
            f"test_nelbo={test['nelbo']:.3f} "
            f"test_recon={test['reconstruction']:.3f} test_kl={test['kl']:.3f}"
        )
    return history
