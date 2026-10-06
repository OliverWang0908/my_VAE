"""Checkpoint save/load helpers shared by training and inference."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import TrainConfig
from .model import VariationalAutoencoder


def save_checkpoint(
    path: Path,
    model: VariationalAutoencoder,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    metrics: dict[str, Any],
) -> Path:
    """Save model, optimizer, configuration, and metrics; return the path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "metrics": metrics,
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[VariationalAutoencoder, TrainConfig, dict[str, Any]]:
    """Restore a model for inference and return model, config, raw payload."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = TrainConfig(**checkpoint["config"])
    model = VariationalAutoencoder(config.hidden_dim, config.latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, checkpoint
