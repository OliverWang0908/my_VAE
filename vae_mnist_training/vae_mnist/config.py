"""Experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """All settings required to reproduce a training run."""

    data_dir: str = "work/data"
    output_dir: str = "work/experiment_z20"
    epochs: int = 50
    batch_size: int = 100
    hidden_dim: int = 500
    latent_dim: int = 20
    learning_rate: float = 0.01
    optimizer: str = "adagrad"
    seed: int = 42
    dynamic_binarization: bool = True
