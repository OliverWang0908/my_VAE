"""Small reproducibility and I/O helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and MPS when available."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def choose_device() -> torch.device:
    """Prefer CUDA, then MPS, and otherwise use CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(value: Any, path: Path) -> None:
    """Write JSON with stable, readable formatting."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
