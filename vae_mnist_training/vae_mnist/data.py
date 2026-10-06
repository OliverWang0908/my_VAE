"""MNIST input pipeline."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import TrainConfig


def build_mnist_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Download MNIST if needed and return seeded train/test loaders."""

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(config.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(config.data_dir, train=False, download=True, transform=transform)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader
