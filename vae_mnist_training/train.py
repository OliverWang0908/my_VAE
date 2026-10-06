#!/usr/bin/env python3
"""Train and evaluate the structured MNIST VAE reproduction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from vae_mnist.checkpoint import save_checkpoint
from vae_mnist.config import TrainConfig
from vae_mnist.data import build_mnist_loaders
from vae_mnist.engine import fit
from vae_mnist.model import VariationalAutoencoder
from vae_mnist.utils import choose_device, save_json, seed_everything
from vae_mnist.visualization import save_all_visualizations


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="work/data")
    parser.add_argument("--output-dir", default="work/experiment_z20")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=500)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("adagrad", "adam"), default="adagrad")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-dynamic-binarization", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        seed=args.seed,
        dynamic_binarization=not args.no_dynamic_binarization,
    )


def build_optimizer(
    model: torch.nn.Module, config: TrainConfig
) -> torch.optim.Optimizer:
    if config.optimizer == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=config.learning_rate)
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def main() -> None:
    args = build_argument_parser().parse_args()
    config = config_from_args(args)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)
    device = choose_device()
    print(f"device={device}")
    train_loader, test_loader = build_mnist_loaders(config)
    model = VariationalAutoencoder(config.hidden_dim, config.latent_dim).to(device)
    optimizer = build_optimizer(model, config)

    history = fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        config=config,
    )
    final = history[-1]
    metrics = {
        "config": asdict(config),
        "device": str(device),
        "train_examples": len(train_loader.dataset),
        "test_examples": len(test_loader.dataset),
        "final_test_nelbo_nats_per_image": final["test_nelbo"],
        "final_test_reconstruction_nats_per_image": final["test_reconstruction"],
        "final_test_kl_nats_per_image": final["test_kl"],
        "final_test_bits_per_dimension": final["test_nelbo"] / (784.0 * np.log(2.0)),
    }

    save_json(asdict(config), output_dir / "config.json")
    save_json(metrics, output_dir / "metrics.json")
    save_checkpoint(
        path=output_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        metrics=metrics,
    )
    save_all_visualizations(
        model, train_loader, test_loader, history, device, output_dir, seed=config.seed
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
