#!/usr/bin/env python3
"""Regenerate all report figures from an existing training checkpoint."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from vae_mnist.checkpoint import load_checkpoint
from vae_mnist.data import build_mnist_loaders
from vae_mnist.utils import choose_device, seed_everything
from vae_mnist.visualization import save_all_visualizations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    device = choose_device()
    model, config, _ = load_checkpoint(checkpoint_path, device)
    config = replace(config, data_dir=args.data_dir, output_dir=args.output_dir)
    seed_everything(config.seed)
    train_loader, test_loader = build_mnist_loaders(config)

    with (output_dir / "history.csv").open(encoding="utf-8") as handle:
        history = [
            {key: int(value) if key == "epoch" else float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    save_all_visualizations(
        model, train_loader, test_loader, history, device, output_dir, seed=config.seed
    )


if __name__ == "__main__":
    main()
