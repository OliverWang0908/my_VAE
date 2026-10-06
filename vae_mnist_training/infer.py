#!/usr/bin/env python3
"""Generate report Figures 2, 4, and 5 from a trained checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from vae_mnist.checkpoint import load_checkpoint
from vae_mnist.data import build_mnist_loaders
from vae_mnist.utils import choose_device, seed_everything
from vae_mnist.visualization import (
    save_latent_interpolations,
    save_prior_samples,
    save_reconstructions,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for checkpoint inference."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--task",
        choices=("all", "reconstruct", "interpolate", "generate"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """Load a checkpoint and write the requested inference figures."""

    args = build_argument_parser().parse_args()
    device = choose_device()
    seed_everything(args.seed)
    model, config, _ = load_checkpoint(Path(args.checkpoint), device)
    config = replace(config, data_dir=args.data_dir, output_dir=args.output_dir)
    _, test_loader = build_mnist_loaders(config)
    figure_dir = Path(args.output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if args.task in ("all", "reconstruct"):
        save_reconstructions(model, test_loader, device, figure_dir)
    if args.task in ("all", "interpolate"):
        save_latent_interpolations(model, test_loader, device, figure_dir)
    if args.task in ("all", "generate"):
        save_prior_samples(model, device, figure_dir)


if __name__ == "__main__":
    main()
