"""Report-ready plots for a high-dimensional VAE."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from .inference import generate, interpolate, reconstruct
from .model import VariationalAutoencoder


def save_training_curves(history: list[dict[str, float]], figure_dir: Path) -> None:
    """Save ELBO and ELBO-decomposition curves plus history.csv."""

    with (figure_dir.parent / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    axes[0].plot(epochs, [row["train_nelbo"] for row in history], marker="o", ms=3, label="Train")
    axes[0].plot(epochs, [row["test_nelbo"] for row in history], marker="o", ms=3, label="Test")
    axes[0].set(xlabel="Epoch", ylabel="Negative ELBO (nats/image)", title="Optimization objective")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, [row["test_reconstruction"] for row in history], marker="o", ms=3, label="Reconstruction")
    axes[1].plot(epochs, [row["test_kl"] for row in history], marker="o", ms=3, label="KL")
    axes[1].set(xlabel="Epoch", ylabel="Nats/image", title="Test objective decomposition")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "training_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_reconstructions(
    model: VariationalAutoencoder,
    test_loader: DataLoader,
    device: torch.device,
    figure_dir: Path,
) -> None:
    """Save test inputs and posterior-mean reconstructions."""

    images, _ = next(iter(test_loader))
    recon = reconstruct(model, images[:12], device)
    comparison = torch.cat([images[:12], recon], dim=0)
    grid = make_grid(comparison, nrow=12, padding=2, pad_value=1.0)
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title("Top: MNIST inputs   |   Bottom: posterior-mean reconstructions")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure_dir / "reconstructions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def collect_posterior_means(
    model: VariationalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    limit: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect posterior means and labels for visualization only."""

    latents, labels = [], []
    count = 0
    for images, y in loader:
        mu, _ = model.encode(images.view(images.size(0), -1).to(device))
        latents.append(mu.cpu().numpy())
        labels.append(y.numpy())
        count += images.size(0)
        if count >= limit:
            break
    return np.concatenate(latents)[:limit], np.concatenate(labels)[:limit]


def save_latent_tsne(
    model: VariationalAutoencoder,
    test_loader: DataLoader,
    device: torch.device,
    figure_dir: Path,
    seed: int,
) -> None:
    """Save a reproducible t-SNE view of the 20-D posterior means."""

    latents, labels = collect_posterior_means(model, test_loader, device)
    projected = TSNE(
        n_components=2,
        perplexity=30.0,
        learning_rate="auto",
        max_iter=1000,
        init="pca",
        random_state=seed,
        method="barnes_hut",
    ).fit_transform(latents)
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    scatter = ax.scatter(projected[:, 0], projected[:, 1], c=labels, s=5, alpha=0.65, cmap="tab10")
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label("Digit label")
    ax.set(
        xlabel="t-SNE dimension 1",
        ylabel="t-SNE dimension 2",
        title="t-SNE of 20-D posterior means (perplexity = 30)",
    )
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(figure_dir / "latent_tsne.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_latent_interpolations(
    model: VariationalAutoencoder,
    test_loader: DataLoader,
    device: torch.device,
    figure_dir: Path,
) -> None:
    """Decode five ten-step interpolations between different digit classes."""

    images, labels = next(iter(test_loader))
    representatives: dict[int, torch.Tensor] = {}
    for image, label in zip(images, labels):
        representatives.setdefault(int(label), image)
        if len(representatives) == 10:
            break
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    starts = torch.stack([representatives[start] for start, _ in pairs])
    ends = torch.stack([representatives[end] for _, end in pairs])
    rows = interpolate(model, starts, ends, device, steps=10)
    grid = make_grid(rows.flatten(0, 1), nrow=10, padding=1, pad_value=1.0)
    fig, ax = plt.subplots(figsize=(10, 5.1))
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title("Latent-space interpolations: 0→1, 2→3, 4→5, 6→7, 8→9")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure_dir / "latent_interpolations.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_prior_samples(
    model: VariationalAutoencoder,
    device: torch.device,
    figure_dir: Path,
) -> None:
    """Save 100 independent samples from the standard-normal prior."""

    samples = generate(model, num_samples=100, device=device)
    grid = make_grid(samples, nrow=10, padding=1, pad_value=1.0)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(r"100 samples with $z \sim \mathcal{N}(0,I)$")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure_dir / "prior_samples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_all_visualizations(
    model: VariationalAutoencoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    history: list[dict[str, float]],
    device: torch.device,
    output_dir: Path,
    seed: int = 42,
) -> None:
    """Create every CSV and figure used by the write-up."""

    del train_loader  # Reserved for future train/test latent comparisons.
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_training_curves(history, figure_dir)
    save_reconstructions(model, test_loader, device, figure_dir)
    save_latent_tsne(model, test_loader, device, figure_dir, seed)
    save_latent_interpolations(model, test_loader, device, figure_dir)
    save_prior_samples(model, device, figure_dir)
