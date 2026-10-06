"""Tensor-level inference operations for a trained MNIST VAE."""

from __future__ import annotations

import torch

from .model import VariationalAutoencoder


def _flatten_images(images: torch.Tensor) -> torch.Tensor:
    """Return a batch in [N, 784] form."""

    if images.ndim == 2 and images.shape[1] == 784:
        return images
    return images.view(images.shape[0], 784)


@torch.no_grad()
def encode(
    model: VariationalAutoencoder,
    images: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return posterior mean and log-variance for an image batch."""

    model.eval()
    return model.encode(_flatten_images(images).to(device))


@torch.no_grad()
def reconstruct(
    model: VariationalAutoencoder,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Decode posterior means and return reconstructed [N, 1, 28, 28] images."""

    mu, _ = encode(model, images, device)
    return torch.sigmoid(model.decode_logits(mu)).view(-1, 1, 28, 28).cpu()


@torch.no_grad()
def interpolate(
    model: VariationalAutoencoder,
    start_images: torch.Tensor,
    end_images: torch.Tensor,
    device: torch.device,
    steps: int = 10,
) -> torch.Tensor:
    """Return decoded linear paths between corresponding image pairs."""

    if start_images.shape[0] != end_images.shape[0]:
        raise ValueError("start_images and end_images must contain the same number of images")
    if steps < 2:
        raise ValueError("steps must be at least 2")
    start_mu, _ = encode(model, start_images, device)
    end_mu, _ = encode(model, end_images, device)
    alpha = torch.linspace(0.0, 1.0, steps, device=device).view(1, steps, 1)
    paths = (1.0 - alpha) * start_mu[:, None, :] + alpha * end_mu[:, None, :]
    decoded = torch.sigmoid(model.decode_logits(paths.reshape(-1, paths.shape[-1])))
    return decoded.view(start_images.shape[0], steps, 1, 28, 28).cpu()


@torch.no_grad()
def generate(
    model: VariationalAutoencoder,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample the prior and return generated [N, 1, 28, 28] images."""

    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    model.eval()
    latent_dim = model.posterior_mean.out_features
    z = torch.randn(num_samples, latent_dim, device=device)
    return torch.sigmoid(model.decode_logits(z)).view(-1, 1, 28, 28).cpu()
