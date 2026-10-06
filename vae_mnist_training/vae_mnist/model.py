"""Variational auto-encoder model definition."""

from __future__ import annotations

import torch
import torch.nn as nn


class VariationalAutoencoder(nn.Module):
    """One-hidden-layer Bernoulli VAE matching the paper's MNIST model."""

    def __init__(self, hidden_dim: int = 500, latent_dim: int = 20) -> None:
        super().__init__()
        self.encoder_hidden = nn.Linear(784, hidden_dim)
        self.posterior_mean = nn.Linear(hidden_dim, latent_dim)
        self.posterior_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, 784)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize linear weights from N(0, 0.01^2), as in the paper."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior mean and log-variance for flattened images."""

        hidden = torch.tanh(self.encoder_hidden(x))
        return self.posterior_mean(hidden), self.posterior_logvar(hidden)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + exp(0.5 logvar) * epsilon."""

        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent vectors to Bernoulli logits for 784 pixels."""

        hidden = torch.tanh(self.decoder_hidden(z))
        return self.decoder_output(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return decoder logits, posterior mean, and posterior log-variance."""

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode_logits(z), mu, logvar
