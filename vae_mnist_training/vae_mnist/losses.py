"""Variational objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def negative_elbo(
    logits: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return summed negative ELBO, reconstruction, and analytic KL terms."""

    reconstruction = F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
    kl = -0.5 * torch.sum(1.0 + logvar - mu.square() - logvar.exp())
    return reconstruction + kl, reconstruction, kl
