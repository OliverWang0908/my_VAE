"""Fast shape and gradient checks for the training project."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from vae_mnist.checkpoint import load_checkpoint, save_checkpoint
from vae_mnist.config import TrainConfig
from vae_mnist.inference import encode, generate, interpolate, reconstruct
from vae_mnist.losses import negative_elbo
from vae_mnist.model import VariationalAutoencoder


class VaeSmokeTest(unittest.TestCase):
    def test_forward_shapes_and_gradients(self) -> None:
        model = VariationalAutoencoder(hidden_dim=32, latent_dim=20)
        x = torch.rand(4, 784)
        logits, mu, logvar = model(x)
        self.assertEqual(logits.shape, (4, 784))
        self.assertEqual(mu.shape, (4, 20))
        self.assertEqual(logvar.shape, (4, 20))
        loss, reconstruction, kl = negative_elbo(logits, x, mu, logvar)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(kl.item(), 0.0)
        self.assertGreater(reconstruction.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(model.encoder_hidden.weight.grad)

    def test_checkpoint_and_inference_interfaces(self) -> None:
        device = torch.device("cpu")
        model = VariationalAutoencoder(hidden_dim=32, latent_dim=5)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
        config = TrainConfig(hidden_dim=32, latent_dim=5, epochs=1)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            returned_path = save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                config,
                metrics={"test_nelbo": 1.0},
            )
            restored, restored_config, payload = load_checkpoint(checkpoint_path, device)

        self.assertEqual(returned_path, checkpoint_path)
        self.assertEqual(restored_config.latent_dim, 5)
        self.assertIn("optimizer_state", payload)

        images = torch.rand(4, 1, 28, 28)
        mu, logvar = encode(restored, images, device)
        reconstructions = reconstruct(restored, images, device)
        interpolations = interpolate(
            restored, images[:2], images[2:], device, steps=6
        )
        samples = generate(restored, num_samples=7, device=device)
        self.assertEqual(mu.shape, (4, 5))
        self.assertEqual(logvar.shape, (4, 5))
        self.assertEqual(reconstructions.shape, (4, 1, 28, 28))
        self.assertEqual(interpolations.shape, (2, 6, 1, 28, 28))
        self.assertEqual(samples.shape, (7, 1, 28, 28))


if __name__ == "__main__":
    unittest.main()
