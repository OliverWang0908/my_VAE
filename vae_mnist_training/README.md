# MNIST VAE Training Project

This project reproduces the MNIST variational auto-encoder from Kingma and
Welling with a 20-dimensional latent space. It uses a 500-unit tanh encoder and
decoder, a diagonal Gaussian posterior, a standard-normal prior, a Bernoulli
decoder, one latent sample per image, and Adagrad.

## Project structure

```text
vae_mnist_training/
├── README.md
├── requirements.txt
├── infer.py
├── render_checkpoint.py
├── train.py
├── tests/
│   └── test_smoke.py
├── vae_mnist/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── config.py
│   ├── data.py
│   ├── engine.py
│   ├── inference.py
│   ├── losses.py
│   ├── model.py
│   ├── utils.py
│   └── visualization.py
└── pretrained/
    ├── checkpoint.pt
    ├── config.json
    ├── history.csv
    ├── metrics.json
    └── figures/
```

## Main names

- `TrainConfig`: all reproducible hyperparameters.
- `build_mnist_loaders`: downloads MNIST and creates train/test loaders.
- `VariationalAutoencoder`: model class.
- `VariationalAutoencoder.encode`: returns posterior mean and log-variance.
- `VariationalAutoencoder.reparameterize`: implements the pathwise sample.
- `VariationalAutoencoder.decode_logits`: returns Bernoulli logits.
- `negative_elbo`: returns total, reconstruction, and KL losses.
- `run_epoch`: executes one training or evaluation epoch.
- `fit`: runs the complete optimization loop.
- `save_checkpoint`: saves model state, optimizer state, configuration, and metrics.
- `load_checkpoint`: restores a model and its configuration for inference.
- `encode`: returns posterior mean and log-variance for input images.
- `reconstruct`: generates posterior-mean reconstructions used in Figure 2.
- `interpolate`: generates decoded latent paths used in Figure 4.
- `generate`: draws prior samples used in Figure 5.
- `save_latent_tsne`: embeds 5,000 posterior means with fixed-seed t-SNE.
- `save_latent_interpolations`: decodes five controlled interpolation paths.
- `save_all_visualizations`: creates curves, reconstructions, the t-SNE
  embedding, latent interpolations, and prior samples.
- `render_checkpoint.py`: regenerates figures from a saved checkpoint without
  retraining.
- `infer.py`: command-line inference entry point for Figures 2, 4, and 5.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 train.py \
  --epochs 50 \
  --latent-dim 20 \
  --hidden-dim 500 \
  --batch-size 100 \
  --learning-rate 0.01 \
  --optimizer adagrad \
  --seed 42 \
  --data-dir ../../work/data \
  --output-dir ../../work/experiment_z20_project
```

The output directory contains `checkpoint.pt`, `config.json`, `metrics.json`,
`history.csv`, and a `figures/` directory. The t-SNE figure uses 5,000 posterior
means, perplexity 30, PCA initialization, 1,000 iterations, and the run seed.

## Inference and report figures

The following command loads the pretrained checkpoint and regenerates Figure 2
(reconstructions), Figure 4 (latent interpolations), and Figure 5 (prior
samples) without training:

```bash
python3 infer.py \
  --checkpoint pretrained/checkpoint.pt \
  --data-dir ../../work/data \
  --output-dir inference_figures \
  --task all \
  --seed 42
```

Use `--task reconstruct`, `--task interpolate`, or `--task generate` to create
only one of the three figures. The output filenames are `reconstructions.png`,
`latent_interpolations.png`, and `prior_samples.png`, respectively.

## Smoke test

```bash
python3 -m unittest tests/test_smoke.py
```

The smoke test uses synthetic tensors and does not download MNIST.
