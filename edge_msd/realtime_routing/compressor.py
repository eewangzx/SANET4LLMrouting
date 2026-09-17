"""Predictive information-bottleneck compressor (X -> Z -> Y_future).

Variational IB with a VAE-style rate term and a prediction ("relevance") term:

    L = E[ KL(q(z|x) || N(0,I)) ]  +  beta * MSE(pred(z), y_future)

Z is the short message content; the same Z is fed to the routing policy as part of
its state, so the TD gradient also reaches the encoder during joint training. This is
a predictive IB, not the original SANet IB method, and not a minimal sufficient
statistic claim.
"""

from __future__ import annotations

import torch
from torch import nn


class PredictiveIBCompressor(nn.Module):
    def __init__(self, features, window, latent_dim=8, hidden=64, horizons=4, out_dim=1):
        super().__init__()
        if min(features, window, latent_dim, hidden, horizons, out_dim) < 1:
            raise ValueError("all dimensions must be positive")
        self.features, self.window = features, window
        self.latent_dim, self.horizons, self.out_dim = latent_dim, horizons, out_dim
        self.gru = nn.GRU(features, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.predict = nn.Linear(latent_dim, horizons * out_dim)

    def encode(self, x):
        """Deterministic posterior mean, used for acting and for rate accounting."""
        _, h = self.gru(x)
        return self.mu(h[-1])

    def forward(self, x):
        _, h = self.gru(x)
        h = h[-1]
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8.0, 8.0)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        pred = self.predict(z).view(-1, self.horizons, self.out_dim)
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(-1)  # per sample
        return z, pred, kl

    def predict_from(self, z):
        return self.predict(z).view(-1, self.horizons, self.out_dim)

    def latent_dim_size(self) -> int:
        return self.latent_dim


def ib_loss(pred, target, kl, beta=1.0, free_bits=0.0):
    """Prediction term + beta * rate term with optional per-dimension free bits."""
    mse = (pred - target).pow(2).mean()
    rate = kl.mean()
    if free_bits > 0.0:
        rate = torch.clamp(kl.mean(), min=free_bits)
    return mse + beta * rate, mse.detach(), rate.detach()
