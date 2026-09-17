"""Three telemetry codecs with one shared receiver interface.

The comparison is bytes-vs-SLA: all three uploads are turned into a fixed-dim
embedding and fed to the same policy, but they carry different amounts of the
node's history over the wire.

  raw      : the whole W x F window          -> W*F*4 + 16 bytes
  stats16  : last/mean/std summary, 16 vals  -> 16*4 + 16 bytes
  proposed : predictive-IB bottleneck of the window -> latent*4 + 16 bytes

raw and stats16 also get a small prediction head so they are pre-trained with the
same future-prediction objective as proposed (fairness), then everything is frozen
before RL training.
"""

from __future__ import annotations

import torch
from torch import nn

from edge_msd.realtime_routing.compressor import PredictiveIBCompressor


class Codec(nn.Module):
    def __init__(self, kind, features, window, latent, horizons, hidden=64, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        if kind not in ("raw", "stats16", "proposed"):
            raise ValueError(f"unknown codec kind: {kind}")
        self.kind, self.features, self.window, self.latent = kind, features, window, latent
        self.horizons = horizons
        if kind == "proposed":
            self.ib = PredictiveIBCompressor(features, window, latent_dim=latent,
                                             hidden=hidden, horizons=horizons, out_dim=features)
            self.receiver = None
            self.head = None
        else:
            self.ib = None
            in_dim = window * features if kind == "raw" else 16
            self.receiver = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                          nn.Linear(hidden, latent))
            self.head = nn.Linear(latent, horizons * features)

    def payload_bytes(self) -> int:
        if self.kind == "proposed":
            return self.latent * 4
        if self.kind == "stats16":
            return 16 * 4
        return self.window * self.features * 4

    def total_bytes(self) -> int:
        return self.payload_bytes() + 16  # 16-byte fixed protocol header

    def _stats16(self, windows):
        last, mean, std = windows[:, -1, :], windows.mean(1), windows.std(1)
        stats = torch.cat([last, mean, std], 1)
        if stats.shape[1] < 16:
            pad = torch.zeros(len(stats), 16 - stats.shape[1], device=stats.device)
            stats = torch.cat([stats, pad], 1)
        return stats[:, :16]

    def embed(self, windows):
        windows = windows.to(self.device)
        if self.kind == "proposed":
            return self.ib.encode(windows)
        if self.kind == "raw":
            return self.receiver(windows.reshape(len(windows), -1))
        return self.receiver(self._stats16(windows))

    def predict(self, windows):
        z = self.embed(windows)
        if self.kind == "proposed":
            return self.ib.predict_from(z)
        return self.head(z).view(-1, self.horizons, self.features)


def pretrain_codec(codec, sample_windows, sample_futures, iters=2000, beta=5e-3, lr=2e-3):
    """Train the codec (and its prediction head) on the future-prediction objective."""
    X = sample_windows.to(codec.device)
    Y = sample_futures.to(codec.device)
    params = list(codec.ib.parameters()) if codec.kind == "proposed" else (
        list(codec.receiver.parameters()) + list(codec.head.parameters()))
    opt = torch.optim.Adam(params, lr=lr)
    n = int(0.85 * len(X))
    for _ in range(iters):
        b = torch.randint(0, n, (256,))
        if codec.kind == "proposed":
            _, pred, kl = codec.ib(X[b])
            loss = (pred - Y[b]).pow(2).mean() + beta * kl.mean()
        else:
            pred = codec.predict(X[b])
            loss = (pred - Y[b]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    codec.eval()
    with torch.no_grad():
        pred = codec.predict(X[n:])
        mse = (pred - Y[n:]).pow(2).mean().item()
        var = Y[n:].var().item()
    return {"val_mse": mse, "r2": 1 - mse / var, "bytes": codec.total_bytes()}
