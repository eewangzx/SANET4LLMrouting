"""Sequence-to-vector prediction bottlenecks and SANet-inspired top-k filtering.

The deployed actor consumes only received Z, its age/missing flag and public
request/model fields. The future-prediction head is a training auxiliary; it is
never given future samples at inference. A mask is transmitted for sparse Z.
"""

import math

import numpy as np
import torch
from torch import nn

from .environment import RoutingEnv
from .learning import Codec, window_stats


class PredictiveNet(nn.Module):
    def __init__(self, history=8, horizon=60, selection="dense", dimension=4, keep=3):
        super().__init__()
        if selection not in ("dense", "importance", "random", "stats", "adaptive"):
            raise ValueError("Unknown selection")
        if not 1 <= dimension <= 16 or not 1 <= keep <= dimension:
            raise ValueError("Invalid bottleneck dimensions")
        if selection in ("importance", "random", "adaptive") and dimension > 8:
            raise ValueError("Sparse wire format supports at most eight candidates")
        if selection == "stats" and dimension != 16:
            raise ValueError("Statistics require 16 dimensions")
        torch.backends.nnpack.set_flags(False)
        self.history, self.horizon = history, horizon
        self.selection, self.latent_dim, self.keep = selection, dimension, keep
        self.mode = "stats" if selection == "stats" else "semantic"
        self.use_forecast = True
        self.encoder = nn.Sequential(
            nn.Conv1d(16, 32, 3, padding=1), nn.GELU(),
            nn.Conv1d(32, 32, 3, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(32 * history, dimension), nn.Tanh(),
        )
        self.importance = nn.Linear(dimension, dimension)
        nn.init.normal_(self.importance.weight, std=0.05)
        nn.init.constant_(self.importance.bias, math.log(keep / max(0.5, dimension - keep)))
        self.budget_actor = nn.Sequential(nn.Linear(dimension, 16), nn.Tanh(), nn.Linear(16, 3))
        self.predictor = nn.Sequential(
            nn.Linear(dimension, 64), nn.GELU(),
            nn.Linear(64, horizon * 3), nn.Sigmoid(),
        )
        # A fixed 16-slot local interface keeps every actor architecture identical.
        # Sparse zeros/dense padding do not consume wire bytes.
        feature_dim = 2 * (16 + 2) + 15
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, 96), nn.Tanh(), nn.Linear(96, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(feature_dim * 2, 96), nn.Tanh(),
            nn.Linear(96, 64), nn.Tanh(), nn.Linear(64, 1),
        )

    @property
    def wire_floats(self):
        return self.keep + 1 if self.selection in ("importance", "random", "adaptive") else self.latent_dim

    def budget_logits(self, windows):
        shape = windows.shape[:-2]
        c = self.encoder(windows.reshape(-1, self.history, 16).transpose(1, 2))
        return self.budget_actor(c).reshape(*shape, 3)

    def representation(self, windows, wire_counts=None):
        shape = windows.shape[:-2]
        if self.selection == "stats":
            c = window_stats(windows)
        else:
            c = self.encoder(windows.reshape(-1, self.history, 16).transpose(1, 2))
            c = c.reshape(*shape, self.latent_dim)
        scores = torch.sigmoid(self.importance(c))
        mask = torch.ones_like(c)
        if self.selection == "random":
            # Reproducible, non-learned pseudorandom subset per source window.
            # Exactly repeatable between wire encoding and PPO recomputation;
            # mask is still transmitted, so the receiver needs no source history.
            flat = windows.flatten(-2).double()
            weights = torch.arange(1, flat.shape[-1] + 1, device=flat.device).double()
            phase = (flat * weights).sum(-1, keepdim=True)
            coordinate = torch.arange(self.latent_dim, device=flat.device).double()
            scores = torch.frac(torch.sin(phase * 17.17 + coordinate * 31.31) * 43758.5453)
            scores = scores.abs().to(c.dtype)
        if self.selection in ("importance", "random", "adaptive"):
            counts = torch.full(shape, self.keep, device=c.device)
            if self.selection == "adaptive" and wire_counts is not None:
                counts = (wire_counts - 1).clamp(1, 3)
            ranks = scores.argsort(dim=-1, descending=True).argsort(dim=-1)
            hard = (ranks < counts[..., None]).to(c.dtype)
            # Forward uses exactly k entries, while backward reaches the scorer.
            mask = hard + (scores - scores.detach()) if self.selection in ("importance", "adaptive") else hard
        z = nn.functional.pad(c * mask, (0, 16 - self.latent_dim))
        return z, scores, mask

    def encode(self, windows, k=None):
        # This experiment fixes k in the model spec. k from the legacy interface
        # counts wire words (including sparse metadata), not selected coordinates.
        return self.representation(windows, k)[0]

    def bandwidth_penalty(self, scores, counts=None):
        if self.selection not in ("importance", "adaptive"):
            return scores.sum() * 0
        # SANet paper Eq. (13), on SOFT scores. Hard top-k count is constant k
        # and would give no useful budget gradient. This is not an IB/KL loss.
        target = self.keep if counts is None else counts
        return torch.exp((scores.sum(-1) - target).clamp(-8, 8)).mean()

    def forecast(self, z):
        return self.predictor(z[..., :self.latent_dim]).reshape(*z.shape[:-1], self.horizon, 3) * 1.2

    def forward_z(self, z, ages, present, public, slot_ms=5.0):
        z = z * present[..., None]
        node = torch.cat((z, (ages / 100).clamp(max=10)[..., None], (1 - present)[..., None]), -1)
        local = node.repeat_interleave(3, dim=-2)
        global_context = node.mean(-2, keepdim=True).expand(*node.shape[:-2], 9, 18)
        features = torch.cat((local, global_context, public), -1)
        logits = self.actor(features).squeeze(-1)
        eligible = public[..., 7] >= public[..., 14]
        eligible = torch.where(eligible.any(-1, keepdim=True), eligible, torch.ones_like(eligible))
        logits = logits.masked_fill(~eligible, -1e9)
        features = features.detach()
        value = self.critic(torch.cat((features.mean(-2), features.max(-2).values), -1)).squeeze(-1)
        return logits, value

    def forward(self, windows, k, ages, present, public, slot_ms=5.0):
        return self.forward_z(self.encode(windows, k), ages, present, public, slot_ms)


class PredictiveCodec(Codec):
    """Actual float32 wire words, including a numeric bitmask for sparse vectors.

    A float32 can represent every 8-bit mask exactly. Spending four bytes on
    this field preserves the existing serializer and conservatively charges its
    full cost. Three values + one mask word + 16-byte header = 32 bytes.
    """

    def __init__(self, model, stochastic=False):
        super().__init__(model.mode, model, model.wire_floats)
        self.codec_id = {"dense": 20, "importance": 21, "random": 22, "stats": 23, "adaptive": 24}[model.selection]
        self.stochastic, self.events, self.sizes, self.keeps = stochastic, [], [], []

    def encode(self, history):
        with torch.no_grad():
            x = torch.as_tensor(history)[None]
            counts = None
            if self.model.selection == "adaptive":
                dist = torch.distributions.Categorical(logits=self.model.budget_logits(x)[0])
                action = dist.sample() if self.stochastic else dist.logits.argmax()
                counts = torch.tensor([int(action) + 2])
                if self.stochastic:
                    self.events.append({"window": np.asarray(history).copy(), "action": int(action),
                                        "logprob": float(dist.log_prob(action))})
            z, _, mask = self.model.representation(x, counts)
            z = z[0].numpy()
            if self.model.selection not in ("importance", "random", "adaptive"):
                self.sizes.append(16 + self.model.latent_dim * 4)
                return z[:self.model.latent_dim].copy()
            indices = np.flatnonzero(mask[0].numpy() > 0.5)
            code = sum(1 << int(i) for i in indices)
            self.sizes.append(20 + len(indices) * 4)
            self.keeps.append(len(indices))
            return np.r_[np.float32(code), z[indices]].astype(np.float32)

    def latent(self, payload):
        payload = np.asarray(payload, np.float32)
        valid_length = (2 <= len(payload) <= 4) if self.model.selection == "adaptive" else len(payload) == self.model.wire_floats
        if not valid_length or not np.isfinite(payload).all():
            raise ValueError("Invalid predictive payload")
        z = np.zeros(16, np.float32)
        if self.model.selection in ("importance", "random", "adaptive"):
            code = int(payload[0])
            if code != payload[0] or not 0 <= code < 2 ** self.model.latent_dim:
                raise ValueError("Invalid sparse mask")
            indices = [i for i in range(self.model.latent_dim) if code & (1 << i)]
            expected = len(payload) - 1 if self.model.selection == "adaptive" else self.model.keep
            if len(indices) != expected:
                raise ValueError("Invalid sparse cardinality")
            z[indices] = payload[1:]
        else:
            z[:len(payload)] = payload
        return z


class ReportingEnv(RoutingEnv):
    """Charge actual bytes sent, including headers, in the common RL reward."""

    def __init__(self, *args, wire_price=1.0, **kwargs):
        self.wire_price = wire_price
        super().__init__(*args, **kwargs)

    def step(self, action):
        before = self.channel.transmitted_bytes
        obs, reward, done, info = super().step(action)
        wire_cost = self.wire_price * (self.channel.transmitted_bytes - before) / 32.0
        info["wire_cost"] = wire_cost
        return obs, reward - wire_cost, done, info
