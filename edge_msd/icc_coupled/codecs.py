"""Causal, explicitly serialized telemetry codecs for the ICC coupled simulator.

Each source observes eight samples of sixteen features. The first nine are light
service-rate multipliers, the next five are ordered outgoing-link multipliers
(absent links have multiplier one), and the last two are past-window summaries.
Targets contain the first fourteen quantities at the sample time (index zero)
and future sample times. No forecast API accepts the source window or labels.

All wire words are float32. Protocol headers are deliberately NOT included in
``payload_bytes``; the reporting simulator must add its common header once.
The importance codec sends three selected values and one exact numeric bitmask.
Its SANet-inspired soft budget penalty is not an information-bottleneck loss.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

KINDS = ("raw", "stats16", "stats4", "dense4", "importance", "ae4")


class CoupledCodec(nn.Module):
    """One source-side encoder and a receiver-side current/future predictor.

    ``forward`` is differentiable (a straight-through mask for importance).
    ``encode_wire`` and ``forecast_wire`` are independent deployed operations.
    A horizon of 61, at sample_ms=5, spans current state through +300 ms.
    """

    def __init__(
        self, kind="dense4", history=8, features=16, horizon=61, targets=14, sample_ms=5.0
    ):
        super().__init__()
        kind = "dense4" if kind == "predictive" else kind
        if kind not in KINDS:
            raise ValueError(f"Unknown codec {kind!r}; expected {KINDS}")
        if features != 16 or targets != 14 or history < 2 or horizon < 1:
            raise ValueError("Requires 16 input features, 14 targets, history >= 2, horizon >= 1")
        if sample_ms <= 0:
            raise ValueError("sample_ms must be positive")
        self.name = self.kind = kind
        self.history, self.features = int(history), int(features)
        self.horizon, self.targets = int(horizon), int(targets)
        self.sample_ms = float(sample_ms)
        self.latent_dim = {
            "raw": history * features, "stats16": 16, "stats4": 4,
            "dense4": 4, "importance": 8, "ae4": 4,
        }[kind]
        self.payload_dim = 4 if kind == "importance" else self.latent_dim
        # A registered buffer supplies dtype/device even for parameter-free stats4.
        self.register_buffer("_anchor", torch.zeros(()))
        if kind in ("dense4", "importance", "ae4"):
            self.encoder = nn.Sequential(
                nn.Flatten(start_dim=1), nn.Linear(history * features, 96), nn.GELU(),
                nn.Linear(96, 48), nn.GELU(), nn.Linear(48, self.latent_dim), nn.Tanh(),
            )
        if kind == "importance":
            self.importance = nn.Linear(8, 8)
            nn.init.normal_(self.importance.weight, std=0.05)
            nn.init.constant_(self.importance.bias, math.log(3 / 5))
        if kind == "ae4":
            self.reconstructor = nn.Sequential(
                nn.Linear(4, 64), nn.GELU(), nn.Linear(64, history * features),
            )
        if kind != "stats4":
            self.predictor = nn.Sequential(
                nn.Linear(self.latent_dim, 96), nn.GELU(),
                nn.Linear(96, horizon * targets), nn.Sigmoid(),
            )

    @property
    def wire_floats(self):
        return self.payload_dim

    @property
    def payload_bytes(self):
        return self.payload_dim * np.dtype(np.float32).itemsize

    def config(self):
        return {
            "kind": self.kind, "history": self.history, "features": self.features,
            "horizon": self.horizon, "targets": self.targets, "sample_ms": self.sample_ms,
        }

    def _validate_x(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.history, self.features):
            raise ValueError(f"Expected [batch, {self.history}, {self.features}], got {x.shape}")

    def representation(self, x):
        """Return receiver latent, soft scores and a hard-forward selection mask."""
        self._validate_x(x)
        if self.kind == "raw":
            z = x.flatten(start_dim=1)
        elif self.kind == "stats16":
            z = x[:, -1, :]
        elif self.kind == "stats4":
            # Link means include all five slots, INCLUDING absent-link padding.
            groups = torch.stack((x[:, :, :9].mean(-1), x[:, :, 9:14].mean(-1)), -1)
            time = torch.arange(self.history, dtype=x.dtype, device=x.device)
            time = time - time.mean()
            slopes = (groups * time[None, :, None]).sum(1) / time.square().sum()
            z = torch.cat((groups[:, -1], slopes), -1)
        else:
            z = self.encoder(x)
        if self.kind != "importance":
            return z, None, None
        scores = torch.sigmoid(self.importance(z))
        indices = scores.topk(3, dim=-1).indices
        hard = torch.zeros_like(scores).scatter(-1, indices, 1)
        mask = hard + (scores - scores.detach())
        return z * mask, scores, mask

    def forecast_latent(self, z):
        """Predict using only the representation reconstructed at the receiver."""
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"Expected [batch, {self.latent_dim}] latent, got {z.shape}")
        if self.kind == "stats4":
            steps = torch.arange(self.horizon, dtype=z.dtype, device=z.device)
            groups = z[:, None, :2] + steps[None, :, None] * z[:, None, 2:]
            return torch.cat(
                (groups[:, :, 0:1].expand(-1, -1, 9), groups[:, :, 1:2].expand(-1, -1, 5)),
                dim=-1,
            ).clamp(0.15, 1.5)
        forecast = self.predictor(z).reshape(-1, self.horizon, self.targets)
        return 0.15 + 1.35 * forecast

    def forward(self, x):
        return self.forecast_latent(self.representation(x)[0])

    def forecast_loss(self, x, y):
        """Prediction MSE and optional soft importance-budget regularization."""
        z, scores, _ = self.representation(x)
        mse = nn.functional.mse_loss(self.forecast_latent(z), y)
        penalty = mse.new_zeros(())
        if scores is not None:
            # Applying this to the hard cardinality would produce a constant.
            penalty = torch.exp((scores.sum(-1) - 3).clamp(-8, 8)).mean()
        return mse + 0.001 * penalty, mse, penalty

    @torch.no_grad()
    def encode_wire(self, history):
        """Encode one past/current source window as fresh float32 wire words."""
        history = np.asarray(history, dtype=np.float32)
        if history.shape != (self.history, self.features) or not np.isfinite(history).all():
            raise ValueError(f"Expected finite source window [{self.history}, {self.features}]")
        x = torch.as_tensor(history, dtype=self._anchor.dtype, device=self._anchor.device)[None]
        z, _, mask = self.representation(x)
        z = z[0].cpu().numpy()
        if self.kind != "importance":
            return z.astype(np.float32, copy=True)
        indices = np.flatnonzero(mask[0].cpu().numpy() > 0.5)
        bitmask = sum(1 << int(i) for i in indices)
        # Numeric integers from 0 to 255 are represented exactly by float32.
        return np.concatenate((np.array([bitmask], np.float32), z[indices])).astype(np.float32)

    def decode_wire(self, payload):
        """Decode a packet, checking exact length, finite values and sparse mask."""
        payload = np.asarray(payload, dtype=np.float32)
        if payload.shape != (self.payload_dim,) or not np.isfinite(payload).all():
            raise ValueError(f"Expected finite float32 payload [{self.payload_dim}]")
        if self.kind != "importance":
            return payload.copy()
        code = int(payload[0])
        if code != float(payload[0]) or not 0 <= code < 256 or code.bit_count() != 3:
            raise ValueError("Importance payload must have an exact 8-bit mask with three set bits")
        indices = [i for i in range(8) if code & (1 << i)]
        z = np.zeros(8, dtype=np.float32)
        z[indices] = payload[1:]
        return z

    @torch.no_grad()
    def forecast_wire(self, payload):
        """Receiver prediction from the packet alone; no source history is used."""
        z = torch.as_tensor(
            self.decode_wire(payload), dtype=self._anchor.dtype, device=self._anchor.device
        )[None]
        return self.forecast_latent(z)[0].cpu().numpy().astype(np.float32, copy=True)

    def encoding_hash(self):
        """Stable hash of the encoder protocol and source-side learned weights."""
        digest = hashlib.sha256(json.dumps(self.config(), sort_keys=True).encode())
        for key, value in sorted(self.state_dict().items()):
            if key.startswith(("encoder.", "importance.")):
                digest.update(key.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"format_version": 1, "config": self.config(), "state_dict": self.state_dict(),
             "encoding_hash": self.encoding_hash()}, path,
        )

    @classmethod
    def load(cls, path, device="cpu"):
        data = torch.load(path, map_location=device, weights_only=True)
        if data.get("format_version") != 1:
            raise ValueError("Unsupported codec format")
        model = cls(**data["config"]).to(device)
        model.load_state_dict(data["state_dict"])
        if model.encoding_hash() != data["encoding_hash"]:
            raise ValueError("Encoder hash does not match checkpoint")
        if model.kind == "ae4":
            model.encoder.requires_grad_(False)
        return model.eval()


def load_codec(path, device="cpu"):
    return CoupledCodec.load(path, device=device)


def _array_tensor(value, name):
    value = np.asarray(value, dtype=np.float32)
    if not value.size or not np.isfinite(value).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return torch.from_numpy(value)


@torch.no_grad()
def _evaluate(model, x, y, reconstruction=False, batch_size=512):
    squared = []
    current, future, counts = 0.0, 0.0, 0
    for start in range(0, len(x), batch_size):
        xb, yb = x[start:start + batch_size], y[start:start + batch_size]
        if reconstruction:
            pred = model.reconstructor(model.encoder(xb)).reshape_as(xb)
            error = (pred - xb).square()
        else:
            error = (model(xb) - yb).square()
            current += float(error[:, 0].sum())
            if model.horizon > 1:
                future += float(error[:, 1:].sum())
            counts += len(xb)
        squared.append((float(error.sum()), error.numel()))
    result = {"mse": sum(v for v, _ in squared) / sum(n for _, n in squared)}
    if not reconstruction:
        result.update({"current_mse": current / (counts * model.targets),
                       "future_mse": future / (counts * max(1, model.horizon - 1) * model.targets)})
    return result


def _fit_stage(model, train_x, train_y, val_x, val_y, epochs, batch_size, seed, reconstruction):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    curve, best_state, best_mse, best_epoch = [], None, float("inf"), 0
    initial = _evaluate(model, val_x, val_y, reconstruction=reconstruction)
    best_mse = initial["mse"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        loss_sum, count = 0.0, 0
        for indices in order.split(batch_size):
            xb, yb = train_x[indices], train_y[indices]
            if reconstruction:
                pred = model.reconstructor(model.encoder(xb)).reshape_as(xb)
                loss = nn.functional.mse_loss(pred, xb)
            else:
                loss = model.forecast_loss(xb, yb)[0]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            count += len(indices)
        model.eval()
        validation = _evaluate(model, val_x, val_y, reconstruction=reconstruction)
        curve.append({"epoch": epoch, "train_loss": loss_sum / count,
                      "validation": validation})
        if validation["mse"] < best_mse:
            best_mse, best_epoch = validation["mse"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"initial_validation": initial, "selected_epoch": best_epoch, "curve": curve}


def train_codecs(
    train_x, train_y, val_x, val_y, output, seed=7, epochs=60, batch_size=128,
    kinds=None, sample_ms=5.0,
):
    """Train all codecs on caller-supplied, trajectory-separated data splits.

    Returns a ``{name: CoupledCodec}`` dictionary and writes ``name.pt`` plus
    ``metrics.json``. Input shapes are [N,H,16] and [N,F,14]; H/F are inferred.
    Labels must start at the final input sample time. The caller is responsible
    for splitting by independent trace, not overlapping random windows.
    Every learned method has ``epochs`` forecast epochs. AE additionally has
    ``epochs`` history-reconstruction epochs, then its encoder stays frozen.
    Validation chooses each stage's checkpoint; no evaluation/test trace is used.
    """
    if epochs < 0 or batch_size < 1:
        raise ValueError("epochs must be nonnegative and batch_size positive")
    train_x = _array_tensor(train_x, "train_x")
    train_y = _array_tensor(train_y, "train_y")
    val_x = _array_tensor(val_x, "val_x")
    val_y = _array_tensor(val_y, "val_y")
    if (train_x.ndim != 3 or train_y.ndim != 3 or val_x.ndim != 3 or val_y.ndim != 3
            or train_x.shape[1:] != val_x.shape[1:] or train_y.shape[1:] != val_y.shape[1:]
            or len(train_x) != len(train_y) or len(val_x) != len(val_y)):
        raise ValueError("Inconsistent training/validation tensor shapes")
    kinds = tuple(KINDS if kinds is None else kinds)
    kinds = tuple("dense4" if k == "predictive" else k for k in kinds)
    if len(set(kinds)) != len(kinds) or any(k not in KINDS for k in kinds):
        raise ValueError("kinds must contain unique supported codec names")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "seed": int(seed), "epochs_per_stage": int(epochs), "sample_ms": float(sample_ms),
        "train_windows": len(train_x), "validation_windows": len(val_x),
        "train_x_shape": list(train_x.shape), "train_y_shape": list(train_y.shape),
        "protocol": "float32 payload, excludes common simulator-charged header",
        "target_time_zero": "last input sample; index h predicts h * sample_ms later",
        "stats4_links": "mean of all five ordered slots, including absent-link padding one",
        "methods": {},
    }
    models = {}
    for kind in kinds:
        # Same seed independent of requested method subset/order.
        torch.manual_seed(seed)
        model = CoupledCodec(
            kind, history=train_x.shape[1], features=train_x.shape[2],
            horizon=train_y.shape[1], targets=train_y.shape[2], sample_ms=sample_ms,
        )
        record = {"payload_floats": model.payload_dim, "payload_bytes": model.payload_bytes,
                  "parameters": sum(p.numel() for p in model.parameters())}
        if kind == "ae4":
            record["reconstruction"] = _fit_stage(
                model, train_x, train_y, val_x, val_y, epochs, batch_size, seed, True
            )
            model.encoder.requires_grad_(False)
            model.reconstructor.requires_grad_(False)
            record["encoder_hash_before_forecast"] = model.encoding_hash()
        if kind != "stats4":
            record["forecast_training"] = _fit_stage(
                model, train_x, train_y, val_x, val_y, epochs, batch_size, seed + 1, False
            )
        model.eval()
        record["training"] = _evaluate(model, train_x, train_y)
        record["validation"] = _evaluate(model, val_x, val_y)
        record["encoding_hash"] = model.encoding_hash()
        if kind == "ae4" and record["encoder_hash_before_forecast"] != record["encoding_hash"]:
            raise RuntimeError("AE encoder changed during frozen forecast training")
        if kind == "importance":
            with torch.no_grad():
                masks = model.representation(val_x)[2].cpu().numpy()
            record["validation_unique_masks"] = len(np.unique(masks, axis=0))
        model.save(output / f"{kind}.pt")
        metrics["methods"][kind] = record
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        models[kind] = model
    return models
