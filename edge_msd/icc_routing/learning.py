"""SANet-inspired bandwidth-conditioned temporal encoding and task losses.

No SANet weights are used. The encoder receives reconstruction, future-resource,
and PPO routing gradients. Prefix transmission gives a real byte budget without
sending dense zero masks. Full histories/future labels are trainer-only data.
"""

from dataclasses import replace

import numpy as np
import torch
from scipy.special import ndtr
from torch import nn
from torch.distributions import Categorical

from .environment import RoutingEnv


def window_stats(x):
    """16 float features, all computed from the SAME causal history as the encoder."""
    if isinstance(x, torch.Tensor):
        last = x[..., -1, :]
        return torch.cat(
            (
                last[..., :7],
                last[..., 9:10],
                x[..., :2].mean(-2),
                x[..., :2].std(-2, unbiased=False),
                x[..., -1, :2] - x[..., 0, :2],
                x[..., 3:5].mean(-2),
            ),
            -1,
        )
    return np.concatenate(
        (
            x[-1, :7],
            x[-1, 9:10],
            x[:, :2].mean(0),
            x[:, :2].std(0),
            x[-1, :2] - x[0, :2],
            x[:, 3:5].mean(0),
        )
    ).astype(np.float32)


def decode_stats(z):
    if isinstance(z, torch.Tensor):
        state = torch.zeros_like(z)
    else:
        state = np.zeros(16, np.float32)
    state[..., :7], state[..., 9] = z[..., :7], z[..., 7]
    return state


class SemanticNet(nn.Module):
    def __init__(self, history=8, horizon=20, mode="semantic", use_forecast=True, latent_dim=16):
        super().__init__()
        # The sandbox CPU backend cannot initialize NNPACK; ordinary CPU convolution works.
        torch.backends.nnpack.set_flags(False)
        self.history, self.horizon, self.mode = history, horizon, mode
        self.use_forecast = use_forecast
        if type(latent_dim) is not int or not 1 <= latent_dim <= 16:
            raise ValueError("Latent dimension must be an integer from 1 to 16")
        if mode == "stats" and latent_dim != 16:
            raise ValueError("The existing statistics baseline has 16 features")
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * history, latent_dim),
        )
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, history * 16))
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, horizon * 3), nn.Sigmoid()
        )
        # Per-node features, global context, and model/request-specific public fields.
        self.node_dim = 16 + 16 + horizon * 3 + 2
        dim = self.node_dim * 2 + 15
        self.actor = nn.Sequential(
            nn.Linear(dim, 96), nn.Tanh(), nn.Linear(96, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(dim * 2, 96), nn.Tanh(), nn.Linear(96, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def encode(self, windows, k):
        shape = windows.shape[:-2]
        if self.mode == "stats":
            z = window_stats(windows)
        else:
            z = self.encoder(windows.reshape(-1, self.history, 16).transpose(1, 2)).reshape(
                *shape, self.latent_dim
            )
            # A true d-dimensional bottleneck. Interface padding preserves the same actor
            # shape; padding carries no information and is never included in wire payloads.
            z = nn.functional.pad(z, (0, 16 - self.latent_dim))
        return z * (torch.arange(16, device=z.device) < k[..., None])

    def decode(self, z):
        if self.mode == "stats":
            return decode_stats(z)[..., None, :].expand(*z.shape[:-1], self.history, 16)
        return self.decoder(z[..., :self.latent_dim]).reshape(*z.shape[:-1], self.history, 16)

    def forecast(self, z):
        return self.predictor(z[..., :self.latent_dim]).reshape(*z.shape[:-1], self.horizon, 3) * 1.2

    def forward_z(self, z, ages, present, public, slot_ms=5.0):
        z = z * present[..., None]
        state = self.decode(z)[..., -1, :]
        prior = torch.zeros_like(state)
        prior[..., 0:2], prior[..., 5:7], prior[..., 9] = 0.65, 0.65, 1.0
        state = torch.where(present[..., None] > 0, state, prior)
        pred = (
            self.forecast(z)
            if self.use_forecast
            else state[..., [1, 5, 6]][..., None, :].expand(*state.shape[:-1], self.horizon, 3)
        )
        prior_forecast = torch.full_like(pred, 0.65)
        pred = torch.where(present[..., None, None] > 0, pred, prior_forecast)
        initial = state[..., [1, 5, 6]][..., None, :]
        sequence = torch.cat((initial, pred), -2)
        index = (ages / slot_ms).long()[..., None] + torch.arange(self.horizon, device=z.device)
        index = index.clamp(0, self.horizon)
        aligned = torch.gather(sequence, -2, index[..., None].expand(*index.shape, 3))
        node = torch.cat(
            (
                z,
                state,
                aligned.flatten(-2),
                (ages / 100).clamp(max=10)[..., None],
                (1 - present)[..., None],
            ),
            -1,
        )
        local = node.repeat_interleave(3, dim=-2)
        global_context = node.mean(-2, keepdim=True).expand(*node.shape[:-2], 9, self.node_dim)
        features = torch.cat((local, global_context, public), -1)
        logits = self.actor(features).squeeze(-1)
        # Public quality predictions, never realized answer labels.
        eligible = public[..., 7] >= public[..., 14]
        eligible = torch.where(eligible.any(-1, keepdim=True), eligible, torch.ones_like(eligible))
        logits = logits.masked_fill(~eligible, -1e9)
        # Value regression must not dominate the encoder's policy/prediction gradients.
        critic_features = features.detach()
        value = self.critic(
            torch.cat((critic_features.mean(-2), critic_features.max(-2).values), -1)
        ).squeeze(-1)
        return logits, value

    def forward(self, windows, k, ages, present, public, slot_ms=5.0):
        return self.forward_z(self.encode(windows, k), ages, present, public, slot_ms)


class Codec:
    def __init__(self, mode="raw", model=None, k=16):
        self.mode, self.model, self.k = mode, model, k
        self.codec_id = {"raw": 10, "latest": 11, "stats": 12, "semantic": 13}[mode]

    def encode(self, history):
        if self.mode == "raw":
            return history.reshape(-1).copy()
        if self.mode == "latest":
            return history[-1].copy()
        if self.mode == "stats":
            return window_stats(history)
        with torch.no_grad():
            return (
                self.model.encode(torch.as_tensor(history)[None], torch.tensor([self.k]))[
                    0, : self.k
                ]
                .numpy()
                .copy()
            )

    def latent(self, payload):
        if self.mode == "raw":
            with torch.no_grad():
                return self.model.encode(
                    torch.as_tensor(payload).reshape(1, -1, 16), torch.tensor([16])
                )[0].numpy()
        z = np.zeros(16, np.float32)
        z[: len(payload)] = payload
        return z

    def decode(self, payload):
        if self.mode == "raw":
            return payload.reshape(-1, 16)[-1].copy()
        if self.mode == "latest":
            return payload.copy()
        if self.mode == "stats":
            return decode_stats(payload)
        with torch.no_grad():
            return self.model.decode(torch.as_tensor(self.latent(payload)))[-1].numpy()

    def forecast(self, payload):
        with torch.no_grad():
            return self.model.forecast(torch.as_tensor(self.latent(payload))).numpy()


def public_features(obs):
    public = np.zeros((9, 15), np.float32)
    r = obs["request"]
    if r is None:
        return public
    edges = {e: i for i, e in enumerate(obs["edges"])}
    for n in range(3):
        nominal = sum(1 / obs["edge_rate"][edges[e]] for e in obs["paths"][r.origin, n])
        for m, profile in enumerate(obs["models"]):
            row = public[n * 3 + m]
            row[r.task] = 1
            row[4:] = [
                r.input_mb / 4,
                r.output_mb / 1.5,
                r.deadline_ms / 100,
                profile["quality"][r.task],
                profile["cost"],
                obs["base_work"][r.task] * profile["work_multiplier"] / 10,
                obs["node_speed"][n],
                r.input_mb * nominal / 100,
                r.output_mb * nominal / 100,
                float(n == r.origin),
                obs["quality_min"],
            ]
    return public


def report_inputs(obs, codec):
    z, ages, present = (
        np.zeros((3, 16), np.float32),
        np.full(3, 1000, np.float32),
        np.zeros(3, np.float32),
    )
    k = np.full(3, 16, np.int64)
    for n, report in enumerate(obs["reports"]):
        if report is not None:
            z[n] = codec.latent(report["payload"])
            ages[n] = obs["now_ms"] - report["tick"] * obs["slot_ms"]
            present[n] = 1
            k[n] = len(report["payload"]) if codec.mode == "semantic" else 16
    return z, ages, present, k


class Policy:
    def __init__(self, model, codec):
        self.model, self.codec = model, codec

    def select(self, obs):
        if obs["request"] is None:
            return None
        z, ages, present, _ = report_inputs(obs, self.codec)
        with torch.no_grad():
            logits, _ = self.model.forward_z(
                *(torch.as_tensor(x)[None] for x in (z, ages, present, public_features(obs))),
                obs["slot_ms"],
            )
        return int(logits[0].argmax())


def _finish_time(amount, start, cumulative, slot):
    horizon = len(cumulative) - 1
    end = horizon * slot
    tail_rate = max(1e-6, (cumulative[-1] - cumulative[-2]) / slot)
    at_start = float(np.interp(min(start, end), np.arange(horizon + 1) * slot, cumulative))
    at_start += max(0, start - end) * tail_rate
    target = at_start + max(0, amount)
    if target >= cumulative[-1]:
        return end + (target - cumulative[-1]) / tail_rate
    index = max(1, int(np.searchsorted(cumulative, target, side="right")))
    return (index - 1) * slot + (target - cumulative[index - 1]) / (
        cumulative[index] - cumulative[index - 1]
    ) * slot


def heuristic_scores(obs, codec, predictive=False, query_only=False):
    r, H, slot = obs["request"], obs["horizon"], obs["slot_ms"]
    if r is None:
        return np.zeros(9, np.float32)
    states = np.zeros((3, 16), np.float32)
    states[:, [0, 1, 5, 6]], states[:, 9] = 0.65, 1
    rates = np.full((3, H, 3), 0.65)
    for n, report in enumerate(obs["reports"]):
        if report is None or query_only:
            continue
        states[n] = codec.decode(report["payload"])
        current = np.clip(states[n, [1, 5, 6]], 0.12, 1.2)
        rates[n] = current
        if predictive and codec.model is not None:
            pred = codec.forecast(report["payload"])
            age = int(round(obs["now_ms"] / slot - report["tick"]))
            seq = np.concatenate((current[None], pred))
            rates[n] = seq[np.minimum(age + np.arange(H), H)]
    rates = np.clip(rates, 0.12, 1.2)
    compute_curves = np.concatenate(
        (np.zeros((3, 1)), np.cumsum(rates[:, :, 0] * obs["node_speed"][:, None] * slot, axis=1)),
        axis=1,
    )
    edge_curves = {}
    for e, (a, b) in enumerate(obs["edges"]):
        j = [x for x in range(3) if x != a].index(b)
        edge_curves[a, b] = np.r_[0, np.cumsum(rates[a, :, 1 + j] * obs["edge_rate"][e] * slot)]
    scores = []
    for n in range(3):
        for model in obs["models"]:
            t = 0.0
            for edge in obs["paths"][r.origin, n]:
                backlog = max(0, states[edge[0], 4]) * 10 / 2
                t = _finish_time(r.input_mb + backlog, t, edge_curves[edge], slot)
            work = obs["base_work"][r.task] * model["work_multiplier"]
            # Existing compute can progress while this request is uploading.
            q_finish = _finish_time(max(0, states[n, 3]) * 100, 0, compute_curves[n], slot)
            t = _finish_time(work, max(t, q_finish), compute_curves[n], slot)
            for edge in obs["paths"][n, r.origin]:
                t = _finish_time(r.output_mb, t, edge_curves[edge], slot)
            quality = ndtr((model["quality"][r.task] - obs["quality_min"]) / 0.025)
            ontime = 1 / (1 + np.exp(np.clip((t / r.deadline_ms - 1) * 5, -30, 30)))
            scores.append(3 * quality * ontime - 0.08 * model["cost"] - 0.05 * t / r.deadline_ms)
    return np.asarray(scores, np.float32)


class Heuristic:
    def __init__(self, codec, predictive=False, query_only=False):
        self.codec, self.predictive, self.query_only = codec, predictive, query_only
        self.count = 0

    def select(self, obs):
        if obs["request"] is None:
            return None
        scores = heuristic_scores(obs, self.codec, self.predictive, self.query_only)
        best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-6))
        action = int(best[self.count % len(best)])
        self.count += 1
        return action


def collect_pretraining(dataset, config, seeds):
    xs, ys = [], []
    for seed in seeds:
        codec = Codec("raw")
        env = RoutingEnv(dataset, config, seed, codec, "instant")
        router = Heuristic(codec)
        collected = -1
        while env.tick < env.total_steps:
            if env.tick != collected and env.tick < config.arrival_steps:
                xs.extend(env.history.copy())
                ys.extend(
                    env.targets[env.tick + 1 : env.tick + 1 + config.horizon]
                    .transpose(1, 0, 2)
                    .copy()
                )
                collected = env.tick
            obs = env.observe()
            env.step(router.select(obs))
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def pretrain(model, train, valid, epochs=20, seed=7, prefix_sizes=None):
    x, y = [torch.as_tensor(a) for a in train]
    vx, vy = [torch.as_tensor(a) for a in valid]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    prefixes = tuple(prefix_sizes or (4, 8, 16))
    if any(p not in (4, 8, 16) or p > model.latent_dim for p in prefixes):
        raise ValueError("Pretraining prefixes must fit the model bottleneck")
    eval_k = max(prefixes)
    for epoch in range(epochs):
        for idx in torch.randperm(len(x), generator=generator).split(128):
            k = torch.full((len(idx),), 16)
            if model.mode != "stats":
                k = torch.tensor(prefixes)[
                    torch.randint(len(prefixes), (len(idx),), generator=generator)
                ]
            z = model.encode(x[idx], k)
            # Ordinary AE baseline: reconstruct first, predictor trained separately below.
            loss = (
                ((model.decode(z) - x[idx]) ** 2).mean()
                if model.mode != "stats"
                else ((model.forecast(z) - y[idx]) ** 2).mean()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        z = model.encode(x, torch.full((len(x),), eval_k)).detach()
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), generator=generator).split(128):
            loss = ((model.forecast(z[idx]) - y[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        vz = model.encode(vx, torch.full((len(vx),), eval_k))
        rec = ((model.decode(vz) - vx) ** 2).mean().item()
        pred = ((model.forecast(vz) - vy) ** 2).mean().item()
        persistence = ((vx[:, -1, [1, 5, 6]][:, None] - vy) ** 2).mean().item()
    return {"reconstruction_mse": rec, "forecast_mse": pred, "persistence_mse": persistence}


def episode_samples(env, model, codec, teacher=False):
    rows = []
    while env.tick < env.total_steps:
        obs = env.observe()
        z, ages, present, k = report_inputs(obs, codec)
        public = public_features(obs)
        windows, targets = env.training_windows()
        with torch.no_grad():
            logits, value = model.forward_z(
                *(torch.as_tensor(a)[None] for a in (z, ages, present, public)), obs["slot_ms"]
            )
            dist = Categorical(logits=logits[0])
            action = int(dist.sample()) if obs["request"] is not None else 0
            if teacher and obs["request"] is not None:
                scores = heuristic_scores(obs, codec, model.use_forecast)
                scores[logits[0].numpy() < -1e8] = -np.inf
                action = int(scores.argmax())
            logprob = float(dist.log_prob(torch.tensor(action)))
        _, reward, done, info = env.step(action if obs["request"] is not None else None)
        rows.append(
            dict(
                windows=windows,
                targets=targets,
                ages=ages,
                present=present,
                k=k,
                public=public,
                action=action,
                logprob=logprob,
                value=float(value),
                reward=reward / 10,
                done=done,
                elapsed=info["elapsed_ms"],
                mask=obs["request"] is not None,
            )
        )
    return rows


def advantages(rows, slot_ms, gamma=0.99, lam=0.95):
    adv = np.zeros(len(rows), np.float32)
    carry = 0.0
    for i in range(len(rows) - 1, -1, -1):
        row = rows[i]
        alive = float(not row["done"])
        next_value = rows[i + 1]["value"] if alive and i + 1 < len(rows) else 0.0
        discount = gamma ** (row["elapsed"] / slot_ms)
        trace = lam ** (row["elapsed"] / slot_ms)
        delta = row["reward"] + discount * next_value * alive - row["value"]
        carry = delta + discount * trace * alive * carry
        adv[i] = carry
    return adv, adv + np.asarray([r["value"] for r in rows], np.float32)


def tensors(rows):
    return {key: torch.as_tensor(np.asarray([r[key] for r in rows])) for key in rows[0]}


def train_policy(
    model, dataset, config, updates=10, seed=7, joint=False,
    reporting="event", report_periods=None, bandwidths=None, prefix_sizes=None,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    periods = tuple(report_periods or (config.report_period_ms,))
    rates = tuple(bandwidths or (32000, 64000, 256000))
    prefixes = tuple(prefix_sizes or (4, 8, 16))
    if reporting not in ("periodic", "event"):
        raise ValueError("Training requires periodic or event reporting")
    if any(not np.isfinite(p) or p <= 0 for p in (*periods, *rates)):
        raise ValueError("Training periods and bandwidths must be positive")
    if any(p not in (4, 8, 16) for p in prefixes):
        raise ValueError("Training prefixes must be 4, 8, or 16")
    if model.mode != "stats" and any(p > model.latent_dim for p in prefixes):
        raise ValueError("Training prefix exceeds the model bottleneck")
    for component in (model.encoder, model.decoder):
        component.requires_grad_(joint and model.mode != "stats")
    codec_mode = "stats" if model.mode == "stats" else "semantic"
    # Warm start uses the same causal reports and forecast-based heuristic.
    warm = []
    warm_prefixes = []
    for s in (3000, 3001):
        warm_k = (
            prefixes[(s - 3000) % len(prefixes)]
            if codec_mode != "stats" and prefix_sizes is not None else 16
        )
        warm_prefixes.append(warm_k)
        codec = Codec(codec_mode, model, warm_k)
        cfg = replace(config, report_period_ms=periods[(s - 3000) % len(periods)])
        env = RoutingEnv(dataset, cfg, s, codec, reporting)
        warm.extend(episode_samples(env, model, codec, True))
    w = tensors([r for r in warm if r["mask"]])
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    for _ in range(20):
        for idx in torch.randperm(len(w["action"])).split(128):
            logits, _ = model(
                *(w[k][idx] for k in ("windows", "k", "ages", "present", "public")), config.slot_ms
            )
            loss = nn.functional.cross_entropy(logits, w["action"][idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    logs = []
    for update in range(updates):
        rows = []
        success = []
        for episode in range(2):
            rate = float(rng.choice(rates))
            k = 16 if codec_mode == "stats" else int(rng.choice(prefixes))
            codec = Codec(codec_mode, model, k)
            period = periods[(2 * update + episode) % len(periods)]
            cfg = replace(config, report_bps=rate, report_period_ms=period)
            env = RoutingEnv(dataset, cfg, 4000 + 2 * update + episode, codec, reporting)
            rows.extend(episode_samples(env, model, codec))
            success.append(env.results()["success_rate"])
        data = tensors(rows)
        adv, returns = advantages(rows, config.slot_ms)
        adv, returns = torch.as_tensor(adv), torch.as_tensor(returns)
        mask = data["mask"]
        adv = (adv - adv[mask].mean()) / adv[mask].std(unbiased=False).clamp_min(1e-6)
        for _ in range(4):
            for idx in torch.randperm(len(rows)).split(128):
                logits, value = model(
                    *(data[k][idx] for k in ("windows", "k", "ages", "present", "public")),
                    config.slot_ms,
                )
                dist = Categorical(logits=logits)
                ratio = (dist.log_prob(data["action"][idx]) - data["logprob"][idx]).exp()
                active = mask[idx]
                policy = (
                    -torch.minimum(ratio * adv[idx], ratio.clamp(0.8, 1.2) * adv[idx])[
                        active
                    ].mean()
                    if active.any()
                    else value.sum() * 0
                )
                value_loss = 0.5 * (value - returns[idx]).square().mean()
                entropy = dist.entropy()[active].mean() if active.any() else value.sum() * 0
                z = model.encode(data["windows"][idx], data["k"][idx])
                valid = data["present"][idx]
                pred_error = (model.forecast(z) - data["targets"][idx]).square().mean((-1, -2))
                pred_loss = (
                    (pred_error * valid).sum() / valid.sum().clamp_min(1)
                    if model.use_forecast else z.sum() * 0
                )
                rec_error = (model.decode(z) - data["windows"][idx]).square().mean((-1, -2))
                rec_loss = (rec_error * valid).sum() / valid.sum().clamp_min(1)
                loss = (
                    policy
                    + value_loss
                    - 0.01 * entropy
                    + 0.5 * pred_loss
                    + (0.1 * rec_loss if joint else 0)
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        logs.append(
            {
                "update": update + 1,
                "success_train": float(np.mean(success)),
                "loss": float(loss.detach()),
            }
        )
        if (update + 1) % 2 == 0:
            print(
                f"{model.mode} joint={joint} update {update + 1}/{updates}: train success {np.mean(success):.3f}",
                flush=True,
            )
    return {
        "joint_encoder_training": joint,
        "warm_examples": len(w["action"]),
        "updates": logs,
        "training_seeds": [4000, 4000 + 2 * updates - 1],
        "gamma_per_slot": 0.99,
        "lambda_per_slot": 0.95,
        "reporting": reporting,
        "report_periods_ms": list(periods),
        "bandwidths": list(rates),
        "prefix_sizes": list(prefixes),
        "warm_prefix_sizes": warm_prefixes,
        "latent_dimension": model.latent_dim,
        "explicit_forecast": model.use_forecast,
        "note": "zero-duration actions within a Poisson batch use discount=1; updates only between full episodes",
    }
