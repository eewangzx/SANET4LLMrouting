"""Small deterministic AE, frozen-latent predictor and categorical PPO.

No SANet weights or pretrained quality claims are used. The AE+PPO implementation
is a baseline foundation; event triggers are fixed rules, not a learned policy.
"""

from dataclasses import replace

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .environment import RoutingEnv
from .routers import HeuristicRouter, policy_features, routing_scores
from .telemetry import RawCodec


class StateAutoencoder(nn.Module):
    def __init__(self, history=8, features=16, latent_dim=16, prediction_steps=20):
        super().__init__()
        self.history, self.features = history, features
        self.latent_dim, self.prediction_steps = latent_dim, prediction_steps
        dim = history * features
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("scale", torch.ones(dim))
        self.encoder = nn.Sequential(nn.Linear(dim, 96), nn.GELU(), nn.Linear(96, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 96), nn.GELU(), nn.Linear(96, dim))
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, prediction_steps * 2), nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder((x.reshape(-1, self.history * self.features) - self.mean) / self.scale)

    def decode(self, z):
        return self.decoder(z) * self.scale + self.mean

    def forecast(self, z):
        return self.predictor(z).reshape(-1, self.prediction_steps, 2) * 1.2


class AECodec:
    codec_id = 2

    def __init__(self, model):
        self.model = model.eval()
        self.latent_dim = self.payload_dim = model.latent_dim

    def encode(self, history):
        with torch.no_grad():
            return self.model.encode(torch.as_tensor(history.copy()).float())[0].numpy().copy()

    def decode(self, payload):
        with torch.no_grad():
            return (
                self.model.decode(torch.as_tensor(payload).float()[None])[0]
                .numpy()
                .reshape(self.model.history, self.model.features)
                .copy()
            )

    def predict(self, payload):
        with torch.no_grad():
            return self.model.forecast(torch.as_tensor(payload).float()[None])[0].numpy().copy()

    def embedding(self, payload):
        return payload.copy()


class RawAECodec(RawCodec):
    """Send raw history; run the SAME encoder centrally after receipt.

    Allows one frozen policy to isolate transmission effects without architecture changes.
    The raw received state remains available to receiver-side heuristics/features.
    """

    def __init__(self, model):
        super().__init__(model.history, model.features)
        self.ae = AECodec(model)
        self.latent_dim = model.latent_dim

    def embedding(self, payload):
        return self.ae.encode(self.decode(payload))

    def predict(self, payload):
        return self.ae.predict(self.embedding(payload))


def collect_state_data(config, seeds, profiles, stride=2):
    states, futures = [], []
    for seed in seeds:
        env = RoutingEnv(replace(config, seed=int(seed)), reporting="instant", profiles=profiles)
        router = HeuristicRouter()
        rng = np.random.default_rng(seed + 123456)
        for tick in range(env.total_steps):
            if tick < config.arrival_steps - config.prediction_steps and tick % stride == 0:
                states.extend(env.history.copy().reshape(6, -1))
                futures.extend(
                    env.capacity[tick + 1 : tick + 1 + config.prediction_steps].transpose(1, 0, 2)
                )
            obs = env.observe()
            if obs["request"] is None:
                action = None
            elif rng.random() < 0.25:
                action = int(rng.integers(6))
            else:
                action = router.select(obs)
            env.step(action)
    return np.asarray(states, dtype=np.float32), np.asarray(futures, dtype=np.float32)


def train_representation(config, profiles, latent_dim=16, epochs=20, seed=7):
    torch.manual_seed(seed)
    train_x, train_y = collect_state_data(config, range(1000, 1008), profiles)
    valid_x, valid_y = collect_state_data(config, (2000, 2001), profiles)
    model = StateAutoencoder(
        config.history, RoutingEnv.n_features, latent_dim, config.prediction_steps
    )
    x, y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    vx, vy = torch.from_numpy(valid_x), torch.from_numpy(valid_y)
    model.mean.copy_(x.mean(0))
    model.scale.copy_(x.std(0).clamp_min(0.05))
    optim = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=1e-3
    )
    logs = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x))
        for idx in order.split(256):
            z = model.encode(x[idx])
            rec = model.decode(z)
            loss = (((rec - x[idx]) / model.scale) ** 2).mean() + 1e-4 * z.square().mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
        with torch.no_grad():
            val = (((model.decode(model.encode(vx)) - vx) / model.scale) ** 2).mean().item()
        logs.append({"epoch": epoch + 1, "validation_normalized_mse": val})
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(
                f"AE epoch {epoch + 1}/{epochs}: held-out reconstruction MSE={val:.4f}", flush=True
            )
    # A distinct predictor trained with encoder frozen: prediction on/off uses
    # identical transmitted representations, separating the two effects.
    with torch.no_grad():
        z, vz = model.encode(x).detach(), model.encode(vx).detach()
    optim = torch.optim.Adam(model.predictor.parameters(), lr=1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(z)).split(256):
            loss = (model.forecast(z[idx]) - y[idx]).square().mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
    with torch.no_grad():
        prediction_mse = (model.forecast(vz) - vy).square().mean().item()
        persistence = vx.reshape(-1, config.history, RoutingEnv.n_features)[:, -1, :2]
        persistence_mse = (persistence[:, None, :] - vy).square().mean().item()
    print(
        f"Forecast held-out MSE={prediction_mse:.4f}; persistence={persistence_mse:.4f}", flush=True
    )
    model.eval().requires_grad_(False)
    return model, {
        "train_samples": len(x),
        "validation_samples": len(vx),
        "training_trace_seeds": list(range(1000, 1008)),
        "validation_trace_seeds": [2000, 2001],
        "ae_epochs": logs,
        "prediction_mse": prediction_mse,
        "persistence_mse": persistence_mse,
    }


class ActorCritic(nn.Module):
    def __init__(self, input_dim, quality_guard=False, separate_value=False):
        super().__init__()
        self.input_dim = input_dim
        self.quality_guard, self.separate_value = quality_guard, separate_value
        self.body = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, 1)
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        if separate_value:
            self.value_body = nn.Sequential(
                nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
            )

    def eligible(self, x):
        # Last 12 features are five public profile fields + seven query fields.
        # Uses predicted quality only, NEVER an actual future answer label.
        mask = x[..., -12] >= x[..., -1]
        return torch.where(mask.any(-1, keepdim=True), mask, torch.ones_like(mask))

    def forward(self, x):
        h = self.body(x)
        logits = self.actor(h).squeeze(-1)
        if self.quality_guard:
            logits = logits.masked_fill(~self.eligible(x), -1e9)
        value_h = self.value_body(x) if self.separate_value else h
        pooled = torch.cat((value_h.mean(dim=-2), value_h.max(dim=-2).values), dim=-1)
        return logits, self.critic(pooled).squeeze(-1)


class PolicyRouter:
    def __init__(self, model, predictive=True):
        self.model, self.predictive = model.eval(), predictive

    def select(self, observation):
        if observation["request"] is None:
            return None
        with torch.no_grad():
            logits, _ = self.model(
                torch.from_numpy(policy_features(observation, self.predictive))[None]
            )
        return int(logits[0].argmax())


def warm_start_policy(policy, ae, config, profiles, epochs=8):
    """Imitation of a causal heuristic; never of an oracle using true state."""
    inputs, targets = [], []
    for seed in range(3000, 3006):
        env = RoutingEnv(replace(config, seed=seed), AECodec(ae), "event", profiles)
        teacher = HeuristicRouter(predictive=True)
        for _ in range(env.total_steps):
            obs = env.observe()
            if obs["request"] is not None:
                inputs.append(policy_features(obs))
                targets.append(routing_scores(obs, predictive=True))
            env.step(teacher.select(obs))
    x, target = torch.from_numpy(np.asarray(inputs)), torch.from_numpy(np.asarray(targets))
    target = (target * 5).softmax(-1)
    if policy.quality_guard:
        target = target * policy.eligible(x)
        target = target / target.sum(-1, keepdim=True).clamp_min(1e-9)
    optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(x)).split(128):
            logits, _ = policy(x[idx])
            loss = -(target[idx] * logits.log_softmax(-1)).sum(-1).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
    print(f"Policy warm start: {len(x)} causal-heuristic examples", flush=True)
    return {"samples": len(x), "trace_seeds": list(range(3000, 3006)), "epochs": epochs}


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        continuing = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * continuing - values[t]
        gae = delta + gamma * lam * continuing * gae
        advantages[t] = gae
    return advantages, advantages + np.asarray(values, dtype=np.float32)


def train_ppo(ae, config, profiles, steps=16384, seed=7, rollout_steps=1024, legacy_policy=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    next_seed = 4000

    def new_env():
        nonlocal next_seed
        cfg = replace(config, seed=next_seed, report_bps=float(rng.choice([32000, 64000, 256000])))
        next_seed += 1
        # Both raw and latent reports are seen in training, with the same public
        # information boundary. This supports a frozen-policy communication ablation.
        codec = RawAECodec(ae) if rng.random() < 0.25 else AECodec(ae)
        reporting = "periodic" if rng.random() < 0.5 else "event"
        return RoutingEnv(cfg, codec, reporting, profiles)

    env = new_env()
    obs = env.observe()
    policy = ActorCritic(
        policy_features(obs).shape[-1],
        quality_guard=not legacy_policy,
        separate_value=not legacy_policy,
    )
    warm = warm_start_policy(policy, ae, config, profiles, epochs=8 if legacy_policy else 40)
    optim = torch.optim.Adam(policy.parameters(), lr=3e-4)
    logs, recent = [], []
    completed_steps = 0
    while completed_steps < steps:
        xs, actions, rewards, dones, values, logps, masks = [], [], [], [], [], [], []
        for _ in range(min(rollout_steps, steps - completed_steps)):
            features = policy_features(obs)
            with torch.no_grad():
                logits, value = policy(torch.from_numpy(features)[None])
                dist = Categorical(logits=logits[0])
                action = dist.sample()
                logp = dist.log_prob(action)
            active = obs["request"] is not None
            next_obs, reward, done, _ = env.step(int(action) if active else None)
            xs.append(features)
            actions.append(int(action))
            rewards.append(reward)
            dones.append(done)
            values.append(float(value[0]))
            logps.append(float(logp))
            masks.append(active)
            obs = next_obs
            if done:
                recent.append(env.results()["success_rate"])
                env = new_env()
                obs = env.observe()
        with torch.no_grad():
            _, last = policy(torch.from_numpy(policy_features(obs))[None])
        advantages, returns = compute_gae(rewards, values, dones, float(last[0]))
        x = torch.from_numpy(np.asarray(xs))
        act, old_lp = torch.tensor(actions), torch.tensor(logps)
        adv, ret, mask = (
            torch.from_numpy(advantages),
            torch.from_numpy(returns),
            torch.tensor(masks),
        )
        if mask.any():
            adv = (adv - adv[mask].mean()) / adv[mask].std(unbiased=False).clamp_min(1e-6)
        policy.train()
        for _ in range(4):
            for idx in torch.randperm(len(x)).split(128):
                logits, value = policy(x[idx])
                dist = Categorical(logits=logits)
                ratios = (dist.log_prob(act[idx]) - old_lp[idx]).exp()
                unclipped = ratios * adv[idx]
                clipped = ratios.clamp(0.8, 1.2) * adv[idx]
                present = mask[idx]
                actor_loss = (
                    -torch.minimum(unclipped, clipped)[present].mean()
                    if present.any()
                    else logits.sum() * 0
                )
                entropy = dist.entropy()[present].mean() if present.any() else logits.sum() * 0
                loss = actor_loss + 0.5 * (value - ret[idx]).square().mean() - 0.01 * entropy
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optim.step()
        completed_steps += len(x)
        row = {
            "steps": completed_steps,
            "training_success_recent": float(np.mean(recent[-5:])) if recent else None,
            "loss": float(loss.detach()),
        }
        logs.append(row)
        print(
            f"PPO {completed_steps}/{steps}: recent training success={row['training_success_recent']}",
            flush=True,
        )
    policy.eval()
    return policy, {
        "warm_start": warm,
        "updates": logs,
        "first_trace_seed": 4000,
        "last_trace_seed": next_seed - 1,
        "steps": completed_steps,
        "gamma_per_slot": 0.99,
        "gae_lambda": 0.95,
        "rollout_steps": rollout_steps,
        "quality_guard": policy.quality_guard,
        "separate_value": policy.separate_value,
    }
