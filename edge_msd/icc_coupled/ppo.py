"""Hierarchical PPO over nine modes of the unchanged ICC deployment solver.

The actor selects eta multiplier x parallelism cap, NOT combinatorial deployment
variables. The ICC controller remains responsible for x/y/path and feasibility;
it must raise the requested cap to the largest occupied parallelism when needed.

Only received telemetry and public controller state enter ``act``. Archived
source windows are used for encoder replay, and future labels ONLY for auxiliary
training. Rollouts keep all weights fixed through complete episodes. This is a
practical clipped policy-gradient method, not a convergence or SLA guarantee.
"""

from __future__ import annotations

import copy
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .codecs import CoupledCodec

MODES = tuple((eta, cap) for eta in (0.25, 1.0, 4.0) for cap in (2, 8, 20))


def _capture_rng(policy):
    numpy_state = np.random.get_state()
    state = {"torch": torch.get_rng_state(), "action": policy.action_rng.get_state(),
             "numpy": {"name": numpy_state[0], "keys": numpy_state[1].tolist(),
                       "position": int(numpy_state[2]), "has_gaussian": int(numpy_state[3]),
                       "cached_gaussian": float(numpy_state[4])},
             "python": random.getstate()}
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(policy, state):
    required = {"torch", "action", "numpy", "python"}
    if not required.issubset(state):
        raise ValueError("Checkpoint lacks RNG state required for exact continuation")
    torch.set_rng_state(state["torch"])
    policy.action_rng.set_state(state["action"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["name"], np.asarray(numpy_state["keys"], np.uint32),
                         numpy_state["position"], numpy_state["has_gaussian"],
                         numpy_state["cached_gaussian"]))
    random.setstate(state["python"])
    if "cuda" in state:
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(state["cuda"]):
            raise ValueError("Exact continuation requires the checkpoint's CUDA device configuration")
        torch.cuda.set_rng_state_all(state["cuda"])


def _write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def _explained_variance(values, returns):
    values, returns = np.asarray(values), np.asarray(returns)
    variance = float(np.var(returns))
    return None if variance < 1e-12 else float(1 - np.var(returns - values) / variance)


def _accounting(value):
    if "accounting" in value:
        value = value["accounting"]
    fields = ("cost", "violations", "report_bytes", "time_ms")
    if not all(k in value and np.isfinite(value[k]) for k in fields):
        raise ValueError(f"Cumulative accounting must contain finite {fields}")
    return {key: float(value[key]) for key in fields}


def interval_reward(before, after):
    """Cost and original DAG-deadline failures, with actual telemetry-byte cost."""
    before, after = _accounting(before), _accounting(after)
    delta = {k: after[k] - before[k] for k in before}
    if any(delta[k] < -1e-6 for k in delta):
        raise ValueError("Episode accounting must be cumulative and monotonic")
    return -delta["cost"] / 100 - 2 * delta["violations"] - 0.01 * delta["report_bytes"] / 32


def _metric_summary(metrics):
    result = {k: v for k, v in metrics.items()
              if v is None or isinstance(v, (str, bool, int, float))}
    for key in ("overall", "costs", "accounting"):
        if isinstance(metrics.get(key), dict):
            result[key] = {k: v for k, v in metrics[key].items()
                           if v is None or isinstance(v, (str, bool, int, float))}
    return result


class _PoolHead(nn.Module):
    def __init__(self, latent_dim, outputs):
        super().__init__()
        self.nodes = nn.Sequential(nn.Linear(latent_dim + 6, 64), nn.Tanh(),
                                   nn.Linear(64, 64), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(133, 64), nn.Tanh(), nn.Linear(64, outputs))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.head[-1].weight, gain=0.01 if outputs > 1 else 1.0)

    def forward(self, latent, ages, present, public, global_state):
        latent = latent * present[..., None]
        features = torch.cat((latent, (ages / 100).clamp(0, 10)[..., None],
                              present[..., None], public), -1)
        nodes = self.nodes(features)
        pooled = torch.cat((nodes.mean(-2), nodes.max(-2).values, global_state), -1)
        return self.head(pooled)


class HierarchicalPPO(nn.Module):
    """Simulator callback plus PPO update; a codec is shared with its simulator.

    Observation fields:
      latents [N,D], ages [N] ms, present [N], public [N,4], global [5],
      accounting {cost, violations, report_bytes, time_ms, ...}.
    training_context optionally contains archived windows [N,H,16] and
    targets [N,F,14] for those *received* reports. Missing entries are masked.
    ``finish`` must include pending requests in the terminal failure accounting.
    """

    def __init__(self, codec, joint=False, seed=7, stochastic=True, gamma=0.995,
                 gae_lambda=0.995, entropy_coefficient=0.005, reward_scale=0.001,
                 gae_lambda_unit="ms"):
        super().__init__()
        self.codec = codec
        self.joint, self.seed, self.stochastic = bool(joint), int(seed), bool(stochastic)
        self.gamma, self.gae_lambda = float(gamma), float(gae_lambda)
        self.reward_scale = float(reward_scale)
        self.gae_lambda_unit = gae_lambda_unit
        self.entropy_coefficient = float(entropy_coefficient)
        if not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1:
            raise ValueError("Invalid discount or GAE lambda")
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("reward_scale must be a fixed finite positive number")
        if gae_lambda_unit not in ("ms", "decision"):
            raise ValueError("gae_lambda_unit must be ms or legacy decision")
        torch.manual_seed(seed)
        self.actor, self.critic = _PoolHead(codec.latent_dim, 9), _PoolHead(codec.latent_dim, 1)
        self.codec.requires_grad_(joint)
        if self.codec.kind == "ae4":
            # This baseline always retains its reconstruction-trained encoder.
            self.codec.encoder.requires_grad_(False)
            self.codec.reconstructor.requires_grad_(False)
        self.action_rng = torch.Generator().manual_seed(seed)
        self.optimizer = None
        self.update_number = 0
        self.reset_episode()

    def reset_episode(self, stochastic=None, action_seed=None):
        if stochastic is not None:
            self.stochastic = bool(stochastic)
        if action_seed is not None:
            self.action_rng.manual_seed(int(action_seed))
        self.transitions = []
        self.episode_finished = False
        self.episode_metrics = None

    def forward(self, latents, ages, present, public, global_state):
        logits = self.actor(latents, ages, present, public, global_state)
        # Independent value network; value fitting never changes the encoder.
        values = self.critic(latents.detach(), ages, present, public, global_state).squeeze(-1)
        return logits, values

    def _observation(self, observation):
        names = ("latents", "ages", "present", "public", "global")
        result = {key: np.asarray(observation[key], np.float32).copy() for key in names}
        n = len(result["latents"])
        shapes = {"latents": (n, self.codec.latent_dim), "ages": (n,),
                  "present": (n,), "public": (n, 4), "global": (5,)}
        if n == 0 or any(result[k].shape != shapes[k] or not np.isfinite(result[k]).all()
                         for k in names):
            raise ValueError(f"Invalid policy observation; expected {shapes}")
        if not np.isin(result["present"], [0, 1]).all():
            raise ValueError("present must be a binary reception mask")
        return result

    def _close_interval(self, accounting, terminal):
        if not self.transitions:
            return
        transition = self.transitions[-1]
        if "reward" in transition:
            raise RuntimeError("Previous interval is already closed")
        transition["reward"] = interval_reward(transition["accounting"], accounting)
        elapsed = accounting["time_ms"] - transition["accounting"]["time_ms"]
        transition["elapsed_ms"] = elapsed
        transition["discount"] = self.gamma ** elapsed
        transition["terminal"] = bool(terminal)

    @torch.no_grad()
    def act(self, observation, training_context=None):
        if self.episode_finished:
            raise RuntimeError("Call reset_episode before starting another episode")
        accounting = _accounting(observation)
        self._close_interval(accounting, terminal=False)
        obs = self._observation(observation)
        tensors = {k: torch.from_numpy(v)[None] for k, v in obs.items()}
        # Critically, inference consumes the received latent, NOT windows/targets.
        logits, values = self(tensors["latents"], tensors["ages"], tensors["present"],
                              tensors["public"], tensors["global"])
        distribution = Categorical(logits=logits[0])
        if self.stochastic:
            action = int(torch.multinomial(distribution.probs, 1, generator=self.action_rng))
        else:
            action = int(logits[0].argmax())
        transition = {"observation": obs, "accounting": accounting, "action": action,
                      "logprob": float(distribution.log_prob(torch.tensor(action))),
                      "value": float(values[0]), "entropy": float(distribution.entropy())}
        # Copy these ONLY after action computation. Future labels are inaccessible
        # to the actor and do not enter any inference-network input.
        if training_context is not None:
            n = len(obs["present"])
            for key, shape in (
                ("windows", (n, self.codec.history, 16)),
                ("targets", (n, self.codec.horizon, 14)),
            ):
                if key in training_context:
                    data = np.asarray(training_context[key], np.float32).copy()
                    if data.shape != shape or not np.isfinite(data).all():
                        raise ValueError(f"training_context {key} must have shape {shape}")
                    transition[key] = data
        self.transitions.append(transition)
        return MODES[action]

    def finish(self, metrics):
        if self.episode_finished:
            raise RuntimeError("finish may only be called once per episode")
        accounting = _accounting(metrics)
        self._close_interval(accounting, terminal=True)
        self.episode_finished = True
        self.episode_metrics = _metric_summary(metrics)

    def episode(self):
        """Return completed trajectory with true terminal MC returns and GAE."""
        if not self.episode_finished or not self.transitions:
            raise RuntimeError("A nonempty, finished episode is required")
        returns, physical_returns, advantages = [], [], []
        next_return, next_value, next_advantage = 0.0, 0.0, 0.0
        next_physical_return = 0.0
        for item in reversed(self.transitions):
            g = item["discount"]
            # Optimization units are a single fixed positive rescaling of the
            # original objective. All physical metric/reward reports stay raw.
            optimization_reward = self.reward_scale * item["reward"]
            mc = optimization_reward + g * next_return
            physical_mc = item["reward"] + g * next_physical_return
            delta = optimization_reward + g * next_value - item["value"]
            lambda_discount = self.gae_lambda ** (
                item["elapsed_ms"] if self.gae_lambda_unit == "ms" else 1)
            advantage = delta + g * lambda_discount * next_advantage
            returns.append(mc)
            physical_returns.append(physical_mc)
            advantages.append(advantage)
            next_return, next_value, next_advantage = mc, item["value"], advantage
            next_physical_return = physical_mc
        returns, physical_returns, advantages = returns[::-1], physical_returns[::-1], advantages[::-1]
        for item, mc, physical_mc, advantage in zip(
                self.transitions, returns, physical_returns, advantages, strict=True):
            item["mc_return"], item["advantage"] = mc, advantage
            item["physical_mc_return"] = physical_mc
            item["optimization_reward"] = self.reward_scale * item["reward"]
        return {"transitions": self.transitions, "metrics": self.episode_metrics,
                "discounted_return": physical_returns[0],
                "optimization_discounted_return": returns[0], "reward_scale": self.reward_scale,
                "reward": sum(r["reward"] for r in self.transitions),
                "entropy": float(np.mean([r["entropy"] for r in self.transitions])),
                "mc_explained_variance": _explained_variance(
                    [r["value"] for r in self.transitions], returns),
                "actions": dict(Counter(r["action"] for r in self.transitions))}

    def _make_optimizer(self):
        groups = [{"params": list(self.actor.parameters()), "lr": 1e-4, "name": "actor"},
                  {"params": list(self.critic.parameters()), "lr": 3e-4, "name": "critic"}]
        if self.joint:
            enc, forecast = [], []
            for name, p in self.codec.named_parameters():
                if not p.requires_grad or name.startswith("reconstructor."):
                    continue
                (enc if name.startswith(("encoder.", "importance.")) else forecast).append(p)
            if enc:
                groups.append({"params": enc, "lr": 2e-5, "name": "encoder"})
            if forecast:
                groups.append({"params": forecast, "lr": 1e-4, "name": "predictor"})
        self.optimizer = torch.optim.Adam(groups)

    def _tensors(self, transitions):
        obs = {k: torch.from_numpy(np.stack([r["observation"][k] for r in transitions]))
               for k in ("latents", "ages", "present", "public", "global")}
        if self.joint:
            if not all("windows" in r and "targets" in r for r in transitions):
                raise ValueError("Joint PPO needs received-report windows and targets for every step")
            obs["windows"] = torch.from_numpy(np.stack([r["windows"] for r in transitions]))
            obs["targets"] = torch.from_numpy(np.stack([r["targets"] for r in transitions]))
        for key in ("logprob", "mc_return", "advantage", "value"):
            obs[key] = torch.tensor([r[key] for r in transitions], dtype=torch.float32)
        obs["action"] = torch.tensor([r["action"] for r in transitions], dtype=torch.long)
        return obs

    def _replay(self, batch):
        prediction_loss, filter_loss = torch.zeros(()), torch.zeros(())
        latents = batch["latents"]
        if self.joint:
            windows = batch["windows"]
            b, n = windows.shape[:2]
            encoded, scores, _ = self.codec.representation(windows.reshape(-1, self.codec.history, 16))
            latents = encoded.reshape(b, n, -1)
            present = batch["present"].flatten()
            forecast = self.codec.forecast_latent(encoded)
            targets = batch["targets"].reshape_as(forecast)
            errors = (forecast - targets).square().mean((1, 2))
            prediction_loss = (errors * present).sum() / present.sum().clamp_min(1)
            if scores is not None:
                penalty = torch.exp((scores.sum(-1) - 3).clamp(-8, 8))
                filter_loss = (penalty * present).sum() / present.sum().clamp_min(1)
        logits, values = self(latents, batch["ages"], batch["present"], batch["public"],
                              batch["global"])
        return Categorical(logits=logits), values, prediction_loss, filter_loss

    def update(self, episodes, epochs=4, minibatch_size=128, target_kl=0.02):
        """One complete-episode, on-policy update with causal encoder replay."""
        transitions = [r for ep in episodes for r in ep["transitions"]]
        if not transitions:
            raise ValueError("PPO update requires transitions")
        if self.optimizer is None:
            self._make_optimizer()
        batch = self._tensors(transitions)
        with torch.no_grad():
            distribution, values, _, _ = self._replay(batch)
            error = float((distribution.log_prob(batch["action"]) - batch["logprob"]).abs().max())
            initial_entropy = float(distribution.entropy().mean())
        if error > 1e-4:
            raise RuntimeError(f"Received-window replay does not reproduce old logprobs: {error:.6g}")
        advantage = batch["advantage"]
        batch["advantage"] = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
        rows = []
        generator = torch.Generator().manual_seed(self.seed + self.update_number * 41)
        stopped = False
        for epoch in range(epochs):
            order = torch.randperm(len(transitions), generator=generator)
            for indices in order.split(minibatch_size):
                mini = {k: v[indices] for k, v in batch.items()}
                distribution, values, prediction, filtering = self._replay(mini)
                logratio = distribution.log_prob(mini["action"]) - mini["logprob"]
                ratio = logratio.exp()
                policy_loss = -torch.minimum(ratio * mini["advantage"],
                                             ratio.clamp(0.8, 1.2) * mini["advantage"]).mean()
                value_loss = nn.functional.mse_loss(values, mini["mc_return"])
                entropy = distribution.entropy().mean()
                loss = (policy_loss + 0.5 * value_loss - self.entropy_coefficient * entropy
                        + 0.5 * prediction + 0.001 * filtering)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite PPO loss; refusing to corrupt weights")
                approx_kl = float(((ratio - 1) - logratio).mean().detach())
                if approx_kl > target_kl:
                    stopped = True
                    break
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norms = {}
                for group in self.optimizer.param_groups:
                    norms[group["name"]] = float(nn.utils.clip_grad_norm_(group["params"], 0.5))
                self.optimizer.step()
                rows.append({"policy_loss": float(policy_loss.detach()),
                             "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach()),
                             "prediction_mse": float(prediction.detach()),
                             "filter_penalty": float(filtering.detach()), "approx_kl": approx_kl,
                             "clip_fraction": float(((ratio - 1).abs() > 0.2).float().mean()),
                             "gradient_norms": norms})
            if stopped:
                break
        with torch.no_grad():
            distribution, values, prediction, _ = self._replay(batch)
            logratio = distribution.log_prob(batch["action"]) - batch["logprob"]
            final_kl = float((logratio.exp() - 1 - logratio).mean())
        self.update_number += 1
        record = {"update": self.update_number, "transitions": len(transitions),
                  "optimizer_steps": len(rows), "stopped_for_kl": stopped,
                  "replay_logprob_max_error": error, "entropy_before": initial_entropy,
                  "entropy_after": float(distribution.entropy().mean()), "approx_kl_after": final_kl,
                  "clip_fraction_after": float(((logratio.exp() - 1).abs() > 0.2).float().mean()),
                  "mc_explained_variance_before": _explained_variance(
                      batch["value"].numpy(), batch["mc_return"].numpy()),
                  "mc_explained_variance_after_fit": _explained_variance(
                      values.numpy(), batch["mc_return"].numpy()),
                  "prediction_mse_after": float(prediction),
                  "train_discounted_return": float(np.mean([ep["discounted_return"] for ep in episodes])),
                  "train_optimization_discounted_return": float(np.mean(
                      [ep["optimization_discounted_return"] for ep in episodes])),
                  "reward_scale": self.reward_scale,
                  "value_loss_units": "(physical reward * reward_scale)^2",
                  "train_reward": float(np.mean([ep["reward"] for ep in episodes])),
                  "episodes": [{k: v for k, v in ep.items() if k != "transitions"} for ep in episodes]}
        for key in ("policy_loss", "value_loss", "entropy", "prediction_mse", "filter_penalty",
                    "approx_kl", "clip_fraction"):
            record[key] = float(np.mean([row[key] for row in rows])) if rows else None
        record["gradient_norms"] = {
            group["name"]: float(np.mean([row["gradient_norms"][group["name"]] for row in rows]))
            if rows else None for group in self.optimizer.param_groups}
        return record

    def save(self, path, training_state=None):
        """Atomically save weights, optimizer, RNG and optional aligned run history.

        Exact training continuation is supported only at an episode boundary.
        A checkpoint at update zero records an uninitialized optimizer explicitly.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"format_version": 2, "codec_config": self.codec.config(),
                    "policy_config": {"joint": self.joint, "seed": self.seed, "gamma": self.gamma,
                                      "gae_lambda": self.gae_lambda,
                                      "gae_lambda_unit": self.gae_lambda_unit,
                                      "reward_scale": self.reward_scale,
                                      "entropy_coefficient": self.entropy_coefficient},
                    "state_dict": self.state_dict(), "update": self.update_number,
                    "encoding_hash": self.codec.encoding_hash(), "modes": MODES,
                    "optimizer_state": None if self.optimizer is None else self.optimizer.state_dict(),
                    "rng_state": _capture_rng(self),
                    "episode_boundary": self.episode_finished or not self.transitions,
                    "training_state": training_state}, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, stochastic=False, resume_optimizer=False):
        """Load deployment weights, or strictly restore a v2 training checkpoint.

        Legacy v1 weights remain usable for evaluation. They cannot reconstruct
        Adam moments/RNG and are refused when resume_optimizer=True.
        """
        data = torch.load(path, map_location="cpu", weights_only=True)
        version = data.get("format_version")
        if version not in (1, 2):
            raise ValueError("Unsupported hierarchical policy checkpoint")
        if resume_optimizer:
            if version != 2 or "optimizer_state" not in data or "rng_state" not in data:
                raise ValueError("Legacy weights-only checkpoint has no optimizer/RNG; exact resume is impossible")
            if not data.get("episode_boundary"):
                raise ValueError("Exact resume requires an episode-boundary checkpoint")
            if data["update"] > 0 and data["optimizer_state"] is None:
                raise ValueError("Trained checkpoint is missing optimizer state; exact resume is impossible")
        codec = CoupledCodec(**data["codec_config"])
        if codec.kind == "ae4":
            codec.encoder.requires_grad_(False)
        policy_config = dict(data["policy_config"])
        # Older checkpoints must retain their original critic units/GAE timing.
        policy_config.setdefault("reward_scale", 1.0)
        policy_config.setdefault("gae_lambda_unit", "decision")
        policy = cls(codec, stochastic=stochastic, **policy_config)
        policy.load_state_dict(data["state_dict"])
        policy.update_number = data["update"]
        if policy.codec.encoding_hash() != data["encoding_hash"]:
            raise ValueError("Policy codec hash mismatch")
        policy.resume_training_state = data.get("training_state")
        if resume_optimizer:
            if data["optimizer_state"] is not None:
                policy._make_optimizer()
                policy.optimizer.load_state_dict(data["optimizer_state"])
            # Constructors consume torch RNG. Restore only after all initialization.
            _restore_rng(policy, data["rng_state"])
        return policy.eval()


def _run_episode(factory, policy, episode_seed, stochastic, action_seed):
    policy.reset_episode(stochastic=stochastic, action_seed=action_seed)
    simulator = factory(policy, episode_seed)
    metrics = simulator.run()
    if not policy.episode_finished:
        policy.finish(metrics)
    return policy.episode()


def evaluate_policy(simulator_factory, policy, seeds=(6000, 6001)):
    """Paired deployment evaluation that does not advance training RNG streams."""
    results = []
    rng, original_stochastic = _capture_rng(policy), policy.stochastic
    try:
        for seed in seeds:
            for stochastic in (False, True):
                episode = _run_episode(simulator_factory, policy, seed, stochastic,
                                       action_seed=seed + 19000)
                results.append({"seed": int(seed), "deployment": "sampled" if stochastic else "greedy",
                                **{k: v for k, v in episode.items() if k != "transitions"}})
    finally:
        _restore_rng(policy, rng)
        policy.stochastic = original_stochastic
    return results


def train_policy(simulator_factory, codec, output, seed=7, updates=60, episodes_per_update=2,
                 validation_seeds=(6000, 6001), validation_every=5, joint=False,
                 train_seed_base=4000, epochs=4, minibatch_size=128, resume_path=None,
                 run_signature=None, reward_scale=None, gae_lambda=None, gae_lambda_unit=None):
    """Train with factory(policy, seed) -> simulator exposing run().

    The factory MUST install ``policy.codec`` as its telemetry codec: this
    function deep-copies the input codec so frozen/joint experiments are isolated.
    Simulator calls act at control instants, then finish with terminal accounting.
    Returns (policy_at_last_update, logs), saving initial/periodic/best/last
    weights. Best is selected by mean sampled validation discounted return.
    ``updates`` is the TARGET TOTAL count, including completed resumed updates.
    New runs use fixed reward_scale=.001 and GAE lambda=.995 per physical ms.
    This rescales critic optimization units, not the physical objective or metrics.
    None-valued scale/GAE arguments inherit the saved values during exact resume.
    Resume restores optimizer/RNG and resumes the episode seed schedule. Embedded
    checkpoint history is authoritative; training.json is a fallback for manually
    saved v2 policies. Pass run_signature to validate simulator configuration;
    the caller must keep its factory/environment and execution backend unchanged.
    No test seed is consumed. This helper does not claim optimizer convergence.
    """
    if updates < 0 or episodes_per_update < 1 or validation_every < 1:
        raise ValueError("Invalid training schedule")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = {"seed": int(seed), "joint": bool(joint),
              "episodes_per_update": int(episodes_per_update),
              "validation_seeds": list(validation_seeds), "validation_every": int(validation_every),
              "train_seed_base": int(train_seed_base), "epochs": int(epochs),
              "minibatch_size": int(minibatch_size), "run_signature": run_signature}
    config = json.loads(json.dumps(config))
    if resume_path is None:
        if codec is None:
            raise ValueError("Fresh training requires a codec")
        policy_kwargs = {k: v for k, v in {"reward_scale": reward_scale,
                                          "gae_lambda": gae_lambda,
                                          "gae_lambda_unit": gae_lambda_unit}.items() if v is not None}
        policy = HierarchicalPPO(copy.deepcopy(codec), joint=joint, seed=seed, **policy_kwargs)
        config.update(reward_scale=policy.reward_scale, gae_lambda=policy.gae_lambda,
                      gae_lambda_unit=policy.gae_lambda_unit)
        logs = {"algorithm": "hierarchical PPO over ICC eta multiplier and parallelism cap",
                "joint": bool(joint), "seed": int(seed), "modes": MODES,
                "reward": "-delta_cost/100 - 2*delta_original_deadline_failures - .01*delta_bytes/32",
                "gamma_per_ms": policy.gamma, "gae_lambda": policy.gae_lambda,
                "gae_lambda_unit": policy.gae_lambda_unit, "reward_scale": policy.reward_scale,
                "reported_reward_units": "unscaled original physical objective",
                "encoder_hash_initial": policy.codec.encoding_hash(), "updates": [], "validation": [],
                "training_config": config}
        start_update = 0
    else:
        policy = HierarchicalPPO.load(resume_path, resume_optimizer=True)
        requested_optimization = {
            "reward_scale": policy.reward_scale if reward_scale is None else float(reward_scale),
            "gae_lambda": policy.gae_lambda if gae_lambda is None else float(gae_lambda),
            "gae_lambda_unit": policy.gae_lambda_unit if gae_lambda_unit is None else gae_lambda_unit,
        }
        config.update(requested_optimization)
        start_update = policy.update_number
        if updates < start_update:
            raise ValueError(f"updates is a target total and must be >= completed count {start_update}")
        state = policy.resume_training_state
        if state is not None:
            logs = copy.deepcopy(state["logs"])
        else:
            log_path = Path(resume_path).parent / "training.json"
            if not log_path.exists():
                raise ValueError("Resume requires embedded training history or adjacent training.json")
            logs = json.loads(log_path.read_text())
        saved_config = logs.get("training_config", {}).copy()
        for key in ("reward_scale", "gae_lambda", "gae_lambda_unit"):
            saved_config.setdefault(key, getattr(policy, key))
        if saved_config != config:
            raise ValueError("Resume training/simulator configuration differs from the saved run")
        logs["training_config"] = saved_config
        logs["updates"] = [row for row in logs["updates"] if row["update"] <= start_update]
        logs["validation"] = [row for row in logs["validation"] if row["update"] <= start_update]
        if [row["update"] for row in logs["updates"]] != list(range(1, start_update + 1)):
            raise ValueError("Resume log history is incomplete for the checkpoint update number")
        logs.setdefault("resumptions", []).append({"checkpoint": str(resume_path),
                                                   "completed_updates": start_update,
                                                   "target_total_updates": int(updates)})
    best_score, best_update = -float("inf"), 0

    for row in logs["validation"]:
        scores = [r["discounted_return"] for r in row["episodes"] if r["deployment"] == "sampled"]
        if scores and float(np.mean(scores)) > best_score:
            best_score, best_update = float(np.mean(scores)), row["update"]
    logs["best_checkpoint_available"] = False
    if resume_path is not None and validation_seeds:
        # A new output directory must retain the earlier best model, even when
        # continued updates do not improve it. Never copy a future checkpoint
        # when resuming an older checkpoint from an already-longer run.
        source = Path(resume_path).parent
        for candidate in (source / "best.pt", source / f"checkpoint_{best_update:03d}.pt"):
            if not candidate.exists():
                continue
            candidate_data = torch.load(candidate, map_location="cpu", weights_only=True)
            if (candidate_data.get("format_version") == 2
                    and candidate_data.get("update") == best_update):
                destination = output / "best.pt"
                if candidate.resolve() != destination.resolve():
                    temporary = destination.with_suffix(".pt.tmp")
                    shutil.copyfile(candidate, temporary)
                    temporary.replace(destination)
                logs["best_checkpoint_available"] = True
                break

    def training_state():
        logs["best_update"] = best_update if validation_seeds else None
        logs["encoder_hash_last"] = policy.codec.encoding_hash()
        logs["completed_updates"] = policy.update_number
        return {"logs": logs}

    def validate(update):
        nonlocal best_score, best_update
        results = evaluate_policy(simulator_factory, policy, validation_seeds)
        logs["validation"].append({"update": update, "episodes": results})
        sampled = [r["discounted_return"] for r in results if r["deployment"] == "sampled"]
        score = float(np.mean(sampled)) if sampled else -float("inf")
        if score > best_score:
            best_score, best_update = score, update
            logs["best_checkpoint_available"] = True
            policy.save(output / "best.pt", training_state=training_state())

    if resume_path is None:
        if validation_seeds:
            validate(0)
        policy.save(output / "checkpoint_000.pt", training_state=training_state())
    for update in range(start_update, updates):
        episodes = []
        for j in range(episodes_per_update):
            episode_seed = train_seed_base + update * episodes_per_update + j
            episodes.append(_run_episode(simulator_factory, policy, episode_seed, True,
                                         action_seed=seed * 100000 + episode_seed))
        logs["updates"].append(policy.update(episodes, epochs=epochs, minibatch_size=minibatch_size))
        progress = logs["updates"][-1]
        ev = progress["mc_explained_variance_before"]
        ev_text = "n/a" if ev is None else f"{ev:.4f}"
        print(f"PPO update={update + 1}/{updates} EV={ev_text} "
              f"raw_discounted_return={progress['train_discounted_return']:.3f} "
              f"entropy={progress['entropy_after']:.4f}", flush=True)
        if (update + 1) % validation_every == 0 or update + 1 == updates:
            if validation_seeds:
                validate(update + 1)
            policy.save(output / f"checkpoint_{update + 1:03d}.pt", training_state=training_state())
        state = training_state()
        _write_json(output / "training.json", logs)
        policy.save(output / "last.pt", training_state=state)
    state = training_state()
    _write_json(output / "training.json", logs)
    policy.save(output / "last.pt", training_state=state)
    return policy, logs
