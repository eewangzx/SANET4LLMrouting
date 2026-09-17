"""Masked Double DQN with a dueling head; legal-action mask applied everywhere."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn


class DuelingQNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.input_norm = nn.LayerNorm(state_dim)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, action_dim)

    def forward(self, state):
        h = self.trunk(self.input_norm(state))
        adv = self.advantage(h)
        logits = self.value(h) + adv - adv.mean(-1, keepdim=True)
        return logits


class CandidateDuelingQNet(DuelingQNet):
    """Learned shared candidate score with explicitly aligned node features."""
    def __init__(self,state_dim,action_dim,latent_dim,include_costs=False,hidden=64):
        super().__init__(state_dim,action_dim,hidden)
        self.actions,self.latent_dim=action_dim,latent_dim
        self.node_blocks=7 if include_costs else 5
        local_dim=latent_dim+self.node_blocks
        self.advantage=nn.Sequential(nn.Linear(hidden+local_dim+action_dim,hidden),
                                    nn.ReLU(),nn.Linear(hidden,1))
        self.register_buffer('node_identity',torch.eye(action_dim))

    def candidate_advantage(self,state):
        normalized=self.input_norm(state)
        h=self.trunk(normalized)
        nodes=self.actions;start=nodes*self.latent_dim
        latent=normalized[...,:start].reshape(*state.shape[:-1],nodes,self.latent_dim)
        fields=normalized[...,start:start+nodes*self.node_blocks].reshape(
            *state.shape[:-1],self.node_blocks,nodes).transpose(-1,-2)
        identity=self.node_identity.expand(*state.shape[:-1],nodes,nodes)
        context=h.unsqueeze(-2).expand(*state.shape[:-1],nodes,h.shape[-1])
        adv=self.advantage(torch.cat((context,latent,fields,identity),dim=-1)).squeeze(-1)
        return h,adv

    def forward(self,state):
        h,adv=self.candidate_advantage(state)
        return self.value(h)+adv-adv.mean(-1,keepdim=True)


class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim, lr=3e-5, gamma=0.99, hidden=128, device="cpu"):
        self.action_dim, self.gamma, self.device = action_dim, gamma, device
        self.online = DuelingQNet(state_dim, action_dim, hidden).to(device)
        self.target = DuelingQNet(state_dim, action_dim, hidden).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.updates = 0
        self.rng = np.random.default_rng(0)
        self.last_metrics = {}
        self._episode_metrics = []
        self._episode_rewards = []

    def act(self, state, mask, epsilon):
        if self.rng.random() < epsilon:
            legal = np.flatnonzero(mask)
            return int(legal[self.rng.integers(len(legal))])
        with torch.no_grad():
            q = self.online(torch.tensor(state, device=self.device).unsqueeze(0))[0]
        q = q.masked_fill(torch.tensor(mask, device=self.device) == 0, -1e9)
        return int(torch.argmax(q).item())

    def greedy(self, state, mask):
        with torch.no_grad():
            q = self.online(torch.tensor(state, device=self.device).unsqueeze(0))[0]
        q = q.masked_fill(torch.tensor(mask, device=self.device) == 0, -1e9)
        return int(torch.argmax(q).item())

    def update(self, batch):
        s = torch.tensor(np.array([b[0] for b in batch]), device=self.device)
        a = torch.tensor([b[1] for b in batch], device=self.device)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.array([b[3] for b in batch]), device=self.device)
        m2 = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32, device=self.device)
        done = torch.tensor([b[5] for b in batch], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q2 = self.online(s2).masked_fill(m2 == 0, -1e9)
            a2 = q2.argmax(-1, keepdim=True)
            y = r + (1.0 - done) * self.gamma * self.target(s2).gather(1, a2).squeeze(1)
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        # clip_grad_norm_ returns the norm BEFORE clipping. Logging only the
        # clipped norm would hide explosions behind the constant threshold.
        grad_norm = nn.utils.clip_grad_norm_(
            self.online.parameters(), 10.0, error_if_nonfinite=True
        )
        grad_norm_post = torch.linalg.vector_norm(torch.stack([
            parameter.grad.detach().norm(2)
            for parameter in self.online.parameters() if parameter.grad is not None
        ]))
        self.opt.step()
        self.updates += 1
        if self.updates % 200 == 0:
            self.target.load_state_dict(self.online.state_dict())
        # Transfer the diagnostic scalars together, avoiding one CUDA sync per
        # metric on these small Q-network updates.
        names = ("loss", "grad_norm_pre", "grad_norm_post", "clipped", "td_abs",
                 "q_abs_max", "target_abs_max", "batch_reward_abs_max", "state_rms")
        values = torch.stack([
            loss.detach(), grad_norm.detach(), grad_norm_post.detach(),
            (grad_norm > 10.0).float(), (y - q).detach().abs().mean(),
            q.detach().abs().max(), y.abs().max(), r.abs().max(),
            s.square().mean().sqrt(),
        ]).cpu().tolist()
        self.last_metrics = dict(zip(names, values))
        self._episode_metrics.append(self.last_metrics)
        return self.last_metrics["loss"]

    def record_reward(self, reward):
        self._episode_rewards.append(float(reward))

    def episode_metrics(self):
        """Summarize this episode's updates, then release the per-update samples."""
        rows, self._episode_metrics = self._episode_metrics, []
        rewards, self._episode_rewards = self._episode_rewards, []
        def mean(key):
            return float(np.mean([row[key] for row in rows])) if rows else None
        def maximum(key):
            return max(row[key] for row in rows) if rows else None
        return {
            "updates_episode": len(rows), "updates_total": self.updates,
            "loss_mean": mean("loss"),
            "grad_norm_pre_mean": mean("grad_norm_pre"),
            "grad_norm_pre_max": maximum("grad_norm_pre"),
            "grad_norm_post_mean": mean("grad_norm_post"),
            "clip_fraction": mean("clipped"),
            "td_abs_mean": mean("td_abs"),
            "q_abs_max": maximum("q_abs_max"),
            "target_abs_max": maximum("target_abs_max"),
            "batch_reward_abs_max": maximum("batch_reward_abs_max"),
            "state_rms_mean": mean("state_rms"),
            "step_reward_mean": float(np.mean(rewards)) if rewards else None,
            "step_reward_abs_max": max(map(abs, rewards)) if rewards else None,
        }


def log_training_episode(agent, path, episode, reward, decisions, sla, epsilon,
                         latent_rms=None, latent_abs_max=None):
    """Write one compact diagnostic row per episode; no stopping heuristic."""
    row = {
        "episode": episode, "epsilon": epsilon,
        "lr": agent.opt.param_groups[0]["lr"], "gamma": agent.gamma,
        "reward": reward, "decisions": decisions, "sla": sla,
        "latent_rms": latent_rms, "latent_abs_max": latent_abs_max,
        **agent.episode_metrics(),
    }
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Episode 1 starts a new run, avoiding accidental appends to an old run.
        with path.open("w" if episode == 1 else "a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if episode == 1:
                writer.writeheader()
            writer.writerow(row)
    if episode == 1 or episode % 10 == 0:
        def fmt(key):
            return "n/a" if row[key] is None else f"{row[key]:.3g}"
        print(
            f"ep {episode:3d} reward={reward:+.1f} SLA={sla:.3f} "
            f"loss={fmt('loss_mean')} grad_pre={fmt('grad_norm_pre_mean')} "
            f"grad_max={fmt('grad_norm_pre_max')} clip={fmt('clip_fraction')} "
            f"Qmax={fmt('q_abs_max')} target_max={fmt('target_abs_max')}",
            flush=True,
        )
    return row
