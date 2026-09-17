"""End-to-end simulation: predictive-IB compression + Double DQN routing.

Steps: (1) pre-train the predictive-IB compressor on the exogenous factor trace;
(2) train a Double DQN whose per-node state contains the compressed Z; (3) compare
against instant-information greedy and a future-aware oracle. Only the "proposed"
codec is wired here; raw/stats16 come next.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from edge_msd.config import Settings
from edge_msd.realtime_routing.agent import DoubleDQNAgent, log_training_episode
from edge_msd.realtime_routing.compressor import PredictiveIBCompressor, ib_loss
from edge_msd.realtime_routing.dynamics import CorrelatedFactorTrace
from edge_msd.realtime_routing.environment import RoutingEnvironment
from edge_msd.realtime_routing.fixed_placement import (
    FixedPlacement,
    build_fixed_placement,
    placement_hash,
)
from edge_msd.realtime_routing.scenario import build_scenario

WINDOW, HORIZON, LATENT, HIDDEN = 16, 4, 6, 64
REPORT_MS = 10.0


class CorrelatedRoutingEnv(RoutingEnvironment):
    """ICC environment plus AR(1) multiplicative factors on light-service rates."""

    def __init__(self, scenario, settings, placement, seed, drain_ms, trace):
        self.factor_trace = trace
        super().__init__(scenario, settings, placement, seed=seed, drain_ms=drain_ms)

    def light_rate(self, instance, now):
        rate = super().light_rate(instance, now)
        slot = int(round(now / self.settings.slot_ms))
        return rate * self.factor_trace.factor(slot, instance.node, instance.service)


def _idx(start, length, cap):
    return np.clip(np.arange(start, start + length), 0, cap - 1)


def node_window(trace, node, now_slot):
    idx = _idx(now_slot - WINDOW + 1, WINDOW, trace.n_slots)
    return np.stack([trace.table[node, s][idx] for s in trace.services], axis=1).astype(np.float32)


def node_future(trace, node, now_slot):
    idx = _idx(now_slot + 1, HORIZON, trace.n_slots)
    return np.stack([trace.table[node, s][idx] for s in trace.services], axis=1).astype(np.float32)


def build_world(seed=2026, level=0.1, duration=40.0, drain=40.0, slot=1.0, rho=0.9,
                core_scale=1.0, light_per_service=2):
    from edge_msd.network import Network
    from edge_msd.placement import place_core
    from edge_msd.realtime_routing.fixed_placement import _pack_light, scenario_hash

    scenario = build_scenario(2026, load_multiplier=level)
    settings = Settings(duration_ms=int(duration), slot_ms=slot,
                        ec_admission_window_ms=1, max_parallelism=20, seed=seed)
    network = Network(scenario, settings.wireless)
    full = {k: v for k, v in place_core(scenario, network, settings, "proposed").counts.items() if v}
    # Tighten the core backbone FIRST, then pack light into the remaining budget.
    core = full if core_scale == 1.0 else {k: max(1, int(round(v * core_scale))) for k, v in full.items()}
    light = _pack_light(scenario, core, light_per_service)
    placement = FixedPlacement(core=core, light=light, core_solver="proposed+scale",
                               scenario_hash=scenario_hash(scenario),
                               placement_hash=placement_hash(core, light))
    effective_drain = max(drain, max(t.deadline_ms for t in scenario.tasks.values()))
    n_slots = int(np.ceil((duration + effective_drain) / slot)) + WINDOW + HORIZON + 2
    trace = CorrelatedFactorTrace(list(scenario.nodes), scenario.light, n_slots, seed=seed, rho=rho)
    return scenario, settings, placement, trace


def pretrain_compressor(nodes, services, seed=0, device="cpu", iters=800, beta=5e-3, n_slots=4000):
    # Train on an independent, long trajectory so the compressor sees enough data;
    # it is applied unchanged to the shorter episode trace.
    trace = CorrelatedFactorTrace(list(nodes), list(services), n_slots, seed=seed)
    X, Y = [], []
    for node in nodes:
        for t in range(WINDOW, trace.n_slots - HORIZON):
            X.append(node_window(trace, node, t))
            Y.append(node_future(trace, node, t))
    X = torch.tensor(np.array(X), device=device)
    Y = torch.tensor(np.array(Y), device=device)
    model = PredictiveIBCompressor(X.shape[2], WINDOW, latent_dim=LATENT, hidden=HIDDEN,
                                   horizons=HORIZON, out_dim=X.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n = int(0.85 * len(X))
    for _ in range(iters):
        b = torch.randint(0, n, (256,))
        _, pred, kl = model(X[b])
        loss, _, _ = ib_loss(pred, Y[b], kl, beta=beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        _, pred, kl = model(X[n:])
        mse = (pred - Y[n:]).pow(2).mean().item()
        var = Y[n:].var().item()
    return model, {"val_mse": mse, "val_var": var, "r2": 1 - mse / var, "kl": float(kl.mean())}


def node_latents(env, trace, model, device="cpu"):
    now_slot = int(round(env.now / env.settings.slot_ms))
    X = np.stack([node_window(trace, n, now_slot) for n in env.node_ids])
    with torch.no_grad():
        z = model.encode(torch.tensor(X, device=device))
    return z.cpu().numpy()


SERVICES = None


def build_state(env, latents):
    global SERVICES
    if SERVICES is None:
        SERVICES = sorted(env.scenario.services)
    obs = env.observe()
    cur = obs["current"]
    svc = np.zeros(len(SERVICES), np.float32)
    if cur["present"]:
        svc[SERVICES.index(cur["service"])] = 1.0
    return np.concatenate([
        latents.reshape(-1),
        obs["ledger"], obs["queue"],
        np.array([cur["present"], cur["age_ms"] / 50.0, cur["deadline_ms"] / 50.0,
                  cur["slack_ms"] / 50.0], np.float32),
        svc,
        np.asarray(obs["legal_mask"], np.float32),
    ]).astype(np.float32)


def state_dim(env):
    return len(env.node_ids) * LATENT + 2 * len(env.node_ids) + 4 + len(sorted(env.scenario.services)) + len(env.node_ids)


def baseline_node(env, trace, horizon):
    stage = env._routable_stage()
    slot = int(round(env.now / env.settings.slot_ms))
    best, best_v = None, -1.0
    for n in env.legal_nodes(stage):
        if stage.service in trace.services:
            v = float(np.mean([trace.factor(slot + k, n, stage.service) for k in range(horizon)]))
        else:
            v = 0.0  # core services carry no AR(1) factor; fall back to first legal node
        if v > best_v:
            best, best_v = n, v
    return best


def run_episode(env, trace, model, agent=None, epsilon=0.0, train=False, buffer=None, device="cpu"):
    env.reset()
    cached, cache_time = None, -1e9
    total, decisions = 0.0, 0
    while not env.terminated:
        if cached is None or env.now - cache_time >= REPORT_MS:
            cached, cache_time = node_latents(env, trace, model, device), env.now
        state = build_state(env, cached)
        mask = np.asarray(env.observe()["legal_mask"])
        action = agent.act(state, mask, epsilon) if train else agent.greedy(state, mask)
        _, reward, term, _, _ = env.step(env.node_ids[action])
        total += reward
        decisions += 1
        if train:
            agent.record_reward(reward)
            if term:
                buffer.append((state, action, reward, state, mask, 1.0))
            else:
                if env.now - cache_time >= REPORT_MS:
                    cached, cache_time = node_latents(env, trace, model, device), env.now
                nxt = build_state(env, cached)
                nmask = np.asarray(env.observe()["legal_mask"])
                buffer.append((state, action, reward, nxt, nmask, 0.0))
            if len(buffer) >= 128:
                agent.update(random.sample(buffer, 64))
    return total, decisions


def run_baseline(env, trace, horizon):
    env.reset()
    total = 0.0
    guard = 0
    while not env.terminated and guard < 100000:
        node = baseline_node(env, trace, horizon)
        _, reward, _, _, _ = env.step(node)
        total += reward
        guard += 1
    return total


def sla(env):
    ontime, late, pending = env.sla_counts()
    arrivals = len(env.requests)
    return {"ontime": ontime, "late": late, "pending": pending, "arrivals": arrivals,
            "sla": ontime / arrivals if arrivals else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=float, default=0.1)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--drain", type=float, default=40.0)
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--core-scale", type=float, default=1.0)
    ap.add_argument("--light-per-service", type=int, default=2)
    ap.add_argument("--pretrain-iters", type=int, default=800)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--log-dir", default="runs/rt_gradient_monitor")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scenario, settings, placement, trace = build_world(
        seed=args.seed, level=args.level, duration=args.duration, drain=args.drain,
        core_scale=args.core_scale, light_per_service=args.light_per_service)
    print(f"scenario: {len(scenario.nodes)} nodes, {len(scenario.services)} services, "
          f"duration={args.duration}ms, level={args.level}")

    model, stats = pretrain_compressor(scenario.nodes, scenario.light, seed=0, iters=args.pretrain_iters)
    print("compressor:", {k: round(v, 4) for k, v in stats.items()})

    env = CorrelatedRoutingEnv(scenario, settings, placement, seed=args.seed, drain_ms=args.drain, trace=trace)
    agent = DoubleDQNAgent(state_dim(env), len(env.node_ids), lr=args.lr)
    buffer: list = []

    for episode in range(args.episodes):
        epsilon = max(0.05, 1.0 - episode / (0.75 * args.episodes))
        reward, decisions = run_episode(env, trace, model, agent, epsilon, True, buffer)
        log_training_episode(
            agent, f"{args.log_dir}/seed{args.seed}/experiment_proposed.csv",
            episode + 1, reward, decisions, sla(env)["sla"], epsilon,
        )

    # evaluation (deterministic policy)
    env_eval = CorrelatedRoutingEnv(scenario, settings, placement, seed=args.seed + 999, drain_ms=args.drain, trace=trace)
    run_episode(env_eval, trace, model, agent, 0.0, False, None)
    dqn = sla(env_eval)
    greedy_env = CorrelatedRoutingEnv(scenario, settings, placement, seed=args.seed + 999, drain_ms=args.drain, trace=trace)
    run_baseline(greedy_env, trace, 1)
    greedy = sla(greedy_env)
    oracle_env = CorrelatedRoutingEnv(scenario, settings, placement, seed=args.seed + 999, drain_ms=args.drain, trace=trace)
    run_baseline(oracle_env, trace, HORIZON)
    oracle = sla(oracle_env)

    print("\n=== evaluation ===")
    print("dqn    :", dqn)
    print("greedy :", greedy)
    print("oracle :", oracle)
    if args.out:
        import json
        with open(args.out, "w") as fh:
            json.dump({"dqn": dqn, "greedy": greedy, "oracle": oracle, "compressor": stats}, fh, indent=2)


if __name__ == "__main__":
    main()
