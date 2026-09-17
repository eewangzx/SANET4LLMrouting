"""Compare telemetry codecs: SLA vs bytes (raw / stats16 / proposed)."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from edge_msd.realtime_routing.agent import DoubleDQNAgent, log_training_episode
from edge_msd.realtime_routing.codec import Codec, pretrain_codec
from edge_msd.realtime_routing.dynamics import CorrelatedFactorTrace
from edge_msd.realtime_routing.experiment import (
    HORIZON, REPORT_MS, WINDOW, CorrelatedRoutingEnv, baseline_node, build_world,
    node_future, node_window, sla,
)

LATENT = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pretrain_data(nodes, services, seed, n_slots=4000):
    tr = CorrelatedFactorTrace(list(nodes), list(services), n_slots, seed=seed)
    X, Y = [], []
    for n in tr.nodes:
        for t in range(WINDOW, tr.n_slots - HORIZON):
            X.append(node_window(tr, n, t))
            Y.append(node_future(tr, n, t))
    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))


def embed_nodes(env, codec, trace):
    slot = int(round(env.now / env.settings.slot_ms))
    windows = np.stack([node_window(trace, n, slot) for n in env.node_ids])
    with torch.no_grad():
        return codec.embed(torch.tensor(windows, device=DEVICE)).cpu().numpy()


def build_state(env, emb, services_sorted):
    obs = env.observe()
    cur = obs["current"]
    svc = np.zeros(len(services_sorted), np.float32)
    if cur["present"]:
        svc[services_sorted.index(cur["service"])] = 1.0
    return np.concatenate([
        emb.reshape(-1), obs["ledger"], obs["queue"],
        np.array([cur["present"], cur["age_ms"] / 50.0,
                  cur["deadline_ms"] / 50.0, cur["slack_ms"] / 50.0], np.float32),
        svc, np.asarray(obs["legal_mask"], np.float32),
    ]).astype(np.float32)


def state_dim(env, latent):
    n = len(env.node_ids)
    return n * latent + 2 * n + 4 + len(env.scenario.services) + n


def run_episode(env, trace, codec, services_sorted, agent=None, epsilon=0.0, train=False, buffer=None):
    env.reset()
    cached, cache_t = None, -1e9
    total, decisions = 0.0, 0
    while not env.terminated:
        if cached is None or env.now - cache_t >= REPORT_MS:
            cached, cache_t = embed_nodes(env, codec, trace), env.now
        state = build_state(env, cached, services_sorted)
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
                if env.now - cache_t >= REPORT_MS:
                    cached, cache_t = embed_nodes(env, codec, trace), env.now
                nxt = build_state(env, cached, services_sorted)
                buffer.append((state, action, reward, nxt,
                               np.asarray(env.observe()["legal_mask"]), 0.0))
            if len(buffer) >= 128:
                agent.update(random.sample(buffer, 64))
    return total, decisions


def run_baseline(env, trace, horizon):
    env.reset()
    guard = 0
    while not env.terminated and guard < 200000:
        env.step(baseline_node(env, trace, horizon))
        guard += 1


def train_method(kind, scenario, settings, placement, trace, X, Y, seed, episodes, drain,
                 latent=LATENT, lr=3e-5, log_dir="runs/rt_gradient_monitor"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    codec = Codec(kind, features=X.shape[2], window=WINDOW, latent=latent, horizons=HORIZON,
                  device=DEVICE).to(DEVICE)
    stats = pretrain_codec(codec, X, Y, iters=2000)
    with torch.no_grad():
        sample_latents = codec.embed(X[-min(512, len(X)):])
        stats["latent_rms"], stats["latent_abs_max"] = torch.stack([
            sample_latents.square().mean().sqrt(), sample_latents.abs().max()
        ]).cpu().tolist()
    print(f"{kind}: pretrained R2={stats['r2']:+.3f} "
          f"latent_rms={stats['latent_rms']:.3g} "
          f"latent_max={stats['latent_abs_max']:.3g}", flush=True)
    env = CorrelatedRoutingEnv(scenario, settings, placement, seed=seed, drain_ms=drain, trace=trace)
    services = sorted(env.scenario.services)
    agent = DoubleDQNAgent(state_dim(env, latent), len(env.node_ids), lr=lr, device=DEVICE)
    buffer: list = []
    for episode in range(episodes):
        epsilon = max(0.05, 1.0 - episode / (0.75 * episodes))
        reward, decisions = run_episode(env, trace, codec, services, agent, epsilon, True, buffer)
        log_training_episode(
            agent, None if log_dir is None else f"{log_dir}/seed{seed}/{kind}.csv",
            episode + 1, reward, decisions, sla(env)["sla"], epsilon,
            latent_rms=stats["latent_rms"], latent_abs_max=stats["latent_abs_max"],
        )
    env_eval = CorrelatedRoutingEnv(scenario, settings, placement, seed=seed + 999, drain_ms=drain, trace=trace)
    run_episode(env_eval, trace, codec, services, agent, 0.0, False, None)
    return sla(env_eval), stats, codec.total_bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=float, default=0.5)
    ap.add_argument("--core-scale", type=float, default=0.5)
    ap.add_argument("--light-per-service", type=int, default=3)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--drain", type=float, default=50.0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--log-dir", default="runs/rt_gradient_monitor")
    args = ap.parse_args()

    scenario, settings, placement, trace = build_world(
        seed=args.seed, level=args.level, duration=args.duration, drain=args.drain,
        core_scale=args.core_scale, light_per_service=args.light_per_service)
    X, Y = pretrain_data(scenario.nodes, scenario.light, seed=0)
    print(f"nodes={len(scenario.nodes)} core_inst={sum(placement.core.values())} "
          f"light_inst={sum(placement.light.values())} pretrain_samples={len(X)}")

    env = CorrelatedRoutingEnv(scenario, settings, placement, seed=args.seed, drain_ms=args.drain, trace=trace)
    rows = {}
    for h, name in ((1, "greedy_now"), (HORIZON, "oracle")):
        e = CorrelatedRoutingEnv(scenario, settings, placement, seed=args.seed + 999, drain_ms=args.drain, trace=trace)
        run_baseline(e, trace, h)
        rows[name] = (sla(e), 0)
    for kind in ("raw", "stats16", "proposed"):
        result, stats, nbytes = train_method(kind, scenario, settings, placement, trace, X, Y,
                                             args.seed, args.episodes, args.drain,
                                             lr=args.lr, log_dir=args.log_dir)
        rows[kind] = (result, nbytes)
        print(f"{kind:9s} bytes={nbytes:4d}  R2={stats['r2']:+.3f}  SLA={result['sla']:.3f} "
              f"(ontime={result['ontime']} late={result['late']} pending={result['pending']} "
              f"arrivals={result['arrivals']})")

    print("\n=== SLA vs bytes ===")
    for name, (d, nb) in rows.items():
        print(f"{name:11s} bytes={nb:4d}  SLA={d['sla']:.3f}  pending={d['pending']}")


if __name__ == "__main__":
    main()
