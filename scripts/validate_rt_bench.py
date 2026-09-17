"""One independently logged validation bench, reusing current routing code.

This checks the current AR-factor/IB prototype, not a complete SANet importance
filter or pure-ICC service process. All methods use independent training/test
seeds and identical pre-generated exogenous service samples for paired tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_msd.realtime_routing import benchmark as bm
from edge_msd.realtime_routing.agent import DoubleDQNAgent, log_training_episode
from edge_msd.realtime_routing.codec import Codec, pretrain_codec
from edge_msd.realtime_routing.dynamics import CorrelatedFactorTrace
from edge_msd.realtime_routing.experiment import CorrelatedRoutingEnv, build_world
from edge_msd.placement import validate_resources


class PairedEnv(CorrelatedRoutingEnv):
    """IID Gamma samples keyed by instance/slot, not by policy's draw order."""
    def __init__(self, scenario, settings, placement, seed, drain_ms, trace):
        self.rate_table = {}
        index = 0
        for (service_name, _node), count in sorted(placement.counts.items()):
            service = scenario.services[service_name]
            for _ in range(count):
                if service.kind == 'light':
                    rng = np.random.default_rng(np.random.SeedSequence([seed, 6201, index]))
                    self.rate_table[index] = rng.gamma(service.gamma_shape, service.gamma_scale,
                                                       trace.n_slots)
                index += 1
        super().__init__(scenario, settings, placement, seed, drain_ms, trace)

    def light_rate(self, instance, now):
        slot = int(round(now / self.settings.slot_ms))
        return (float(self.rate_table[instance.id][slot])
                * self.factor_trace.factor(slot, instance.node, instance.service))


class FullStatsCodec(Codec):
    """Complete last/mean/std; do not arbitrarily drop two services and all std."""
    def __init__(self, features, window, latent, horizons, device):
        super().__init__('stats16', features, window, latent, horizons, device=device)
        self.receiver = nn.Sequential(nn.Linear(3 * features, 64), nn.ReLU(),
                                      nn.Linear(64, latent))

    def _stats16(self, windows):
        return torch.cat([windows[:, -1, :], windows.mean(1), windows.std(1)], 1)

    def payload_bytes(self):
        return 3 * self.features * 4


class ScaledAgent(DoubleDQNAgent):
    """Fixed positive reward scaling preserves the undiscounted SLA objective."""
    def __init__(self, *args, reward_scale=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_scale = reward_scale

    def update(self, batch):
        return super().update([(s, a, r / self.reward_scale, s2, m2, done)
                               for s, a, r, s2, m2, done in batch])


def summarize(env):
    ontime, late, pending = env.sla_counts()
    delays = [r.completion_ms - r.arrival_ms for r in env.requests
              if r.completion_ms is not None]
    return {'arrivals': len(env.requests), 'ontime': ontime, 'late': late,
            'pending': pending, 'sla': ontime / len(env.requests) if env.requests else None,
            'completion_rate': len(delays) / len(env.requests) if env.requests else None,
            'mean_completed_latency_ms': float(np.mean(delays)) if delays else None,
            'p95_completed_latency_ms': float(np.percentile(delays, 95)) if delays else None,
            'arrival_sha256': env.arrival_hash.hexdigest(),
            'report_messages': getattr(env, '_report_messages', 0),
            'report_bytes': getattr(env, '_report_bytes', 0),
            'horizon_ms': env.horizon_ms}


def earliest_finish_node(env):
    stage = env._routable_stage()
    service = env.scenario.services[stage.service]
    scores = {}
    for node in env.legal_nodes(stage):
        pool = [i for i in env.instances if i.service == stage.service and i.node == node]
        eligible = [i for i in pool if (not i.jobs if service.kind == 'core'
                                       else len(i.jobs) < env.settings.max_parallelism)]
        queued = min(len(i.jobs) for i in eligible)
        scores[node] = (max(env.now, env.network.data_ready(stage, node))
                        + (queued + 1) * service.mean_processing_ms)
    return min(scores, key=lambda node: (scores[node], node))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', choices=['raw', 'stats27', 'proposed', 'baselines'], required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--duration', type=int, default=8)
    parser.add_argument('--level', type=float, default=1.)
    parser.add_argument('--core-scale', type=float, default=.3)
    parser.add_argument('--light-per-service', type=int, default=3)
    parser.add_argument('--pretrain-iters', type=int, default=600)
    parser.add_argument('--pretrain-slots', type=int, default=1500)
    parser.add_argument('--test-seeds', default='20000,20001,20002')
    parser.add_argument('--q-device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--gamma', type=float, default=1.)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def write(name, data):
        temporary = output / (name + '.tmp')
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
        temporary.replace(output / name)

    write('status.json', {'status': 'running', 'stage': 'building_world'})
    scenario, settings, placement, anchor_trace = build_world(
        seed=args.seed, level=args.level, duration=args.duration, drain=0.,
        core_scale=args.core_scale, light_per_service=args.light_per_service)
    validate_resources(scenario, placement.counts)
    expected_arrivals = args.duration / 1000 * sum(
        sum(u.rates_per_second.values()) for u in scenario.users)
    reward_scale = max(expected_arrivals, 1.)
    test_seeds = [int(seed) for seed in args.test_seeds.split(',')]
    manifest = {'args': vars(args), 'settings': asdict(settings), 'reward_scale': reward_scale,
                'train_seeds': [10000 + args.seed * 100 + i for i in range(args.episodes)],
                'test_seeds': test_seeds, 'pretrain_seed': 50000, 'prediction_validation_seed': 60000,
                'placement_hash': placement.placement_hash, 'scenario_hash': placement.scenario_hash,
                'physics': 'ICC Gamma base rates times current AR(1) factor extension',
                'report_model': '10ms decision-time cached reports; byte accounting, no bandwidth delay',
                'core_instances': sum(placement.core.values()),
                'light_instances': sum(placement.light.values()),
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path.cwd()/'edge_msd/realtime_routing').glob('*.py')}}
    write('manifest.json', manifest)
    print('START', args.bench, 'expected_arrivals=', round(expected_arrivals, 2), flush=True)

    def make_env(seed):
        trace = CorrelatedFactorTrace(scenario.nodes, scenario.light, anchor_trace.n_slots, seed=seed)
        return PairedEnv(scenario, settings, placement, seed, 0., trace), trace

    if args.bench == 'baselines':
        results = {}
        for method in ('earliest_finish', 'greedy_now', 'future_factor_heuristic'):
            rows = []
            for seed in test_seeds:
                env, trace = make_env(seed)
                if method == 'earliest_finish':
                    while not env.terminated:
                        env.step(earliest_finish_node(env))
                else:
                    bm.run_baseline(env, trace, 1 if method == 'greedy_now' else 4)
                row = {'seed': seed, **summarize(env)}
                rows.append(row)
                print(method, seed, row, flush=True)
            results[method] = rows
        write('result.json', {'baselines': results, 'elapsed_seconds': time.perf_counter()-started})
    else:
        write('status.json', {'status': 'running', 'stage': 'pretraining_codec'})
        X, Y = bm.pretrain_data(scenario.nodes, scenario.light, 50000, args.pretrain_slots)
        if args.bench == 'stats27':
            codec = FullStatsCodec(X.shape[2], 16, 8, 4, bm.DEVICE).to(bm.DEVICE)
        else:
            codec = Codec(args.bench, X.shape[2], 16, 8, 4, device=bm.DEVICE).to(bm.DEVICE)
        stats = pretrain_codec(codec, X, Y, iters=args.pretrain_iters)
        XV, YV = bm.pretrain_data(scenario.nodes, scenario.light, 60000, 400)
        with torch.no_grad():
            pred = codec.predict(XV)
            target = YV.to(bm.DEVICE)
            latents = codec.embed(XV)
            mse = (pred-target).square().mean().item()
            variance = target.var().item()
            stats.update({'independent_val_mse': mse, 'independent_val_r2': 1-mse/variance,
                          'persistence_val_mse': (XV[:,-1,:].unsqueeze(1)-YV).square().mean().item(),
                          'latent_rms': latents.square().mean().sqrt().item(),
                          'latent_abs_max': latents.abs().max().item()})
        write('codec_metrics.json', stats)
        print('CODEC', args.bench, stats, flush=True)
        example, _ = make_env(10000 + args.seed * 100)
        agent = ScaledAgent(bm.state_dim(example, 8), len(example.node_ids), lr=args.lr,
                            gamma=args.gamma, device=args.q_device, reward_scale=reward_scale)
        agent.rng = np.random.default_rng(args.seed)
        original_embed = bm.embed_nodes
        zero_embedding = [False]
        def counted_embed(env, codec_arg, trace):
            env._report_messages = getattr(env, '_report_messages', 0) + len(env.node_ids)
            env._report_bytes = getattr(env, '_report_bytes', 0) + len(env.node_ids)*codec_arg.total_bytes()
            embedding = original_embed(env, codec_arg, trace)
            return np.zeros_like(embedding) if zero_embedding[0] else embedding
        bm.embed_nodes = counted_embed
        buffer = []
        services = sorted(scenario.services)
        train_start = time.perf_counter()
        for episode, seed in enumerate(manifest['train_seeds'], 1):
            env, trace = make_env(seed)
            epsilon = max(.05, 1-(episode-1)/(.75*args.episodes))
            reward, decisions = bm.run_episode(env, trace, codec, services, agent, epsilon, True, buffer)
            row = log_training_episode(agent, output/'training.csv', episode, reward, decisions,
                                       bm.sla(env)['sla'], epsilon,
                                       stats['latent_rms'], stats['latent_abs_max'])
            print('TRAIN', args.bench, episode, row, flush=True)
            write('status.json', {'status': 'running', 'stage': 'routing_training',
                                 'episode': episode, 'updates': agent.updates,
                                 'elapsed_seconds': time.perf_counter()-started})
        train_elapsed = time.perf_counter()-train_start
        torch.save({'online': agent.online.state_dict(), 'target': agent.target.state_dict(),
                    'optimizer': agent.opt.state_dict(), 'codec': codec.state_dict(),
                    'manifest': manifest}, output/'last.pt')
        write('status.json', {'status': 'running', 'stage': 'independent_evaluation'})
        evaluation = []
        zero_evaluation = []
        for zero in ([False, True] if args.bench == 'proposed' else [False]):
            zero_embedding[0] = zero
            for seed in test_seeds:
                env, trace = make_env(seed)
                reward, decisions = bm.run_episode(env, trace, codec, services, agent)
                row = {'seed': seed, 'zero_embedding': zero, 'reward': reward,
                       'decisions': decisions, **summarize(env)}
                (zero_evaluation if zero else evaluation).append(row)
                print('TEST', args.bench, row, flush=True)
        write('result.json', {'bench': args.bench, 'codec': stats, 'evaluation': evaluation,
                             'zero_embedding_evaluation': zero_evaluation,
                             'bytes_per_node_report': codec.total_bytes(), 'updates': agent.updates,
                             'training_seconds': train_elapsed,
                             'elapsed_seconds': time.perf_counter()-started})
    write('status.json', {'status': 'complete', 'elapsed_seconds': time.perf_counter()-started})
    print('DONE', args.bench, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
