"""Bounded hierarchical-PPO training on the preserved ICC deployment simulator."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from edge_msd.config import Settings
from edge_msd.icc_paper import load_scenario
from edge_msd.network import Network
from edge_msd.placement import place_core

from .codecs import load_codec
from .environment import CoupledSimulator, TelemetrySettings
from .ppo import HierarchicalPPO, evaluate_policy, train_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['train','evaluate'], default='train')
    p.add_argument('--dataset', type=Path, default=Path('data/icc_paper/scenario_2026.json'))
    p.add_argument('--codec', type=Path, default=Path('runs/icc_coupled_models/importance.pt'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--resume', type=Path,
                   help='Resume a v2 optimizer/RNG checkpoint; --updates is the target total')
    p.add_argument('--load', type=float, default=.2)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--updates', type=int, default=60)
    p.add_argument('--episodes-per-update', type=int, default=2)
    p.add_argument('--train-seed-base', type=int, default=100000)
    p.add_argument('--validation-every', type=int, default=20)
    p.add_argument('--validation-seeds', type=int, nargs='*', default=[8400,8401])
    p.add_argument('--test-seeds', type=int, nargs='+', default=[8500,8501])
    p.add_argument('--test-last', action='store_true',
                   help='Additionally evaluate last.pt; the primary training test uses validation-selected best.pt')
    p.add_argument('--joint', action='store_true')
    p.add_argument('--arrival-ms', type=int, default=120)
    p.add_argument('--drain-ms', type=int, default=150)
    p.add_argument('--report-ms', type=int, default=100)
    p.add_argument('--report-bps', type=float, default=64000)
    args = p.parse_args()
    if args.resume is not None and args.stage != 'train':
        p.error('--resume applies only to --stage train; use --checkpoint for evaluation')
    if args.stage == 'train':
        if not args.validation_seeds:
            p.error('Training requires validation seeds to select best.pt before testing')
        train_end = args.train_seed_base + args.updates * args.episodes_per_update
        held_out = set(args.validation_seeds) | set(args.test_seeds)
        overlap = sorted(seed for seed in held_out if args.train_seed_base <= seed < train_end)
        if overlap:
            p.error(f'Target-total training seed range overlaps held-out seeds: {overlap}')
        if set(args.validation_seeds) & set(args.test_seeds):
            p.error('Validation and test seeds must be disjoint')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario(args.dataset, args.load)
    settings = Settings(duration_ms=args.arrival_ms+args.drain_ms, slot_ms=1,
                        seed=args.seed, ec_admission_window_ms=1.)
    placement = place_core(scenario, Network(scenario), settings)
    telemetry = TelemetrySettings(arrival_ms=args.arrival_ms, control_ms=1,
                                  report_ms=args.report_ms, report_bps=args.report_bps)

    def factory(policy, episode_seed):
        from dataclasses import replace
        return CoupledSimulator(scenario, replace(settings, seed=episode_seed), telemetry,
            policy.codec, 'periodic', placement.counts, policy=policy)

    if args.stage == 'train':
        codec = None if args.resume is not None else load_codec(args.codec)
        print('Training ICC hierarchical PPO', 'joint' if args.joint else 'frozen', flush=True)
        run_signature = {
            'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            'load': args.load, 'arrival_ms': args.arrival_ms, 'drain_ms': args.drain_ms,
            'report_ms': args.report_ms, 'report_bps': args.report_bps,
            'control_ms': 1, 'physical_ms': 1, 'ec_admission_window_ms': 1.0,
        }
        policy, logs = train_policy(factory, codec, args.output, seed=args.seed,
            updates=args.updates, episodes_per_update=args.episodes_per_update,
            validation_seeds=tuple(args.validation_seeds), validation_every=args.validation_every,
            joint=args.joint, train_seed_base=args.train_seed_base, resume_path=args.resume,
            run_signature=run_signature)
        print('Training completed', len(logs['updates']), flush=True)
        selected_checkpoint = args.output / 'best.pt'
        if not logs.get('best_checkpoint_available') or not selected_checkpoint.exists():
            raise RuntimeError('The validation-selected best checkpoint is unavailable; refusing test-based selection')
        selected_policy = HierarchicalPPO.load(selected_checkpoint)
        if selected_policy.update_number != logs['best_update']:
            raise RuntimeError('best.pt does not match the validation-selected update')
        result = evaluate_policy(factory, selected_policy, args.test_seeds)
        (args.output / 'test_best.json').write_text(json.dumps(result, indent=2) + '\n')
        if args.test_last:
            last_result = (result if selected_policy.update_number == policy.update_number
                           else evaluate_policy(factory, policy, args.test_seeds))
            (args.output / 'test_last.json').write_text(json.dumps(last_result, indent=2) + '\n')
        selection = 'maximum mean unscaled discounted return on sampled validation episodes only'
    else:
        if args.checkpoint is None:
            p.error('--checkpoint is required for evaluation')
        policy = HierarchicalPPO.load(args.checkpoint)
        selected_policy, selected_checkpoint = policy, args.checkpoint
        result = evaluate_policy(factory, policy, args.test_seeds)
        (args.output / 'test_checkpoint.json').write_text(json.dumps(result, indent=2) + '\n')
        selection = 'explicit --checkpoint; no checkpoint selection using test data'
    metadata = {**{k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                'task_sla': 'ICC complete DAG deadline; no quality condition',
                'action': 'hierarchical eta multiplier and parallelism cap; original constrained solver chooses counts/routes',
                'control_ms': 1, 'physical_ms': 1, 'ec_admission_window_ms': 1,
                'core_counts': {f'{s}@{n}':v for (s,n),v in placement.counts.items() if v},
                'training_seeds_start': args.train_seed_base,
                'training_seeds_stop_exclusive': args.train_seed_base + args.updates * args.episodes_per_update,
                'primary_test_checkpoint': str(selected_checkpoint),
                'primary_test_update': selected_policy.update_number,
                'checkpoint_selection': selection,
                'updates_semantics': 'target total updates, including completed resumed updates',
                'resume_semantics': 'v2 optimizer/RNG continuation only; v1 weights-only cannot resume exactly',
                'method_limit': 'joint updates are a heuristic composite with discrete filtering and constrained lower-level solver; no exact joint-policy-gradient or SLA guarantee is claimed'}
    (args.output/'experiment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Test episodes', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
