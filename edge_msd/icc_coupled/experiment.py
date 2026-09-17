"""Reproduce ICC-coupled telemetry training and paired control evaluations."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from edge_msd.config import Settings
from edge_msd.icc_paper import load_scenario
from edge_msd.network import Network
from edge_msd.placement import place_core

from .codecs import load_codec, train_codecs
from .dynamics import training_arrays
from .environment import CoupledSimulator, TelemetrySettings


def train(args):
    scenario = load_scenario(args.dataset)
    x, y, train_hashes = training_arrays(scenario, range(8000, 8008))
    vx, vy, val_hashes = training_arrays(scenario, [8100, 8101])
    args.models.mkdir(parents=True, exist_ok=True)
    meta = {'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            'train_seeds': list(range(8000, 8008)), 'validation_seeds': [8100, 8101],
            'training_windows': len(x), 'validation_windows': len(vx),
            'train_trace_hashes': train_hashes, 'validation_trace_hashes': val_hashes,
            'source': 'ICC node/service/link parameters; new explicitly generated correlated background dynamics',
            'history_ms': 40, 'prediction_horizon_ms': 300,
            'quality_gate': False, 'epochs': args.epochs}
    (args.models/'data_provenance.json').write_text(json.dumps(meta, indent=2)+'\n')
    print('train arrays', x.shape, y.shape, 'validation', vx.shape, flush=True)
    train_codecs(x, y, vx, vy, args.models, seed=7, epochs=args.epochs)
    print('all codecs saved', flush=True)


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    rows, full, placements = [], [], {}
    codecs = {m: load_codec(args.models/f'{m}.pt') for m in args.methods if m not in ('instant', 'prior')}
    for load in args.loads:
        scenario = load_scenario(args.dataset, load)
        placement_settings = Settings(duration_ms=args.arrival_ms+args.drain_ms,
            slot_ms=1, seed=args.seeds[0], ec_admission_window_ms=args.ec_window_ms)
        placement = place_core(scenario, Network(scenario), placement_settings)
        placements[load] = placement.counts
        for seed in args.seeds:
            settings = Settings(duration_ms=args.arrival_ms+args.drain_ms, slot_ms=1,
                                seed=seed, ec_admission_window_ms=args.ec_window_ms)
            for period in args.periods:
                hashes = set()
                for method in args.methods:
                    telemetry = TelemetrySettings(arrival_ms=args.arrival_ms,
                                    control_ms=args.control_ms, report_ms=period,
                                    report_bps=args.report_bps, dynamic=not args.static)
                    started = perf_counter()
                    sim = CoupledSimulator(scenario, settings, telemetry, codecs.get(method),
                        method if method in ('prior', 'instant') else 'periodic', placement.counts)
                    result = sim.run()
                    elapsed = perf_counter()-started
                    hashes.add((result['arrival_sha256'], result['resource_sha256']))
                    result.update(telemetry=asdict(telemetry), label=method, load=load, wall_seconds=elapsed)
                    full.append(result)
                    row = {'method': method, 'load': load, 'seed': seed, 'report_ms': period,
                        **result['overall'], 'total_cost': result['costs']['total'],
                        'report_bytes': result['report_bytes'], 'report_bps': result['report_bps'],
                        'state_age_ms': result['mean_state_age_ms'],
                        'mean_light_instances': float(np.mean([r['light_instances'] for r in result['control_trace']])),
                        'peak_parallelism': max(r['max_parallelism'] for r in result['control_trace']),
                        'wall_seconds': elapsed}
                    rows.append(row)
                    with (args.output/'episodes.csv').open('w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    (args.output/'episodes.json').write_text(json.dumps(full, indent=2)+'\n')
                    print(f'load={load:g} seed={seed} period={period} {method}: '
                          f'ICC ontime={100*row["ontime_rate"]:.2f}% cost={row["total_cost"]:.1f} '
                          f'report={row["report_bps"]:.0f}bps wall={elapsed:.1f}s', flush=True)
                assert len(hashes) == 1, 'Methods did not share the same external scenario'
    metadata = {'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                'quality_gate': False, 'original_dag': True,
                'resource_instance_parallelism_checks': 'original Simulator._deploy',
                'dynamic_core_placement': False,
                'ec_admission_window_ms': args.ec_window_ms,
                'physical_slot_ms': 1, 'control_ms': args.control_ms,
                'paired_arrival_and_resource_hashes': 'passed',
                'coupled_control_optimized': ['light instance count', 'parallelism', 'stage routing'],
                'core_placement': 'same offline ICC MILP result within each load',
                'arrival_seeds': args.seeds, 'original_author_samples': False,
                'learned_controller': False, 'notes': 'Original ICC controller, new observation mechanism. '
                'Current-state instant reference sees no future. Original independent-transfer network '
                'abstraction is retained with fixed nominal paths and integrated time-varying rates; '
                'no shared data-link queue is claimed. Core processing remains deterministic.'}
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    lines = ['# ICC 原 DAG 与部署/并行约束下的遥测实验', '',
             'SLA = 完整任务在原 deadline 内完成；没有质量门槛。所有到达计入分母。', '',
             '| 方法 | 负载 | 上报周期 ms | 按时完成率 % | 成本 | 实发 kbit/s |',
             '|---|---:|---:|---:|---:|---:|']
    for load in args.loads:
        for period in args.periods:
            for method in args.methods:
                group = [r for r in rows if r['load']==load and r['report_ms']==period and r['method']==method]
                lines.append(f'| {method} | {load:g} | {period} | '
                    f'{100*np.mean([r["ontime_rate"] for r in group]):.2f} | '
                    f'{np.mean([r["total_cost"] for r in group]):.1f} | '
                    f'{np.mean([r["report_bps"] for r in group])/1000:.2f} |')
    lines += ['', '这是相同 ICC 控制器下的表征/预测实验，尚未包含 PPO 控制增益。',
              '即时状态参考免费、无未来；原型数据仍是公开参数重建。细节见 metadata.json。']
    (args.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['train','evaluate'], required=True)
    p.add_argument('--dataset', type=Path, default=Path('data/icc_paper/scenario_2026.json'))
    p.add_argument('--models', type=Path, default=Path('runs/icc_coupled_models'))
    p.add_argument('--output', type=Path, default=Path('runs/icc_coupled_evaluation'))
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--loads', nargs='+', type=float, default=[.2])
    p.add_argument('--seeds', nargs='+', type=int, default=[8200,8201])
    p.add_argument('--periods', nargs='+', type=int, default=[100])
    p.add_argument('--arrival-ms', type=int, default=200)
    p.add_argument('--drain-ms', type=int, default=150)
    p.add_argument('--control-ms', type=int, default=1)
    p.add_argument('--ec-window-ms', type=float, default=1.)
    p.add_argument('--report-bps', type=float, default=64000)
    p.add_argument('--methods', nargs='+', default=['prior','instant','stats16','stats4','dense4','importance','ae4','raw'])
    p.add_argument('--static', action='store_true')
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    train(args) if args.stage == 'train' else evaluate(args)


if __name__ == '__main__':
    main()
