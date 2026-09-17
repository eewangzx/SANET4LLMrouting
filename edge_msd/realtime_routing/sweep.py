"""Multi-seed SLA-vs-bytes sweep in a congested setting."""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from edge_msd.realtime_routing.benchmark import pretrain_data, run_baseline, train_method
from edge_msd.realtime_routing.experiment import (
    HORIZON, CorrelatedRoutingEnv, build_world, sla,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,8,9")
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--core-scale", type=float, default=0.3)
    ap.add_argument("--light-per-service", type=int, default=3)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--drain", type=float, default=50.0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--log-dir", default="runs/rt_gradient_monitor")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    out = defaultdict(list)
    bytes_map = {}
    for seed in seeds:
        scenario, settings, placement, trace = build_world(
            seed=seed, level=args.level, duration=args.duration, drain=args.drain,
            core_scale=args.core_scale, light_per_service=args.light_per_service)
        X, Y = pretrain_data(scenario.nodes, scenario.light, seed=seed)
        for h, name in ((1, "greedy_now"), (HORIZON, "oracle")):
            env = CorrelatedRoutingEnv(scenario, settings, placement, seed=seed + 999,
                                       drain_ms=args.drain, trace=trace)
            run_baseline(env, trace, h)
            out[name].append(sla(env)["sla"])
            bytes_map.setdefault(name, 0)
        for kind in ("raw", "stats16", "proposed"):
            result, _stats, nbytes = train_method(kind, scenario, settings, placement, trace,
                                                  X, Y, seed, args.episodes, args.drain,
                                                  lr=args.lr, log_dir=args.log_dir)
            out[kind].append(result["sla"])
            bytes_map[kind] = nbytes
        means = {k: round(float(np.mean(v)), 3) for k, v in out.items()}
        print("seed", seed, "done:", means, flush=True)

    print("\n=== mean over seeds ===")
    for name, values in out.items():
        arr = np.array(values)
        print(f"{name:11s} bytes={bytes_map.get(name, 0):4d}  SLA={arr.mean():.3f} +- {arr.std():.3f}  n={len(arr)}")


if __name__ == "__main__":
    main()
