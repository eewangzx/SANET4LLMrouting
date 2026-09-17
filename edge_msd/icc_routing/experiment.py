"""Train task-aware telemetry routing on an explicit ICC parameter dataset."""

import argparse
import copy
import csv
import hashlib
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .environment import DEFAULT_MODELS, Config, RoutingEnv
from .learning import (
    Codec,
    Heuristic,
    Policy,
    SemanticNet,
    collect_pretraining,
    pretrain,
    train_policy,
)


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def evaluate(env, router):
    seconds, count = 0.0, 0
    while env.tick < env.total_steps:
        obs = env.observe()
        start = perf_counter()
        action = router.select(obs)
        if action is not None:
            seconds += perf_counter() - start
            count += 1
        env.step(action)
    result = env.results()
    result["router_cpu_ms_per_request"] = seconds * 1000 / max(1, count)
    return result


def suite(config, models):
    ae, semantic, stats = models["ae"], models["semantic"], models["stats"]
    raw = Codec("raw")
    latest = Codec("latest")
    stat = Codec("stats", stats)
    ae_codec = Codec("semantic", ae, 16)
    sem = Codec("semantic", semantic, 16)
    adaptive = Codec(
        "semantic",
        semantic,
        4 if config.report_bps <= 32000 else 8 if config.report_bps <= 64000 else 16,
    )
    raw_period = (
        np.ceil(
            (16 + config.history * 16 * 4) * 3 * 8000 / (0.8 * config.report_bps) / config.slot_ms
        )
        * config.slot_ms
    )
    slow = replace(config, report_period_ms=max(config.report_period_ms, float(raw_period)))
    return [
        ("query_only", config, raw, "none", Heuristic(raw, query_only=True)),
        ("raw_periodic", config, raw, "periodic", Heuristic(raw)),
        ("raw_budget_periodic", slow, raw, "periodic", Heuristic(raw)),
        ("latest_event", config, latest, "event", Heuristic(latest)),
        ("window_stats_trajectory", config, stat, "event", Heuristic(stat, True)),
        ("window_stats_ppo", config, stat, "event", Policy(stats, stat)),
        ("ae_trajectory", config, ae_codec, "event", Heuristic(ae_codec, True)),
        ("ae_reactive", config, ae_codec, "event", Heuristic(ae_codec)),
        ("semantic_trajectory", config, sem, "event", Heuristic(sem, True)),
        ("semantic_reactive", config, sem, "event", Heuristic(sem)),
        ("ae_ppo", config, ae_codec, "event", Policy(ae, ae_codec)),
        ("semantic_fixed16_ppo", config, sem, "event", Policy(semantic, sem)),
        ("semantic_adaptive_ppo", config, adaptive, "event", Policy(semantic, adaptive)),
        ("semantic_periodic_ppo", config, sem, "periodic", Policy(semantic, sem)),
        ("instant_full_heuristic", config, raw, "instant", Heuristic(raw)),
        ("instant_semantic_ppo", config, sem, "instant", Policy(semantic, sem)),
    ]


def aggregate(rows):
    result = []
    for load, rate, method in sorted(
        {(r["load_multiplier"], r["capacity_bps"], r["method"]) for r in rows}
    ):
        group = [
            r
            for r in rows
            if (r["load_multiplier"], r["capacity_bps"], r["method"]) == (load, rate, method)
        ]
        entry = {"load_multiplier": load, "capacity_bps": rate, "method": method, "n": len(group)}
        for key in (
            "success_rate",
            "ontime_rate",
            "p95_completed_latency_ms",
            "report_bps_actual",
            "mean_state_age_ms",
            "missing_fraction",
            "cost_per_arrival",
            "router_cpu_ms_per_request",
        ):
            values = [r[key] for r in group if r[key] is not None]
            entry[key + "_mean"] = float(np.mean(values)) if values else None
            entry[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        result.append(entry)
    return result


def plot(summary, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    load = min(r["load_multiplier"] for r in summary)
    methods = [
        "raw_budget_periodic",
        "latest_event",
        "window_stats_ppo",
        "ae_ppo",
        "semantic_fixed16_ppo",
        "semantic_adaptive_ppo",
    ]
    for method in methods:
        group = sorted(
            (r for r in summary if r["load_multiplier"] == load and r["method"] == method),
            key=lambda r: r["capacity_bps"],
        )
        x = [r["capacity_bps"] / 1000 for r in group]
        axes[0].errorbar(
            x,
            [100 * r["success_rate_mean"] for r in group],
            yerr=[100 * r["success_rate_std"] for r in group],
            marker="o",
            capsize=3,
            label=method,
        )
        axes[1].plot(x, [r["mean_state_age_ms_mean"] for r in group], marker="o", label=method)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Reporting capacity (kbit/s)")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Quality + deadline success (%)")
    axes[1].set_ylabel("Received state age (ms)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"ICC parameter background + surrogate models; load multiplier {load:g}")
    fig.savefig(out / "comparison.png", dpi=180)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/icc_paper/scenario_2026.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/icc_semantic_routing"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[201, 202, 203])
    parser.add_argument("--bandwidths", type=float, nargs="+", default=[32000, 64000, 256000])
    parser.add_argument("--loads", type=float, nargs="+", default=[0.05])
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--static", action="store_true")
    parser.add_argument(
        "--arrival-steps",
        type=int,
        help="Explicit evaluation-horizon override when loading a checkpoint",
    )
    args = parser.parse_args()
    if any(1000 <= s < 100000 for s in args.eval_seeds):
        raise SystemExit("Evaluation seeds must be outside reserved training/validation range")
    out = args.output
    if (out / "results.csv").exists():
        raise SystemExit("Choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.train_seed)
    start = perf_counter()
    cfg = Config()
    dataset = json.loads(args.dataset.read_text())
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved["dataset_hash"] != dataset_hash:
            raise SystemExit("Checkpoint and ICC dataset differ")
        saved_config = dict(saved["config"])
        saved_config.setdefault("service_process", "sampled_inverse_rate")
        cfg = Config(**saved_config)
        models = {}
        for name, state in saved["models"].items():
            net = SemanticNet(cfg.history, cfg.horizon, "stats" if name == "stats" else "semantic")
            net.load_state_dict(state)
            models[name] = net.eval()
        training = saved["training"]
    else:
        print("Collecting ICC-background training and held-out validation histories", flush=True)
        train = collect_pretraining(args.dataset, cfg, range(1000, 1006))
        valid = collect_pretraining(args.dataset, cfg, (2000, 2001))
        base = SemanticNet(cfg.history, cfg.horizon)
        rep = pretrain(base, train, valid, args.epochs, args.train_seed)
        print(f"AE/predictor pretraining: {rep}", flush=True)
        stats = SemanticNet(cfg.history, cfg.horizon, "stats")
        stats_rep = pretrain(stats, train, valid, args.epochs, args.train_seed)
        models = {"ae": copy.deepcopy(base), "semantic": copy.deepcopy(base), "stats": stats}
        training = {
            "pretraining": rep,
            "stats_pretraining": stats_rep,
            "train_windows": len(train[0]),
            "validation_windows": len(valid[0]),
            "train_seeds": list(range(1000, 1006)),
            "validation_seeds": [2000, 2001],
            "training_seed": args.train_seed,
        }
        for name, model in models.items():
            training[name] = train_policy(
                model, args.dataset, cfg, args.updates, args.train_seed, joint=name == "semantic"
            )
            model.eval()
        saved = {
            "models": {name: m.state_dict() for name, m in models.items()},
            "config": asdict(cfg),
            "dataset_hash": dataset_hash,
            "training": training,
            "surrogate_models": DEFAULT_MODELS,
        }
        torch.save(saved, out / "checkpoint.pt")
    dump(out / "icc_scenario.json", dataset)
    dump(out / "training.json", training)
    dump(out / "surrogate_models.json", DEFAULT_MODELS)
    rows = []
    for load in args.loads:
        for rate in args.bandwidths:
            for seed in args.eval_seeds:
                case = replace(
                    cfg,
                    load_multiplier=load,
                    report_bps=rate,
                    dynamic=not args.static,
                    arrival_steps=args.arrival_steps or cfg.arrival_steps,
                )
                trace_hash = None
                for name, config, codec, reporting, router in suite(case, models):
                    if args.methods and name not in args.methods:
                        continue
                    env = RoutingEnv(args.dataset, config, seed, codec, reporting)
                    row = evaluate(env, router)
                    if trace_hash is not None and row["trace_hash"] != trace_hash:
                        raise RuntimeError("Paired workloads differ")
                    trace_hash = row["trace_hash"]
                    row.update(
                        method=name,
                        seed=seed,
                        capacity_bps=rate,
                        load_multiplier=load,
                        dataset_hash=dataset_hash,
                        dynamic=not args.static,
                        report_period_ms=config.report_period_ms,
                    )
                    rows.append(row)
                with (out / "results.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"Evaluated load={load:g}, reporting={rate:g}, seed={seed}", flush=True)
    summary = aggregate(rows)
    dump(out / "summary.json", summary)
    metadata = {
        "config": asdict(cfg),
        "evaluation_loads": args.loads,
        "evaluation_seeds": args.eval_seeds,
        "bandwidths": args.bandwidths,
        "dynamic": not args.static,
        "dataset_hash": dataset_hash,
        "dataset_kind": "ICC Table I / Figures reconstruction, not author-original samples",
        "surrogate_models": True,
        "microservice_dag_execution": "aggregate work proxy, not original DAG scheduling",
        "wireless_access": "not modeled; requests enter at their ICC-associated ES",
        "temporal_process": "new correlated sinusoid + AR noise, not ICC iid model",
        "encoder_ms": "configured 0.2 ms; decoder/router CPU time excluded from physical clock",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "elapsed_seconds": perf_counter() - start,
        "episodes": len(rows),
        "checkpoint_source": str(args.checkpoint or out / "checkpoint.pt"),
        "evaluation_arrival_steps": args.arrival_steps or cfg.arrival_steps,
        "methods": sorted({r["method"] for r in rows}),
    }
    dump(out / "metadata.json", metadata)
    lines = [
        "# ICC 背景下的任务驱动压缩与路由",
        "",
        "ICC 公开参数重建 + 明确新增的备选模型与时变过程；不是真实 LLM 测量或 ICC 原论文复现。",
        "模型相同部署在三个 ES。任务 DAG 的服务时间总和用于计算工作量代理；没有把不同功能的 core MS 当作替代 LLM。",
        "主实验负载乘数见表；1.0 才是这份 ICC 场景未经缩放的到达强度。",
        "一个模型训练种子；均值和标准差按独立测试轨迹计算。完整瞬时状态为参考而非最优上界。",
        "",
        "| 方法 | 负载 | 上报容量 kbit/s | 联合达标率 % | 标准差 pp | 实发 kbit/s | 状态年龄 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['method']} | {r['load_multiplier']:g} | {r['capacity_bps'] / 1000:g} | "
            f"{100 * r['success_rate_mean']:.2f} | {100 * r['success_rate_std']:.2f} | "
            f"{r['report_bps_actual_mean'] / 1000:.2f} | {r['mean_state_age_ms_mean']:.2f} |"
        )
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    plot(summary, out)
    print(f"Saved {len(rows)} episodes to {out}; {perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
