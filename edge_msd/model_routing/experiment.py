"""Train once, evaluate paired network traces, save checkpoints and diagnostics.

Example: python -m edge_msd.model_routing.experiment --output runs/narrowband_mvp
All default workloads/profiles are synthetic. No method is presumed to win.
"""

import argparse
import csv
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .environment import RoutingEnv
from .learning import (
    ActorCritic,
    AECodec,
    PolicyRouter,
    RawAECodec,
    StateAutoencoder,
    train_ppo,
    train_representation,
)
from .routers import HeuristicRouter
from .telemetry import RawCodec, StatsCodec
from .types import DEFAULT_PROFILES, ModelProfile, RoutingConfig


def write_json(path, obj):
    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            default=lambda v: v.item() if isinstance(v, np.generic) else str(v),
        )
        + "\n"
    )


def run_episode(env, router):
    elapsed = 0.0
    decisions = 0
    for _ in range(env.total_steps):
        observation = env.observe()
        start = time.perf_counter()
        action = router.select(observation)
        if action is not None:
            elapsed += time.perf_counter() - start
            decisions += 1
        env.step(action)
    result = env.results()
    result["measured_router_ms_per_request"] = elapsed * 1000 / max(1, decisions)
    return result


def suite(config, model, policy):
    # Slow full reports are a necessary baseline: raw reports may change period
    # to fit the SAME physical channel instead of being forced to overload it.
    raw_bytes = 16 + config.history * RoutingEnv.n_features * 4
    budget_period = max(config.report_period_ms, raw_bytes * 6 * 8000 / (0.8 * config.report_bps))
    budget_period = np.ceil(budget_period / config.slot_ms) * config.slot_ms
    slow = replace(config, report_period_ms=float(budget_period))
    return [
        (
            "query_only",
            config,
            RawCodec(config.history, 16),
            "none",
            HeuristicRouter(query_only=True),
        ),
        (
            "full_state_heuristic",
            config,
            RawCodec(config.history, 16),
            "instant",
            HeuristicRouter(),
        ),
        ("raw_periodic", config, RawCodec(config.history, 16), "periodic", HeuristicRouter()),
        ("raw_budget_periodic", slow, RawCodec(config.history, 16), "periodic", HeuristicRouter()),
        ("stats_periodic", config, StatsCodec(config.history, 16), "periodic", HeuristicRouter()),
        ("stats_event", config, StatsCodec(config.history, 16), "event", HeuristicRouter()),
        ("ae_periodic", config, AECodec(model), "periodic", HeuristicRouter()),
        ("ae_event", config, AECodec(model), "event", HeuristicRouter()),
        (
            "ae_periodic_predict",
            config,
            AECodec(model),
            "periodic",
            HeuristicRouter(predictive=True),
        ),
        ("ae_event_predict", config, AECodec(model), "event", HeuristicRouter(predictive=True)),
        ("ae_event_ppo", config, AECodec(model), "event", PolicyRouter(policy)),
        ("raw_budget_ppo", slow, RawAECodec(model), "periodic", PolicyRouter(policy)),
        ("full_state_ppo", config, RawAECodec(model), "instant", PolicyRouter(policy)),
    ]


def aggregate(rows):
    result = []
    keys = sorted({(r["report_bps"], r["method"]) for r in rows})
    metrics = (
        "success_rate",
        "ontime_rate",
        "p95_completed_latency_ms",
        "cost_per_arrival",
        "report_transmitted_bps",
        "mean_observation_age_ms",
        "missing_observation_fraction",
        "measured_router_ms_per_request",
    )
    for rate, method in keys:
        group = [r for r in rows if r["report_bps"] == rate and r["method"] == method]
        entry = {"report_bps": rate, "method": method, "evaluation_seeds": len(group)}
        for key in metrics:
            values = np.asarray([r[key] for r in group if r[key] is not None])
            entry[key + "_mean"] = float(values.mean()) if len(values) else None
            entry[key + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result.append(entry)
    return result


def plot_results(summary, output):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    methods = ["raw_budget_periodic", "stats_event", "ae_event", "ae_event_predict", "ae_event_ppo"]
    colors = ["#64748b", "#16a34a", "#d97706", "#9333ea", "#2563eb"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for method, color in zip(methods, colors):
        group = sorted((r for r in summary if r["method"] == method), key=lambda r: r["report_bps"])
        x = [r["report_bps"] / 1000 for r in group]
        axes[0].errorbar(
            x,
            [100 * r["success_rate_mean"] for r in group],
            yerr=[100 * r["success_rate_std"] for r in group],
            marker="o",
            capsize=3,
            label=method,
            color=color,
        )
        axes[1].plot(
            x,
            [r["mean_observation_age_ms_mean"] for r in group],
            marker="o",
            color=color,
            label=method,
        )
    axes[0].set_ylabel("Quality + deadline success (%)")
    axes[1].set_ylabel("Mean age of received state (ms)")
    for ax in axes:
        ax.set_xlabel("Shared reporting capacity (kbit/s)")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle("Synthetic profiles and workloads — prototype results", fontsize=13)
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    return True


def write_report(output, summary, training, metadata):
    lines = [
        "# 窄带模型路由：首版运行结果",
        "",
        "**合成模型 profile 与合成工作负载；不是实际 LLM 测量，也不是论文性能结论。**",
        "",
        "AE、预测器和 PPO 已分别训练；PPO 先模仿可见信息下的启发式，再在线训练。",
        "事件上报使用固定阈值。没有实现任务驱动 encoder 微调、学习式触发或 SANet 门控。",
        "",
        "同一测试 seed 的所有策略使用相同请求、网络演化及候选模型随机结果。",
        "误差条为不同测试轨迹的标准差，不包含多次独立模型训练的不确定性。",
        "",
        "所有可部署方法受同一物理上报容量约束；实际发送量可能不同。",
        "`full_state_*` 使用即时完整观测，是信息充分参考，未计通信开销，也非最优上界。",
        "",
        "| 容量 kbit/s | 方法 | 质量且按时达标 % | 报告 kbit/s | 状态年龄 ms |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['report_bps'] / 1000:g} | {r['method']} | "
            f"{100 * r['success_rate_mean']:.1f} ± {100 * r['success_rate_std']:.1f} | "
            f"{r['report_transmitted_bps_mean'] / 1000:.2f} | "
            f"{r['mean_observation_age_ms_mean']:.1f} |"
        )
    lines += [
        "",
        "## 训练与重现",
        "",
        f"- 训练 seed：{metadata['training_seed']}；测试 seeds：{metadata['evaluation_seeds']}。",
        f"- PPO 物理时隙步数：{training['ppo']['steps']}。",
        f"- 验证集预测 MSE：{training['representation']['prediction_mse']:.5f}；"
        f"持久性预测：{training['representation']['persistence_mse']:.5f}。",
        "",
        "查看 `metadata.json`、`training.json`、`results.csv` 和 `summary.json`。",
        "`checkpoint.pt` 保存 AE、预测器和最终策略；没有基于测试结果挑选 checkpoint。",
        "",
        "## 实现边界",
        "",
        "固定模型部署与 FIFO；无连续 batching、KV cache 或运行中迁移。",
        "请求通道与上报通道分离。编码时延采用配置值；实测 router CPU 时间单独报告，",
        "当前未反馈进仿真决策时钟。正式实验需校准这些开销并接入真实质量/耗时数据。",
        "",
        "下一步先检查强统计量基线是否已足够，再决定是否投入任务驱动编码和联合上报优化。",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/narrowband_mvp"))
    parser.add_argument(
        "--checkpoint", type=Path, help="Evaluate an existing checkpoint without training"
    )
    parser.add_argument(
        "--profiles", type=Path, help="JSON array of three ModelProfile dictionaries"
    )
    parser.add_argument("--ae-epochs", type=int, default=20)
    parser.add_argument("--rl-steps", type=int, default=16384)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument(
        "--legacy-policy",
        action="store_true",
        help="Reproduce the first pilot's shared actor/critic and short warm start",
    )
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--bandwidths", type=float, nargs="+", default=[32000, 64000, 256000])
    parser.add_argument("--arrival-steps", type=int, default=400)
    parser.add_argument("--drain-steps", type=int, default=120)
    args = parser.parse_args()
    if args.ae_epochs <= 0 or args.rl_steps <= 0 or args.latent_dim <= 0:
        parser.error("Training steps, epochs and latent size must be positive")
    if any(1000 <= seed < 100000 for seed in args.eval_seeds):
        parser.error("Evaluation seeds 1000..99999 are reserved for training/validation")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.csv").exists():
        parser.error("Output already contains results; choose another directory")
    torch.set_num_threads(1)
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    profiles = DEFAULT_PROFILES
    if args.profiles:
        profiles = tuple(ModelProfile(**p) for p in json.loads(args.profiles.read_text()))
    config = RoutingConfig(arrival_steps=args.arrival_steps, drain_steps=args.drain_steps)
    start = time.perf_counter()
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = RoutingConfig(**saved["config"])
        profiles = tuple(ModelProfile(**p) for p in saved["profiles"])
        model = StateAutoencoder(**saved["model_spec"])
        model.load_state_dict(saved["ae"])
        model.eval().requires_grad_(False)
        policy = ActorCritic(saved["policy_input_dim"], **saved.get("policy_options", {}))
        policy.load_state_dict(saved["policy"])
        training = saved["training"]
    else:
        model, rep_log = train_representation(
            config, profiles, args.latent_dim, args.ae_epochs, args.train_seed
        )
        policy, ppo_log = train_ppo(
            model,
            config,
            profiles,
            args.rl_steps,
            args.train_seed,
            legacy_policy=args.legacy_policy,
        )
        training = {"representation": rep_log, "ppo": ppo_log}
        saved = {
            "ae": model.state_dict(),
            "policy": policy.state_dict(),
            "model_spec": {
                "history": config.history,
                "features": RoutingEnv.n_features,
                "latent_dim": model.latent_dim,
                "prediction_steps": config.prediction_steps,
            },
            "policy_input_dim": policy.input_dim,
            "policy_options": {
                "quality_guard": policy.quality_guard,
                "separate_value": policy.separate_value,
            },
            "config": asdict(config),
            "profiles": [asdict(p) for p in profiles],
            "training": training,
            "training_seed": args.train_seed,
        }
        torch.save(saved, output / "checkpoint.pt")
    write_json(output / "training.json", training)
    write_json(output / "profiles.json", [asdict(p) for p in profiles])
    rows = []
    for rate in args.bandwidths:
        for seed in args.eval_seeds:
            cfg = replace(config, seed=seed, report_bps=rate)
            trace_hash = None
            for name, method_cfg, codec, mode, router in suite(cfg, model, policy):
                env = RoutingEnv(method_cfg, codec, mode, profiles)
                result = run_episode(env, router)
                if trace_hash is not None and result["trace_sha256"] != trace_hash:
                    raise RuntimeError("Counterfactual comparison used different exogenous traces")
                trace_hash = result["trace_sha256"]
                result.update(
                    method=name,
                    seed=seed,
                    report_bps=rate,
                    report_period_ms=method_cfg.report_period_ms,
                )
                rows.append(result)
            print(f"Evaluated {rate / 1000:g} kbit/s, trace seed {seed}", flush=True)
            # Checkpoint results after each complete paired trace.
            with (output / "results.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    summary = aggregate(rows)
    write_json(output / "summary.json", summary)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    metadata = {
        "synthetic_workload": True,
        "profiles_source": str(args.profiles or "synthetic defaults"),
        "training_seed": saved.get("training_seed", args.train_seed),
        "evaluation_seeds": args.eval_seeds,
        "report_bps": args.bandwidths,
        "config": asdict(config),
        "icc_base_commit": git_sha,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "elapsed_seconds": time.perf_counter() - start,
        "checkpoint_source": str(args.checkpoint or output / "checkpoint.pt"),
    }
    write_json(output / "metadata.json", metadata)
    plot_results(summary, output)
    write_report(output, summary, training, metadata)
    print(f"Saved paired results and report to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
