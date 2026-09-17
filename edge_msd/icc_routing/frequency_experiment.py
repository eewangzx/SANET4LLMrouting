"""Controlled periodic-report sweep, with optional matched low-frequency training."""

import argparse
import copy
import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .environment import DEFAULT_MODELS, Config, RoutingEnv
from .experiment import aggregate, dump, evaluate
from .learning import (
    Codec,
    Heuristic,
    Policy,
    SemanticNet,
    collect_pretraining,
    pretrain,
    train_policy,
)


def periodic_suite(config, models):
    """Every method samples and generates reports at exactly the same fixed period."""
    latest = Codec("latest")
    result = [("latest_periodic", latest, Heuristic(latest))]
    for name in ("stats", "ae", "semantic", "semantic_no_forecast"):
        if name not in models:
            continue
        model = models[name]
        codec = Codec("stats" if name == "stats" else "semantic", model, 16)
        result.append((name + "_ppo", codec, Policy(model, codec)))
    codec = Codec("semantic", models["semantic"], 16)
    result.extend([
        ("semantic_trajectory", codec, Heuristic(codec, True)),
        ("semantic_reactive", codec, Heuristic(codec, False)),
    ])
    return result


def summarize(rows):
    summary = []
    for period in sorted({r["report_period_ms"] for r in rows}):
        for item in aggregate([r for r in rows if r["report_period_ms"] == period]):
            item["report_period_ms"] = period
            summary.append(item)
    return summary


def paired_differences(rows):
    result = []
    pairs = (
        ("semantic_ppo", "stats_ppo"),
        ("semantic_ppo", "ae_ppo"),
        ("semantic_ppo", "semantic_no_forecast_ppo"),
        ("semantic_trajectory", "semantic_reactive"),
    )
    for period in sorted({r["report_period_ms"] for r in rows}):
        indexed = {(r["method"], r["seed"]): r for r in rows if r["report_period_ms"] == period}
        for proposed, reference in pairs:
            seeds = sorted(s for m, s in indexed if m == proposed and (reference, s) in indexed)
            if not seeds:
                continue
            diffs = [100 * (indexed[proposed, s]["success_rate"]
                            - indexed[reference, s]["success_rate"]) for s in seeds]
            result.append({
                "report_period_ms": period, "proposed": proposed, "reference": reference,
                "seeds": seeds, "differences_pp": diffs,
                "mean_pp": float(np.mean(diffs)),
                "std_pp": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0,
            })
    return result


def forecast_errors(models, validation, slot_ms):
    """Held-out local-history diagnostics; these labels never enter routing inputs."""
    x, y = [torch.as_tensor(a) for a in validation]
    result = []
    with torch.no_grad():
        actual_persistence = x[:, -1, [1, 5, 6]][:, None]
        for name, model in models.items():
            if not model.use_forecast:
                continue
            z = model.encode(x, torch.full((len(x),), 16))
            predicted = model.forecast(z)
            decoded_persistence = model.decode(z)[:, -1, [1, 5, 6]][:, None]
            for low, high in ((0, 25), (25, 50), (50, 100), (100, 200), (200, 300)):
                first, last = int(low / slot_ms), min(model.horizon, int(high / slot_ms))
                if first >= last:
                    continue
                target = y[:, first:last]
                result.append({
                    "model": name, "lead_start_ms_exclusive": low,
                    "lead_end_ms_inclusive": last * slot_ms,
                    "forecast_mse": float((predicted[:, first:last] - target).square().mean()),
                    "decoded_persistence_mse": float((decoded_persistence - target).square().mean()),
                    "actual_persistence_mse": float((actual_persistence - target).square().mean()),
                    "validation_source": "instant-state heuristic histories, seeds 2000 and 2001",
                    "target": "compute and outgoing-link availability only, excludes iid Gamma rate",
                })
    return result


def write_report(rows, metadata, out):
    summary = summarize(rows)
    dump(out / "summary.json", summary)
    dump(out / "paired_differences.json", paired_differences(rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    periods = sorted({r["report_period_ms"] for r in rows})
    lines = [
        "# 统一降低上报频率的配对实验", "",
        f"训练条件：{metadata['training_regime']}。",
        "所有方法均周期上报，每节点每报文 80 B；本组不改变压缩维度，也不使用事件触发。",
        f"共享上报容量 {metadata['bandwidth_bps'] / 1000:g} kbit/s；"
        f"负载为 ICC 重建场景的 {metadata['load_multiplier']:g} 倍；"
        f"到达阶段 {metadata['evaluation_arrival_ms']:g} ms。",
        f"预测跨度 {metadata['forecast_horizon_ms']:g} ms；模型不是实际 LLM profiling。",
        "联合达标率按全部到达请求计。以下为轨迹均值，单位 %，并非统计显著性结论。",
        "", "| 方法 | " + " | ".join(f"{p:g} ms" for p in periods) + " |",
        "|---|" + "---:|" * len(periods),
    ]
    lookup = {(r["method"], r["report_period_ms"]): r for r in summary}
    for method in methods:
        lines.append("| " + method + " | " + " | ".join(
            f"{100 * lookup[method, p]['success_rate_mean']:.2f}" for p in periods
        ) + " |")
    lines.extend([
        "", "stats_ppo 和 ae_ppo 同样具有预测器与 PPO 长期回报优化。",
        "semantic_no_forecast_ppo 独立训练：策略输入的显式未来轨迹替换为当前解码状态的保持值，"
        "并移除预测辅助损失。latent/actor 仍可能隐式编码趋势，不能称为完全无法预测。",
        "semantic_trajectory 与 semantic_reactive 使用同一编码器和启发式，只切换显式预测。",
        "启发式与 PPO 的差异不能单独归因于长期优化；没有做短视 RL 的训练对照。",
        "预测跨度以报告的采样时间为起点；跨度外延用预测终值，不等于仍有可靠预测。",
        "每个阶段内部核对所有方法及所有上报周期的外生轨迹哈希；不同预测跨度改变"
        "仿真预生成数组长度，因此两个阶段不能直接相减归因。",
    ])
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    labels = {
        "latest_periodic": "Latest + heuristic",
        "stats_ppo": "Statistics + forecast + PPO",
        "ae_ppo": "Frozen AE + forecast + PPO",
        "semantic_ppo": "Joint encoder + forecast + PPO",
        "semantic_no_forecast_ppo": "Joint encoder + PPO, no explicit forecast",
        "semantic_trajectory": "Joint encoder + forecast heuristic",
        "semantic_reactive": "Joint encoder + reactive heuristic",
    }
    for method in methods:
        if method.startswith("semantic_") and method.endswith(("trajectory", "reactive")):
            continue
        group = [lookup[method, p] for p in periods]
        axes[0].errorbar(periods, [100 * r["success_rate_mean"] for r in group],
                        yerr=[100 * r["success_rate_std"] for r in group],
                        marker="o", capsize=2, label=labels[method])
        if method == methods[0]:
            axes[1].plot(periods, [r["mean_state_age_ms_mean"] for r in group],
                         marker="o", color="#374151")
            axes[2].plot(periods, [r["report_bps_actual_mean"] / 1000 for r in group],
                         marker="o", color="#374151")
    for ax in axes:
        ax.set_xlabel("Reporting interval (ms)")
        ax.set_xticks(periods)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Quality + deadline success (%)")
    axes[0].set_title("Mean and trajectory standard deviation", fontsize=9)
    axes[1].set_ylabel("Received state age (ms)")
    axes[2].set_ylabel("Actual reporting (kbit/s)")
    axes[1].set_title("Identical across methods", fontsize=9)
    axes[2].set_title("Identical across methods", fontsize=9)
    axes[0].legend(fontsize=6.5, loc="best")
    fig.suptitle(f"{metadata['training_regime']}; fixed 16 floats; "
                 f"{metadata['bandwidth_bps'] / 1000:g} kbit/s; load {metadata['load_multiplier']:g}")
    fig.savefig(out / "frequency.png", dpi=180)
    fig.savefig(out / "frequency.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/icc_paper/scenario_2026.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Replay old or matched frequency weights")
    parser.add_argument("--periods", type=float, nargs="+", default=[20, 50, 100, 200])
    parser.add_argument("--bandwidth", type=float, default=64000)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[231, 232, 233])
    parser.add_argument("--arrival-steps", type=int, default=400)
    parser.add_argument("--train-arrival-steps", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if any(not np.isfinite(p) or p <= 0 or p % 5 for p in args.periods):
        raise SystemExit("Periods must be positive multiples of the 5 ms sampling slot")
    if any(1000 <= s < 100000 for s in args.eval_seeds):
        raise SystemExit("Evaluation seeds overlap reserved training/validation range")
    if args.output.exists():
        raise SystemExit("Choose a fresh output directory")
    args.output.mkdir(parents=True)
    out, started = args.output, perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(args.train_seed)
    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved["dataset_hash"] != digest:
            raise SystemExit("Checkpoint dataset differs")
        raw_config = dict(saved["config"])
        raw_config.setdefault("service_process", "sampled_inverse_rate")
        config = Config(**raw_config)
        models = {}
        for name, weights in saved["models"].items():
            model = SemanticNet(config.history, config.horizon,
                                "stats" if name == "stats" else "semantic",
                                use_forecast=name != "semantic_no_forecast")
            model.load_state_dict(weights)
            models[name] = model.eval()
        training = saved["training"]
        regime = saved.get("training_regime", "Frozen old event-trained checkpoint")
    else:
        config = Config(arrival_steps=args.train_arrival_steps, horizon=args.horizon,
                        report_bps=args.bandwidth)
        print("Pretraining shared encoder initialization and forecasting baselines", flush=True)
        train = collect_pretraining(args.dataset, config, range(1000, 1006))
        valid = collect_pretraining(args.dataset, config, (2000, 2001))
        base = SemanticNet(config.history, config.horizon)
        initial = pretrain(base, train, valid, args.epochs, args.train_seed)
        stats = SemanticNet(config.history, config.horizon, "stats")
        stats_initial = pretrain(stats, train, valid, args.epochs, args.train_seed)
        models = {"ae": copy.deepcopy(base), "semantic": copy.deepcopy(base),
                  "stats": stats, "semantic_no_forecast": copy.deepcopy(base)}
        models["semantic_no_forecast"].use_forecast = False
        training = {"training_seed": args.train_seed, "pretraining": initial,
                    "stats_pretraining": stats_initial, "periods_ms": args.periods,
                    "epochs": args.epochs, "train_windows": len(train[0]),
                    "validation_windows": len(valid[0])}
        for name, model in models.items():
            print(f"Training {name}, matched periodic reporting {args.periods}", flush=True)
            training[name] = train_policy(
                model, args.dataset, config, args.updates, args.train_seed,
                joint=name.startswith("semantic"), reporting="periodic",
                report_periods=args.periods, bandwidths=[args.bandwidth], prefix_sizes=[16],
            )
            model.eval()
            dump(out / "training.json", training)
        dump(out / "forecast_diagnostics.json", forecast_errors(models, valid, config.slot_ms))
        regime = "Matched periodic training, 300 ms forecast" if config.horizon == 60 else (
            f"Matched periodic training, {config.horizon * config.slot_ms:g} ms forecast")
        saved = {"config": asdict(config), "dataset_hash": digest, "training": training,
                 "models": {name: model.state_dict() for name, model in models.items()},
                 "surrogate_models": DEFAULT_MODELS, "training_regime": regime}
        torch.save(saved, out / "checkpoint.pt")
    metadata = {
        "training_regime": regime, "config": asdict(config), "dataset_hash": digest,
        "report_periods_ms": args.periods, "bandwidth_bps": args.bandwidth,
        "load_multiplier": config.load_multiplier, "evaluation_seeds": args.eval_seeds,
        "evaluation_arrival_ms": args.arrival_steps * config.slot_ms,
        "forecast_horizon_ms": config.horizon * config.slot_ms,
        "checkpoint_source": str(args.checkpoint or out / "checkpoint.pt"),
        "all_reporting": "periodic", "all_packet_bytes": 80,
        "quality_min": config.quality_min,
        "encoder_delay_ms": config.encoder_ms,
        "model_status": "surrogate, not measured LLM profiles",
        "data_status": "ICC parameter reconstruction, not author-original samples",
    }
    dump(out / "training.json", training)
    dump(out / "icc_scenario.json", json.loads(args.dataset.read_text()))
    dump(out / "surrogate_models.json", DEFAULT_MODELS)
    if args.train_only:
        metadata["elapsed_seconds"] = perf_counter() - started
        dump(out / "metadata.json", metadata)
        print(f"Saved training checkpoint to {out}", flush=True)
        return
    rows, trace_hashes = [], {}
    for period in args.periods:
        config_eval = replace(config, report_period_ms=period, report_bps=args.bandwidth,
                              arrival_steps=args.arrival_steps)
        for seed in args.eval_seeds:
            for name, codec, router in periodic_suite(config_eval, models):
                env = RoutingEnv(args.dataset, config_eval, seed, codec, "periodic")
                if seed in trace_hashes and env.trace_hash != trace_hashes[seed]:
                    raise RuntimeError("Methods or periods changed the exogenous trace")
                trace_hashes[seed] = env.trace_hash
                row = evaluate(env, router)
                row.update(method=name, report_period_ms=period, seed=seed,
                           capacity_bps=args.bandwidth, load_multiplier=config.load_multiplier,
                           dataset_hash=digest,
                           report_generated_bytes=env.channel.generated_bytes,
                           report_replacements=env.channel.replaced_reports,
                           nominal_reports_per_second_per_node=1000 / period,
                           report_packet_bytes=80)
                rows.append(row)
            with (out / "results.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Evaluated period={period:g} ms, seed={seed}, episodes={len(rows)}", flush=True)
    metadata.update(episodes=len(rows), elapsed_seconds=perf_counter() - started,
                    paired_trace_hashes=trace_hashes)
    dump(out / "metadata.json", metadata)
    write_report(rows, metadata, out)
    print(f"Saved {len(rows)} episodes to {out}; {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
