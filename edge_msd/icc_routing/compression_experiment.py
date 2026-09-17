"""True 4/8-dimensional telemetry versus 16-float statistics, with matched PPO training."""

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
from .experiment import dump, evaluate
from .frequency_experiment import summarize
from .learning import Codec, Policy, SemanticNet, collect_pretraining, pretrain, train_policy


def load_low_checkpoint(path, dataset_hash):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["dataset_hash"] != dataset_hash:
        raise ValueError("Checkpoint dataset differs")
    models = {}
    for name, weights in saved["models"].items():
        spec = saved["model_specs"][name]
        model = SemanticNet(saved["config"]["history"], saved["config"]["horizon"],
                            latent_dim=spec["latent_dim"])
        model.load_state_dict(weights)
        models[name] = model.eval()
    return saved, models


def train(args, dataset_hash):
    cfg = Config(arrival_steps=200, horizon=60, report_bps=args.bandwidth)
    d = args.dimension
    print(f"Collecting histories for true {d}-dimensional bottleneck", flush=True)
    training_data = collect_pretraining(args.dataset, cfg, range(1000, 1006))
    validation = collect_pretraining(args.dataset, cfg, (2000, 2001))
    initial = SemanticNet(cfg.history, cfg.horizon, latent_dim=d)
    initial_metrics = pretrain(initial, training_data, validation,
                               args.epochs, args.train_seed, prefix_sizes=[d])
    models = {f"ae{d}": copy.deepcopy(initial), f"semantic{d}": copy.deepcopy(initial)}
    records = {
        "training_seed": args.train_seed, "pretraining": initial_metrics,
        "epochs": args.epochs, "train_windows": len(training_data[0]),
        "validation_windows": len(validation[0]),
        "pretraining_prefix_sizes": [d], "periods_ms": args.periods,
    }
    for name, model in models.items():
        print(f"Training {name}: {d} transmitted float32 values in ALL stages", flush=True)
        records[name] = train_policy(
            model, args.dataset, cfg, args.updates, args.train_seed,
            joint=name.startswith("semantic"), reporting="periodic",
            report_periods=args.periods, bandwidths=[args.bandwidth], prefix_sizes=[d],
        )
        model.eval()
        assert records[name]["warm_prefix_sizes"] == [d, d]
        assert records[name]["prefix_sizes"] == [d]
        assert all(torch.isfinite(p).all() for p in model.parameters())
        # Preserve a recoverable checkpoint after each completed policy.
        specs = {key: {"latent_dim": d, "wire_floats": d} for key in models if key in records}
        torch.save({
            "config": asdict(cfg), "dataset_hash": dataset_hash,
            "models": {key: models[key].state_dict() for key in specs},
            "model_specs": specs, "training": records,
            "surrogate_models": DEFAULT_MODELS,
        }, args.output / "checkpoint.pt")
        dump(args.output / "training.json", records)
    x, y = [torch.as_tensor(v) for v in validation]
    validation_metrics = {}
    with torch.no_grad():
        for name, model in models.items():
            z = model.encode(x, torch.full((len(x),), d))
            assert not torch.count_nonzero(z[:, d:])
            validation_metrics[name] = {
                "reconstruction_mse": float((model.decode(z) - x).square().mean()),
                "forecast_mse": float((model.forecast(z) - y).square().mean()),
                "persistence_mse": float((x[:, -1, [1, 5, 6]][:, None] - y).square().mean()),
                "validation_source": "instant-state heuristic histories, seeds 2000 and 2001",
            }
    dump(args.output / "validation.json", validation_metrics)
    return {"stage": "train", "dimension": d, "config": asdict(cfg),
            "training_seed": args.train_seed, "report_periods_ms": args.periods,
            "packet_bytes": 16 + 4 * d, "models": list(models)}


def read_rows(path):
    rows = []
    for row in csv.DictReader(Path(path).open()):
        for key, value in row.items():
            if key not in ("method", "trace_hash", "dataset_hash"):
                row[key] = None if value == "" else int(value) if value.isdigit() else float(value)
        rows.append(row)
    return rows


def paired_comparisons(rows):
    result = []
    for period in sorted({r["report_period_ms"] for r in rows}):
        index = {(r["method"], r["seed"]): r for r in rows if r["report_period_ms"] == period}
        for proposed, reference in (
            ("semantic4", "stats16"), ("semantic8", "stats16"),
            ("semantic4", "ae4"), ("semantic8", "ae8"),
            ("semantic4", "semantic16"), ("semantic8", "semantic16"),
        ):
            seeds = sorted(s for m, s in index if m == proposed and (reference, s) in index)
            if not seeds:
                continue
            diffs = [100 * (index[proposed, s]["success_rate"]
                            - index[reference, s]["success_rate"]) for s in seeds]
            saving = [1 - index[proposed, s]["report_transmitted_bytes"]
                      / index[reference, s]["report_transmitted_bytes"] for s in seeds]
            result.append({"report_period_ms": period, "proposed": proposed,
                           "reference": reference, "seeds": seeds,
                           "differences_pp": diffs, "mean_pp": float(np.mean(diffs)),
                           "std_pp": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0,
                           "actual_sent_byte_reduction_mean": float(np.mean(saving)),
                           "wins": int(sum(d > 0 for d in diffs))})
    return result


def write_results(rows, metadata, out):
    out = Path(out)
    summary = summarize(rows)
    dump(out / "summary.json", summary)
    dump(out / "paired_comparisons.json", paired_comparisons(rows))
    periods = sorted({r["report_period_ms"] for r in rows})
    order = [m for m in ("stats16", "semantic4", "semantic8", "ae4", "ae8",
                         "ae16", "semantic16") if any(r["method"] == m for r in rows)]
    lookup = {(r["method"], r["report_period_ms"]): r for r in summary}
    sizes = {r["method"]: int(r["packet_bytes"]) for r in rows}
    lines = [
        "# 4/8 维压缩对 16 维统计量", "",
        "主设置为真正的 4 维编码器瓶颈，8 维为消融；16 维学习模型仅作为辅助对照。",
        "所有值均为 float32，每报文另计 16 B 报头。编码器、预测器和路由策略均按"
        "各自实际瓶颈维度训练；接收端接口的补零不传输、不携带信息。",
        f"共享容量 {metadata['bandwidth_bps'] / 1000:g} kbit/s；"
        f"ICC 重建场景到达率乘数 {metadata['load_multiplier']:g}；"
        f"每条轨迹到达阶段 {metadata['evaluation_arrival_ms']:g} ms。",
        "环境与上一轮一致，没有新增耦合过程；代理模型仍不是真实 LLM 测量。",
        "下表为质量与截止时间联合达标率的轨迹均值，单位 %，分母为全部到达请求。",
        "", "| 方法 | 报文 B | " + " | ".join(f"{p:g} ms" for p in periods) + " |",
        "|---|---:|" + "---:|" * len(periods),
    ]
    for method in order:
        lines.append(f"| {method} | {sizes[method]} | " + " | ".join(
            f"{100 * lookup[method, p]['success_rate_mean']:.2f}" for p in periods) + " |")
    lines.extend([
        "", "## 实际发送与状态时效", "",
        "| 周期 ms | 方法 | 实发 kbit/s | 决策时信息年龄 ms |",
        "|---:|---|---:|---:|",
    ])
    for p in periods:
        for method in ("stats16", "semantic4", "semantic8"):
            r = lookup[method, p]
            lines.append(f"| {p:g} | {method} | {r['report_bps_actual_mean'] / 1000:.2f} | "
                         f"{r['mean_state_age_ms_mean']:.2f} |")
    lines.extend([
        "", "单报文 80→32 B 减少 60%，80→48 B 减少 40%。但链路饱和时，"
        "实际发送量受容量限制，不能把单包比例直接套到实际发送速率上。",
        "16 维基线复用此前同配置、同训练种子、同训练轨迹与 PPO 预算的权重，并在"
        "本轮测试种子上重新评估。预训练瓶颈架构不同，不宣称神经网络参数逐项相同。",
        "固定 float32 比较未包括量化或删减后的统计量，因此没有证明最低比特或普遍最优。",
        "仅一个训练种子；预测和长期回报在统计量与 AE 基线中同样存在。",
    ])
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    palette = {"stats16": "#475569", "semantic4": "#dc2626", "semantic8": "#2563eb"}
    labels = {"stats16": "Statistics, 16D / 80 B", "semantic4": "Proposed, 4D / 32 B",
              "semantic8": "Proposed, 8D / 48 B"}
    for method in palette:
        group = [lookup[method, p] for p in periods]
        axes[0].errorbar(periods, [100 * r["success_rate_mean"] for r in group],
                        yerr=[100 * r["success_rate_std"] for r in group],
                        marker="o", capsize=3, color=palette[method], label=labels[method])
        axes[1].plot(periods, [r["report_bps_actual_mean"] / 1000 for r in group],
                     marker="o", color=palette[method])
        axes[2].plot(periods, [r["mean_state_age_ms_mean"] for r in group],
                     marker="o", color=palette[method])
    for ax in axes:
        ax.set_xlabel("Reporting interval (ms)")
        ax.set_xticks(periods)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Quality + deadline success (%)")
    axes[0].set_title("Mean and trajectory standard deviation", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Actual reporting (kbit/s)")
    axes[2].set_ylabel("Decision-time state age (ms)")
    fig.suptitle(f"True low-dimensional bottlenecks; {metadata['bandwidth_bps'] / 1000:g} kbit/s; "
                 f"ICC parameter reconstruction; load {metadata['load_multiplier']:g}")
    fig.savefig(out / "compression.png", dpi=180)
    fig.savefig(out / "compression.pdf")
    plt.close(fig)


def evaluate_checkpoints(args, dataset_hash):
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    if reference["dataset_hash"] != dataset_hash:
        raise ValueError("Reference dataset differs")
    cfg = Config(**reference["config"])
    models, dimensions = {}, {}
    for key, name in (("stats", "stats16"), ("ae", "ae16"), ("semantic", "semantic16")):
        model = SemanticNet(cfg.history, cfg.horizon, "stats" if key == "stats" else "semantic")
        model.load_state_dict(reference["models"][key])
        models[name], dimensions[name] = model.eval(), 16
    low_sources = {}
    for path in args.low_checkpoints:
        saved, low_models = load_low_checkpoint(path, dataset_hash)
        if saved["config"] != reference["config"]:
            raise ValueError("Low-dimensional and reference training configurations differ")
        if saved["training"]["periods_ms"] != reference["training"]["periods_ms"]:
            raise ValueError("Training report periods differ")
        if saved["training"]["training_seed"] != reference["training"]["training_seed"]:
            raise ValueError("Training random seeds differ")
        if saved["training"]["epochs"] != reference["training"]["epochs"]:
            raise ValueError("Pretraining budgets differ")
        for name, model in low_models.items():
            if len(saved["training"][name]["updates"]) != len(reference["training"]["stats"]["updates"]):
                raise ValueError("PPO update budgets differ")
            if name in models:
                raise ValueError("Duplicate method")
            models[name], dimensions[name] = model, saved["model_specs"][name]["wire_floats"]
            low_sources[name] = str(path)
    if not {"ae4", "ae8", "semantic4", "semantic8"} <= models.keys():
        raise ValueError("Both complete 4D and 8D checkpoints are required")
    rows, hashes = [], {}
    for period in args.periods:
        case = replace(cfg, arrival_steps=args.arrival_steps,
                       report_period_ms=period, report_bps=args.bandwidth)
        for seed in args.eval_seeds:
            for name in ("stats16", "semantic4", "semantic8", "ae4", "ae8", "ae16", "semantic16"):
                codec = Codec("stats" if name == "stats16" else "semantic", models[name], dimensions[name])
                env = RoutingEnv(args.dataset, case, seed, codec, "periodic")
                if seed in hashes and hashes[seed] != env.trace_hash:
                    raise RuntimeError("Exogenous trajectories changed across dimensions or periods")
                hashes[seed] = env.trace_hash
                row = evaluate(env, Policy(models[name], codec))
                packet_bytes = 16 + 4 * dimensions[name]
                expected_reports = 3 * int(np.ceil(env.total_steps * case.slot_ms / period))
                assert env.channel.generated_bytes == expected_reports * packet_bytes
                assert env.channel.transmitted_bytes <= case.report_bps * env.now_ms / 8000 + 1e-6
                row.update(method=name, seed=seed, report_period_ms=period,
                           capacity_bps=args.bandwidth, load_multiplier=case.load_multiplier,
                           dataset_hash=dataset_hash, latent_dim=dimensions[name],
                           packet_bytes=packet_bytes, report_generated_bytes=env.channel.generated_bytes,
                           report_replacements=env.channel.replaced_reports)
                rows.append(row)
            with (args.output / "results.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Evaluated period={period:g} ms seed={seed}; {len(rows)} episodes", flush=True)
    metadata = {
        "stage": "evaluate", "config": asdict(cfg), "episodes": len(rows),
        "evaluation_seeds": args.eval_seeds, "report_periods_ms": args.periods,
        "bandwidth_bps": args.bandwidth, "load_multiplier": cfg.load_multiplier,
        "evaluation_arrival_ms": args.arrival_steps * cfg.slot_ms,
        "reference_checkpoint": str(args.reference), "low_checkpoints": low_sources,
        "paired_trace_hashes": hashes, "dimensions": dimensions,
        "primary_method": "semantic4", "dimension_ablation": "semantic8",
        "compression_dtype": "float32", "header_bytes": 16,
        "environment_changed": False, "quantized_statistics_included": False,
    }
    dump(args.output / "metadata.json", metadata)
    write_results(rows, metadata, args.output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/icc_paper/scenario_2026.json"))
    parser.add_argument("--dimension", type=int, choices=(4, 8), default=4)
    parser.add_argument("--periods", type=float, nargs="+", default=[20, 50, 100, 200])
    parser.add_argument("--bandwidth", type=float, default=64000)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[241, 242, 243])
    parser.add_argument("--arrival-steps", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--reference", type=Path, default=Path("runs/icc_frequency_matched/checkpoint.pt"))
    parser.add_argument("--low-checkpoints", type=Path, nargs="+", default=[
        Path("runs/icc_compression_d4/checkpoint.pt"), Path("runs/icc_compression_d8/checkpoint.pt")])
    args = parser.parse_args()
    if any(not np.isfinite(p) or p <= 0 or p % 5 for p in args.periods):
        raise SystemExit("Periods must be positive multiples of the 5 ms slot")
    if any(1000 <= s < 100000 for s in args.eval_seeds):
        raise SystemExit("Evaluation seeds overlap the reserved training/validation range")
    if args.output.exists():
        raise SystemExit("Choose a fresh output directory")
    args.output.mkdir(parents=True)
    started = perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(args.train_seed)
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    metadata = train(args, dataset_hash) if args.stage == "train" else evaluate_checkpoints(args, dataset_hash)
    metadata.update(elapsed_seconds=perf_counter() - started, dataset_hash=dataset_hash,
                    data_status="ICC published-parameter reconstruction; not author samples",
                    model_status="Synthetic surrogate profiles; not measured LLM performance")
    dump(args.output / "metadata.json", metadata)
    dump(args.output / "icc_scenario.json", json.loads(args.dataset.read_text()))
    dump(args.output / "surrogate_models.json", DEFAULT_MODELS)
    print(f"Saved {args.stage} artifacts to {args.output}; {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
