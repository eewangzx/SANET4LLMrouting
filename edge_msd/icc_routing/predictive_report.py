"""Aggregate predictive-compression experiments without selecting on test results."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .experiment import dump
from .predictive_experiment import SPECS

LABELS = {
    "dense_joint": "Dense 4D + joint PPO",
    "if_joint": "Importance 3/8 + joint PPO",
    "if_frozen": "Importance 3/8, frozen encoder",
    "random_joint": "Random 3/8 + joint PPO",
    "stats16": "Statistics 16D + PPO",
    "adaptive_joint": "Importance + PPO budget (1-3/8)",
    "public_rule": "Public rule; zero telemetry",
}
COLORS = dict(zip(SPECS, ("#2563eb", "#dc2626", "#a16207", "#9333ea", "#475569", "#059669")))
COLORS["public_rule"] = "#111827"


def read_episodes(paths):
    rows = []
    for path in paths:
        for source in csv.DictReader(Path(path).open()):
            r = {}
            for key, value in source.items():
                if key in ("method", "checkpoint", "trace_hash"):
                    r[key] = value
                else:
                    r[key] = float(value) if value else None
            # Counterfactual accounting only: same actions/outcomes, add back fee.
            r["task_reward_per_arrival"] = (r["reward_per_arrival"] +
                r["report_transmitted_bytes"] / (32 * max(1, r["arrivals"])))
            rows.append(r)
    keys = [(r["method"], r["train_seed"], r["test_seed"], r["period_ms"], r["checkpoint"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate evaluation episode")
    for seed in {r["test_seed"] for r in rows}:
        if len({r["trace_hash"] for r in rows if r["test_seed"] == seed}) != 1:
            raise ValueError("Exogenous traces do not match")
    return rows


def aggregate(rows):
    out = []
    metrics = ("success_rate", "reward_per_arrival", "task_reward_per_arrival", "report_bps_actual",
               "mean_state_age_ms", "mean_packet_bytes", "mean_selected_dimensions",
               "p95_completed_latency_ms", "budget_k1_fraction", "budget_k2_fraction", "budget_k3_fraction")
    for name, period, checkpoint in sorted({(r["method"], r["period_ms"], r["checkpoint"]) for r in rows}):
        group = [r for r in rows if (r["method"], r["period_ms"], r["checkpoint"]) == (name, period, checkpoint)]
        entry = dict(method=name, period_ms=period, checkpoint=checkpoint, n=len(group))
        for key in metrics:
            values = [r[key] for r in group if r[key] is not None]
            entry[key + "_mean"] = float(np.mean(values)) if values else None
            entry[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        out.append(entry)
    return out


def comparisons(rows):
    out = []
    index = {(r["method"], r["train_seed"], r["test_seed"], r["period_ms"], r["checkpoint"]): r for r in rows}
    pairs = (("if_joint", "dense_joint"), ("if_joint", "if_frozen"), ("if_joint", "random_joint"),
             ("if_joint", "stats16"), ("adaptive_joint", "if_joint"), ("adaptive_joint", "stats16"))
    for period, checkpoint in sorted({(r["period_ms"], r["checkpoint"]) for r in rows}):
        for proposed, reference in pairs:
            differences, reward, bytes_saved = [], [], []
            for train_seed, test_seed in sorted({(r["train_seed"], r["test_seed"]) for r in rows}):
                a = index.get((proposed, train_seed, test_seed, period, checkpoint))
                b = index.get((reference, train_seed, test_seed, period, checkpoint))
                if a is None or b is None:
                    continue
                differences.append(100 * (a["success_rate"] - b["success_rate"]))
                reward.append(a["reward_per_arrival"] - b["reward_per_arrival"])
                bytes_saved.append(1 - a["report_transmitted_bytes"] / b["report_transmitted_bytes"])
            out.append(dict(proposed=proposed, reference=reference, period_ms=period, checkpoint=checkpoint,
                            sla_differences_pp=differences, mean_sla_pp=float(np.mean(differences)),
                            mean_reward_difference=float(np.mean(reward)),
                            mean_actual_byte_reduction=float(np.mean(bytes_saved)),
                            wins=int(sum(d > 0 for d in differences)), n=len(differences)))
    return out


def training_diagnostics(models):
    out = []
    for directory in sorted(Path(models).glob("seed*/*")):
        if not (directory / "training_summary.json").exists():
            continue
        valid = json.loads((directory / "validation.json").read_text())
        ppo = json.loads((directory / "ppo.json").read_text())
        pre = json.loads((directory / "pretraining.json").read_text())
        meta = json.loads((directory / "metadata.json").read_text())
        tail = valid[-4:]
        sla = np.asarray([r["validation_success"] for r in tail])
        reward = np.asarray([r["validation_reward"] for r in tail])
        span = float(np.ptp(sla) * 100)
        relative = float(np.ptp(reward) / max(abs(reward.mean()), 0.1))
        out.append(dict(method=directory.name, train_seed=meta["train_seed"],
                        tail_sla_span_pp=span, tail_reward_relative_span=relative,
                        greedy_validation_metrics_flat=bool(span <= 2 and relative <= 0.05),
                        full_policy_convergence_assessed=False,
                        first_validation_success=valid[0]["validation_success"],
                        final_validation_success=valid[-1]["validation_success"],
                        first_validation_reward=valid[0]["validation_reward"],
                        final_validation_reward=valid[-1]["validation_reward"],
                        pretraining_initial_mse=pre[0]["forecast_mse"],
                        pretraining_final_mse=pre[-1]["forecast_mse"],
                        final_forecast_mse=valid[-1]["forecast_mse"],
                        persistence_mse=valid[-1]["persistence_mse"],
                        max_kl=max(r["approx_kl"] for r in ppo),
                        final_entropy=ppo[-1]["entropy"],
                        final_value_loss=ppo[-1]["value_loss"],
                        max_wire_logprob_error=max(r["initial_max_logprob_difference"] for r in ppo),
                        final_unique_masks=valid[-1].get("unique_masks"),
                        final_selection_frequencies=valid[-1].get("selection_frequencies"),
                        final_deployed_forecast_mse=valid[-1].get("deployed_forecast_mse"),
                        final_deployed_mean_keep=valid[-1].get("deployed_mean_keep")))
    return out


def plots(models, summary, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for name in SPECS:
        records = [json.loads(p.read_text()) for p in sorted(Path(models).glob(f"seed*/{name}/validation.json"))]
        if not records:
            continue
        updates = [r["update"] for r in records[0]]
        for ax, key, scale in ((axes[0, 0], "validation_success", 100),
                               (axes[0, 1], "validation_reward", 1),
                               (axes[1, 0], "forecast_mse", 1)):
            values = np.array([[r[key] * scale for r in group] for group in records])
            ax.plot(updates, values.mean(0), label=LABELS[name], color=COLORS[name])
            ax.fill_between(updates, values.min(0), values.max(0), color=COLORS[name], alpha=0.09)
        pp = [json.loads(p.read_text()) for p in sorted(Path(models).glob(f"seed*/{name}/ppo.json"))]
        values = np.array([[r["entropy"] for r in group] for group in pp])
        axes[1, 1].plot(np.arange(1, len(values[0]) + 1), values.mean(0), color=COLORS[name])
    for ax, label in zip(axes.flat, ("Validation quality + deadline success (%)",
                                    "Validation reward per arrival", "Held-out future prediction MSE",
                                    "Routing policy entropy (nats)")):
        ax.set_xlabel("PPO update")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=7, loc="best")
    axes[0, 0].text(0.04, 0.12, "Overlapping curves can reflect a fixed routing rule.\nSee zero-Z and zero-telemetry controls.",
                    transform=axes[0, 0].transAxes, fontsize=8)
    axes[1, 0].set_title("Adaptive curve uses k=3; deployed-k MSE is reported separately", fontsize=8)
    fig.suptitle("Independent validation; shading = two training-seed range (not confidence interval)")
    fig.savefig(out / "convergence.png", dpi=170)
    fig.savefig(out / "convergence.pdf")
    plt.close(fig)

    best = [r for r in summary if r["checkpoint"] == "best"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    for name in list(SPECS) + ["public_rule"]:
        records = sorted([r for r in best if r["method"] == name], key=lambda r: r["period_ms"])
        if not records:
            continue
        x = [r["period_ms"] for r in records]
        axes[0].errorbar(x, [100*r["success_rate_mean"] for r in records],
                         yerr=[100*r["success_rate_std"] for r in records],
                         color=COLORS[name], marker="o", capsize=2, label=LABELS[name])
        axes[1].plot(x, [r["report_bps_actual_mean"]/1000 for r in records], color=COLORS[name], marker="o")
        axes[2].plot(x, [r["reward_per_arrival_mean"] for r in records], color=COLORS[name], marker="o")
    for ax, ylabel in zip(axes, ("Quality + deadline success (%)", "Actual reporting (kbit/s)", "Reward per arrival")):
        ax.set_xlabel("Reporting interval (ms)")
        ax.set_xticks([50, 100, 200])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=6.5)
    fig.suptitle("Held-out test; checkpoints selected by validation reward; error bars = descriptive SD")
    fig.savefig(out / "test_results.png", dpi=180)
    fig.savefig(out / "test_results.pdf")
    plt.close(fig)


def write_markdown(summary, convergence, pairs, out, n):
    lookup = {(r["method"], r["period_ms"], r["checkpoint"]): r for r in summary}
    lines = ["# 预测压缩与 importance filter 联合 PPO：本轮结果", "",
             "**结论：实现与训练流程跑通，但本轮没有验证出网络感知路由增益。零上报公共规则达到同样 SLA；不能把相同 SLA 下减少报文量归为有效的学习压缩收益。**", "",
             f"共 {n} 次测试评估：6 个方法、2 个训练种子、3 个外生测试种子、3 个上报周期，"
             "分别评估验证选出的 best 与最后的 last。重复使用同一外生测试场景，因此不将所有点视为独立置信区间样本。",
             "", "## SLA 与通信开销", "",
             "下表按验证 reward 选 checkpoint，测试集不参与选择。SLA 为质量与 deadline 联合达标率 / 全部到达请求。",
             "", "| 方法 | 50 ms SLA % | 100 ms SLA % | 200 ms SLA % | 100 ms 实发 kbit/s | 平均包长 B |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in list(SPECS) + ["public_rule"]:
        if (name, 100.0, "best") not in lookup:
            continue
        r = lookup[name, 100.0, "best"]
        lines.append(f"| {name} | " + " | ".join(f"{100*lookup[name,p,'best']['success_rate_mean']:.2f}" for p in (50.,100.,200.)) +
                     f" | {r['report_bps_actual_mean']/1000:.2f} | {r['mean_packet_bytes_mean']:.2f} |")
    lines += ["", "## 最后 checkpoint，避免只展示峰值", "",
              "| 方法 | 50 ms SLA % | 100 ms SLA % | 200 ms SLA % |", "|---|---:|---:|---:|"]
    for name in SPECS:
        lines.append(f"| {name} | " + " | ".join(f"{100*lookup[name,p,'last']['success_rate_mean']:.2f}" for p in (50.,100.,200.)) + " |")
    lines += ["", "## 贪心验证指标的平坦程度", "",
              "最后 4 个验证 checkpoint 的 argmax 策略 SLA 跨度 ≤2 pp 且 reward 相对跨度 ≤5% 才标记指标平坦。"
              "此项没有检验完整策略分布、采样回报或 critic，不能用于判断 PPO 是否收敛。"
              "update 0 已包含统一的公开信息规则热启动，后续差值才是 PPO 阶段的变化。", "",
              "| 方法 | 种子 | 贪心验证 SLA 初始→最终 % | 末段 SLA 跨度 pp | reward 相对跨度 | 贪心指标平坦 |",
              "|---|---:|---:|---:|---:|---|"]
    for r in convergence:
        lines.append(f"| {r['method']} | {r['train_seed']} | {100*r['first_validation_success']:.2f}→"
                     f"{100*r['final_validation_success']:.2f} | {r['tail_sla_span_pp']:.2f} | "
                     f"{100*r['tail_reward_relative_span']:.1f}% | {'是' if r['greedy_validation_metrics_flat'] else '否'} |")
    lines += ["", "## 消融的配对差值", "",
              "同训练种子、同外生测试种子逐一配对；6 次配对的胜负不是显著性检验。正 SLA 差值代表前者更高。", "",
              "| 前者 / 后者 | 周期 ms | SLA 差 pp | 实际字节减少 | reward 差 | SLA 胜数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in pairs:
        if r["checkpoint"] == "best":
            lines.append(f"| {r['proposed']} / {r['reference']} | {r['period_ms']:g} | {r['mean_sla_pp']:+.2f} | "
                         f"{100*r['mean_actual_byte_reduction']:.1f}% | {r['mean_reward_difference']:+.3f} | {r['wins']}/{r['n']} |")
    lines += ["", "数据仍为 ICC 参数重建，LLM 模型为合成代理配置。一次标量 reward 权重实验不能证明满足任意 SLA 约束，"
              "也没有证明最小维数或 importance filter 必然胜出。", "",
              "原始无热启动 pilot 及源代码快照保留。该轮路由 entropy 接近均匀，验证策略波动；诊断后为全部方法增加"
              "相同的 20 epoch 公共信息规则热启动，再从头按相同预算训练。本轮提升不能全部归因于 PPO 或筛选模块。",
              "", "完整建模、损失、数据口径见 `docs/ICC_PREDICTIVE_FILTER_DESIGN.md`。"]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


def public_summary(path):
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows = []
    for source in data["rows"]:
        for period in (50., 100., 200.):
            for checkpoint in ("best", "last"):
                rows.append({**source, "method": "public_rule", "period_ms": period,
                             "checkpoint": checkpoint, "task_reward_per_arrival": source["reward_per_arrival"],
                             "mean_packet_bytes": 0., "mean_selected_dimensions": 0.,
                             "budget_k1_fraction": 0., "budget_k2_fraction": 0., "budget_k3_fraction": 0.})
    return aggregate(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", nargs="+", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--public-baseline", type=Path, default=Path("runs/icc_predictive_public_rule_test.json"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_episodes(args.episodes)
    with (args.output / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary, convergence, paired = aggregate(rows), training_diagnostics(args.models), comparisons(rows)
    summary.extend(public_summary(args.public_baseline))
    dump(args.output / "summary.json", summary)
    dump(args.output / "convergence.json", convergence)
    dump(args.output / "paired_comparisons.json", paired)
    plots(args.models, summary, args.output)
    write_markdown(summary, convergence, paired, args.output, len(rows))
    print(f"Aggregated {len(rows)} test episodes and {len(convergence)} trained policies")


if __name__ == "__main__":
    main()
