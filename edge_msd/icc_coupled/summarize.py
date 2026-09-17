"""Read completed or in-progress logs; never infer convergence from entropy alone."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(baselines, policy_dirs, output, fixed=None):
    output.mkdir(parents=True, exist_ok=True)
    lines = ['# ICC 耦合环境：结果快照', '',
             'SLA 为完整 DAG 的原 deadline；表内仅列实际完成的评估。'
             '两个环境种子、一个训练种子是初步实验，不构成统计显著或收敛证明。', '']
    records = {'baselines': [], 'policies': []}
    if (baselines / 'episodes.csv').exists():
        rows = list(csv.DictReader((baselines / 'episodes.csv').open()))
        lines += ['## 固定 ICC 控制器的遥测对照', '',
                  '| 方法 | 场景数 | 按时率 % | 平均成本 | 实发 kbit/s |',
                  '|---|---:|---:|---:|---:|']
        for name in dict.fromkeys(r['method'] for r in rows):
            group = [r for r in rows if r['method'] == name]
            result = {'method': name, 'episodes': len(group),
                      'ontime_pct': 100 * np.mean([float(r['ontime_rate']) for r in group]),
                      'cost': np.mean([float(r['total_cost']) for r in group]),
                      'kbps': np.mean([float(r['report_bps']) for r in group]) / 1000}
            records['baselines'].append(result)
            lines.append(f"| {name} | {len(group)} | {result['ontime_pct']:.2f} | "
                         f"{result['cost']:.1f} | {result['kbps']:.2f} |")
    if fixed is not None and (fixed / 'summary.json').exists():
        data = json.loads((fixed / 'summary.json').read_text())
        records['fixed_mode'] = data
        lines += ['', '## 验证集选出的固定控制模式', '',
                  f"仅在验证 seed {data['selection_seed']} 比较九个模式，按与 PPO 相同的"
                  '原单位折扣回报选取；PPO 的 checkpoint 使用两个验证种子，因此验证预算不同。',
                  f"选中 (eta 倍率, 并行上限)={data['selected_action']}。独立测试两个种子："
                  f"按时率 {100 * data['test_mean_ontime_rate']:.2f}%，"
                  f"成本 {data['test_mean_total_cost']:.1f}，"
                  f"折扣回报 {data['test_mean_discounted_return']:.2f}。"]
    curves = []
    for path in policy_dirs:
        if not (path / 'training.json').exists():
            continue
        log = json.loads((path / 'training.json').read_text())
        updates = log['updates']
        result = {'directory': str(path), 'updates': len(updates),
                  'joint': log['joint'], 'best_update': log.get('best_update'),
                  'encoder_changed': log['encoder_hash_initial'] != log.get('encoder_hash_last'),
                  'test_complete': (path / 'test_best.json').exists(), 'tests': []}
        lines += ['', f'## {path.name}', '',
                  f"已完成 {len(updates)} updates；验证集选中的 checkpoint 为 "
                  f"update {log.get('best_update')}。"]
        if updates:
            last = updates[-1]
            result.update(entropy=last['entropy_after'],
                          value_ev=last['mc_explained_variance_after_fit'],
                          replay_error_max=max(u['replay_logprob_max_error'] for u in updates))
            lines += [f"最后一轮 entropy={result['entropy']:.4f}（9 动作均匀分布为 "
                      f"{np.log(9):.4f}），同批次拟合后 critic MC EV={result['value_ev']:.4f}。"
                      '该 EV 是训练诊断，不是独立泛化成绩。']
            curves.append((path.name, updates, log['validation']))
        test_path = path / 'test_best.json'
        if test_path.exists():
            tests = json.loads(test_path.read_text())
            lines += ['', '| 部署方式 | 场景数 | 按时率 % | 平均成本 | 原单位折扣回报 |',
                      '|---|---:|---:|---:|---:|']
            for deployment in ('greedy', 'sampled'):
                group = [r for r in tests if r['deployment'] == deployment]
                if not group:
                    continue
                a = [r['metrics']['accounting'] for r in group]
                data = {'deployment': deployment, 'episodes': len(group),
                        'ontime_pct': 100 * np.mean([1 - r['violations'] / r['arrivals'] for r in a]),
                        'cost': np.mean([r['cost'] for r in a]),
                        'discounted_return': np.mean([r['discounted_return'] for r in group])}
                result['tests'].append(data)
                lines.append(f"| {deployment} | {len(group)} | {data['ontime_pct']:.2f} | "
                             f"{data['cost']:.1f} | {data['discounted_return']:.2f} |")
        else:
            lines += ['独立测试尚未完成，不能用训练轨迹代替测试结果。']
        records['policies'].append(result)
    lines += ['', '冻结与联合实验必须使用同一环境版本、训练场景序列、验证及测试划分。'
              '即时状态只是无上报延迟的当前状态参考，使用同一近似求解器，不是最优上界。',
              '原始详细记录保留各环境 seed，不能把请求数量当作独立实验重复次数。']
    (output / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    (output / 'summary.json').write_text(json.dumps(records, indent=2) + '\n')
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        for name, updates, validation in curves:
            label = 'Joint encoder' if 'joint' in name else 'Frozen encoder'
            x = [r['update'] for r in updates]
            axes[0].plot(x, [r['entropy_after'] for r in updates], label=label)
            axes[1].plot(x, [r['mc_explained_variance_after_fit'] for r in updates], label=label)
            vx = [r['update'] for r in validation]
            vy = [np.mean([e['discounted_return'] for e in r['episodes']
                           if e['deployment'] == 'sampled']) for r in validation]
            axes[2].plot(vx, vy, marker='o', label=label)
        axes[0].axhline(np.log(9), color='gray', ls=':', lw=1)
        axes[0].set_ylim(0, np.log(9) * 1.03)
        axes[1].set_ylim(min(0, min(r['mc_explained_variance_after_fit']
                                   for _, updates, _ in curves for r in updates)), 1.03)
        for ax, title in zip(axes, ['Policy entropy', 'Training critic MC EV',
                                    'Validation sampled return'], strict=True):
            ax.set(title=title, xlabel='PPO update')
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / 'training_diagnostics.png', dpi=180)
        fig.savefig(output / 'training_diagnostics.pdf')
        plt.close(fig)
    print(output / 'SUMMARY.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baselines', type=Path, required=True)
    parser.add_argument('--policies', type=Path, nargs='+', required=True)
    parser.add_argument('--fixed', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.baselines, args.policies, args.output, args.fixed)


if __name__ == '__main__':
    main()
