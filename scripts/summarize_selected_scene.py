"""Compare the DQN and PPO plans for the selected homogeneous inter-ES scene."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dqn', type=Path, required=True)
    parser.add_argument('--ppo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {'Proposed (DQN)': args.dqn, 'Proposed (PPO)': args.ppo}
    summaries = {name: json.loads((root / 'summary.json').read_text())
                 for name, root in runs.items()}
    dqn = summaries['Proposed (DQN)']
    ppo = summaries['Proposed (PPO)']
    for key in ('scenario_variant', 'scenario_change', 'test_seeds', 'joint_init_sha256'):
        if dqn['experiment'][key] != ppo['experiment'][key]:
            raise ValueError(f'DQN/PPO {key} mismatch')
    for seed in dqn['experiment']['test_seeds']:
        reference = json.loads((args.dqn / 'greedy' / f'test_{seed}.json').read_text())
        for root in runs.values():
            for method in dqn['experiment']['methods']:
                row = json.loads((root / method / f'test_{seed}.json').read_text())
                for key in ('arrivals', 'arrival_sha256', 'resource_sha256'):
                    if row[key] != reference[key]:
                        raise ValueError(f'{root}/{method}/seed{seed}: {key} mismatch')
    policies = {name: summary['proposed'] for name, summary in summaries.items()}
    policies.update({label: dqn[name] for label, name in
                     [('Greedy', 'greedy'), ('Shortest Queue', 'shortest_queue'),
                      ('Random', 'random')]})
    reversal_rows = [json.loads((args.dqn / 'rank_reversal' /
                                 f'summary_{seed}.json').read_text())
                     for seed in dqn['experiment']['test_seeds']]
    reversal = {}
    for stage in ('light_stages', 'audio_preprocess'):
        decisions = sum(row[stage]['decisions'] for row in reversal_rows)
        reversed_count = sum(row[stage]['rank_reversed'] for row in reversal_rows)
        worst = sum(row[stage]['became_worst'] for row in reversal_rows)
        reversal[stage] = {'decisions': decisions, 'rank_reversed': reversed_count,
                           'became_worst': worst,
                           'fraction': reversed_count / decisions}
    result = {'policies': policies,
              'plans': {name: {'model_selection': summary['model_selection'],
                               'training': summary['training']}
                        for name, summary in summaries.items()},
              'rank_reversal': reversal,
              'experiment': dqn['experiment']}
    lines = ['# 选定动态驾驶场景：DQN、PPO 与传统基线', '',
             dqn['experiment']['scenario_change'] + '.', '',
             '| 方法 | SLA | 按时/到达 | 失败 | 未完成 | 已完成平均延迟 | P95 | 上报量/轨迹 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in policies.items():
        lines.append(f"| {name} | {100*row['sla']:.2f}% | {row['ontime']}/{row['arrivals']} | "
                     f"{row['failures']} | {row['pending']} | "
                     f"{row['mean_completed_latency_ms']:.2f} ms | "
                     f"{row['p95_completed_latency_ms']:.2f} ms | "
                     f"{row['report_wire_bytes_per_trace']:.0f} B |")
    lines += ['', '## 逐条测试轨迹', '',
              '| seed | DQN | PPO | Greedy | Shortest Queue | Random |',
              '|---|---:|---:|---:|---:|---:|']
    for i, seed in enumerate(dqn['experiment']['test_seeds']):
        values = [dqn['per_seed'][i]['proposed'], ppo['per_seed'][i]['proposed'],
                  *[dqn['per_seed'][i][name] for name in
                    ('greedy', 'shortest_queue', 'random')]]
        lines.append('| ' + str(seed) + ' | ' +
                     ' | '.join(f'{100*value:.2f}%' for value in values) + ' |')
    lines += ['', '全部到达请求计入 SLA 分母，未完成判失败；平均/P95 只统计已完成请求，须同时读取未完成数量。', '',
              '## 验证选择与梯度', '',
              '| 算法 | 所选轮 | 验证 SLA | 更新数 | 梯度最大值 | 裁剪率 |',
              '|---|---:|---:|---:|---:|---:|']
    for name, summary in summaries.items():
        selection, training = summary['model_selection'], summary['training']
        lines.append(f"| {name} | {selection['episode']} | "
                     f"{100*selection['validation_sla']:.2f}% | {training['updates']} | "
                     f"{training['grad_norm_pre_max']:.3f} | "
                     f"{100*training['clip_fraction']:.3f}% |")
    lines += ['', '模型仅按三条独立验证轨迹选择，测试结果不参与本轮 checkpoint 选择。相同测试 seed 在开发期间已多次使用，因此不称为未接触的最终 holdout。', '',
              '## 过时状态导致的排序反转', '']
    for label, key in [('全部 light 阶段', 'light_stages'),
                       ('audio_preprocess', 'audio_preprocess')]:
        row = reversal[key]
        lines.append(f"- {label}：{row['rank_reversed']}/{row['decisions']} "
                     f"({100*row['fraction']:.2f}%)；其中成为最差候选 {row['became_worst']} 次。")
    lines += ['', '该诊断保持决策时队列不变，用候选输入就绪时的实际外部链路/服务轨迹重算排序；三次重放逐请求与原 Greedy 结果一致。light 阶段只有两个合法候选，因此排序反转即原最优变为最差。它验证状态会过时，但不是替代策略的完整反事实执行。', '',
              '任务量、固定部署、SLA、上报周期、共享上报信道、到达和外部轨迹在所有方法间一致；逐 seed 的 arrival/resource hash 已核对。', '',
              '## 本轮判断', '',
              '统一提高跨边缘速率后，DQN 低于 Greedy，PPO 仅略高于 Greedy且总体 P95 更差；这轮不取代此前有明确增益的选定场景。不能把验证接近 100% 的成绩写成测试成绩。', '',
              'Azure 请求量和输入 token 量映射为背景资源及链路压力；前台仍为 ICC 驾驶 DAG。链路和资源是模型映射，并非 Azure 实测网络。', '',
              '完整指标缺口、训练曲线、原始结果及模型分别见两套 run 的 RESULTS.md。', '']
    (args.output / 'COMPARISON.md').write_text('\n'.join(lines))
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: row['sla'] for name, row in policies.items()}, indent=2))


if __name__ == '__main__':
    main()
