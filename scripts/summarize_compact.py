"""Summarize the single compact-state experiment; do not launch other runs."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args()
    root=Path(args.root)
    names=('proposed','raw','baseline')
    manifests={n:json.loads((root/n/'manifest.json').read_text()) for n in names}
    results={n:json.loads((root/n/'result.json').read_text()) for n in names}
    assert all(json.loads((root/n/'status.json').read_text())['status']=='complete' for n in names)
    assert len({m['placement_hash'] for m in manifests.values()})==1
    assert manifests['proposed']['train_seeds']==manifests['raw']['train_seeds']
    assert not set(manifests['proposed']['train_seeds']).intersection(manifests['proposed']['test_seeds'])
    data=[]
    for name in names:data.extend(dict(method=name,**r) for r in results[name]['evaluation'])
    for load in (2.,3.):
        for seed in manifests['proposed']['test_seeds']:
            paired=[r for r in data if r['load']==load and r['seed']==seed]
            assert len(paired)==3
            assert len({r['arrival_sha256'] for r in paired})==1
    with (root/'test_results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    training={}
    for name in names[:2]:
        with (root/name/'training.csv').open() as stream:
            training[name]=[{k:float(v) for k,v in row.items() if v!=''} for row in csv.DictReader(stream)]
        assert len(training[name])==manifests[name]['args']['episodes']
        assert all(math.isfinite(v) for row in training[name] for v in row.values())
        assert all(row['clip_fraction']==0 for row in training[name])
    def aggregate(load,name):
        rows=[r for r in data if r['load']==load and r['method']==name]
        arrivals=sum(r['arrivals'] for r in rows);ontime=sum(r['ontime'] for r in rows)
        complete=sum(r['arrivals']-r['pending'] for r in rows)
        return {'arrivals':arrivals,'ontime':ontime,'sla':ontime/arrivals,
                'pending':sum(r['pending'] for r in rows),
                'mean_completed_latency_ms':sum(r['mean_completed_latency_ms']*(r['arrivals']-r['pending'])
                    for r in rows)/complete,
                'mean_seed_p95_ms':float(np.mean([r['p95_completed_latency_ms'] for r in rows])),
                'mean_report_bytes':float(np.mean([r['report_bytes'] for r in rows]))}
    summary={n:{str(l):aggregate(l,n) for l in (2.,3.)} for n in names}
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# 精简状态路由主实验','',
        '只运行一版精简状态、小型 Double DQN，不做状态消融或参数搜索。',
        '复用既有 frozen compressor；raw 复用其既有 receiver，与传输／排队贪心作基本比较。','',
        'Q 输入：节点表征、节点占用记录、各候选节点传输耗时、当前服务、任务类型、剩余期限。',
        '删除汇总入口等待数、常量 present、重复 age/deadline 和 Q 输入中的 mask；合法动作约束仍用于动作选择和 TD target。',
        f"输入 {manifests['proposed']['state_dim']} 维，两层 64 单元，在线 Q 共 {manifests['proposed']['q_parameters']} 参数。",
        '固定 1×部署，在 2×负载下训练 20 条独立轨迹；一个模型初始化 seed，2×与3×各测试三条新轨迹。',
        '三种方法的部署哈希、各测试轨迹的到达请求哈希一致；Gamma 基础速率采用相同的逐实例／时间槽外生样本。',
        f"学习率 3e-5，gamma=1，学习用奖励除以固定 {manifests['proposed']['reward_scale']:.6f}；原始奖励和 SLA 不变。",'',
        '| 方法 | 2×准时率 | 3×准时率 | 字节/节点/报告，含头部 |',
        '|---|---:|---:|---:|']
    for n in names:
        size=manifests[n].get('report_bytes_per_node','未测')
        lines.append(f"| {n} | {summary[n]['2.0']['sla']:.2%} | {summary[n]['3.0']['sla']:.2%} | {size} |")
    lines.extend(['','| 负载 | 方法 | 准时/到达 | 未完成 | 已完成请求平均延迟 ms | 三条轨迹 p95 的均值 ms |',
                  '|---|---|---:|---:|---:|---:|'])
    for load in (2.,3.):
        for n in names:
            a=summary[n][str(load)]
            lines.append(f"| {load:g}× | {n} | {a['ontime']}/{a['arrivals']} | {a['pending']} | "
                         f"{a['mean_completed_latency_ms']:.2f} | {a['mean_seed_p95_ms']:.2f} |")
    lines.extend(['','| 方法 | 总 Q 更新 | 最后五轮平均梯度范数 | 最大裁剪前梯度范数 | 最后五轮平均 TD loss | 训练准时率范围 |',
                  '|---|---:|---:|---:|---:|---:|'])
    for n in names[:2]:
        rows=training[n]
        lines.append(f"| {n} | {rows[-1]['updates_total']:.0f} | "
            f"{np.mean([r['grad_norm_pre_mean'] for r in rows[-5:]]):.6f} | "
            f"{max(r['grad_norm_pre_max'] for r in rows):.4f} | "
            f"{np.mean([r['loss_mean'] for r in rows[-5:]]):.6g} | "
            f"{min(r['sla'] for r in rows):.2%}–{max(r['sla'] for r in rows):.2%} |")
    lines.extend(['','梯度和日志有限性检查通过。短跑曲线用于监测训练，不能单独证明策略收敛。'])
    for load in (2.,3.):
        a=summary['proposed'][str(load)]
        lines.append(f"{load:g}×下 proposed 相对 raw 准时率差 "
            f"{100*(a['sla']-summary['raw'][str(load)]['sla']):+.2f} 个百分点；相对贪心差 "
            f"{100*(a['sla']-summary['baseline'][str(load)]['sla']):+.2f} 个百分点。")
    lines.extend(['','## 结果边界','',
        '- 沿用当前 ICC Gamma × AR 因子扩展，压缩器学习预测这些因子；没有把人工扩展写成 ICC 原始物理模型。',
        '- 上报仍按 10ms 决策时缓存，仅统计字节，不模拟带宽限制和报文到达时延。消息尺寸收益不等于已经验证了带宽导致的 SLA 收益。',
        '- 调度／完成反馈采用原代码的即时可靠反馈假设，链路参数固定；传输上下文不读取未来服务速率或未完成父阶段的完成时间。',
        '- 本轮没有消融，也没有多模型初始化 seed；不能声称证明了重要性滤波或统计显著优势。',
        '- 所有到达请求都留在 SLA 分母，结束时未完成的请求算未准时。延迟只统计已完成请求。',
        '- 3×测试为2×训练策略的负载泛化，不是3×独立训练结果。','',
        '完整日志、权重、配置与代码哈希保存在各方法目录。'])
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    colors={'proposed':'#309175','raw':'#4277b6','baseline':'#777777'}
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for n in names[:2]:
        rows=training[n];x=[r['episode'] for r in rows]
        axes[0,0].plot(x,[r['grad_norm_pre_mean'] for r in rows],label=n,color=colors[n])
        axes[0,1].plot(x,[100*r['sla'] for r in rows],label=n,color=colors[n])
        axes[1,0].plot(x,[r['loss_mean'] for r in rows],label=n,color=colors[n])
    for n in names:
        axes[1,1].plot([2,3],[100*summary[n][str(l)]['sla'] for l in (2.,3.)],
                        marker='o',label=n,color=colors[n])
    axes[0,0].set(title='Gradient norm before clipping',xlabel='Training episode',yscale='log')
    axes[0,1].set(title='Training SLA',xlabel='Training episode',ylabel='On-time (%)')
    axes[1,0].set(title='TD loss',xlabel='Training episode',yscale='log')
    axes[1,1].set(title='Independent test SLA',xlabel='Load multiplier',ylabel='On-time (%)',xticks=[2,3])
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Compact routing: one training seed, three test trajectories per load')
    fig.savefig(root/'diagnostics.png',dpi=170)
    print((root/'RESULTS.md').read_text())


if __name__=='__main__':main()
