"""Compare completed routing plans, preserving initialization and tail deficits."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--driving',action='append',required=True,help='label=run_root')
    parser.add_argument('--azure',action='append',required=True,help='label=run_root')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cases={'driving':args.driving,'azure':args.azure}
    comparison={};lines=['# 联合预测压缩路由：修复后的算法比较','',
        'DQN/PPO均从同一场景的已有DQN actor与联合codec继续训练，初始化路由argmax一致。每个正式计划48轮；优化时重算收到表征对应的预测代价，任务梯度可以到达编码器、重要性选择和预测头。固定部署、物理动态、SLA及传统基线保持一致。','',
        '模型按验证SLA选择，初始模型也可入选；mean/P95仅用于指定的平局规则。表中的测试是开发期间已多次检查的配对轨迹，不称为未接触的最终holdout。','']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    vfig,vaxes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for panel,(case,entries) in enumerate(cases.items()):
        runs={label:Path(root) for label,root in (entry.split('=',1) for entry in entries)}
        if len(runs)!=len(entries):raise ValueError('Duplicate run label')
        loaded={label:json.loads((root/'summary.json').read_text()) for label,root in runs.items()}
        reference=next(iter(runs.values()))
        seeds=next(iter(loaded.values()))['experiment']['test_seeds']
        policies={label:summary['proposed'] for label,summary in loaded.items()}
        policies.update({label:next(iter(loaded.values()))[name] for label,name in
                         [('Greedy','greedy'),('Shortest Queue','shortest_queue'),('Random','random')]})
        comparison[case]={'policies':policies,'plans':{}}
        lines+=['## '+case,'','| 方法 | SLA | 按时/到达 | 未完成 | 已完成平均 | 已完成P95 | 上报量/轨迹 |',
                '|---|---:|---:|---:|---:|---:|---:|']
        for label,d in policies.items():
            lines.append(f"| {label} | {100*d['sla']:.2f}% | {d['ontime']}/{d['arrivals']} | {d['pending']} | {d['mean_completed_latency_ms']:.2f}ms | {d['p95_completed_latency_ms']:.2f}ms | {d['report_wire_bytes_per_trace']:.0f}B |")
        lines+=['','| 训练计划 | 所选轮 | 验证SLA | 更新数 | 耗时 | 梯度最大值 | 裁剪次数 |',
                '|---|---:|---:|---:|---:|---:|---:|']
        for label,root in runs.items():
            s=loaded[label];status=json.loads((root/'proposed/status.json').read_text())
            if status['status']!='complete':raise ValueError(f'Incomplete: {root}')
            if s['experiment']['test_seeds']!=seeds:raise ValueError('Test seeds differ')
            for seed in seeds:
                ref=json.loads((reference/'greedy'/f'test_{seed}.json').read_text())
                for method in s['experiment']['methods']:
                    d=json.loads((root/method/f'test_{seed}.json').read_text())
                    for key in ('arrivals','arrival_sha256','resource_sha256'):
                        if ref[key]!=d[key]:raise ValueError(f'{root}/{method}: paired {key} mismatch')
            rows=list(csv.DictReader((root/'proposed/training.csv').open()))
            updates=sum(int(r['updates_episode']) for r in rows)
            clipped=sum(round(float(r['clip_fraction'])*int(r['updates_episode'])) for r in rows)
            select=s['model_selection'];epoch=select['episode'];training=s['training']
            manifest=json.loads((root/'proposed/manifest.json').read_text())
            info={'root':str(root),'selected_episode':epoch,'validation_sla':select['validation_sla'],
                  'episodes':len(rows),'updates':updates,'clipped_updates':clipped,
                  'elapsed_seconds':status['elapsed_seconds'],'gradient_max':training['grad_norm_pre_max'],
                  'initial_sha256':s['experiment']['joint_init_sha256'],
                  'git_commit':s['experiment']['git_commit'],
                  'latency_weight':manifest['args'].get('latency_weight',0.),
                  'selection_metric':manifest['args'].get('selection_metric','mean'),
                  'individual_deficits':s['individual_trajectory_deficits']}
            comparison[case]['plans'][label]=info
            lines.append(f"| {label} | {epoch} | {100*select['validation_sla']:.2f}% | {updates} | {status['elapsed_seconds']/60:.2f}min | {training['grad_norm_pre_max']:.3f} | {clipped} |")
            vals=s['validation']
            vaxes[panel].plot([v['episode'] for v in vals],[100*v['sla'] for v in vals],'o-',ms=4,label=label)
        if len({d['initial_sha256'] for d in comparison[case]['plans'].values()})!=1:
            raise ValueError('Initialization checkpoints differ')
        lines+=['','逐轨迹缺口（对Greedy、Random、Shortest Queue检查SLA/平均/P95）：','']
        for label,info in comparison[case]['plans'].items():
            deficits=info['individual_deficits']
            lines.append(f"- {label}: "+('; '.join(f"seed{d['seed']} {d['metric']}={d['proposed']:.4f}，{d['baseline']}={d['reference']:.4f}" for d in deficits) if deficits else '三个性能指标在全部已测轨迹上均不差于三种传统基线。'))
            if info['selected_episode']==0:lines.append('  所选为初始候选，不能把其测试成绩归因于本计划的训练增益。')
        lines+=['','完整记录：'+', '.join(f'[{label}](../{root.name}/RESULTS.md)' for label,root in runs.items()),'']
        values=[100*d['sla'] for d in policies.values()]
        colors=['#3070B3','#9D61A9','#497EA6'][:len(runs)]+['#DD8A36','#45957C','#9C596F']
        axes[panel].bar(range(len(values)),values,color=colors)
        for i,value in enumerate(values):axes[panel].text(i,value+.8,f'{value:.2f}',ha='center',fontsize=8)
        labels=[label+(' (ep.0)' if label in loaded and loaded[label]['model_selection']['episode']==0 else '') for label in policies]
        axes[panel].set_xticks(range(len(values)),labels,rotation=25,ha='right')
        axes[panel].set(title=case,ylabel='Request SLA success (%)',ylim=(0,105))
        axes[panel].grid(axis='y',alpha=.2)
        vaxes[panel].set(title=case,xlabel='Additional training episode',ylabel='Validation SLA success (%)')
        vaxes[panel].grid(alpha=.2);vaxes[panel].legend(fontsize=8)
    lines+=['## 说明','',
        'DQN每轮1000次更新；PPO每轮4遍当前轨迹、minibatch256。预算不同，不能将耗时差解释为相同优化预算下的算法速度。耗时包含训练/验证/测试、不含Slurm排队，并受到并行任务影响；驾驶使用4060Ti，Azure使用L20。梯度稳定不是最优收敛保证。','',
        '全部到达请求进入SLA分母，未完成计失败。平均/P95只统计已完成请求，须同时读未完成数量；不以总体P95掩盖单条轨迹退化。32B压缩报告和80B原始报告共享64kbit/s信道，实际上报1280B和3120B/轨迹，减少58.97%。Random的共同监测开销不是其路由选择的必要开销。','',
        'Azure使用Code一周轨迹的5秒聚合请求量及输入token量，4/1/2天划分训练/验证/测试，训练段归一化。固定条件筛选繁忙变化连续片段，不按策略成绩选轨迹；真实5秒对应仿真5ms。轨迹映射为背景资源/链路压力，前台仍为ICC驾驶DAG。结果不代表Azure实测网络或全周平均。','',
        '![SLA](algorithm_sla.png)','','![Validation](algorithm_validation.png)','']
    for figure,name in [(fig,'algorithm_sla'),(vfig,'algorithm_validation')]:
        for suffix in ('pdf','png'):figure.savefig(args.output/f'{name}.{suffix}',dpi=180)
        plt.close(figure)
    (args.output/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    (args.output/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps({case:{label:d['sla'] for label,d in result['policies'].items()} for case,result in comparison.items()},indent=2))


if __name__=='__main__':main()
