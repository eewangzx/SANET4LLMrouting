"""Compare completed driving/Azure DQN and PPO runs, preserving both DQN stages."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--driving-dqn48',type=Path,required=True)
    p.add_argument('--driving-dqn96',type=Path,required=True)
    p.add_argument('--driving-ppo',type=Path,required=True)
    p.add_argument('--azure-dqn',type=Path,required=True)
    p.add_argument('--azure-ppo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    cases={'driving':{'DQN +48':a.driving_dqn48,'DQN +96':a.driving_dqn96,'PPO 48':a.driving_ppo},
           'azure':{'DQN +48':a.azure_dqn,'PPO 48':a.azure_ppo}}
    colors={'DQN +48':'#3070B3','DQN +96':'#7B94BF','PPO 48':'#9D61A9',
            'Greedy':'#DD8A36','Shortest Queue':'#45957C','Random':'#9C596F'}
    summary={};lines=['# DQN、PPO 与传统路由：完成结果','',
        '共同框架是预测压缩、learned top-3 重要性选择与路由 RL 联合训练。节点实例部署固定。当前用 importance codec，不是显式 IB 目标；接收端解码未来资源/链路预测，不还原整段原始历史。', '',
        'RL 回报沿同一请求的完整 DAG 传播，目标为最终请求 SLA 成功率；不宣称优化跨所有请求的无限期网络总收益。节点排队及时间变化仍在环境中实际执行。', '',
        '每个训练计划均由验证 SLA 选模型、平均已完成延迟打破平局；测试只评估被选模型。PPO 压缩网络继承原模型、策略/价值头全新初始化，DQN 保留已训练 Q；因此这是实用实现与耗时比较，并非初始化受控的算法因果比较。', '']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    vfig,vaxes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for panel,(case,runs) in enumerate(cases.items()):
        loaded={name:json.loads((root/'summary.json').read_text()) for name,root in runs.items()}
        base=next(iter(runs.values()))
        ex=json.loads((base/'experiment.json').read_text());seeds=ex['test_seeds']
        for root in runs.values():
            for seed in seeds:
                ref=json.loads((base/'greedy'/f'test_{seed}.json').read_text())
                proposed=json.loads((root/'proposed'/f'test_{seed}.json').read_text())
                for key in ('arrivals','arrival_sha256','resource_sha256'):
                    if ref[key]!=proposed[key]:raise ValueError(f'{case} {root}: {key} differs')
        policies={name:result['proposed'] for name,result in loaded.items()}
        policies.update({label:next(iter(loaded.values()))[key] for label,key in
                         [('Greedy','greedy'),('Shortest Queue','shortest_queue'),('Random','random')]})
        summary[case]={'policies':policies,'training':{},'source_runs':{k:str(v) for k,v in runs.items()}}
        title='驾驶动态场景' if case=='driving' else 'Azure Code 繁忙变化场景'
        lines+=['## '+title,'',
                '| 方法 | SLA | 失败/到达 | 已完成平均延迟 | 已完成 P95 | 上报量/轨迹 |',
                '|---|---:|---:|---:|---:|---:|']
        for label,d in policies.items():
            lines.append(f"| {label} | {100*d['sla']:.2f}% | {d['failures']}/{d['arrivals']} | {d['mean_completed_latency_ms']:.2f} ms | {d['p95_completed_latency_ms']:.2f} ms | {d['report_wire_bytes_per_trace']:.0f} B |")
        lines+=['','| RL 训练计划 | 所选训练轮 | 验证 SLA | 优化更新数 | 本计划耗时 | 梯度最大值 | 裁剪更新数 |',
                '|---|---:|---:|---:|---:|---:|---:|']
        for label,root in runs.items():
            status=json.loads((root/'proposed/status.json').read_text())
            if status['status']!='complete':raise ValueError(f'incomplete: {root}')
            rows=list(csv.DictReader((root/'proposed/training.csv').open()))
            elapsed=status['elapsed_seconds'];updates=sum(int(r['updates_episode']) for r in rows)
            clipped=sum(round(float(r['clip_fraction'])*int(r['updates_episode'])) for r in rows)
            maximum=max(float(r['grad_norm_pre_max']) for r in rows)
            if label=='DQN +96':
                preceding=list(csv.DictReader((root/'training_stage1.csv').open()))
                elapsed+=json.loads((a.driving_dqn48/'proposed/status.json').read_text())['elapsed_seconds']
                updates+=sum(int(r['updates_episode']) for r in preceding)
                clipped+=sum(round(float(r['clip_fraction'])*int(r['updates_episode'])) for r in preceding)
                maximum=max(maximum,max(float(r['grad_norm_pre_max']) for r in preceding))
            select=loaded[label]['model_selection'];episode=select.get('global_episode',select['episode'])
            summary[case]['training'][label]={'elapsed_seconds':elapsed,'updates':updates,
                'selected_episode':episode,'validation_sla':select['validation_sla'],
                'gradient_max':maximum,'clipped_updates':clipped,'selection_uses_test':False}
            lines.append(f"| {label} | {episode} | {100*select['validation_sla']:.2f}% | {updates} | {elapsed/60:.2f} min | {maximum:.3f} | {clipped} |")
            vals=loaded[label]['validation']
            vaxes[panel].plot([r['episode'] for r in vals],[100*r['sla'] for r in vals],
                              'o-',color=colors[label],label=label,ms=4)
        vals=[100*d['sla'] for d in policies.values()]
        ax=axes[panel];ax.bar(range(len(vals)),vals,color=[colors[k] for k in policies])
        for i,value in enumerate(vals):ax.text(i,value+.8,f'{value:.2f}',ha='center',fontsize=8)
        display=[('PPO (ep. 0)' if key=='PPO 48' and loaded[key]['model_selection']['episode']==0 else key)
                 for key in policies]
        ax.set_xticks(range(len(vals)),display,rotation=25,ha='right')
        ax.set(ylabel='Request SLA success (%)',ylim=(0,105),title='Driving dynamics' if case=='driving' else 'Azure Code: busy transitions')
        ax.grid(axis='y',alpha=.2)
        vaxes[panel].set(xlabel='Additional training episode',ylabel='Validation SLA success (%)',title=ax.get_title())
        vaxes[panel].grid(alpha=.2);vaxes[panel].legend(fontsize=8)
        lines+=['','原始报告：'+', '.join(f'[{name}](../{root.name}/RESULTS.md)' for name,root in runs.items()),'']
        ppo_selection=loaded['PPO 48']['model_selection']
        if ppo_selection['episode']==0:
            lines+=['PPO 此次训练没有超过初始策略的验证成绩，最终测试使用初始候选（第0轮）。该结果不能作为 PPO 学习带来额外增益的证据；它反映继承预测器及预测代价先验的表现。','']
    for obj,name in [(fig,'algorithm_sla'),(vfig,'algorithm_validation')]:
        for suffix in ('pdf','png'):obj.savefig(out/f'{name}.{suffix}',dpi=180)
        plt.close(obj)
    lines+=['## 结果边界与使用','',
        '本轮建议保留 DQN 为主算法：两套场景的 DQN 均有验证提升；PPO 训练48轮后均回退到初始候选，虽然初始候选的 Azure 测试成绩更高，却没有显示 PPO 学习增益。PPO 的本轮计划用时更短，但更新预算更小。', '',
        '驾驶 DQN 的 +96 轮计划提高了验证 SLA，但测试 SLA 低于已完成的 +48 轮计划；不能宣称训练越多越好。两个计划的结果完整保留，未按测试成绩重新指定 +96 轮计划内的模型。', '',
        'Azure 实验使用 Code 完整一周 trace 的请求量和输入 token 量，4/1/2 天分别训练/验证/测试；从每段中按固定负载条件抽取繁忙且变化的连续片段，未按策略成绩挑轨迹。真实 5 秒对应仿真 5ms。网络/资源状态由背景压力映射产生，业务仍是 ICC 驾驶 DAG。这是选定条件下的真实负载驱动仿真，不是 Azure 实测网络结果或全周平均收益。源数据说明：[Azure 2024 官方文档](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md)。', '',
        '全部到达请求计入 SLA，未完成计失败；均值/P95 仅统计已完成请求。所有 RL 与基线的到达/资源 hash 已核对。通信节省针对 Greedy：32B versus 80B/packet，实际上报 1280B versus 3120B/trace，降低58.97%。Random 的共同监测流量不代表其必要上报量。', '',
        '耗时包含训练、验证与测试，不含 Slurm 等待；同一场景使用同一类设备（驾驶4060Ti、Azure L20），仍受到并行任务影响。PPO 每轮4遍当前轨迹，DQN每轮1000次更新，更新预算不同。梯度监测与验证曲线共同判断训练稳定性，不将有限非零梯度作为最优收敛证明。', '',
        '![SLA](algorithm_sla.png)','','![Validation](algorithm_validation.png)','']
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    (out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
