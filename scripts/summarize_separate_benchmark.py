"""Summarize restored-forecast-state DQN against the paired main experiment."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();root=args.root
    experiment=json.loads((root/'experiment.json').read_text());reference=root/'reference'
    ref=json.loads((reference/'summary.json').read_text());method=root/'decoded_separate'
    seeds=experiment['test_seeds']
    data={name:[json.loads((reference/name/f'test_{seed}.json').read_text()) for seed in seeds]
          for name in ('proposed','greedy','shortest_queue','random')}
    data['decoded_separate']=[json.loads((method/f'test_{seed}.json').read_text()) for seed in seeds]
    for i,seed in enumerate(seeds):
        for name,values in data.items():
            for key in ('arrivals','arrival_sha256','resource_sha256'):
                if values[i][key]!=data['greedy'][i][key]:raise ValueError(f'{seed}/{name}/{key} mismatch')
    def aggregate(rows):
        arrivals=sum(r['arrivals'] for r in rows);ontime=sum(r['ontime'] for r in rows)
        delays=[q['completion_ms']-q['arrival_ms'] for r in rows for q in r['requests']
                if q['completion_ms'] is not None]
        return {'arrivals':arrivals,'ontime':ontime,'failures':arrivals-ontime,
            'sla':ontime/arrivals,'pending':sum(r['pending'] for r in rows),
            'mean_completed_latency_ms':float(np.mean(delays)),
            'p95_completed_latency_ms':float(np.percentile(delays,95)),
            'report_wire_bytes_per_trace':float(np.mean([r['report_transmitted_bytes_active_window'] for r in rows])),
            'mean_received_age_ms':float(np.mean([r['mean_received_age_ms'] for r in rows]))}
    policies={name:aggregate(rows) for name,rows in data.items()}
    manifest=json.loads((method/'manifest.json').read_text())
    selection=json.loads((method/'model_selection.json').read_text())
    if not experiment.get('initial_policy_eligible_for_selection',True) and selection['episode']<=0:
        raise ValueError('Experiment required a trained RL checkpoint but selected episode 0')
    status=json.loads((method/'status.json').read_text())
    if status['status']!='complete':raise ValueError('RL run incomplete')
    frozen=json.loads((method/'joint_training.json').read_text())['parameter_delta_l2']
    if any(value!=0 for value in frozen.values()):raise ValueError('Codec changed during separate RL')
    forecast=json.loads((root/'forecast_codec/metrics.json').read_text())
    if forecast['codec_sha256']!=manifest['codec_init_sha256']:raise ValueError('RL codec differs from selected predictor')
    rows=list(csv.DictReader((method/'training.csv').open()))
    training={'episodes':len(rows),'updates':sum(int(r['updates_episode']) for r in rows),
        'elapsed_seconds':status['elapsed_seconds'],
        'grad_norm_pre_mean':float(np.mean([float(r['grad_norm_pre_mean']) for r in rows])),
        'grad_norm_pre_max':max(float(r['grad_norm_pre_max']) for r in rows),
        'clip_fraction':float(np.mean([float(r['clip_fraction']) for r in rows])),
        'codec_parameter_delta_l2':frozen}
    validation=json.loads((method/'validation.json').read_text())
    per_seed=[{'seed':seed,**{name:data[name][i]['sla'] for name in data}}
              for i,seed in enumerate(seeds)]
    summary={'policies':policies,'per_seed':per_seed,'model_selection':selection,
        'training':training,'forecast_pretraining':forecast,'manifest':manifest,'experiment':experiment,
        'sla_gain_over_greedy_pp':100*(policies['decoded_separate']['sla']-policies['greedy']['sla']),
        'sla_difference_from_joint_pp':100*(policies['decoded_separate']['sla']-policies['proposed']['sla'])}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    labels={'proposed':'Joint Proposed','decoded_separate':'Decoded forecast + separate DQN',
            'greedy':'Greedy','shortest_queue':'Shortest Queue','random':'Random'}
    order=list(labels)
    lines=['# 恢复预测状态、独立训练基线','',
        '压缩预测网络先用训练/验证资源轨迹单独训练并冻结；控制器收到32B的Z后恢复61×14预测序列，将它连同可观测队列、任务及剩余SLA输入新DQN。RL训练不读取源历史或未来标签。','',
        '| 方法 | SLA | 按时/到达 | 失败 | 未完成 | 已完成平均延迟 | P95 | 上报量/轨迹 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in order:
        d=policies[name];lines.append(f"| {labels[name]} | {100*d['sla']:.2f}% | {d['ontime']}/{d['arrivals']} | {d['failures']} | {d['pending']} | {d['mean_completed_latency_ms']:.2f} ms | {d['p95_completed_latency_ms']:.2f} ms | {d['report_wire_bytes_per_trace']:.0f} B |")
    lines+=['',f"独立基线相对Greedy的SLA差值为 {summary['sla_gain_over_greedy_pp']:+.2f}pp，相对联合Proposed为 {summary['sla_difference_from_joint_pp']:+.2f}pp。",'',
        '| seed | Joint Proposed | Decoded separate | Greedy | Shortest Queue | Random |','|---|---:|---:|---:|---:|---:|']
    for row in per_seed:lines.append('| '+str(row['seed'])+' | '+' | '.join(f"{100*row[n]:.2f}%" for n in order)+' |')
    clipped=round(training['clip_fraction']*training['updates'])
    lines+=['','## 两阶段训练','',
        f"预测阶段共{len(forecast['curve'])}轮，验证MSE从 {forecast['initial_validation_mse']:.6f} 降至 {forecast['selected_validation_mse']:.6f}，选择第{forecast['selected_epoch']}轮；耗时 {forecast['elapsed_seconds']:.2f}s。预测阶段不使用RL任务损失。",'',
        f"DQN共{training['episodes']}轮/{training['updates']}次更新，验证选择第{selection['episode']}轮（SLA {100*selection['validation_sla']:.2f}%）；耗时 {training['elapsed_seconds']:.2f}s。裁剪前梯度最大 {training['grad_norm_pre_max']:.3f}，裁剪 {clipped} 次。编码器、重要性层和预测头参数变化均为0。",'',
        f"恢复状态形状为 {manifest['forecast_state_shape']}，Q网络输入维度 {manifest['state_dim']}；线上仍只发送Z，恢复不产生网络字节。",'',
        ('第0轮预测代价策略不允许参与模型选择；被测Q至少经过8个episode的RL更新。' if not experiment.get('initial_policy_eligible_for_selection',True) else '第0轮预测代价策略允许参与模型选择。')+' Joint Proposed使用其正式实验中按验证选择的联合模型；独立基线从随机importance codec及新Q开始分别训练。因此结果比较最终方案，不把差异严格归因于单一训练因素。相同测试seed在开发中已经检查，不称为未接触holdout。全部到达进入SLA分母，未完成判失败；平均/P95仅统计已完成请求。','']
    fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True)
    values=[100*policies[n]['sla'] for n in order];ax.bar(range(len(order)),values,color=['#3070B3','#8B5FBF','#DD8A36','#45957C','#9C596F'])
    for i,value in enumerate(values):ax.text(i,value+.8,f'{value:.2f}',ha='center',fontsize=8)
    ax.set_xticks(range(len(order)),['Joint','Decoded\nseparate','Greedy','Shortest\nQueue','Random'])
    ax.set(ylabel='Request SLA success (%)',ylim=(0,105));ax.grid(axis='y',alpha=.2)
    for suffix in ('pdf','png'):fig.savefig(root/f'separate_benchmark_sla.{suffix}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
    ax.plot([v['episode'] for v in validation],[100*v['sla'] for v in validation],'o-',label='Decoded separate')
    ax.plot([v['episode'] for v in ref['validation']],[100*v['sla'] for v in ref['validation']],'o-',label='Joint reference')
    ax.axvline(selection['episode'],color='gray',ls='--',label='Selected separate model')
    ax.set(xlabel='Training episode',ylabel='Validation SLA (%)');ax.grid(alpha=.2);ax.legend()
    for suffix in ('pdf','png'):fig.savefig(root/f'separate_benchmark_validation.{suffix}',dpi=180)
    plt.close(fig)
    lines+=['![SLA](separate_benchmark_sla.png)','','![Validation](separate_benchmark_validation.png)','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'case':experiment['dynamics_profile'],**{name:policies[name]['sla'] for name in order}},indent=2))


if __name__=='__main__':main()
