"""Summarize paired four-policy routing experiments and training diagnostics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS={'proposed':'Proposed','greedy':'Current-state Greedy',
        'random':'Random','shortest_queue':'Shortest Queue'}
COLORS={'proposed':'#3070B3','greedy':'#DD8A36','random':'#9C596F','shortest_queue':'#45957C'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root
    experiment=json.loads((root/'experiment.json').read_text())
    algorithm=experiment.get('algorithm','dqn')
    manifest=json.loads((root/'proposed/manifest.json').read_text())
    if algorithm=='ppo':LABELS['proposed']='Proposed (PPO)'
    methods=experiment['methods'];seeds=experiment['test_seeds']
    data={name:[json.loads((root/name/f'test_{seed}.json').read_text()) for seed in seeds]
          for name in methods}
    for i in range(len(seeds)):
        for name in methods:
            for key in ('arrival_sha256','resource_sha256','arrivals'):
                if data[name][i][key]!=data['greedy'][i][key]:
                    raise ValueError(f'Paired {key} mismatch: {name}, seed={seeds[i]}')
    def delays(rows):
        return [r['completion_ms']-r['arrival_ms'] for d in rows for r in d['requests']
                if r['completion_ms'] is not None]
    def aggregate(rows):
        arrived=sum(x['arrivals'] for x in rows);ontime=sum(x['ontime'] for x in rows)
        values=delays(rows)
        return {'arrivals':arrived,'ontime':ontime,'failures':arrived-ontime,
                'sla':ontime/arrived,'pending':sum(x['pending'] for x in rows),
                'mean_completed_latency_ms':float(np.mean(values)) if values else None,
                'p95_completed_latency_ms':float(np.percentile(values,95)) if values else None,
                'report_bytes_per_packet':rows[0]['report_bytes_per_packet'],
                'report_wire_bytes_per_trace':float(np.mean([x['report_transmitted_bytes_active_window'] for x in rows])),
                'mean_received_age_ms':float(np.mean([x['mean_received_age_ms'] for x in rows]))}
    summary={name:aggregate(data[name]) for name in methods}
    p,g=summary['proposed'],summary['greedy']
    summary['comparison']={'sla_gain_pp':100*(p['sla']-g['sla']),
                           'wire_reduction_percent':100*(1-p['report_wire_bytes_per_trace']/g['report_wire_bytes_per_trace'])}
    summary['per_seed']=[{'seed':seed,**{name:data[name][i]['sla'] for name in methods}}
                         for i,seed in enumerate(seeds)]
    checks=[];deficits=[]
    metrics=[('sla',1),('mean_completed_latency_ms',-1),('p95_completed_latency_ms',-1)]
    for i,seed in enumerate(seeds):
        p=data['proposed'][i]
        for name in methods:
            if name=='proposed':continue
            baseline=data[name][i]
            for metric,direction in metrics:
                value,reference=p[metric],baseline[metric]
                passed=value is not None and reference is not None and direction*(value-reference)>=-1e-9
                check={'seed':seed,'baseline':name,'metric':metric,'proposed':value,
                       'reference':reference,'passed':passed}
                checks.append(check)
                if not passed:deficits.append(check)
    summary['individual_trajectory_checks']=checks
    summary['individual_trajectory_deficits']=deficits
    summary['dominates_all_traditional_on_reported_metrics']=not deficits
    rows=list(csv.DictReader((root/'proposed/training.csv').open()))
    validation=json.loads((root/'proposed/validation.json').read_text())
    selection=json.loads((root/'proposed/model_selection.json').read_text())
    stage_offset=experiment.get('previous_episodes',0)
    if stage_offset:
        previous=list(csv.DictReader((root/'training_stage1.csv').open()))
        previous_validation=json.loads((root/'validation_stage1.json').read_text())
        rows=previous+[{**r,'episode':int(r['episode'])+stage_offset} for r in rows]
        validation=previous_validation+[{**v,'episode':v['episode']+stage_offset}
                                        for v in validation if v['episode']>0]
    selected_global_episode=selection['episode']+stage_offset
    summary['total_additional_episodes']=len(rows)
    selection['global_episode']=selected_global_episode
    summary['training']={key:float(np.mean([float(r[key]) for r in rows])) for key in
                         ('grad_norm_pre_mean','codec_rl_grad_norm_mean','importance_rl_grad_norm_mean','clip_fraction')}
    summary['training']['grad_norm_pre_max']=max(float(r['grad_norm_pre_max']) for r in rows)
    summary['training']['algorithm']=algorithm
    summary['training']['updates']=sum(int(r['updates_episode']) for r in rows)
    summary['training']['elapsed_seconds']=json.loads((root/'proposed/status.json').read_text())['elapsed_seconds']
    summary['training']['device']=manifest['device']
    if stage_offset:
        summary['training']['elapsed_seconds_note']='second continuation stage only; preceding stage time recorded separately'
    if algorithm=='ppo':
        for key in ('policy_loss_mean','value_loss_mean','entropy_mean','approx_kl_mean','policy_clip_fraction_mean'):
            summary['training'][key]=float(np.mean([float(r[key]) for r in rows]))
    summary['training']['last8']={key:float(np.mean([float(r[key]) for r in rows[-8:]])) for key in
                                 ('sla','prediction_mse_mean','grad_norm_pre_mean','td_loss_mean')}
    summary['model_selection']=selection;summary['validation']=validation
    summary['experiment']=experiment
    summary['analysis_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    def save(fig,name):
        for suffix in ('png','pdf'):fig.savefig(root/f'{name}.{suffix}',dpi=180)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.5),constrained_layout=True)
    values=[100*summary[name]['sla'] for name in methods]
    ax.bar(range(len(methods)),values,color=[COLORS[name] for name in methods])
    for i,v in enumerate(values):ax.text(i,v+1,f'{v:.2f}%',ha='center')
    ax.set_xticks(range(len(methods)),[LABELS[name].replace('Current-state ','') for name in methods])
    ax.set(ylabel='Driving SLA success (%)',ylim=(0,105));ax.grid(axis='y',alpha=.2)
    save(fig,'sla_comparison')

    fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
    for name in methods:
        values=sorted(delays(data[name]))
        ax.step(values,np.arange(1,len(values)+1)/summary[name]['arrivals'],where='post',
                label=LABELS[name],color=COLORS[name])
    deadline=next(t['deadline_ms'] for t in json.loads((root/'scenario.json').read_text())['scenario']['tasks']
                  if t['name']=='multimodal_driving')
    ax.axvline(deadline,color='gray',ls='--',label='Driving SLA deadline')
    ax.set(xlabel='End-to-end latency (ms)',ylabel='Fraction of all arrivals',ylim=(0,1.01))
    ax.legend(fontsize=8);ax.grid(alpha=.2);save(fig,'latency_cdf')

    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    ep=[int(r['episode']) for r in rows]
    def curve(ax,key,label):ax.plot(ep,[float(r[key]) for r in rows],label=label)
    curve(axes[0,0],'sla','Training, exploration enabled')
    axes[0,0].plot([v['episode'] for v in validation],[v['sla'] for v in validation],
                   'o-',label='Independent validation, no exploration')
    axes[0,0].axvline(selected_global_episode,color='gray',ls='--',label='Selected model')
    axes[0,0].set(ylabel='SLA success fraction',title='Whole-request outcomes')
    curve(axes[0,1],'td_loss_mean','PPO task loss' if algorithm=='ppo' else 'Task BCE')
    if algorithm=='ppo':curve(axes[0,1],'value_loss_mean','Value MSE')
    curve(axes[0,1],'prediction_mse_mean','Task-window forecast MSE')
    axes[0,1].set(ylabel='Loss',title='Joint optimization')
    curve(axes[1,0],'grad_norm_pre_mean','Before clipping: mean')
    curve(axes[1,0],'grad_norm_pre_max','Before clipping: maximum')
    axes[1,0].axhline(10,color='gray',ls='--',label='Clipping threshold')
    axes[1,0].set(yscale='log',ylabel='Gradient norm',title='Gradient monitoring')
    curve(axes[1,1],'codec_rl_grad_norm_mean','Compressor: RL task only')
    curve(axes[1,1],'importance_rl_grad_norm_mean','Importance: RL task only')
    curve(axes[1,1],'codec_grad_norm_mean','Compressor: combined')
    axes[1,1].set(yscale='log',ylabel='Gradient norm',title='Task gradients into semantic compression')
    for ax in axes.flat:
        ax.set_xlabel('Additional training episode');ax.grid(alpha=.2);ax.legend(fontsize=7)
    save(fig,'training_diagnostics')

    sys.path.insert(0,str(root/'source'))
    from edge_msd.icc_paper import load_scenario
    from edge_msd.icc_coupled.dynamics import ResourceTrace
    scenario=load_scenario(root/'scenario.json',experiment['load'])
    horizon=experiment['warmup_ms']+experiment['arrival_ms']+max(
        experiment['drain_ms'],max(t.deadline_ms for t in scenario.tasks.values()))
    kwargs={'azure_path':root/'azure_trace.npz','azure_split':'test',
            'azure_case':experiment.get('azure_case','busy_transitions')} if experiment['dynamics_profile']=='azure' else {}
    trace=ResourceTrace(scenario,horizon,seeds[0],profile=experiment['dynamics_profile'],**kwargs)
    if trace.sha256!=data['greedy'][0]['resource_sha256']:raise ValueError('Dynamics figure trace hash mismatch')
    time=np.arange(experiment['warmup_ms'],experiment['warmup_ms']+experiment['arrival_ms'],5.)
    fig,axes=plt.subplots(2,1,figsize=(8,5.5),sharex=True,constrained_layout=True)
    for target in ('es2','es3'):
        n=trace.node_index['es1'];port=9+trace.neighbors['es1'].index(target)
        nominal=next(rate for a,b,rate,_ in scenario.links if {a,b}=={'es1',target})
        axes[0].plot(time,[nominal*trace.current(t)[n,port] for t in time],label=f'ES1 to {target.upper()}')
        n=trace.node_index[target];service=trace.service_index['audio_preprocess']
        axes[1].plot(time,[trace.current(t)[n,service] for t in time],label=target.upper())
    axes[0].set(ylabel='Available link rate (Gbit/s)',title='First held-out environment trace')
    axes[1].set(ylabel='Audio preprocessing multiplier',xlabel='Physical simulation time (ms)')
    for ax in axes:
        for t in np.arange(experiment['warmup_ms'],time[-1]+1,experiment['report_ms']):
            ax.axvline(t,color='gray',ls='--',alpha=.4)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,'environment_dynamics')

    if experiment['dynamics_profile']=='azure':
        with np.load(root/'azure_trace.npz',allow_pickle=False) as asset:
            counts=asset['request_count'].reshape(-1,12).sum(1)/60.
            tokens=asset['context_tokens'].reshape(-1,12).sum(1)/60.
        days=np.arange(len(counts))/1440.
        fig,axes=plt.subplots(2,1,figsize=(9,5.5),sharex=True,constrained_layout=True)
        axes[0].plot(days,counts,lw=.7);axes[1].plot(days,tokens,lw=.7)
        axes[0].set(ylabel='Requests / second',title='Azure Code: full seven-day background workload')
        axes[1].set(ylabel='Input tokens / second',xlabel='Days from 2024-05-10 00:00 UTC')
        for ax in axes:
            for a,b,label,color in [(0,4,'Training','#DCEAF7'),(4,5,'Validation','#FFF0D7'),(5,7,'Test','#DFEFE7')]:
                ax.axvspan(a,b,color=color,alpha=.6,zorder=-1)
                ax.text((a+b)/2,.97,label,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
            ax.grid(alpha=.2)
        save(fig,'azure_workload')

    lines=['# '+('Azure Code 真实轨迹驱动驾驶实验' if experiment['dynamics_profile']=='azure' else '驾驶场景续训及四种路由方法'),'',
           '| 方法 | SLA | 按时完成/到达 | 失败 | 未完成 | 已完成平均延迟 | P95 | 上报量/轨迹 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in methods:
        d=summary[name]
        lines.append(f"| {LABELS[name]} | {100*d['sla']:.2f}% | {d['ontime']}/{d['arrivals']} | {d['failures']} | {d['pending']} | {d['mean_completed_latency_ms']:.2f} ms | {d['p95_completed_latency_ms']:.2f} ms | {d['report_wire_bytes_per_trace']:.0f} B |")
    lines+=['',f"Proposed 相对 Greedy 的 SLA 差值为 {summary['comparison']['sla_gain_pp']:+.2f} 个百分点，上报量降低 {summary['comparison']['wire_reduction_percent']:.2f}%。全部到达请求计入 SLA 分母，未完成判失败；平均/P95 只统计已完成请求。",'',
            '| 测试 seed | Proposed | Greedy | Random | Shortest Queue |','|---|---:|---:|---:|---:|']
    for d in summary['per_seed']:
        lines.append('| '+str(d['seed'])+' | '+' | '.join(f'{100*d[n]:.2f}%' for n in methods)+' |')
    lines+=['','## 逐条轨迹核对','',
            '核对每条轨迹的 SLA、已完成平均延迟和已完成 P95；相同指标允许持平，不用总体平均掩盖单条退化。该有限测试核对不构成任意未来网络下的最优保证。','']
    if deficits:
        lines+=['以下指标尚未领先：','',
                '| seed | 基线 | 指标 | Proposed | 基线值 |','|---|---|---|---:|---:|']
        for d in deficits:
            lines.append(f"| {d['seed']} | {LABELS[d['baseline']]} | {d['metric']} | {d['proposed']} | {d['reference']} |")
    else:lines+=['全部已测轨迹的三个指标均不差于这三种传统基线。','']
    if algorithm=='ppo':
        initialization=('压缩网络和策略偏好继承已有DQN联合模型，初始路由argmax不变'
                        if manifest['args'].get('ppo_warm_start') else
                        '压缩网络继承原联合模型，策略/价值输出头重新初始化')
        training_note=f"PPO 训练 {len(rows)} 个 episode，每轮 {experiment['ppo_epochs']} 遍当前轨迹、minibatch={experiment['ppo_batch_size']}，共 {summary['training']['updates']} 次更新。{initialization}；联合优化压缩、重要性、预测头及 actor/critic。采用 lr=3e-5、LayerNorm、64宽网络、相同预测代价先验与有界长期修正，clip=0.2，temperature=0.25，entropy weight=0.001。每条请求采用完整 DAG 的终端 SLA 成功回报，gamma=1、lambda=1，当前轨迹 advantage 标准化；训练策略随机采样，验证/测试取最大概率动作。"
    else:
        training_note=f"从保留的驾驶联合模型额外训练 {len(rows)} 个 episode，每轮 {experiment['updates_per_episode']} 次更新，每个续训阶段重新建立优化器。训练仍联合更新编码器、重要性层、预测头及 Q，使用 lr=3e-5、LayerNorm、小型64宽网络和有界长期修正。训练 SLA 包含 epsilon=0.05 的探索。"
    if manifest['args'].get('end_to_end_forecast'):
        training_note+=' 优化器重新计算与已收到报告对应的预测传输/处理代价及剩余DAG压力，RL任务梯度穿过编码器、重要性与预测头。未来标签仍只供辅助监督。'
    if manifest['args'].get('latency_weight',0.):
        weight=manifest['args']['latency_weight']
        training_note+=f' 完整请求终端目标为 (SLA成功 + {weight}×[1-min(延迟/(3×deadline),1)^2])/(1+{weight})，未完成目标为0；目标始终在[0,1]，SLA判定和物理deadline不变。'
    tie_break='已完成P95、随后平均延迟' if manifest['args'].get('selection_metric')=='p95' else '平均已完成延迟'
    lines+=['','## 训练和模型选择','',training_note,'',
            f"验证初始 SLA 为 {100*validation[0]['sla']:.2f}%，选择累计额外训练第 {selected_global_episode} 轮（本阶段第{selection['episode']}轮），验证 SLA 为 {100*selection['validation_sla']:.2f}%。最高验证 SLA 优先、{tie_break}打破平局；初始模型也可入选。最终测试没有参与模型选择。best.pt 是被测模型，last.pt 是最后一轮模型。",'',
            f"裁剪前梯度均值 {summary['training']['grad_norm_pre_mean']:.4f}、最大 {summary['training']['grad_norm_pre_max']:.4f}，裁剪率 {100*summary['training']['clip_fraction']:.2f}%。梯度稳定不等于证明最优收敛。",'',
            'Greedy 估计当前下一阶段的通信、排队和处理延迟；Shortest Queue 优先最小每实例排队量，当前延迟打破平局；Random 在合法部署节点间均匀抽取。Greedy/Shortest Queue 读取原始当前状态上报，不使用 proposed 的预测器。Random 接收共同监测上报，但选择不使用其测量值；其字节统计是实验共同监测协议开销，不是随机路由的必要开销。通信节省主张针对 Greedy。全部方法使用相同部署、到达、外部资源轨迹和上报信道。到达与资源 hash 已逐条核对。', '',
            f"源码提交 `{experiment['git_commit']}`。部署为72 core、54 light；load=0.8，业务到达240ms，预热200ms，排空150ms，上报周期100ms，共享64kbit/s遥测信道，物理槽1ms、资源采样5ms。驾驶 SLA 82.9785ms。输入等待不占计算实例，允许复用已收到的预测表征。",'']
    if experiment['dynamics_profile']=='azure':
        lines+=['## Azure 映射','',
                '完整 Code 一周数据共16,803,695条。按5秒聚合请求量及输入token量，前4天训练、第5天验证、最后2天测试；仅用训练段p5/p95归一化，未使用输出token作为到达时输入。选定繁忙变化场景：所属时间段内连续片段的平均归一化压力至少0.45，任一压力特征的最大最小差至少0.45，候选起点间隔50真实秒；从满足条件的片段均匀抽样。筛选依据负载条件，不依据策略测试成绩。这不代表全周均匀负载下的平均收益。', '',
                '真实5秒映射为仿真5ms，1000倍时间加速。35ms历史对应真实35秒，100ms上报对应真实100秒，300ms预测对应真实300秒。light资源余量及链路速率由背景压力映射产生，没有沿用人工正弦/互补链路规律。这是明确缩放的真实轨迹驱动仿真；Azure不提供本实验的实测链路或资源状态。业务仍为ICC驾驶DAG的原到达模型。','',
                '原始CSV和聚合数据hash、归一化参数及时间切分见azure_trace.json和experiment.json；各节点源片段位置见test文件里的azure_provenance。', '',
                '![Azure workload](azure_workload.png)','']
    if experiment.get('scenario_change'):
        lines+=['## 本轮场景改动','',experiment['scenario_change']+'.','',
                '该改动用于检验预测路由在远端候选可用时的收益；任务量、部署、SLA、上报周期和共享上报信道均保持不变。','']
    lines+=['![SLA](sla_comparison.png)','','![Latency](latency_cdf.png)','',
            '![Training](training_diagnostics.png)','','![Dynamics](environment_dynamics.png)','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({k:summary[k] for k in [*methods,'comparison','model_selection','training']},indent=2))


if __name__=='__main__':main()
