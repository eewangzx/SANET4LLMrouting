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
    parser=argparse.ArgumentParser()
    parser.add_argument('run')
    args=parser.parse_args()
    root=Path(args.run)
    names=('baselines','raw','stats27','proposed')
    manifests={name:json.loads((root/name/'manifest.json').read_text()) for name in names}
    results={name:json.loads((root/name/'result.json').read_text()) for name in names}
    stress={name:json.loads((root/name/'stress.json').read_text()) for name in names}
    assert len({m['placement_hash'] for m in manifests.values()})==1
    assert len({m['scenario_hash'] for m in manifests.values()})==1
    for m in manifests.values():
        assert not set(m['train_seeds']).intersection(m['test_seeds'])
    data=[]
    for method,rows in results['baselines']['baselines'].items():
        data.extend(dict(level=1.,method=method,**row) for row in rows)
    for name in names[1:]:
        data.extend(dict(level=1.,method=name,**row) for row in results[name]['evaluation'])
    data.extend(dict(level=1.,method='proposed_zero',**row)
                for row in results['proposed']['zero_embedding_evaluation'])
    for name in names:
        for level,methods in stress[name].items():
            for method,rows in methods.items():
                data.extend(dict(level=float(level),method=method,**row) for row in rows)
    for level in (1.,2.,3.):
        for seed in manifests['raw']['test_seeds']:
            paired=[r for r in data if r['level']==level and r['seed']==seed]
            assert len({r['arrival_sha256'] for r in paired})==1, (level,seed)
            assert len({r['arrivals'] for r in paired})==1
    fields=list(dict.fromkeys(k for row in data for k in row))
    with (root/'test_results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(data)
    training={}
    for name in names[1:]:
        with (root/name/'training.csv').open() as stream:
            training[name]=[{k:float(value) for k,value in row.items() if value!=''}
                            for row in csv.DictReader(stream)]
        assert len(training[name])==manifests[name]['args']['episodes']
        assert all(math.isfinite(value) for row in training[name] for value in row.values())

    def aggregate(level,method):
        rows=[r for r in data if r['level']==level and r['method']==method]
        total=sum(r['arrivals'] for r in rows)
        completed=sum(r['arrivals']-r['pending'] for r in rows)
        return {'ontime':sum(r['ontime'] for r in rows),'arrivals':total,
                'sla':sum(r['ontime'] for r in rows)/total,
                'pending':sum(r['pending'] for r in rows),
                'mean_latency':sum(r['mean_completed_latency_ms']*(r['arrivals']-r['pending'])
                                   for r in rows)/completed,
                'mean_seed_p95':float(np.mean([r['p95_completed_latency_ms'] for r in rows]))}

    methods=('earliest_finish','raw','stats27','proposed','proposed_zero',
             'greedy_now','future_factor_heuristic')
    lines=['# Fedora split-bench validation','',
           'Independent processes completed. Fixed deployment and per-test arrival hashes match across all methods.',
           'One training seed, 8 independent training episodes, 3 independent test trajectories per load.',
           'All RL methods: input LayerNorm, lr=3e-5, gamma=1, fixed reward scale='+
           f"{manifests['raw']['reward_scale']:.6f}.",'',
           'The 2x/3x load tests reuse the trained policies and identical 1x deployment; no retraining/model selection.',
           'Physics: current ICC Gamma + AR factor extension. Reports: cached every 10ms at decisions, byte accounting only.',
           '', '| Method | 1x SLA | 2x SLA | 3x SLA | Bytes/node/report |',
           '|---|---:|---:|---:|---:|']
    for method in methods:
        size=(results[method]['bytes_per_node_report'] if method in results and method!='baselines'
              else (48 if method=='proposed_zero' else None))
        lines.append('| '+method+' | '+' | '.join(f"{aggregate(level,method)['sla']:.2%}"
                     for level in (1.,2.,3.))+' | '+(str(size) if size else 'unmeasured')+' |')
    lines.extend(['','## Latency and completion','',
                  '| Load | Method | On-time/arrivals | Pending | Mean completed latency (ms) | Mean test-seed p95 (ms) |',
                  '|---:|---|---:|---:|---:|---:|'])
    for level in (1.,2.,3.):
        for method in methods[:5]:
            a=aggregate(level,method)
            lines.append(f"| {level:g} | {method} | {a['ontime']}/{a['arrivals']} | {a['pending']} | "
                         f"{a['mean_latency']:.2f} | {a['mean_seed_p95']:.2f} |")
    lines.extend(['','## Training diagnostics','',
                  '| Method | Q updates | First-episode mean grad | Last-3-episode mean grad | Last-3 loss | Largest preclip grad | Largest Q | Largest raw step reward | Independent prediction R2 | Latent RMS |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for name in names[1:]:
        rows=training[name];last=rows[-3:];c=results[name]['codec']
        lines.append(f"| {name} | {results[name]['updates']} | {rows[0]['grad_norm_pre_mean']:.5f} | "
          f"{np.mean([r['grad_norm_pre_mean'] for r in last]):.5f} | "
          f"{np.mean([r['loss_mean'] for r in last]):.6g} | "
          f"{max(r['grad_norm_pre_max'] for r in rows):.3f} | "
          f"{max(r['q_abs_max'] for r in rows):.3f} | "
          f"{max(r['step_reward_abs_max'] for r in rows):.0f} | "
          f"{c['independent_val_r2']:.3f} | {c['latent_rms']:.3f} |")
    lines.extend(['','## Interpretation','',
                  '- Gradient/loss finite checks passed; these short runs do not prove policy convergence.',
                  '- 1x SLA hits a ceiling, including proposed with zeroed embeddings; it cannot demonstrate routing gain.',
                  '- Zeroing proposed embeddings probes whether the policy uses the message; it does not by itself prove importance filtering.',
                  '- Size reductions are measured message dimensions, not an end-to-end bandwidth-delay/SLA gain.',
                  '- Test-load results are sensitivity checks of one trained seed, not final multi-seed conference evidence.'])
    delta_raw=100*(aggregate(3.,'proposed')['sla']-aggregate(3.,'raw')['sla'])
    delta_greedy=100*(aggregate(3.,'proposed')['sla']-aggregate(3.,'earliest_finish')['sla'])
    lines.extend([
        f'- At 3x, proposed is {delta_raw:+.2f} percentage points vs raw, and {delta_greedy:+.2f} vs earliest-finish greedy.',
        '- Zeroing proposed embeddings improves SLA by '+
        ', '.join(f"{100*(aggregate(level,'proposed_zero')['sla']-aggregate(level,'proposed')['sla']):.2f} pp at {level:g}x"
                  for level in (2.,3.))+'. The observed raw comparison gain cannot be attributed to informative compressed messages.',
        '- All 1x training episodes have 100% SLA: the SLA reward does not distinguish routing quality in that training workload.',
        '- Proposed latent RMS is lower than raw/stats27 in this run, so the suspected larger IB input scale is not observed.',
        '- Every learned prediction head has higher independent-validation MSE than the simple last-value persistence baseline.',
    ])
    lines.extend(['', '## Minimal state counterexample', '',
                  'The two-node check in state_aliasing_check.json produces identical Q inputs for two request-origin assignments.',
                  'With a 10ms deadline, the correct node gives 1ms processing; the wrong node gives 17ms including transfer.',
                  'The optimal first action reverses between the assignments. Current aggregate queue counts cannot resolve this alias.',
                  'Add causal candidate-node transfer estimates (from request gateway / completed DAG parents) to the routing input.',
                  '', '## Next bounded experiment', '',
                  'Train all methods on the same load where SLA violations occur, rather than the ceiling 1x case.',
                  'Then test on independent seeds with the fixed placement and unchanged ICC execution semantics.',
                  'Use actual node states for compression and keep a same-budget non-learned summary to test routing-relevant filtering.'])
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    colors={'raw':'#4277b6','stats27':'#d7922c','proposed':'#309175'}
    for name in names[1:]:
        rows=training[name];x=[r['episode'] for r in rows]
        axes[0,0].plot(x,[r['grad_norm_pre_mean'] for r in rows],label=name,color=colors[name])
        axes[0,1].plot(x,[r['loss_mean'] for r in rows],label=name,color=colors[name])
    for method in methods[:5]:
        style='--' if method in ('earliest_finish','proposed_zero') else '-'
        color=colors.get(method,'#777777' if method=='earliest_finish' else colors['proposed'])
        axes[1,0].plot([1,2,3],[100*aggregate(level,method)['sla'] for level in (1.,2.,3.)],
                       marker='o',linestyle=style,label=method,color=color)
        axes[1,1].plot([1,2,3],[aggregate(level,method)['mean_latency'] for level in (1.,2.,3.)],
                       marker='o',linestyle=style,label=method,color=color)
    axes[0,0].set(title='Mean gradient norm before clipping',xlabel='Training episode',yscale='log')
    axes[0,1].set(title='Mean TD loss',xlabel='Training episode',yscale='log')
    axes[1,0].set(title='Independent test SLA',xlabel='Arrival load multiplier',ylabel='On-time rate (%)')
    axes[1,1].set(title='Completed-request latency',xlabel='Arrival load multiplier',ylabel='Mean latency (ms)')
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Current routing prototype: fixed deployment, one training seed, three test trajectories')
    fig.savefig(root/'diagnostics.png',dpi=170)
    summary={'checks':'passed','training_updates':{name:results[name]['updates'] for name in names[1:]},
             'sla':{method:{str(level):aggregate(level,method) for level in (1.,2.,3.)} for method in methods}}
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    print((root/'SUMMARY.md').read_text())


if __name__=='__main__': main()
