"""Plot the requested SLA-cost tradeoff and two individual metric figures."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


NAMES={'proposed':'Proposed','decoded_separate':'Decoded + RL (separate)','current_budget':'CS-DPP','predictive_budget':'PM-DPP',
       'predictive_deadline':'PDH (matched codec)','latency_only':'LOM',
       'greedy':'LOM','random':'Random','shortest_queue':'Shortest Queue'}
COLORS={'Proposed':'#1764a1','Decoded + RL (separate)':'#a1436e','CS-DPP':'#267e77','PM-DPP':'#d37922','PDH (matched codec)':'#8057a8',
        'LOM':'#4c4c4c','Random':'#a6a6a6','Shortest Queue':'#488b67'}
MARKERS={'Proposed':'o','Decoded + RL (separate)':'v','CS-DPP':'h','PM-DPP':'s','PDH (matched codec)':'^',
         'LOM':'X','Random':'d','Shortest Queue':'P'}


def read_rows(root):
    rows=[]
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path/'result.json').exists():continue
        result=json.loads((path/'result.json').read_text())
        manifest=json.loads((path/'manifest.json').read_text())
        bench=result['bench'];values=result['evaluation']
        if bench not in NAMES or not values:continue
        arrivals=sum(v['arrivals'] for v in values)
        reference=sum(v.get('budget_reference_sum',0.) for v in values)
        cost=(sum(v['budget_cost_sum'] for v in values)/reference if reference else
              sum(v['mean_normalized_resource_cost']*v['arrivals'] for v in values)/arrivals)
        budget=manifest['args'].get('average_resource_budget') or None
        episode=None
        if (path/'model_selection.json').exists():
            episode=json.loads((path/'model_selection.json').read_text())['episode']
        rows.append({'method':NAMES[bench],'bench':bench,'nominal_budget':budget,
            'selected_episode':episode,'arrivals':arrivals,
            'sla':sum(v['ontime'] for v in values)/arrivals,
            'resource_charge':cost,
            'mean_normalized_resource_cost':sum(v['mean_normalized_resource_cost']*v['arrivals']
                                                 for v in values)/arrivals,
            'pending':sum(v['pending'] for v in values),
            'budget_feasible':cost<=budget+1e-6 if budget else None,
            'path':str(path.resolve())})
    return rows


def style(ax):
    ax.grid(alpha=.18,linewidth=.6)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=10)


def save(fig,output,name):
    fig.tight_layout()
    for suffix in ('pdf','png'):fig.savefig(output/(name+'.'+suffix),dpi=240,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();rows=read_rows(args.root)
    if not rows:raise RuntimeError('No completed experiment results found')
    output=args.output or args.root/'figures';output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'ps.fonttype':42})
    budgets=sorted({r['nominal_budget'] for r in rows if r['bench']=='proposed'})
    if not budgets:budgets=sorted({r['nominal_budget'] for r in rows if r['nominal_budget']})
    methods=[name for name in NAMES.values() if name in {r['method'] for r in rows}]
    methods=list(dict.fromkeys(methods))
    fig,ax=plt.subplots(figsize=(5.7,3.6))
    for name in methods:
        group=sorted([r for r in rows if r['method']==name],key=lambda r:r['resource_charge'])
        x=[r['resource_charge'] for r in group];y=[100*r['sla'] for r in group]
        if name in ('Proposed','PM-DPP'):ax.plot(x,y,color=COLORS[name],lw=1.5,alpha=.8)
        for i,r in enumerate(group):
            ax.scatter(x[i],y[i],s=48,marker=MARKERS[name],edgecolors=COLORS[name],
                facecolors=COLORS[name] if r['budget_feasible'] else 'white',
                label=name if i==0 else None,zorder=3)
            if name=='Proposed':
                ax.annotate('B='+format(r['nominal_budget'],'.2f'),(x[i],y[i]),
                            xytext=(4,6),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Realized normalized resource charge')
    ax.set_ylabel('On-time completion (%)');ax.set_ylim(0,102)
    style(ax);ax.legend(fontsize=8,frameon=False,ncol=2)
    save(fig,output,'sla_cost_frontier')
    for metric,label,filename in (('sla','On-time completion (%)','sla_by_budget'),
                                   ('resource_charge','Realized normalized resource charge','cost_by_budget')):
        fig,ax=plt.subplots(figsize=(5.7,3.6))
        for name in methods:
            group=sorted([r for r in rows if r['method']==name],key=lambda r:r['nominal_budget'] or 0.)
            fixed=group[0]['bench'] in ('latency_only','greedy','random','shortest_queue')
            x=budgets if fixed else [r['nominal_budget'] for r in group]
            y=[group[0][metric]]*len(x) if fixed else [r[metric] for r in group]
            if metric=='sla':y=[100*v for v in y]
            ax.plot(x,y,color=COLORS[name],marker=MARKERS[name],label=name,lw=1.5,ms=5)
        if metric=='resource_charge' and budgets:
            ax.plot(budgets,budgets,'--',color='#777777',label='Budget bound',lw=1.)
        if metric=='sla':ax.set_ylim(0,102)
        ax.set_xlabel('Average resource budget B');ax.set_ylabel(label)
        if budgets:ax.set_xticks(budgets)
        style(ax);ax.legend(fontsize=8,frameon=False,ncol=2)
        save(fig,output,filename)
    (output/'summary.json').write_text(json.dumps(rows,indent=2))
    with (output/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# SLA-cost results','',
        'Hollow frontier markers exceed the method\'s nominal finite-trace budget. '
        'Three selected budgets show an empirical tradeoff, not a proof of global Pareto optimality.',
        '', '| Method | B | Selected RL episode | SLA | Charge | Budget feasible | Pending |',
        '|---|---:|---:|---:|---:|---|---:|']
    for r in rows:
        lines.append('| {method} | {budget} | {episode} | {sla:.2%} | {resource_charge:.4f} | {feasible} | {pending} |'.format(
            **r,budget=format(r['nominal_budget'],'.2f') if r['nominal_budget'] else '—',
            episode=r['selected_episode'] if r['selected_episode'] is not None else '—',
            feasible='yes' if r['budget_feasible'] else 'no'))
    lines+=['','RL contribution at the same budget (both policies must be feasible):','']
    for budget in budgets:
        proposed=next((r for r in rows if r['bench']=='proposed' and r['nominal_budget']==budget),None)
        myopic=next((r for r in rows if r['bench']=='predictive_budget' and r['nominal_budget']==budget),None)
        if proposed and myopic:
            eligible=proposed['budget_feasible'] and myopic['budget_feasible']
            lines.append('- B={:.2f}: Proposed minus PM-DPP SLA {:+.2f} pp; charge {:+.4f}; both feasible: {}.'.format(
                budget,100*(proposed['sla']-myopic['sla']),proposed['resource_charge']-myopic['resource_charge'],eligible))
    (output/'results.md').write_text('\n'.join(lines)+'\n')
    print(output.resolve())


if __name__=='__main__':main()
