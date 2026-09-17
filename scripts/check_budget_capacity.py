"""Optimistic average charge lower bound under fixed core-flow capacities."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edge_msd.icc_paper import load_scenario
from edge_msd.realtime_routing.fixed_placement import load_placement
from edge_msd.realtime_routing.dynamic_routing import capacity_rank_resource_prices


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,default=Path('data/icc_paper/scenario_2026.json'))
    p.add_argument('--placement',type=Path,default=Path('configs/fixed_placement_routing_main.json'))
    p.add_argument('--load',type=float,default=.5)
    p.add_argument('--output',type=Path)
    args=p.parse_args();s=load_scenario(args.dataset,args.load);placement=load_placement(args.placement)
    prices=capacity_rank_resource_prices(s)
    rates={t:sum(u.rates_per_second.get(t,0.) for u in s.users)/1000. for t in s.tasks}
    total=sum(rates.values())
    hosts={name:[n for (sn,n),count in placement.counts.items() if sn==name and count]
           for name in s.services}
    refs={t:sum(s.services[name].work_mb*max(prices[n] for n in hosts[name])
                for name in task.order) for t,task in s.tasks.items()}
    variables=[(t,name,n) for t,task in s.tasks.items() for name in task.order for n in hosts[name]]
    cost=np.array([s.services[name].work_mb*prices[n]/refs[t]/total for t,name,n in variables])
    eq=[];beq=[]
    for t,task in s.tasks.items():
        for name in task.order:
            eq.append([float(v[0]==t and v[1]==name) for v in variables]);beq.append(rates[t])
    ub=[];bub=[]
    for (name,node),count in placement.core.items():
        ub.append([float(v[1]==name and v[2]==node) for v in variables])
        bub.append(count/np.ceil(s.services[name].mean_processing_ms))
    result=linprog(cost,A_ub=np.array(ub),b_ub=bub,A_eq=np.array(eq),b_eq=beq,
                   bounds=(0,None),method='highs')
    values={'load':args.load,'core_flow_feasible':bool(result.success),
        'optimistic_stable_core_cost_lower_bound':float(result.fun) if result.success else None,
        'definition':'route all offered stage flows; each core host flow <= fixed instances / slot-rounded processing ms; light capacities, transfer and deadlines relaxed',
        'limitation':'necessary optimistic bound for stable complete-flow service, not sufficient for deadline SLA',
        'resource_prices':prices,'placement_hash':placement.placement_hash}
    text=json.dumps(values,indent=2);print(text)
    if args.output:args.output.write_text(text+'\n')


if __name__=='__main__':main()
