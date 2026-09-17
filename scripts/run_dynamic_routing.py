"""Routing-only benchmark with existing time-varying ICC and prediction codecs."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edge_msd.config import Settings
from edge_msd.icc_paper import load_scenario
from edge_msd.icc_coupled.codecs import load_codec
from edge_msd.icc_coupled.environment import TelemetrySettings
from edge_msd.realtime_routing.agent import DoubleDQNAgent,log_training_episode
from edge_msd.realtime_routing.dynamic_routing import DynamicRoutingEnvironment,CurrentMeasurementCodec
from edge_msd.realtime_routing.fixed_placement import FixedPlacement,scenario_hash,load_placement,save_placement
from edge_msd.realtime_routing.experiment import build_world
from edge_msd.placement import validate_resources


class SLAAgent(DoubleDQNAgent):
    def __init__(self,*args,reward_scale=1.,**kwargs):
        super().__init__(*args,**kwargs);self.reward_scale=reward_scale
    def update(self,batch):
        return super().update([(s,a,r/self.reward_scale,s2,m,d) for s,a,r,s2,m,d in batch])


def rollout(env,agent=None,epsilon=0.,buffer=None,n_step=8):
    env.reset()
    total=0.;decisions=0;pending=deque()
    def append_transition():
        first=pending[0];last=pending[-1]
        buffer.append((first[0],first[1],sum(t[2] for t in pending),last[3],last[4],last[5]))
        pending.popleft()
    while not env.terminated:
        state,mask=env.state()
        if agent is None:node=env.greedy_node();action=env.node_ids.index(node)
        else:
            action=agent.act(state,mask,epsilon) if buffer is not None else agent.greedy(state,mask)
            node=env.node_ids[action]
        _,reward,term,_,_=env.step(node)
        total+=reward;decisions+=1
        if buffer is not None:
            agent.record_reward(reward)
            nxt,m2=(state,mask) if term else env.state()
            pending.append((state,action,reward,nxt,m2,float(term)))
            if term:
                while pending:append_transition()
            elif len(pending)>=n_step:append_transition()
            if len(buffer)>=128:agent.update(random.sample(buffer,64))
    env.update_telemetry()
    assert abs(total-(2*env.sla_counts()[0]-len(env.requests)))<1e-6
    return total,decisions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bench',choices=['proposed','raw','greedy'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--placement',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('data/icc_paper/scenario_2026.json'))
    p.add_argument('--models',type=Path,default=Path('runs/icc_coupled_models'))
    p.add_argument('--load',type=float,default=.4)
    p.add_argument('--arrival-ms',type=int,default=120)
    p.add_argument('--drain-ms',type=float,default=150.)
    p.add_argument('--episodes',type=int,default=8)
    p.add_argument('--test-seeds',default='55000,55001,55002')
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    scenario=load_scenario(args.dataset,args.load)
    settings=Settings(duration_ms=args.arrival_ms,seed=args.seed,slot_ms=1,
                      ec_admission_window_ms=1,record_trace=True)
    if args.prepare:
        # Keep the SAME previously validated routing deployment, rather than
        # re-solving the ICC deployment problem for the new dynamic experiment.
        _,_,original,_=build_world(seed=7,level=1.,duration=args.arrival_ms,
            drain=args.drain_ms,core_scale=.3,light_per_service=3)
        placement=FixedPlacement(original.core,original.light,original.core_solver,
                                 scenario_hash(scenario),original.placement_hash)
        validate_resources(scenario,placement.counts)
        save_placement(placement,args.placement)
        print(json.dumps({'placement_hash':placement.placement_hash,
              'core_instances':sum(placement.core.values()),'light_instances':sum(placement.light.values())}))
        return
    placement=load_placement(args.placement)
    codec_name='current' if args.bench=='greedy' else ('raw' if args.bench=='raw' else 'importance')
    codec_path=None if args.bench=='greedy' else args.models/(codec_name+'.pt')
    codec=CurrentMeasurementCodec() if codec_path is None else load_codec(codec_path).eval().requires_grad_(False)
    telemetry=TelemetrySettings(arrival_ms=args.arrival_ms,control_ms=1,report_ms=100,
                                report_bps=64000,dynamic=True)
    def world(seed):
        return DynamicRoutingEnvironment(scenario,settings,placement,codec,telemetry,
                                          seed=seed,drain_ms=args.drain_ms)
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    def write(name,value):
        tmp=root/(name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(root/name)
    write('status.json',{'status':'running','stage':'setup'})
    example=world(52000);state,mask=example.state()
    if args.smoke:
        assert not any(report is not None for report in example.received)
        assert example.channel.transmitted_bytes==0
        before=state.copy();future=example.resource.values.copy()
        k=example.resource.index(example.now)
        example.resource.values[k+1:,:,9:14]=.15
        example.observed_network._cache.clear()
        example.observed_network.ready_cache.clear()
        assert np.array_equal(before,example.state()[0]),'Hidden future leaked into state'
        example.resource.values[:]=future
        before_counts=dict(placement.counts)
        elapsed=time.perf_counter()
        reward,decisions=rollout(example)
        diagnostic=example.diagnostics()
        a,b,*_=scenario.links[0]
        probe=[{'start_ms':t,'delay_ms':example.network._delay_at(a,b,1.,t)}
               for t in (0.,50.,100.,150.,200.)]
        assert np.ptp([row['delay_ms'] for row in probe])>1e-4,'Link did not vary with time'
        assert {key:len(pool) for key,pool in example.pools.items()}==before_counts
        write('smoke.json',{'reward':reward,'decisions':decisions,'state_dim':len(state),
              'wall_seconds':time.perf_counter()-elapsed,'link_probe':{'source':a,'target':b,
              'data_mb':1.,'samples':probe},**diagnostic})
        print('SMOKE',json.dumps({k:v for k,v in json.loads((root/'smoke.json').read_text()).items()
                                 if k not in ('requests','tasks')},indent=2),flush=True)
        write('status.json',{'status':'complete','stage':'smoke'})
        return
    train_seeds=list(range(52000,52000+args.episodes))
    test_seeds=[int(x) for x in args.test_seeds.split(',')]
    reward_scale=args.arrival_ms/1000*sum(sum(u.rates_per_second.values()) for u in scenario.users)
    sources=[Path(__file__),Path.cwd()/'edge_msd/realtime_routing/dynamic_routing.py',
        Path.cwd()/'edge_msd/icc_coupled/dynamics.py',Path.cwd()/'edge_msd/icc_coupled/environment.py',
        Path.cwd()/'edge_msd/icc_coupled/codecs.py',Path.cwd()/'edge_msd/realtime_routing/environment.py',
        Path.cwd()/'edge_msd/realtime_routing/agent.py']
    manifest={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'settings':asdict(settings),'telemetry':asdict(telemetry),'placement_hash':placement.placement_hash,
        'core_instances':sum(placement.core.values()),'light_instances':sum(placement.light.values()),
        'placement_source':'same previously validated fixed routing counts: anchor load 1, core scale .3, light/service 3',
        'codec':codec_name,'codec_sha256':hashlib.sha256(codec_path.read_bytes()).hexdigest() if codec_path else None,
        'codec_encoding_hash':codec.encoding_hash() if codec_path else None,'codec_config':codec.config(),
        'state_dim':len(state),'hidden':64,'gamma':1.,'n_step':8,'lr':3e-5,
        'reward_scale':reward_scale,'train_seeds':train_seeds,'test_seeds':test_seeds,
        'state_fields':['received_node_representation','current_service_backlog',
                        'received_forecast_transfer_estimate','report_age','missing_report',
                        'service_onehot','task_onehot','request_gateway_onehot','remaining_deadline'],
        'actions':'choose service execution node only; instance counts fixed',
        'future_access':'physical simulator only; policy consumes delivered messages and known ledger',
        'observer':'last received raw current measurements, no learned compression/prediction' if args.bench=='greedy' else 'received codec forecasts',
        'data_network':'existing fixed nominal paths, time-varying integrated rates; independent transfers',
        'control_traffic':'existing separate reliable dispatch/completion feedback assumption',
        'source_sha256':{str(path.relative_to(Path.cwd())):hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sources}}
    if args.bench!='greedy':
        torch.manual_seed(args.seed)
        agent=SLAAgent(len(state),len(example.node_ids),hidden=64,lr=3e-5,gamma=1.,
                       reward_scale=reward_scale,device='cpu')
        agent.rng=np.random.default_rng(args.seed)
        manifest['q_parameters']=sum(p.numel() for p in agent.online.parameters())
    write('manifest.json',manifest)
    print('START',args.bench,manifest,flush=True)
    if args.bench!='greedy':
        buffer=[]
        for episode,seed in enumerate(train_seeds,1):
            env=world(seed);epsilon=max(.05,1-(episode-1)/(.75*args.episodes))
            reward,decisions=rollout(env,agent,epsilon,buffer)
            diagnostics=env.diagnostics()
            log_training_episode(agent,root/'training.csv',episode,reward,decisions,
                                 diagnostics['sla'],epsilon)
            write(f'train_{episode:02d}.json',diagnostics)
            print('TRAIN',args.bench,episode,'sla',diagnostics['sla'],flush=True)
            write('status.json',{'status':'running','stage':'training','episode':episode,
                                'updates':agent.updates,'elapsed_seconds':time.perf_counter()-started})
        torch.save({'online':agent.online.state_dict(),'target':agent.target.state_dict(),
                    'optimizer':agent.opt.state_dict(),'manifest':manifest},root/'last.pt')
    evaluations=[]
    for seed in test_seeds:
        env=world(seed)
        reward,decisions=rollout(env,None if args.bench=='greedy' else agent)
        diagnostics=env.diagnostics()
        write(f'test_{seed}.json',diagnostics)
        rows={k:v for k,v in diagnostics.items() if k!='requests'}
        evaluations.append({'seed':seed,'reward':reward,'decisions':decisions,**rows})
        write('evaluation.json',evaluations)
        print('TEST',args.bench,seed,'sla',diagnostics['sla'],flush=True)
        write('status.json',{'status':'running','stage':'evaluation','seed':seed,
                            'elapsed_seconds':time.perf_counter()-started})
    write('result.json',{'bench':args.bench,'evaluation':evaluations})
    write('status.json',{'status':'complete','elapsed_seconds':time.perf_counter()-started})


if __name__=='__main__':main()
