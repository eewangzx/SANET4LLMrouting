"""One compact-state routing configuration; reuses existing frozen codecs."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_rt_bench as v
from edge_msd.realtime_routing.compact_state import build_compact_state
from edge_msd.realtime_routing.scenario import build_scenario
from edge_msd.placement import validate_resources


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bench',choices=['proposed','raw','baseline'],required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--codec-run',required=True)
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--test-seeds',default='31000,31001,31002')
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    # Frozen GRU and tiny Q run on CPU: no pretraining or GPU synchronization.
    v.bm.DEVICE=torch.device('cpu')
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    def write(name,value):
        temporary=root/(name+'.tmp')
        temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
        temporary.replace(root/name)
    write('status.json',{'status':'running','stage':'setup'})
    _,settings,placement,anchor=v.build_world(seed=args.seed,level=1.,duration=8,
        drain=0.,core_scale=.3,light_per_service=3)
    scenario=build_scenario(2026,load_multiplier=2.)
    validate_resources(scenario,placement.counts)
    services=sorted(scenario.services);tasks=sorted(scenario.tasks)
    test_seeds=[int(x) for x in args.test_seeds.split(',')]
    train_seeds=list(range(41000,41000+args.episodes))
    reward_scale=8/1000*sum(sum(u.rates_per_second.values()) for u in scenario.users)
    manifest={'args':vars(args),'train_seeds':train_seeds,'test_seeds':test_seeds,
        'training_load':2.,'test_loads':[2.,3.],'duration_ms':8,
        'placement_anchor_load':1.,'placement_hash':placement.placement_hash,
        'reward_scale':reward_scale,'gamma':1.,'lr':3e-5,'hidden':64,
        'state_fields':['node_embedding','node_ledger','candidate_transfer_ms/50',
                        'current_service_onehot','task_type_onehot','remaining_deadline_ms/50'],
        'information_assumption':'immediate reliable dispatch/completion feedback; static link parameters',
        'reports':'10ms decision-time caching; byte accounting only, no delivery-delay/capacity model',
        'physics':'current ICC Gamma base multiplied by AR factors',
        'source_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__),Path.cwd()/'edge_msd/realtime_routing/compact_state.py',
                         Path.cwd()/'edge_msd/realtime_routing/agent.py',
                         Path.cwd()/'edge_msd/realtime_routing/benchmark.py',
                         Path.cwd()/'scripts/validate_rt_bench.py']}}
    write('manifest.json',manifest)
    def world(level,seed):
        sc=build_scenario(2026,load_multiplier=level)
        trace=v.CorrelatedFactorTrace(sc.nodes,sc.light,anchor.n_slots,seed=seed)
        return v.PairedEnv(sc,settings,placement,seed,0.,trace),trace
    if args.bench!='baseline':
        codec_path=Path(args.codec_run)/args.bench/'last.pt'
        checkpoint=torch.load(codec_path,map_location='cpu',weights_only=False)
        codec=v.Codec(args.bench,9,16,8,4,device='cpu').to('cpu')
        codec.load_state_dict(checkpoint['codec']);codec.eval()
        for parameter in codec.parameters():parameter.requires_grad_(False)
        manifest['codec_checkpoint']=str(codec_path)
        manifest['codec_sha256']=hashlib.sha256(codec_path.read_bytes()).hexdigest()
        # Embed is shared by the actual next action and replay next state through
        # the existing corrected 10ms report cache.
        original_embed=v.bm.embed_nodes
        def counted_embed(env,codec_arg,trace):
            env._report_messages=getattr(env,'_report_messages',0)+len(env.node_ids)
            env._report_bytes=getattr(env,'_report_bytes',0)+len(env.node_ids)*codec_arg.total_bytes()
            return original_embed(env,codec_arg,trace)
        v.bm.embed_nodes=counted_embed
        v.bm.build_state=lambda env,emb,svc:build_compact_state(env,emb,svc,tasks)
        example,trace=world(2.,train_seeds[0])
        with torch.no_grad():
            embedding=v.bm.embed_nodes(example,codec,trace)
        state=build_compact_state(example,embedding,services,tasks)
        agent=v.ScaledAgent(len(state),len(example.node_ids),hidden=64,lr=3e-5,
                           gamma=1.,device='cpu',reward_scale=reward_scale)
        agent.rng=np.random.default_rng(args.seed)
        manifest['state_dim']=len(state)
        manifest['q_parameters']=sum(p.numel() for p in agent.online.parameters())
        manifest['report_bytes_per_node']=codec.total_bytes()
        write('manifest.json',manifest)
        print('START',args.bench,manifest,flush=True)
        buffer=[];training=[]
        for episode,seed in enumerate(train_seeds,1):
            env,trace=world(2.,seed)
            epsilon=max(.05,1-(episode-1)/(.75*args.episodes))
            reward,decisions=v.bm.run_episode(env,trace,codec,services,agent,epsilon,True,buffer)
            row=v.log_training_episode(agent,root/'training.csv',episode,reward,decisions,
                                      v.summarize(env)['sla'],epsilon)
            training.append({'episode':episode,'seed':seed,**v.summarize(env)})
            print('TRAIN',args.bench,episode,'sla',row['sla'],'grad',row['grad_norm_pre_mean'],
                  'loss',row['loss_mean'],flush=True)
            write('status.json',{'status':'running','stage':'training','episode':episode,
                                'updates':agent.updates,'elapsed_seconds':time.perf_counter()-start})
        torch.save({'online':agent.online.state_dict(),'target':agent.target.state_dict(),
                    'optimizer':agent.opt.state_dict(),'codec':codec.state_dict(),
                    'manifest':manifest},root/'last.pt')
        write('training_requests.json',training)
    evaluation=[]
    for level in (2.,3.):
        for seed in test_seeds:
            env,trace=world(level,seed)
            if args.bench=='baseline':
                while not env.terminated:env.step(v.earliest_finish_node(env))
            else:
                v.bm.run_episode(env,trace,codec,services,agent)
            row={'load':level,'seed':seed,**v.summarize(env)}
            evaluation.append(row)
            write('evaluation.json',evaluation)
            print('TEST',args.bench,row,flush=True)
            write('status.json',{'status':'running','stage':'evaluation','load':level,'seed':seed,
                                'elapsed_seconds':time.perf_counter()-start})
    write('result.json',{'bench':args.bench,'evaluation':evaluation,
                        'elapsed_seconds':time.perf_counter()-start})
    write('status.json',{'status':'complete','elapsed_seconds':time.perf_counter()-start})
    print('DONE',args.bench,flush=True)


if __name__=='__main__':main()
