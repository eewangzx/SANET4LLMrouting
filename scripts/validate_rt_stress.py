"""Load sensitivity of fixed, already-trained policies; never select a model."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_rt_bench as v
from edge_msd.realtime_routing.scenario import build_scenario


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bench',choices=['raw','stats27','proposed','baselines'],required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--levels',default='2,3')
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output=Path(args.run)/args.bench
    manifest=json.loads((output/'manifest.json').read_text())
    cfg=manifest['args']
    _,settings,placement,anchor=v.build_world(seed=cfg['seed'],level=cfg['level'],
        duration=cfg['duration'],drain=0.,core_scale=cfg['core_scale'],
        light_per_service=cfg['light_per_service'])
    assert placement.placement_hash==manifest['placement_hash']
    started=time.perf_counter()
    result={}
    zero_embedding=[False]
    if args.bench!='baselines':
        checkpoint=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        if args.bench=='stats27':
            codec=v.FullStatsCodec(9,16,8,4,v.bm.DEVICE).to(v.bm.DEVICE)
        else:
            codec=v.Codec(args.bench,9,16,8,4,device=v.bm.DEVICE).to(v.bm.DEVICE)
        codec.load_state_dict(checkpoint['codec'])
        codec.eval()
        agent=v.DoubleDQNAgent(129,10,device='cpu')
        agent.online.load_state_dict(checkpoint['online'])
        agent.online.eval()
        original_embed=v.bm.embed_nodes
        def counted_embed(env,codec_arg,trace):
            env._report_messages=getattr(env,'_report_messages',0)+len(env.node_ids)
            env._report_bytes=getattr(env,'_report_bytes',0)+len(env.node_ids)*codec_arg.total_bytes()
            embedding=original_embed(env,codec_arg,trace)
            return np.zeros_like(embedding) if zero_embedding[0] else embedding
        v.bm.embed_nodes=counted_embed
    for level in map(float,args.levels.split(',')):
        scenario=build_scenario(2026,load_multiplier=level)
        methods=(['earliest_finish','greedy_now','future_factor_heuristic']
                 if args.bench=='baselines' else
                 ([args.bench,args.bench+'_zero'] if args.bench=='proposed' else [args.bench]))
        rows={}
        for method in methods:
            zero_embedding[0]=method.endswith('_zero')
            tests=[]
            for seed in manifest['test_seeds']:
                trace=v.CorrelatedFactorTrace(scenario.nodes,scenario.light,anchor.n_slots,seed=seed)
                env=v.PairedEnv(scenario,settings,placement,seed,0.,trace)
                if args.bench=='baselines':
                    if method=='earliest_finish':
                        while not env.terminated: env.step(v.earliest_finish_node(env))
                    else:
                        v.bm.run_baseline(env,trace,1 if method=='greedy_now' else 4)
                else:
                    v.bm.run_episode(env,trace,codec,sorted(scenario.services),agent)
                row={'seed':seed,**v.summarize(env)}
                tests.append(row)
                print('STRESS',level,method,row,flush=True)
                (output/'stress_status.json').write_text(json.dumps({'status':'running',
                    'level':level,'method':method,'seed':seed,
                    'elapsed_seconds':time.perf_counter()-started},indent=2))
            rows[method]=tests
        result[str(level)]=rows
        (output/'stress.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (output/'stress_status.json').write_text(json.dumps({'status':'complete',
        'elapsed_seconds':time.perf_counter()-started},indent=2))
    print('STRESS DONE',args.bench,flush=True)


if __name__=='__main__': main()
