"""Chronological Azure workload segments mapped to simulated background pressure."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np


@lru_cache(maxsize=4)
def load_azure(path):
    with np.load(path,allow_pickle=False) as archive:
        pressure=archive['pressure'].copy()
        metadata=json.loads(str(archive['metadata']))
    pressure.flags.writeable=False
    return pressure,metadata


@lru_cache(maxsize=24)
def segment_starts(path,split,steps,case):
    pressure,metadata=load_azure(path)
    lo,hi=metadata['ranges'][split]
    if hi-lo<steps:raise ValueError('Azure split is shorter than the requested trajectory')
    if case=='all':return np.arange(lo,hi-steps+1)
    if case!='busy_transitions':raise ValueError('Unknown Azure workload case')
    # Select a stated busy, variable workload condition using trace statistics,
    # not any routing policy result. Entire forecast windows stay in this split.
    windows=np.lib.stride_tricks.sliding_window_view(pressure[lo:hi],steps,axis=0)[::10]
    mean=windows.mean((1,2))
    span=(windows.max(2)-windows.min(2)).max(1)
    starts=lo+10*np.flatnonzero((mean>=.45)&(span>=.45))
    if not len(starts):raise ValueError('No eligible busy transition segments in Azure split')
    return starts


def azure_multipliers(scenario,nodes,neighbors,steps,seed,path,split,case):
    path=str(Path(path).resolve())
    pressure,metadata=load_azure(path)
    if metadata['sim_ms_per_bin']!=5.:
        raise ValueError('Azure asset must match the 5ms resource sampling grid')
    rng=np.random.default_rng(np.random.SeedSequence([seed,803]))
    eligible=segment_starts(path,split,steps,case)
    starts=rng.choice(eligible,size=len(nodes),replace=True)
    local=np.stack([pressure[start:start+steps] for start in starts],axis=1)
    values=np.ones((steps,len(nodes),14),np.float32)
    for s,name in enumerate(scenario.light):
        resources=np.asarray(scenario.services[name].resources[:2])
        weight=resources/max(resources.sum(),1e-9)
        jitter=rng.normal(0,.025,(steps,len(nodes)))
        values[:,:,s]=np.clip(1-.85*(local*weight).sum(-1)+jitter,.15,1.1)
    index={node:i for i,node in enumerate(nodes)}
    # Different endpoint loads create route-specific future bottlenecks.
    # No sinusoidal regime or complementary path multiplier is added here.
    for i,node in enumerate(nodes):
        for j,target in enumerate(neighbors[node]):
            traffic=.5*(local[:,i,0]+local[:,index[target],0])
            values[:,i,9+j]=np.clip(1-.85*traffic,.15,1.)
    provenance={'asset_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                'source_sha256':metadata['source_sha256'],'split':split,
                'case':case,'eligible_segment_count':int(len(eligible)),
                'segment_condition':'mean normalized pressure >= 0.45 and max feature span >= 0.45; stride 10 bins' if case=='busy_transitions' else 'all continuous segments',
                'node_segment_start_bins':dict(zip(nodes,map(int,starts))),
                'real_seconds_per_bin':metadata['real_seconds_per_bin'],
                'sim_ms_per_bin':metadata['sim_ms_per_bin'],
                'normalization_fit_split':'train','mapping':metadata['resource_mapping']}
    return values,provenance
