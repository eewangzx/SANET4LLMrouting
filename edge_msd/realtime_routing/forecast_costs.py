"""Differentiable version of the existing received-forecast routing estimates.

Only static paths, received report ages and the controller's own queue ledger
enter the context. Physical future realizations and prediction labels never do.
Transfers integrate the same piecewise-constant rates at absolute 5ms boundaries.
"""
import numpy as np
import torch


def forecast_costs(decoded,contexts,sample_ms=5.):
    device,dtype=decoded.device,decoded.dtype
    def stack(key):
        return torch.as_tensor(np.stack([c[key] for c in contexts]),device=device)
    batch,nodes,horizon,_=decoded.shape
    ages=stack('age_bins').long()
    bins=(ages[...,None]+torch.arange(horizon,device=device)).clamp_max(horizon-1)
    forecast=decoded.gather(2,bins[...,None].expand(-1,-1,-1,decoded.shape[-1]))
    phase=stack('phase').to(dtype)
    paths=stack('paths').to(dtype)
    cursor=stack('starts').to(dtype)
    amounts=stack('amounts').to(dtype)
    # Every route is padded to the task's parent count and the graph's maximum
    # simple path length. Inactive hops have no transmission or propagation.
    prefix_shape=cursor.shape
    durations=torch.full((*prefix_shape,horizon),sample_ms,device=device,dtype=dtype)
    durations[...,0]=sample_ms-phase[:,None,None]
    left=torch.cat((torch.zeros_like(durations[...,:1]),durations.cumsum(-1)[...,:-1]),-1)
    batch_idx=torch.arange(batch,device=device)[:,None,None,None]
    time_idx=torch.arange(horizon,device=device)[None,None,None,:]
    for hop in range(paths.shape[-2]):
        edge=paths[...,hop,:]
        active=edge[...,4]>0
        if not active.any():continue
        origin=edge[...,0].long();port=edge[...,1].long()
        rates=forecast[batch_idx,origin[...,None],time_idx,port[...,None]]/edge[...,2,None].clamp_min(1e-8)
        capacity=(rates*durations).cumsum(-1)
        prefix=torch.cat((torch.zeros_like(capacity[...,:1]),capacity[...,:-1]),-1)
        k=((cursor+phase[:,None,None])/sample_ms).floor().long().clamp(0,horizon-1)
        pick=lambda x,indices:x.gather(-1,indices[...,None]).squeeze(-1)
        at_start=pick(prefix,k)+pick(rates,k)*(cursor-pick(left,k))
        goal=at_start+amounts
        index=torch.searchsorted(capacity.contiguous(),goal[...,None].contiguous()).squeeze(-1).clamp_max(horizon-1)
        finish=pick(left,index)+(goal-pick(prefix,index))/pick(rates,index)+edge[...,3]
        cursor=torch.where(active,finish,cursor)
    transfer=cursor.max(-1).values
    light=stack('light').bool();service=stack('service_index').long()
    service_rates=forecast.gather(-1,service[:,None,None,None].clamp_min(0).expand(-1,nodes,horizon,1)).squeeze(-1)
    mean=stack('mean_processing_ms').to(dtype)
    backlog=stack('backlogs').to(dtype)
    offset=transfer+backlog*mean[:,None]/service_rates[...,0].clamp_min(.15)
    k=(offset/sample_ms).floor().long().clamp(0,horizon-1)
    indices=k[...,None]+torch.arange(3,device=device)
    valid=indices<horizon
    rate=(service_rates.gather(-1,indices.clamp_max(horizon-1))*valid).sum(-1)/valid.sum(-1)
    per_job=torch.where(light[:,None],mean[:,None]/rate.clamp_min(.15),
                        stack('core_processing_ms').to(dtype)[:,None])
    costs=transfer+(backlog+1)*per_job
    costs=torch.where(stack('deployed').bool(),costs,torch.full_like(costs,500.))/50.
    availability=forecast[:,:,:11,:9].mean(2).clamp_min(.15)
    downstream=stack('downstream_core_ms').to(dtype)+(stack('downstream_light_work').to(dtype)/availability).sum(-1)
    pressure=torch.log1p(downstream/50.)
    return pressure,costs
