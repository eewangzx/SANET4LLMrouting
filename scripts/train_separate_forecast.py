"""Train a predictive compression codec independently of routing rewards."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edge_msd.icc_paper import load_scenario
from edge_msd.icc_coupled.codecs import CoupledCodec
from edge_msd.icc_coupled.dynamics import ResourceTrace
from edge_msd.realtime_routing.fixed_placement import load_placement


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--seed',type=int,default=7)
    args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(args.seed)
    experiment=json.loads((args.reference/'experiment.json').read_text())
    scenario=load_scenario(args.reference/'scenario.json',experiment['load'])
    placement=load_placement(args.reference/'fixed_placement.json')
    duration=experiment['warmup_ms']+experiment['arrival_ms']+experiment['drain_ms']
    device='cuda' if torch.cuda.is_available() else 'cpu'
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    def write(name,value):
        (root/name).write_text(json.dumps(value,indent=2)+'\n')
    def arrays(seeds,split):
        xs=[];ys=[];hashes=[]
        for seed in seeds:
            trace=ResourceTrace(scenario,duration,seed,profile=experiment['dynamics_profile'],
                azure_path=args.reference/'azure_trace.npz' if experiment['dynamics_profile']=='azure' else None,
                azure_split=split,azure_case=experiment.get('azure_case','busy_transitions'))
            hashes.append(trace.sha256)
            for when in np.arange(0,duration,20.):
                xs.extend(trace.window(when));ys.extend(trace.future(when))
        return torch.as_tensor(np.asarray(xs),device=device),torch.as_tensor(np.asarray(ys),device=device),hashes
    write('status.json',{'status':'running','stage':'dataset'})
    train_x,train_y,train_hashes=arrays(experiment['train_seeds'],'train')
    val_x,val_y,val_hashes=arrays(experiment['validation_seeds'],'validation')
    codec=CoupledCodec(kind='importance').to(device)
    active={service for user in scenario.users for task,rate in user.rates_per_second.items()
            if rate>0 for service in scenario.tasks[task].order}
    example=ResourceTrace(scenario,duration,experiment['train_seeds'][0],profile=experiment['dynamics_profile'],
        azure_path=args.reference/'azure_trace.npz' if experiment['dynamics_profile']=='azure' else None,
        azure_split='train',azure_case=experiment.get('azure_case','busy_transitions'))
    features=np.zeros((len(example.nodes),14),np.float32)
    for i,node in enumerate(example.nodes):
        for name,j in example.service_index.items():
            features[i,j]=float(name in active and bool(placement.counts.get((name,node))))
        features[i,9:9+len(example.neighbors[node])]=1.
    feature=torch.as_tensor(features,device=device)
    train_weights=feature.repeat(len(train_x)//len(feature),1)
    val_weights=feature.repeat(len(val_x)//len(feature),1)
    def objective(x,y,weights):
        z,scores,_=codec.representation(x)
        error=(codec.forecast_latent(z)-y).square()
        mse=(error*weights[:,None,:]).sum()/(codec.horizon*weights.sum()).clamp_min(1.)
        budget=torch.exp((scores.sum(-1)-3).clamp(-8,8)).mean()
        return mse+.001*budget,mse
    @torch.no_grad()
    def validate():
        codec.eval();weighted=0.;count=0.
        for start in range(0,len(val_x),256):
            _,mse=objective(val_x[start:start+256],val_y[start:start+256],val_weights[start:start+256])
            n=float(val_weights[start:start+256].sum());weighted+=float(mse)*n;count+=n
        return weighted/count
    opt=torch.optim.Adam(codec.parameters(),lr=1e-3)
    generator=torch.Generator().manual_seed(args.seed)
    best=validate();best_epoch=0;initial_mse=best
    codec.save(root/'importance.pt')
    curve=[]
    for epoch in range(1,args.epochs+1):
        codec.train();losses=[];norms=[]
        for indices in torch.randperm(len(train_x),generator=generator).split(256):
            indices=indices.to(device)
            loss,_=objective(train_x[indices],train_y[indices],train_weights[indices])
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(codec.parameters(),10.,error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
        mse=validate()
        if mse<best:
            best=mse;best_epoch=epoch;codec.save(root/'importance.pt')
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_mse':mse,
             'grad_norm_pre_mean':float(np.mean(norms)),'grad_norm_pre_max':max(norms),
             'clipped_updates':sum(norm>10 for norm in norms),'selected_epoch':best_epoch}
        curve.append(row)
        write('status.json',{'status':'running','stage':'forecast_pretraining',**row,
                             'elapsed_seconds':time.perf_counter()-started})
        if epoch==1 or epoch%10==0:print('FORECAST',json.dumps(row),flush=True)
    write('metrics.json',{'training':'forecast MSE on deployed active-task light rates and outgoing links + 0.001 soft selection budget; no RL task loss',
        'initial_validation_mse':initial_mse,'selected_epoch':best_epoch,'selected_validation_mse':best,
        'curve':curve,'train_seeds':experiment['train_seeds'],'validation_seeds':experiment['validation_seeds'],
        'train_resource_hashes':train_hashes,'validation_resource_hashes':val_hashes,
        'train_windows':len(train_x),'validation_windows':len(val_x),'device':device,
        'codec_sha256':hashlib.sha256((root/'importance.pt').read_bytes()).hexdigest(),
        'training_stride_ms':20,'duration_ms':duration,'feature_mask':features.tolist(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'elapsed_seconds':time.perf_counter()-started})
    write('status.json',{'status':'complete','selected_epoch':best_epoch,
                         'selected_validation_mse':best,'elapsed_seconds':time.perf_counter()-started})


if __name__=='__main__':main()
