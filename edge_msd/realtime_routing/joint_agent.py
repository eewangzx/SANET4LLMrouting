"""Joint predictive compression and masked Double DQN for routing."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from edge_msd.realtime_routing.agent import DoubleDQNAgent,CandidateDuelingQNet


def gradient_norm(parameters):
    values=[p.grad.detach().norm() for p in parameters if p.grad is not None]
    return torch.linalg.vector_norm(torch.stack(values)) if values else torch.tensor(0.)


def terminal_utility(request,deadline_ms,latency_weight=0.,resource_weight=0.,data_weight=0.,
                     resource_cost=0.,data_cost=0.):
    """Bounded full-request SLA utility with delay and route-cost preferences.

    Pending requests get zero, rather than a favorable censored latency estimate.
    Costs must already be normalized to [0, 1]. At zero weights this is exactly
    the previous binary terminal target.
    """
    if request.completion_ms is None:return 0.
    delay=max(0.,request.completion_ms-request.arrival_ms)
    success=float(delay<=deadline_ms+1e-9)
    score=1.-(min(delay/deadline_ms,3.)/3.)**2
    resource_score=1.-min(1.,max(0.,resource_cost))
    data_score=1.-min(1.,max(0.,data_cost))
    return ((success+latency_weight*score+resource_weight*resource_score+data_weight*data_score)
            /(1.+latency_weight+resource_weight+data_weight))


class JointRoutingAgent(DoubleDQNAgent):
    def __init__(self, state_dim, action_dim, codec, reward_scale,
                 hidden=64, lr=3e-5, prediction_weight=.1, device='cpu',train_codec=True,
                 candidate_features=False,include_route_costs=False):
        super().__init__(state_dim,action_dim,hidden=hidden,lr=lr,gamma=1.,device=device)
        if candidate_features:
            self.online=CandidateDuelingQNet(state_dim,action_dim,codec.latent_dim,
                include_costs=include_route_costs,hidden=hidden).to(device)
            self.target=deepcopy(self.online).requires_grad_(False)
        self.train_codec=train_codec
        self.codec=codec.to(device).eval().requires_grad_(train_codec)
        self.target_codec=deepcopy(codec).eval().requires_grad_(False)
        self.prediction_weight=prediction_weight
        self.reward_scale=reward_scale
        self.prediction_feature_mask=None
        self.latency_weight=0.
        self.resource_cost_weight=0.
        self.data_cost_weight=0.
        self.route_cost_state=False
        self.global_return=False
        self.budget_state=False
        self._episode_utilities=[]
        self.parameters=list(self.online.parameters())+(list(self.codec.parameters()) if train_codec else [])
        self.opt=torch.optim.Adam(self.parameters,lr=lr)

    def embed(self, observations, codec, labels=False):
        states=torch.as_tensor(np.stack([o[0] for o in observations]),device=self.device)
        if not self.train_codec:return states,None,None,None,None
        nodes=self.action_dim
        records=[r for o in observations for r in o[1]]
        valid=torch.as_tensor([r is not None for r in records],device=self.device)
        empty=np.zeros((codec.history,codec.features),np.float32)
        histories=torch.as_tensor(np.stack([empty if r is None else r[0] for r in records]),device=self.device)
        z,scores,_=codec.representation(histories)
        latent=(z*valid[:,None]).reshape(len(observations),nodes*codec.latent_dim)
        state=torch.cat((latent,states[:,nodes*codec.latent_dim:]),dim=-1)
        if len(observations[0])==3:
            from edge_msd.realtime_routing.forecast_costs import forecast_costs
            decoded=codec.forecast_latent(z).reshape(len(observations),nodes,codec.horizon,codec.targets)
            decoded=torch.where(valid.reshape(len(observations),nodes,1,1),decoded,torch.ones_like(decoded))
            pressure,costs=forecast_costs(decoded,[o[2] for o in observations],codec.sample_ms)
            start=nodes*codec.latent_dim
            state=torch.cat((state[:,:start+nodes],pressure,costs,state[:,start+3*nodes:]),dim=-1)
        targets=None
        if labels:
            target_empty=np.zeros((codec.horizon,codec.targets),np.float32)
            targets=torch.as_tensor(np.stack([target_empty if r is None else r[1] for r in records]),device=self.device)
        return state,z,scores,valid,targets

    def forecast_loss(self,s,z,scores,valid,targets):
        if not self.train_codec:return s.new_zeros(()),s.new_zeros(())
        if valid.any():
            prediction=self.codec.forecast_latent(z[valid])
            if self.prediction_feature_mask is None:
                prediction_loss=nn.functional.mse_loss(prediction,targets[valid])
            else:
                # Concentrate scarce representation capacity on the deployed
                # task's quantities and forecast times used before its deadline.
                nodes=self.action_dim
                start=nodes*(self.codec.latent_dim+3+(2 if self.route_cost_state else 0))
                ages=s[:,start:start+nodes].reshape(-1)*100.
                slack=(s[:,-3 if self.budget_state else -2]*50.).clamp(10.,100.)
                slack=slack[:,None].expand(-1,nodes).reshape(-1)
                times=torch.arange(self.codec.horizon,device=self.device)*self.codec.sample_ms
                ages=ages.clamp(0.,times[-1])
                temporal=(times[None,:]>=ages[:,None]-10.)&(times[None,:]<=ages[:,None]+slack[:,None])
                feature=self.prediction_feature_mask.repeat(s.shape[0],1)
                weights=(temporal[:,:,None]*feature[:,None,:])[valid]
                prediction_loss=((prediction-targets[valid]).square()*weights).sum()/weights.sum().clamp_min(1.)
            budget_loss=torch.exp((scores[valid].sum(-1)-3).clamp(-8,8)).mean()
            auxiliary=prediction_loss+.001*budget_loss
        else:
            prediction_loss=z.sum()*0.;auxiliary=prediction_loss
        return prediction_loss,auxiliary

    def update(self, batch):
        observations=[b[0] for b in batch];next_observations=[b[3] for b in batch]
        s,z,scores,valid,targets=self.embed(observations,self.codec,labels=True)
        a=torch.as_tensor([b[1] for b in batch],device=self.device)
        reward=torch.as_tensor([b[2] for b in batch],dtype=torch.float32,device=self.device)
        if self.global_return:reward=reward/self.reward_scale
        masks=torch.as_tensor(np.stack([b[4] for b in batch]),device=self.device)
        done=torch.as_tensor([b[5] for b in batch],dtype=torch.float32,device=self.device)
        with torch.no_grad():
            online_next=self.embed(next_observations,self.codec)[0]
            target_next=self.embed(next_observations,self.target_codec)[0]
            a2=self.online(online_next).masked_fill(masks==0,-1e9).argmax(-1,keepdim=True)
            next_q=self.target(target_next)
            if self.global_return:
                discount=torch.as_tensor([b[6] for b in batch],dtype=torch.float32,device=self.device)
                y=reward+(1-done)*discount*next_q.gather(1,a2).squeeze(1)
            else:y=reward+(1-done)*next_q.sigmoid().gather(1,a2).squeeze(1)
        logits=self.online(s).gather(1,a[:,None]).squeeze(1)
        q=logits if self.global_return else logits.sigmoid()
        td_loss=(nn.functional.smooth_l1_loss(q,y) if self.global_return else
                 nn.functional.binary_cross_entropy_with_logits(logits,y))
        prediction_loss,auxiliary=self.forecast_loss(s,z,scores,valid,targets)
        self.opt.zero_grad()
        td_loss.backward(retain_graph=True)
        # This is task-loss gradient, measured BEFORE adding prediction loss.
        codec_rl_norm=gradient_norm(self.codec.parameters()).to(self.device)
        importance_rl_norm=gradient_norm(self.codec.importance.parameters()).to(self.device)
        if self.train_codec:(self.prediction_weight*auxiliary).backward()
        q_norm=gradient_norm(self.online.parameters()).to(self.device)
        codec_norm=gradient_norm(self.codec.parameters()).to(self.device)
        pre=nn.utils.clip_grad_norm_(self.parameters,10.,error_if_nonfinite=True)
        post=gradient_norm(self.parameters).to(self.device)
        self.opt.step();self.updates+=1
        if self.updates%200==0:
            self.target.load_state_dict(self.online.state_dict())
            self.target_codec.load_state_dict(self.codec.state_dict())
        loss=td_loss+self.prediction_weight*auxiliary
        names=('loss','grad_norm_pre','grad_norm_post','clipped','td_abs','q_abs_max',
               'target_abs_max','batch_reward_abs_max','state_rms','td_loss',
               'prediction_mse','codec_rl_grad_norm','codec_grad_norm','q_grad_norm','importance_rl_grad_norm')
        values=torch.stack((loss.detach(),pre.detach(),post.detach(),(pre>10).float(),
            (y-q).detach().abs().mean(),q.detach().abs().max(),y.abs().max(),reward.abs().max(),
            s.detach().square().mean().sqrt(),td_loss.detach(),prediction_loss.detach(),
            codec_rl_norm,codec_norm,q_norm,importance_rl_norm)).cpu().tolist()
        self.last_metrics=dict(zip(names,values));self._episode_metrics.append(self.last_metrics)
        return float(loss.detach())

    def episode_metrics(self):
        rows=self._episode_metrics
        extra={key+'_mean':float(np.mean([r[key] for r in rows])) if rows else None
               for key in ('td_loss','prediction_mse','codec_rl_grad_norm','codec_grad_norm','q_grad_norm','importance_rl_grad_norm')}
        extra['codec_rl_grad_nonzero_fraction']=float(np.mean([r['codec_rl_grad_norm']>0 for r in rows])) if rows else None
        extra['codec_grad_norm_max']=max((r['codec_grad_norm'] for r in rows),default=None)
        extra['request_utility_mean']=float(np.mean(self._episode_utilities)) if self._episode_utilities else None
        self._episode_utilities=[]
        return {**super().episode_metrics(),**extra}
