"""Masked categorical PPO jointly trained with predictive semantic compression."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from edge_msd.realtime_routing.joint_agent import JointRoutingAgent,gradient_norm,terminal_utility


class RoutingActorCritic(nn.Module):
    def __init__(self,state_dim,actions,hidden=64,temperature=.25):
        super().__init__()
        self.input_norm=nn.LayerNorm(state_dim)
        self.trunk=nn.Sequential(nn.Linear(state_dim,hidden),nn.ReLU(),
                                 nn.Linear(hidden,hidden),nn.ReLU())
        self.advantage=nn.Linear(hidden,actions);self.value=nn.Linear(hidden,1)
        self.temperature=temperature

    def forward(self,state):
        hidden=self.trunk(self.input_norm(state))
        logits=self.advantage(hidden)
        return logits/self.temperature,self.value(hidden).squeeze(-1).sigmoid()


class JointPPOAgent(JointRoutingAgent):
    def __init__(self,*args,temperature=.25,clip_ratio=.2,entropy_weight=.001,**kwargs):
        super().__init__(*args,**kwargs)
        self.online=RoutingActorCritic(self.online.input_norm.normalized_shape[0],
                                       self.action_dim,temperature=temperature).to(self.device)
        self.target=deepcopy(self.online).requires_grad_(False)
        self.parameters=list(self.online.parameters())+list(self.codec.parameters())
        self.opt=torch.optim.Adam(self.parameters,lr=kwargs.get('lr',3e-5))
        self.clip_ratio,self.entropy_weight=clip_ratio,entropy_weight

    @torch.no_grad()
    def sample(self,state,mask):
        state=torch.as_tensor(state,device=self.device).unsqueeze(0)
        logits,value=self.online(state)
        logits=logits[0].masked_fill(torch.as_tensor(mask,device=self.device)==0,-1e9)
        probs=logits.softmax(-1).cpu().numpy().astype(np.float64)
        probs/=probs.sum()
        action=int(self.rng.choice(self.action_dim,p=probs))
        return action,float(np.log(probs[action])),float(value.item())

    @torch.no_grad()
    def greedy(self,state,mask):
        logits,_=self.online(torch.as_tensor(state,device=self.device).unsqueeze(0))
        return int(logits[0].masked_fill(torch.as_tensor(mask,device=self.device)==0,-1e9).argmax().item())

    def update(self,batch):
        # obs, action, old log-probability, Monte Carlo return, advantage, mask
        s,z,scores,valid,targets=self.embed([b[0] for b in batch],self.codec,labels=True)
        actions=torch.as_tensor([b[1] for b in batch],device=self.device)
        old=torch.as_tensor([b[2] for b in batch],device=self.device,dtype=torch.float32)
        returns=torch.as_tensor([b[3] for b in batch],device=self.device,dtype=torch.float32)
        advantages=torch.as_tensor([b[4] for b in batch],device=self.device,dtype=torch.float32)
        masks=torch.as_tensor(np.stack([b[5] for b in batch]),device=self.device)
        logits,values=self.online(s)
        distribution=torch.distributions.Categorical(logits=logits.masked_fill(masks==0,-1e9))
        logp=distribution.log_prob(actions);ratio=(logp-old).exp()
        policy_loss=-torch.minimum(ratio*advantages,
            ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*advantages).mean()
        value_loss=nn.functional.mse_loss(values,returns)
        entropy=distribution.entropy().mean()
        task_loss=policy_loss+.5*value_loss-self.entropy_weight*entropy
        prediction_loss,auxiliary=self.forecast_loss(s,z,scores,valid,targets)
        self.opt.zero_grad();task_loss.backward(retain_graph=True)
        codec_rl=gradient_norm(self.codec.parameters()).to(self.device)
        importance_rl=gradient_norm(self.codec.importance.parameters()).to(self.device)
        (self.prediction_weight*auxiliary).backward()
        policy_norm=gradient_norm(self.online.parameters()).to(self.device)
        codec_norm=gradient_norm(self.codec.parameters()).to(self.device)
        pre=nn.utils.clip_grad_norm_(self.parameters,10.,error_if_nonfinite=True)
        post=gradient_norm(self.parameters).to(self.device)
        self.opt.step();self.updates+=1
        loss=task_loss+self.prediction_weight*auxiliary
        names=('loss','grad_norm_pre','grad_norm_post','clipped','td_abs','q_abs_max',
               'target_abs_max','batch_reward_abs_max','state_rms','td_loss','prediction_mse',
               'codec_rl_grad_norm','codec_grad_norm','q_grad_norm','importance_rl_grad_norm',
               'policy_loss','value_loss','entropy','approx_kl','policy_clip_fraction')
        metrics=torch.stack((loss.detach(),pre,post,(pre>10).float(),
            (returns-values).detach().abs().mean(),values.detach().abs().max(),returns.abs().max(),
            returns.abs().max(),s.detach().square().mean().sqrt(),task_loss.detach(),
            prediction_loss.detach(),codec_rl,codec_norm,policy_norm,importance_rl,
            policy_loss.detach(),value_loss.detach(),entropy.detach(),
            ((ratio-1)-(logp-old)).detach().mean(),((ratio-1).abs()>self.clip_ratio).float().mean()))
        self.last_metrics=dict(zip(names,metrics.detach().cpu().tolist()))
        self._episode_metrics.append(self.last_metrics)
        return self.last_metrics['loss']

    def episode_metrics(self):
        rows=self._episode_metrics
        extra={key+'_mean':float(np.mean([r[key] for r in rows])) if rows else None
               for key in ('policy_loss','value_loss','entropy','approx_kl','policy_clip_fraction')}
        return {**super().episode_metrics(),**extra}


def collect_ppo(env,agent):
    env.reset();reward=0.;decisions=0;paths={}
    while not env.terminated:
        state,mask=env.state();obs=env.learning_observation(state)
        request=env._routable_stage().request
        action,logp,value=agent.sample(state,mask)
        paths.setdefault(request.id,[]).append((obs,action,logp,value,mask))
        _,r,_,_,_=env.step(env.node_ids[action])
        reward+=r;decisions+=1;agent.record_reward(r)
    env.update_telemetry();rollout=[]
    for request in env.requests:
        costs=env.request_costs(request)
        utility=terminal_utility(request,env.scenario.tasks[request.task].deadline_ms,
            agent.latency_weight,agent.resource_cost_weight,agent.data_cost_weight,
            costs['resource_normalized'],costs['data_normalized'])
        agent._episode_utilities.append(utility)
        for obs,action,logp,value,mask in paths.get(request.id,[]):
            rollout.append((obs,action,logp,utility,utility-value,mask))
    # gamma=1, lambda=1: each same-request path has its full terminal return.
    adv=np.asarray([r[4] for r in rollout],np.float64)
    adv=(adv-adv.mean())/(adv.std()+1e-8)
    rollout=[(*r[:4],float(a),r[5]) for r,a in zip(rollout,adv)]
    return reward,decisions,rollout
