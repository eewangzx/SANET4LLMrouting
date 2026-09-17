"""Verify the scientific cost/gradient repair against the unchanged estimator."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from edge_msd.config import Settings
from edge_msd.icc_paper import load_scenario
from edge_msd.icc_coupled.codecs import load_codec
from edge_msd.icc_coupled.environment import TelemetrySettings
from edge_msd.realtime_routing.dynamic_routing import DynamicRoutingEnvironment
from edge_msd.realtime_routing.fixed_placement import load_placement
from edge_msd.realtime_routing.joint_agent import JointRoutingAgent,terminal_utility


ROOT=Path(__file__).resolve().parents[1]


def observations():
    torch.set_num_threads(1)
    codec=load_codec(ROOT/'runs/icc_coupled_models/importance.pt')
    scenario=load_scenario(ROOT/'data/icc_paper/scenario_driving_predictive.json',.8)
    placement=load_placement(ROOT/'configs/fixed_placement_driving_predictive.json')
    env=DynamicRoutingEnvironment(scenario,Settings(duration_ms=260,slot_ms=1,seed=47),
        placement,codec,TelemetrySettings(arrival_ms=60,report_ms=100),warmup_ms=200,
        record_training=True,dynamics_profile='driving',end_to_end_forecast=True)
    seen=set();rows=[]
    while not env.terminated:
        state,mask=env.state();stage=env._routable_stage()
        key=stage.service,int(env.now%5)
        if key not in seen:
            seen.add(key);rows.append(env.learning_observation(state))
        env.step(env.greedy_node())
        if len({s for s,p in seen})==8 and len(rows)>=16:break
    assert len({s for s,p in seen})==8
    return codec,rows


def test_recomputed_forecast_features_equal_existing_dag_estimator():
    codec,rows=observations()
    agent=JointRoutingAgent(len(rows[0][0]),10,codec,1.)
    computed=agent.embed(rows,codec)[0]
    expected=torch.as_tensor(np.stack([r[0] for r in rows]))
    torch.testing.assert_close(computed,expected,atol=2e-5,rtol=2e-5)
    # Altering future training targets cannot change the recomputed actor state.
    poisoned=[(state,tuple(None if r is None else (r[0],np.full_like(r[1],10000))
                          for r in records),context) for state,records,context in rows]
    torch.testing.assert_close(agent.embed(poisoned,codec,labels=True)[0],computed)


def test_routing_cost_task_gradients_reach_encoder_selector_and_forecast_head():
    codec,rows=observations()
    agent=JointRoutingAgent(len(rows[0][0]),10,codec,1.)
    state=agent.embed(rows,codec)[0]
    # Only forecast-derived routing features, excluding direct latent inputs.
    state[:,90:110].sum().backward()
    for module in (codec.encoder,codec.importance,codec.predictor):
        gradients=[p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients)>0


def test_terminal_delay_preference_keeps_success_above_every_failure():
    requests=[SimpleNamespace(arrival_ms=0.,completion_ms=t) for t in (10.,60.,99.,101.,160.,250.,None)]
    binary=[terminal_utility(r,100.) for r in requests]
    assert binary==[1.,1.,1.,0.,0.,0.,0.]
    weighted=[terminal_utility(r,100.,.02) for r in requests]
    assert all(0.<=value<=1. for value in weighted)
    assert weighted==sorted(weighted,reverse=True)
    assert min(weighted[:3])>max(weighted[3:])+.98
    assert weighted[-1]==0.
    cheap_success=terminal_utility(requests[0],100.,resource_weight=.2,data_weight=.1,
                                   resource_cost=0.,data_cost=0.)
    costly_success=terminal_utility(requests[0],100.,resource_weight=.2,data_weight=.1,
                                    resource_cost=1.,data_cost=1.)
    cheap_failure=terminal_utility(requests[3],100.,resource_weight=.2,data_weight=.1,
                                   resource_cost=0.,data_cost=0.)
    assert cheap_success>costly_success>cheap_failure


def test_route_cost_accounting_is_action_dependent_and_bounded():
    codec=load_codec(ROOT/'runs/icc_coupled_models/importance.pt')
    scenario=load_scenario(ROOT/'data/icc_paper/scenario_driving_predictive.json',.8)
    placement=load_placement(ROOT/'configs/fixed_placement_driving_predictive.json')
    env=DynamicRoutingEnvironment(scenario,Settings(duration_ms=260,slot_ms=1,seed=47),
        placement,codec,TelemetrySettings(arrival_ms=60,report_ms=100),warmup_ms=200,
        dynamics_profile='driving',include_route_costs=True)
    stage=env._routable_stage();legal=env.legal_nodes(stage)
    costs=[env.incremental_costs(stage,node) for node in legal]
    assert min(env.resource_prices.values())==3.
    assert max(env.resource_prices.values())==10.
    assert len(set(costs))>1
    while not env.terminated:env.step(env.greedy_node())
    for request in env.requests:
        values=env.request_costs(request)
        assert 0.<=values['resource_normalized']<=1.
        assert 0.<=values['data_normalized']<=1.


def test_average_budget_queue_updates_on_the_selected_stage():
    codec=load_codec(ROOT/'runs/icc_coupled_models/importance.pt')
    scenario=load_scenario(ROOT/'data/icc_paper/scenario_driving_predictive.json',.8)
    placement=load_placement(ROOT/'configs/fixed_placement_driving_predictive.json')
    env=DynamicRoutingEnvironment(scenario,Settings(duration_ms=260,slot_ms=1,seed=47),
        placement,codec,TelemetrySettings(arrival_ms=60,report_ms=100),warmup_ms=200,
        dynamics_profile='driving',include_route_costs=True,
        average_resource_budget=.85,dpp_v=1.,data_cost_weight=.05)
    state,mask=env.state();node=env.node_ids[int(np.flatnonzero(mask)[-1])]
    _,reward,_,_,info=env.step(node)
    assert env.cost_virtual_queue==max(0.,info['budget_increment'])
    assert env.budget_cost_sum>0 and env.budget_reference_sum>0
    assert np.isclose(reward,.5*info['raw_sla_reward']-.05*env.last_data_increment)
    assert state[-1]==0.
    if not env.terminated:assert np.isclose(env.state()[0][-1],np.log1p(env.cost_virtual_queue))


def test_chronological_credit_reaches_other_requests_after_same_ms_decisions():
    from scripts.run_joint_routing import chronological_replay
    # A and B are routed at the same physical time; B's success settles only
    # after the third decision advances time. A's replay must see that system
    # reward too, unlike same-request terminal credit or a one-action target.
    tau=100.
    path=[('A',0,0.,'B',np.ones(2),0.,1.,0.),
          ('B',1,0.,'C',np.ones(2),0.,1.,0.),
          ('C',0,1.,'D',np.ones(2),0.,np.exp(-2/tau),2.),
          ('D',1,2.,'end',np.ones(2),1.,np.exp(-3/tau),3.)]
    replay=chronological_replay(path,n_step=1,min_elapsed_ms=2.,return_time_ms=tau)
    assert np.isclose(replay[0][2],1.) and replay[0][3]=='D'
    assert np.isclose(replay[1][2],1.) and replay[1][3]=='D'
    assert np.isclose(replay[0][6],np.exp(-2/tau))
    assert replay[-1][5]==1. and np.isclose(replay[-1][2],2.)


def test_candidate_q_has_no_hardcoded_latency_or_cost_action_prior():
    from edge_msd.realtime_routing.agent import CandidateDuelingQNet
    net=CandidateDuelingQNet(38,3,4,include_costs=True,hidden=16)
    state=torch.randn(2,38)
    with torch.no_grad():
        net.advantage[-1].weight.zero_();net.advantage[-1].bias.zero_()
    q=net(state)
    torch.testing.assert_close(q,q[:,:1].expand_as(q))


def test_budget_ppo_credits_other_requests_and_scales_both_objectives():
    from edge_msd.realtime_routing.ppo_agent import chronological_ppo_rollout
    path=[('A',0,0.,0.,np.ones(2),-2.,0.),
          ('B',1,0.,0.,np.ones(2),3.,2.)]
    rows=chronological_ppo_rollout(path,reward_scale=2.,return_time_ms=100.)
    assert np.isclose(rows[0][3],.5) and np.isclose(rows[1][3],1.5)
