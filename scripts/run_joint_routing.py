"""Joint predictive routing versus explicitly defined routing baselines."""
from collections import Counter, deque
from dataclasses import asdict
import argparse
import hashlib
import json
import math
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
from edge_msd.realtime_routing.agent import log_training_episode
from edge_msd.realtime_routing.dynamic_routing import DynamicRoutingEnvironment,CurrentMeasurementCodec
from edge_msd.realtime_routing.fixed_placement import load_placement
from edge_msd.realtime_routing.joint_agent import JointRoutingAgent,terminal_utility
from edge_msd.realtime_routing.ppo_agent import JointPPOAgent,collect_ppo
from edge_msd.placement import validate_resources


def request_utility(env,agent,request):
    costs=env.request_costs(request)
    return terminal_utility(request,env.scenario.tasks[request.task].deadline_ms,
        agent.latency_weight,agent.resource_cost_weight,agent.data_cost_weight,
        costs['resource_normalized'],costs['data_normalized'])


def mean_objective_utility(env,latency_weight,resource_weight,data_weight):
    if env.average_resource_budget:
        arrivals=len(env.requests)
        return (env.sla_counts()[0]/arrivals-data_weight*sum(
            env.request_costs(r)['data_normalized'] for r in env.requests)/arrivals) if arrivals else None
    values=[]
    for request in env.requests:
        costs=env.request_costs(request)
        values.append(terminal_utility(request,env.scenario.tasks[request.task].deadline_ms,
            latency_weight,resource_weight,data_weight,
            costs['resource_normalized'],costs['data_normalized']))
    return float(np.mean(values)) if values else None


def collect(env,agent=None,epsilon=0.,buffer=None,n_step=3,policy='greedy'):
    env.reset();total=0.;decisions=0;trajectories={};global_path=[]
    policy_rng=np.random.default_rng(np.random.SeedSequence([env.settings.seed,901]))
    while not env.terminated:
        state,mask=env.state()
        obs=env.learning_observation(state) if buffer is not None else None
        request=env._routable_stage().request
        if agent is None:
            if policy=='random':
                action=int(policy_rng.choice(np.flatnonzero(mask)))
                node=env.node_ids[action]
            else:
                node=env.shortest_queue_node() if policy=='shortest_queue' else env.greedy_node()
                action=env.node_ids.index(node)
        else:
            action=agent.act(state,mask,epsilon) if buffer is not None else agent.greedy(state,mask)
            node=env.node_ids[action]
        _,reward,term,_,info=env.step(node)
        total+=reward;decisions+=1
        if buffer is not None:
            agent.record_reward(reward)
            if agent.global_return:
                if term:nxt,m2=obs,mask
                else:
                    s2,m2=env.state();nxt=env.learning_observation(s2)
                discount=math.exp(-info['elapsed_ms']/agent.return_time_ms)
                global_path.append((obs,action,reward,nxt,m2,float(term),discount))
            else:trajectories.setdefault(request.id,[]).append((obs,action,mask))
    env.update_telemetry()
    if buffer is not None:
        if agent.global_return:
            for i,row in enumerate(global_path):
                reward=0.;discount=1.;end=row
                for end in global_path[i:i+n_step]:
                    reward+=discount*end[2];discount*=end[6]
                    if end[5]:break
                buffer.append((row[0],row[1],reward,end[3],end[4],end[5],discount))
            return total,decisions
        for request in env.requests:
            utility=request_utility(env,agent,request)
            agent._episode_utilities.append(utility)
            path=trajectories.get(request.id,[])
            for i,(obs,action,mask) in enumerate(path):
                j=i+n_step
                if j<len(path):
                    nxt,_,m2=path[j]
                    buffer.append((obs,action,0.,nxt,m2,0.))
                else:
                    buffer.append((obs,action,utility,obs,mask,1.))
    return total,decisions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bench',choices=['proposed','decoded_separate','latency_only','greedy','random','shortest_queue'],required=True,
                   help='latency_only is cost-unaware myopic routing; greedy is its legacy alias')
    p.add_argument('--algorithm',choices=['dqn','ppo'],default='dqn')
    p.add_argument('--ppo-epochs',type=int,default=4)
    p.add_argument('--ppo-batch-size',type=int,default=256)
    p.add_argument('--ppo-temperature',type=float,default=.25)
    p.add_argument('--ppo-warm-start',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--placement',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('data/icc_paper/scenario_2026.json'))
    p.add_argument('--codec-init',type=Path,default=Path('runs/icc_coupled_models/importance.pt'))
    p.add_argument('--joint-init',type=Path)
    p.add_argument('--fresh-q',action='store_true')
    p.add_argument('--predictive-cost-prior',action='store_true')
    p.add_argument('--route-residual-bound',type=float,default=.25,
                   help='action-logit residual bound; zero allows an unrestricted learned correction')
    p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--task-prediction',action='store_true')
    p.add_argument('--end-to-end-forecast',action='store_true')
    p.add_argument('--latency-weight',type=float,default=0.)
    p.add_argument('--resource-cost-weight',type=float,default=0.)
    p.add_argument('--data-cost-weight',type=float,default=0.)
    p.add_argument('--average-resource-budget',type=float,default=0.)
    p.add_argument('--dpp-v',type=float,default=1.)
    p.add_argument('--return-time-ms',type=float,default=100.)
    p.add_argument('--selection-metric',choices=['mean','p95'],default='mean')
    p.add_argument('--dynamics-profile',choices=['icc','driving','azure'],default='icc')
    p.add_argument('--azure-trace',type=Path)
    p.add_argument('--azure-case',choices=['all','busy_transitions'],default='busy_transitions')
    p.add_argument('--train-seed-base',type=int,default=52000)
    p.add_argument('--load',type=float,default=.8)
    p.add_argument('--arrival-ms',type=int,default=120)
    p.add_argument('--warmup-ms',type=int,default=200)
    p.add_argument('--drain-ms',type=float,default=150.)
    p.add_argument('--report-ms',type=int,default=100)
    p.add_argument('--report-bps',type=float,default=64000.)
    p.add_argument('--episodes',type=int,default=16)
    p.add_argument('--updates-per-episode',type=int,default=1000)
    p.add_argument('--n-step',type=int,default=3)
    p.add_argument('--prediction-weight',type=float,default=1.)
    p.add_argument('--test-seeds',default='55000,55001,55002')
    p.add_argument('--validation-seeds',default='')
    p.add_argument('--validate-every',type=int,default=8)
    p.add_argument('--exclude-initial-selection',action='store_true')
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--epsilon-start',type=float,default=1.)
    p.add_argument('--deadline-aware-prior',action='store_true',
                   help='requests predicted to exceed the heuristic remaining time budget are steered to the most loaded node; requires --predictive-cost-prior')
    p.add_argument('--buffer-cap',type=int,default=0,help='keep only the newest transitions (0: unbounded replay)')
    args=p.parse_args()
    learned=args.bench in ('proposed','decoded_separate')
    separate=args.bench=='decoded_separate'
    if separate and (args.algorithm!='dqn' or args.joint_init or args.end_to_end_forecast):
        p.error('decoded_separate uses DQN and an independently pretrained frozen --codec-init, without --joint-init or --end-to-end-forecast')
    if args.report_ms <= 0 or args.report_bps <= 0 or args.warmup_ms < 0:
        p.error("report period/bandwidth must be positive and warmup nonnegative")
    if args.dynamics_profile=='azure' and args.azure_trace is None:
        p.error('Azure dynamics require --azure-trace')
    if args.ppo_epochs<1 or args.ppo_batch_size<1 or args.ppo_temperature<=0:
        p.error('PPO epochs, batch size and temperature must be positive')
    if args.ppo_warm_start and (args.algorithm!='ppo' or args.joint_init is None or args.fresh_q):
        p.error('PPO warm start requires --algorithm ppo --joint-init and no --fresh-q')
    objective_weights=(args.latency_weight,args.resource_cost_weight,args.data_cost_weight)
    if any(not math.isfinite(value) or value<0 for value in objective_weights):
        p.error('objective weights must be finite and nonnegative')
    if sum(objective_weights)>=1:
        p.error('sum of non-SLA weights must be below one so every SLA success beats every failure')
    if not math.isfinite(args.average_resource_budget) or not 0<=args.average_resource_budget<=1:
        p.error('average resource budget must be in [0,1]; zero disables')
    if args.dpp_v<=0 or args.return_time_ms<=0 or args.lr<=0 or not math.isfinite(args.dpp_v+args.return_time_ms+args.lr):
        p.error('DPP V, physical return time and learning rate must be positive and finite')
    if args.average_resource_budget and (separate or args.algorithm!='dqn' or args.resource_cost_weight):
        p.error('average-budget training uses joint DQN and queue-weighted resource cost, without a fixed resource weight')
    if not math.isfinite(args.route_residual_bound) or args.route_residual_bound<0:
        p.error('route residual bound must be finite and nonnegative')
    if args.deadline_aware_prior and (not args.predictive_cost_prior or separate):
        p.error('--deadline-aware-prior requires --predictive-cost-prior on the joint proposed method')
    if args.exclude_initial_selection and not args.validation_seeds:
        p.error('--exclude-initial-selection requires validation seeds')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    scenario=load_scenario(args.dataset,args.load)
    placement=load_placement(args.placement)
    validate_resources(scenario,placement.counts)
    settings=Settings(duration_ms=args.warmup_ms+args.arrival_ms,seed=args.seed,slot_ms=1,
                      ec_admission_window_ms=1,record_trace=True)
    telemetry=TelemetrySettings(arrival_ms=args.arrival_ms,control_ms=1,
        report_ms=args.report_ms,report_bps=args.report_bps,dynamic=True)
    codec=load_codec(args.codec_init).eval().requires_grad_(not separate) if learned else CurrentMeasurementCodec()
    report_state_dim=codec.horizon*codec.targets if separate else codec.latent_dim
    def world(seed,training=False,split='test'):
        return DynamicRoutingEnvironment(scenario,settings,placement,codec,telemetry,
                 seed=seed,drain_ms=args.drain_ms,record_training=training,warmup_ms=args.warmup_ms,
                 dynamics_profile=args.dynamics_profile,azure_path=args.azure_trace,
                 azure_split='train' if training else split,azure_case=args.azure_case,
                 end_to_end_forecast=args.end_to_end_forecast,
                 state_representation='forecast' if separate else 'latent',
                 include_route_costs=bool(args.resource_cost_weight or args.data_cost_weight or args.average_resource_budget),
                 average_resource_budget=args.average_resource_budget,dpp_v=args.dpp_v,
                 data_cost_weight=args.data_cost_weight,deadline_aware=args.deadline_aware_prior)
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    def write(name,value):
        tmp=root/(name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(root/name)
    write('status.json',{'status':'running','stage':'setup'})
    example=world(52000,split='train');state,mask=example.state()
    train_seeds=(list(range(args.train_seed_base,args.train_seed_base+args.episodes))
                 if learned else [])
    test_seeds=[int(s) for s in args.test_seeds.split(',')]
    validation_seeds=[int(s) for s in args.validation_seeds.split(',') if s]
    if args.validate_every<=0:
        p.error('validation interval must be positive')
    if set(validation_seeds)&(set(test_seeds)|set(train_seeds)):
        p.error('validation seeds must be separate from train and test seeds')
    if set(test_seeds)&set(train_seeds):
        p.error('test seeds must be separate from training seeds')
    reward_scale=1.
    stage_demand=Counter()
    for user in scenario.users:
        for task,rate in user.rates_per_second.items():
            for service in scenario.tasks[task].order:stage_demand[service]+=rate/1000.
    core_capacity={service:sum(count for (s,_),count in placement.core.items() if s==service)
        / (math.ceil(scenario.services[service].mean_processing_ms/settings.slot_ms)*settings.slot_ms)
        for service in scenario.core}
    manifest={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
       'settings':asdict(settings),'telemetry':asdict(telemetry),'placement_hash':placement.placement_hash,
       'codec':codec.config(),'state_dim':len(state),'hidden':64,'gamma':1.,'lr':args.lr,
       'algorithm':args.algorithm,
       'reward_scale':reward_scale,'n_step':args.n_step,
       'trajectory':'successive node choices of the SAME request, through its complete DAG',
       'critic':'eventual request SLA-success probability; bounded sigmoid value',
       'objective':'maximize mean request SLA success; equivalent to mean terminal +1/-1',
       'routing_cost_model':{
           'resource':'sum_s lambda_js * p_node; lambda_js=service work_mb',
           'prices':example.resource_prices,
           'price_source':'Wang et al. JSAC 2023 range [3,10]; CPU/GPU capacity-rank assignment',
           'data':'sum of predecessor payload MB times nominal path hops (MB-hop)',
           'resource_weight':args.resource_cost_weight,
           'data_weight':args.data_cost_weight},
       'train_seeds':train_seeds,'test_seeds':test_seeds,'validation_seeds':validation_seeds,
       'model_selection':'validation SLA; mean completed latency breaks ties; initial checkpoint eligible' if validation_seeds else 'last episode',
       'training':f'joint task TD loss + {args.prediction_weight} * predictive MSE/budget loss' if args.bench=='proposed' else 'none',
       'state_fields':['delivered_latent','current_service_backlog','unfinished_DAG_queue_summary',
                       'received_forecast_stage_latency','incremental_resource_cost',
                       'incremental_data_cost','report_age','missing_report','service','task',
                       'request_gateway','remaining_deadline','remaining_episode_time'],
       'actor_access':'serialized delivered messages + known dispatch/completion ledger only',
       'training_access':'local recorded history/future labels; replay re-encodes only delivered histories',
       'update_schedule':'one joint optimizer; parameters constant during each collected trajectory',
       'actions':'node routing only; service instance counts fixed',
       'state_acquisition':'initial state is delivered during common telemetry warmup; new requests reuse delivered node caches during periodic refresh, with no latest-round waiting',
       'latency_accounting':'arrival through acquisition waiting, routing, transfer, queues and execution; original deadlines unchanged',
       'data_transfer':'starts after node selection; node input waiting does not reserve compute instances',
       'remaining_DAG_pressure':'known waiting stages by input locality plus mean assigned reservations, normalized by instance count; light work uses delivered next-50ms availability; log1p scaling',
       'mean_stage_arrivals_per_ms':dict(stage_demand),
       'core_discrete_execution_upper_capacity_per_ms':core_capacity,
       'load_regime':'finite Poisson arrival burst; workload doubled versus load=0.4; average offered demand can exceed service capacity',
       'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (
           Path('scripts/run_joint_routing.py'),Path('edge_msd/realtime_routing/joint_agent.py'),
           Path('edge_msd/realtime_routing/ppo_agent.py'),
           Path('edge_msd/realtime_routing/forecast_costs.py'),
           Path('edge_msd/realtime_routing/dynamic_routing.py'))}}
    if args.bench in ('latency_only','greedy'):
        manifest['method_name']='Latency-Only Myopic (LOM)'
        manifest['baseline_definition']={
            'decision':'minimum estimated current stage completion time, including transfer and per-instance backlog',
            'information':'latest delivered raw current measurements and the dispatch/completion ledger',
            'prediction':False,'resource_cost_considered':False,'data_cost_considered':False,
            'average_budget_considered':False,'oracle':False}
    if not (args.resource_cost_weight or args.data_cost_weight or args.average_resource_budget):
        manifest['state_fields']=[name for name in manifest['state_fields']
                                  if name not in ('incremental_resource_cost','incremental_data_cost')]
    if args.azure_trace:
        manifest['azure_asset_sha256']=hashlib.sha256(args.azure_trace.read_bytes()).hexdigest()
        manifest['load_regime']='same ICC driving Poisson arrivals; Azure background workload dynamics'
        manifest['azure_time_mapping']='5 real seconds per 5 simulation milliseconds; chronological 4/1/2 day split'
    if args.latency_weight:
        manifest.update(critic='bounded expected full-request SLA/delay utility',
            objective=f'maximize (terminal SLA success + {args.latency_weight} * quadratic delay score)/(1+weight)',
            latency_weight=args.latency_weight,
            delay_score='1-min(completed latency/(3*deadline),1)^2; pending terminal target zero')
    if args.resource_cost_weight or args.data_cost_weight:
        manifest.update(critic='bounded expected full-request SLA and route-cost utility',
            objective=('maximize (SLA success + latency_weight*delay_score + '
                       'resource_weight*(1-normalized resource cost) + '
                       'data_weight*(1-normalized MB-hop))/(1+sum weights)'),
            route_cost_priority=('sum of non-SLA weights < 1, so every completed SLA '
                                 'success has higher utility than every completed failure'),
            model_selection=('validation objective utility; SLA, normalized resource cost, '
                             'normalized MB-hop and latency break ties; initial checkpoint eligible'))
    if args.selection_metric=='p95':
        if args.resource_cost_weight or args.data_cost_weight:
            manifest['model_selection']=('validation objective utility; SLA, normalized resource cost, '
                'normalized MB-hop, completed P95 and mean latency break ties; initial checkpoint eligible')
        else:
            manifest['model_selection']='validation SLA; completed P95 then mean latency break ties; initial checkpoint eligible'
    if separate:
        manifest.update(training='RL only; independently pretrained prediction codec frozen',
            state_representation='full age-aligned decoded forecast from delivered Z',
            forecast_state_shape=[len(example.node_ids),codec.horizon,codec.targets],
            update_schedule='prediction pretraining first; freeze encoder/importance/predictor; then train Q only',
            training_access='RL receives only actor state; no source histories or future labels',
            state_fields=['delivered_decoded_forecast',*manifest['state_fields'][1:]])
    if args.exclude_initial_selection:
        manifest['model_selection']=manifest['model_selection'].replace(
            'initial checkpoint eligible','only checkpoints after RL updates eligible')
    if learned:
        device='cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(args.seed)
        agent_cls=JointPPOAgent if args.algorithm=='ppo' else JointRoutingAgent
        extra={'temperature':args.ppo_temperature} if args.algorithm=='ppo' else {'train_codec':not separate}
        agent=agent_cls(len(state),len(example.node_ids),codec,reward_scale,
                    lr=args.lr,prediction_weight=args.prediction_weight,device=device,**extra)
        agent.latency_weight=args.latency_weight
        agent.resource_cost_weight=args.resource_cost_weight
        agent.data_cost_weight=args.data_cost_weight
        agent.route_cost_state=bool(args.resource_cost_weight or args.data_cost_weight)
        agent.route_cost_state=agent.route_cost_state or bool(args.average_resource_budget)
        agent.global_return=bool(args.average_resource_budget)
        agent.budget_state=bool(args.average_resource_budget)
        agent.return_time_ms=args.return_time_ms
        matching=False
        if args.joint_init:
            checkpoint=torch.load(args.joint_init,map_location=device,weights_only=True)
            matching=checkpoint.get('manifest',{}).get('algorithm','dqn')==args.algorithm
            if not args.fresh_q and (matching or args.ppo_warm_start):
                agent.online.load_state_dict(checkpoint['online'])
                agent.target.load_state_dict(checkpoint['target'])
            codec.load_state_dict(checkpoint['codec'])
            agent.target_codec.load_state_dict(checkpoint['target_codec'])
            manifest['joint_init_sha256']=hashlib.sha256(args.joint_init.read_bytes()).hexdigest()
            manifest['joint_init_note']='prior joint codec; fresh decision network' if args.fresh_q or not matching else 'continue prior joint model; fresh optimizer'
            if args.ppo_warm_start:
                manifest['joint_init_note']='joint codec and actor warm-started from trained DQN; same initial masked routing argmax'
        residual=(f'{args.route_residual_bound}*tanh(action residual)'
                  if args.route_residual_bound else 'unrestricted action residual')
        if args.predictive_cost_prior:
            for net in (agent.online,agent.target):
                net.route_cost_start=len(example.node_ids)*(report_state_dim+2)
                net.route_residual_bound=args.route_residual_bound or None
            if args.deadline_aware_prior:
                layout=tuple(len(state)+index for index in example.deadline_layout())
                agent.deadline_layout=layout
                for net in (agent.online,agent.target):net.route_cost_start=layout[0]
                manifest['deadline_aware_prior']=('predicted slack = remaining deadline - heuristic unfinished processing work - '
                    'fastest predicted stage latency; if negative the prior steers to the most loaded legal node')
                manifest['state_fields']=[*manifest['state_fields'][:-2],'deadline_prior_cost','deadline_budget','doomed',*manifest['state_fields'][-2:]]
            if separate or args.fresh_q or (args.algorithm=='ppo' and not matching and not args.ppo_warm_start):
                for head in (agent.online.value,agent.online.advantage):
                    torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
                agent.target.load_state_dict(agent.online.state_dict())
            prior_description='deadline-aware routing prior' if args.deadline_aware_prior else 'predicted stage latency / 50ms'
            manifest['q_parameterization']=f'sigmoid of state value + {residual} - 2 * {prior_description}'
        if args.algorithm=='ppo':
            manifest.pop('q_parameterization',None)
            manifest.update(critic='sigmoid state value with Monte Carlo terminal SLA targets and MSE',
                n_step=None,
                training='clipped PPO surrogate + 0.5 value MSE - 0.001 entropy + weighted forecast/budget',
                policy_parameterization=f'masked softmax of ({residual} - 2*{prior_description})/temperature' if args.predictive_cost_prior else 'masked learned action logits / temperature',
                ppo_clip_ratio=.2,ppo_temperature=args.ppo_temperature,ppo_epochs=args.ppo_epochs,
                ppo_batch_size=args.ppo_batch_size,trajectory_return='same-request terminal success, gamma=1 lambda=1',
                replay='current episode only; fixed old log-probabilities and advantages')
            if any(objective_weights):
                manifest['critic']='sigmoid expected bounded SLA/delay/route-cost utility with Monte Carlo terminal targets and MSE'
                manifest['trajectory_return']='same-request terminal bounded objective utility, gamma=1 lambda=1'
        if args.task_prediction:
            active={service for user in scenario.users for task,rate in user.rates_per_second.items()
                    if rate>0 for service in scenario.tasks[task].order}
            features=np.zeros((len(example.node_ids),codec.targets),np.float32)
            for index,node in enumerate(example.node_ids):
                for name,j in example.resource.service_index.items():
                    features[index,j]=float(name in active and bool(placement.counts.get((name,node))))
                features[index,9:9+len(example.resource.neighbors[node])]=1.
            agent.prediction_feature_mask=torch.as_tensor(features,device=device)
            manifest['prediction_supervision']='deployed active-task light rates and real outgoing links; forecast times from received age minus 10ms through remaining deadline'
        agent.rng=np.random.default_rng(args.seed)
        initial={k:v.detach().cpu().clone() for k,v in codec.state_dict().items()}
        manifest.update(device=device,q_parameters=sum(p.numel() for p in agent.online.parameters()),
                        codec_parameters=sum(p.numel() for p in codec.parameters()),
                        codec_init_sha256=hashlib.sha256(args.codec_init.read_bytes()).hexdigest())
    if args.average_resource_budget:
        manifest.update(
            trajectory='chronological decisions across all requests; system-wide settlement rewards',
            critic='unbounded global Double DQN return, smooth-L1 TD loss',
            gamma='exp(-elapsed physical milliseconds / return_time_ms)',
            objective='maximize SLA success rate minus weighted MB-hop, subject to average resource budget',
            average_resource_budget=args.average_resource_budget,dpp_v=args.dpp_v,
            return_time_ms=args.return_time_ms,
            cost_queue='Q_next=max(0,Q+normalized stage charge-budget_fraction*maximum legal stage charge)',
            reward='0.5*system SLA +/-1 settlements - (Q_before/V)*budget_increment - data_weight*incremental normalized MB-hop',
            budget_accounting='credit only for dispatched stages; full DAG credits sum to one request budget; no credit for un-dispatched pending work',
            model_selection='validation budget feasibility first, then SLA-minus-data objective; least violation if no feasible checkpoint; initial checkpoint excluded',
            q_parameterization=(f'state value + {residual} - 2 * {prior_description}'
                if learned and args.predictive_cost_prior else
                ('state value + action advantage' if learned else 'no Q network; fixed heuristic policy')),
            state_fields=[*manifest['state_fields'],'cost_virtual_queue_divided_by_V'])
        manifest.pop('route_cost_priority',None)
    write('manifest.json',manifest)
    print('START',args.bench,json.dumps(manifest),flush=True)
    if learned:
        buffer=[]
        best_score=None;best_episode=None;best_validation=None;validation_rows=[]
        def checkpoint(path,episode):
            torch.save({'online':agent.online.state_dict(),'target':agent.target.state_dict(),
                        'codec':codec.state_dict(),'target_codec':agent.target_codec.state_dict(),
                        'optimizer':agent.opt.state_dict(),'manifest':manifest,
                        'episode':episode},path)
        def validate(episode,eligible=True):
            nonlocal best_score,best_episode,best_validation
            values=[]
            for seed in validation_seeds:
                env=world(seed,split='validation');collect(env,agent)
                d=env.diagnostics()
                d['mean_objective_utility']=mean_objective_utility(env,args.latency_weight,
                    args.resource_cost_weight,args.data_cost_weight)
                write(f'validation_{episode:02d}_{seed}.json',d)
                values.append(d)
            arrivals=sum(d['arrivals'] for d in values)
            sla=sum(d['ontime'] for d in values)/arrivals
            delays=[r['completion_ms']-r['arrival_ms'] for d in values for r in d['requests']
                    if r['completion_ms'] is not None]
            mean=float(np.mean(delays)) if delays else float('inf')
            p95=float(np.percentile(delays,95)) if delays else float('inf')
            utility=sum(d['mean_objective_utility']*d['arrivals'] for d in values)/arrivals
            resource=sum(d['mean_normalized_resource_cost']*d['arrivals'] for d in values)/arrivals
            data=sum(d['mean_normalized_data_cost']*d['arrivals'] for d in values)/arrivals
            budget_ratio=None;budget_feasible=None
            if args.average_resource_budget:
                reference=sum(d['budget_reference_sum'] for d in values)
                budget_ratio=sum(d['budget_cost_sum'] for d in values)/reference
                violation=max(0.,budget_ratio-args.average_resource_budget)
                budget_feasible=violation<=1e-6
                score=(True,utility,sla,-budget_ratio,-mean) if budget_feasible else (
                    False,-violation,utility,sla,-mean)
            elif args.resource_cost_weight or args.data_cost_weight:
                score=(utility,sla,-resource,-data,-p95,-mean) if args.selection_metric=='p95' else (
                       utility,sla,-resource,-data,-mean)
            else:
                score=(sla,-p95,-mean) if args.selection_metric=='p95' else (sla,-mean)
            if eligible and (best_score is None or score>best_score):
                best_score=score;best_episode=episode
                best_validation={'validation_sla':sla,'validation_objective_utility':utility,
                    'validation_mean_normalized_resource_cost':resource,
                    'validation_mean_normalized_data_cost':data,
                    'validation_budget_cost_ratio':budget_ratio,
                    'validation_budget_feasible':budget_feasible,
                    'validation_mean_completed_latency_ms':mean,
                    'validation_p95_completed_latency_ms':p95}
                checkpoint(root/'best.pt',episode)
            validation_rows.append({'episode':episode,'arrivals':arrivals,'sla':sla,
                                    'mean_objective_utility':utility,
                                    'mean_normalized_resource_cost':resource,
                                    'mean_normalized_data_cost':data,
                                    'budget_cost_ratio':budget_ratio,
                                    'budget_feasible':budget_feasible,
                                    'mean_completed_latency_ms':mean,
                                    'p95_completed_latency_ms':p95,
                                    'eligible_for_selection':eligible,
                                    'selected_episode':best_episode})
            write('validation.json',validation_rows)
            print('VALIDATION',episode,'sla',sla,'best_episode',best_episode,flush=True)
        if validation_seeds:validate(0,not args.exclude_initial_selection)
        for episode,seed in enumerate(train_seeds,1):
            epsilon=max(.05,args.epsilon_start*(1-(episode-1)/(.6*args.episodes)))
            env=world(seed,training=True)
            if args.algorithm=='ppo':
                epsilon=0.;reward,decisions,rollout=collect_ppo(env,agent)
            else:
                reward,decisions=collect(env,agent,epsilon,buffer,args.n_step)
                if args.buffer_cap and len(buffer)>args.buffer_cap:del buffer[:len(buffer)-args.buffer_cap]
            diagnostics=env.diagnostics();write(f'train_{episode:02d}.json',diagnostics)
            if args.algorithm=='ppo':
                for _ in range(args.ppo_epochs):
                    random.shuffle(rollout)
                    for start in range(0,len(rollout),args.ppo_batch_size):
                        agent.update(rollout[start:start+args.ppo_batch_size])
            else:
                for _ in range(args.updates_per_episode):agent.update(random.sample(buffer,64))
            log_training_episode(agent,root/'training.csv',episode,reward,decisions,diagnostics['sla'],epsilon)
            print('TRAIN',episode,'sla',diagnostics['sla'],
                  'rollout' if args.algorithm=='ppo' else 'replay',
                  len(rollout) if args.algorithm=='ppo' else len(buffer),flush=True)
            write('status.json',{'status':'running','stage':'training','episode':episode,
                  'updates':agent.updates,'elapsed_seconds':time.perf_counter()-started})
            if validation_seeds and (episode%args.validate_every==0 or episode==args.episodes):
                validate(episode)
        delta={prefix:float(torch.sqrt(sum((v.detach().cpu()-initial[k]).square().sum()
                    for k,v in codec.state_dict().items() if k.startswith(prefix))))
               for prefix in ('encoder.','importance.','predictor.')}
        write('joint_training.json',{'parameter_delta_l2':delta,'loss':manifest['training'],
             'actor_reads_source_history':False,'future_labels_used_by_actor':False})
        if separate and any(value!=0. for value in delta.values()):
            raise RuntimeError('Frozen separate predictor changed during RL training')
        checkpoint(root/'last.pt',args.episodes)
        if validation_seeds:
            selected=torch.load(root/'best.pt',map_location=agent.device,weights_only=True)
            agent.online.load_state_dict(selected['online'])
            agent.target.load_state_dict(selected['target'])
            codec.load_state_dict(selected['codec'])
            agent.target_codec.load_state_dict(selected['target_codec'])
            write('model_selection.json',{'episode':best_episode,**best_validation,
                  'checkpoint':'best.pt','test_access_used_for_selection':False})
    evaluations=[]
    for seed in test_seeds:
        env=world(seed)
        reward,decisions=collect(env,agent if learned else None,policy=args.bench)
        diagnostics=env.diagnostics()
        diagnostics['mean_objective_utility']=mean_objective_utility(env,args.latency_weight,
            args.resource_cost_weight,args.data_cost_weight)
        write(f'test_{seed}.json',diagnostics)
        evaluations.append({'seed':seed,'reward':reward,'decisions':decisions,
                          **{k:v for k,v in diagnostics.items() if k!='requests'}})
        write('evaluation.json',evaluations)
        print('TEST',args.bench,seed,'sla',diagnostics['sla'],flush=True)
        write('status.json',{'status':'running','stage':'evaluation','seed':seed,
                            'elapsed_seconds':time.perf_counter()-started})
    write('result.json',{'bench':args.bench,'evaluation':evaluations})
    write('status.json',{'status':'complete','elapsed_seconds':time.perf_counter()-started})


if __name__=='__main__':main()
