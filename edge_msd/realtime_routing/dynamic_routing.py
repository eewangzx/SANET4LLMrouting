"""Fixed-deployment routing using the project's existing dynamic ICC physics.

Reuses ResourceTrace, integrated time-varying link rates, delayed serialized
telemetry and existing prediction codecs. Actions change only the service node.
"""
from __future__ import annotations

from collections import Counter
from math import ceil
import networkx as nx
import numpy as np

from edge_msd.icc_coupled.dynamics import ResourceTrace
from edge_msd.icc_coupled.environment import MultipliersNetwork, TelemetrySettings
from edge_msd.model_routing.telemetry import NarrowbandChannel, pack_report, unpack_report
from edge_msd.realtime_routing.environment import RoutingEnvironment
from edge_msd.models import Job


def capacity_rank_resource_prices(scenario, low=3., high=10.):
    """Return deterministic per-work-unit prices in Wang et al.'s [3, 10] range.

    CPU and GPU capacities are normalized separately before averaging, then the
    capacity score is min-max mapped to the price range.  This is an explicit
    premium-compute experiment assumption, not a measured bill.
    """
    nodes=sorted(scenario.nodes)
    compute=np.asarray([scenario.nodes[node].resources[:2] for node in nodes],np.float64)
    score=(compute/np.maximum(compute.max(0),1e-12)).mean(1)
    span=float(score.max()-score.min())
    normalized=(score-score.min())/span if span>1e-12 else np.zeros_like(score)
    return {node:float(low+(high-low)*value) for node,value in zip(nodes,normalized)}


class CurrentMeasurementCodec:
    """Traditional controller reports current float32 measurements verbatim."""
    name = 'current'
    latent_dim = 16
    payload_bytes = 64

    def eval(self): return self
    def requires_grad_(self, enabled): return self
    def encode_wire(self, history): return np.asarray(history[-1],np.float32).copy()
    def decode_wire(self, payload): return np.asarray(payload,np.float32).copy()
    def config(self): return {'kind':'raw_current','features':16,'prediction':False}


class DispatchedNetwork(MultipliersNetwork):
    """The selected destination is known only at dispatch, then data can move."""
    def data_ready(self, stage, target):
        key=stage.key,target
        if key not in self.ready_cache:
            req=stage.request
            parents=self.scenario.tasks[req.task].dependencies[stage.service]
            if not parents:
                amount=self.scenario.services[stage.service].input_mb
                if amount is None:amount=self.scenario.tasks[req.task].input_mb
                sources=[(req.gateway,req.uplink_ready_ms,amount)]
            else:
                sources=[(*req.finished[p],self.scenario.services[p].output_mb) for p in parents]
            self.ready_cache[key]=max(
                max(self.now,ready)+self._delay_at(source,target,amount,max(self.now,ready))
                for source,ready,amount in sources)
        return self.ready_cache[key]


class DynamicRoutingEnvironment(RoutingEnvironment):
    def __init__(self, scenario, settings, placement, codec, telemetry=None,
                 seed=None, drain_ms=150., requests=None, record_training=False,
                 warmup_ms=0., dynamics_profile='icc',azure_path=None,azure_split='train',
                 azure_case='busy_transitions',end_to_end_forecast=False,
                 state_representation='latent',include_route_costs=False,
                 average_resource_budget=0.,dpp_v=1.,data_cost_weight=0.):
        if state_representation not in ('latent','forecast'):
            raise ValueError('State representation must be latent or forecast')
        self.state_representation=state_representation
        self.include_route_costs=bool(include_route_costs)
        self.average_resource_budget=float(average_resource_budget)
        self.dpp_v=float(dpp_v)
        self.dpp_data_weight=float(data_cost_weight)
        self.codec = codec.eval()
        self.resource_prices=capacity_rank_resource_prices(scenario)
        self._cost_references={}
        self._cost_paths={}
        self.record_training = record_training
        self.end_to_end_forecast=end_to_end_forecast
        self.training_paths={}
        self.warmup_ms=float(warmup_ms)
        self.dynamics_profile=dynamics_profile
        self.azure_path,self.azure_split=azure_path,azure_split
        self.azure_case=azure_case
        self.telemetry = telemetry or TelemetrySettings(arrival_ms=settings.duration_ms)
        self.rate_tables = {}
        self.positions = {}
        self.pools = {}
        graph=nx.Graph();graph.add_nodes_from(scenario.nodes)
        graph.add_weighted_edges_from((a,b,8./rate) for a,b,rate,_ in scenario.links)
        self.components={node:index for index,nodes in enumerate(nx.connected_components(graph))
                         for node in nodes}
        distances=dict(nx.all_pairs_dijkstra_path_length(graph))
        self.nearest_hosts={
            (service,source):min(
                (node for (s,node),count in placement.counts.items() if s==service and count),
                key=lambda node:(distances[source].get(node,float('inf')),node))
            for service in scenario.services for source in scenario.nodes}
        super().__init__(scenario,settings,placement,seed,drain_ms,requests)

    def reset(self, seed=None):
        if seed is not None:
            from dataclasses import replace
            self.settings=replace(self.settings,seed=int(seed))
        self.resource = ResourceTrace(self.scenario, self.horizon_ms, self.settings.seed,
                                      self.telemetry.dynamic, profile=self.dynamics_profile,
                                      azure_path=self.azure_path,azure_split=self.azure_split,
                                      azure_case=self.azure_case)
        self.network=DispatchedNetwork(self.scenario,self.resource,physical=True)
        self.observed_network=DispatchedNetwork(self.scenario,self.resource)
        self.channel=NarrowbandChannel(self.telemetry.report_bps,self.telemetry.propagation_ms)
        self.received=[None]*len(self.resource.nodes)
        self.report_rounds={}
        self.completed_report_tick=-1
        self.current_report_tick=-1
        self.report_completions=[]
        self.request_acquisition={}
        self.report_start_counters=None
        self.local_training_windows={}
        self.received_forecasts=[None]*len(self.resource.nodes)
        self.forecast=np.ones((len(self.resource.nodes),self.resource.horizon,14),np.float32)
        self.last_telemetry=-1.
        self.next_report=0.
        self.age_samples=[];self.missing_samples=[]
        self.rate_tables={};self.positions={};self.pools={}
        self.node_queues={}
        self.request_resource_cost={};self.request_data_cost={}
        self.cost_virtual_queue=0.;self.budget_cost_sum=0.;self.budget_reference_sum=0.
        self.cost_virtual_queue_peak=0.
        self.budget_trace=[];self.last_budget_increment=0.;self.last_data_increment=0.
        self._cost_references={};self._cost_paths={}
        self.admitted_jobs=0
        return super().reset()

    def _generate(self, now):
        if now>=self.warmup_ms:
            super()._generate(now)

    def _new_instance(self, service, node):
        pool=self.pools.setdefault((service,node),[])
        super()._new_instance(service,node)
        instance=self.instances[-1]
        self.positions[instance.id]=len(pool)
        pool.append(instance)

    def light_rate(self, instance, now):
        tr=self.resource
        key=instance.service,instance.node
        if key not in self.rate_tables:
            service=self.scenario.services[instance.service]
            cap=min(int(c/d+1e-9) for c,d in zip(
                self.scenario.nodes[instance.node].resources,service.resources) if d>0)
            rng=np.random.default_rng(np.random.SeedSequence([
                self.settings.seed,802,tr.node_index[instance.node],tr.service_index[instance.service]]))
            self.rate_tables[key]=rng.gamma(service.gamma_shape,service.gamma_scale,
                (int(np.ceil(self.horizon_ms/self.settings.slot_ms))+1,cap))
        tick=round(now/self.settings.slot_ms)
        gamma=self.rate_tables[key][tick,self.positions[instance.id]]
        return float(gamma*tr.current(now)[tr.node_index[instance.node],tr.service_index[instance.service]])

    def legal_nodes(self, stage):
        # Reachability and known dispatch/completion reservations only. In
        # particular, do not query the physical future-integrated transfer time.
        parents=self.scenario.tasks[stage.request.task].dependencies[stage.service]
        sources=([stage.request.finished[p][0] for p in parents]
                 if parents else [stage.request.gateway])
        legal=[]
        for node in self.node_ids:
            pool=self.pools.get((stage.service,node),[])
            if not pool or any(self.components[s]!=self.components[node] for s in sources):continue
            # A route selects a deployed node, not a free compute instance.
            # Node input queues do not reserve CPU/GPU during transmission.
            legal.append(node)
        return legal

    def _assign(self, stage, node):
        acquisition=self.request_acquisition[stage.request.id]
        if acquisition['acquired_ms'] is None:
            raise RuntimeError('Cannot dispatch before acquiring the request state')
        if acquisition['first_decision_ms'] is None:
            acquisition['first_decision_ms']=self.now
        self.network.update(self.now)
        service=self.scenario.services[stage.service]
        ready=self.network.data_ready(stage,node)
        resource_cost,data_cost=self.incremental_costs(stage,node)
        request_id=stage.request.id
        self.request_resource_cost[request_id]=self.request_resource_cost.get(request_id,0.)+resource_cost
        self.request_data_cost[request_id]=self.request_data_cost.get(request_id,0.)+data_cost
        if self.average_resource_budget:
            resource_ref,data_ref=self._cost_reference(stage.request.task,stage.request.gateway)
            maximum=service.work_mb*max(self.resource_prices[host]
                for (name,host),count in self.fixed.counts.items() if name==stage.service and count)
            cost=resource_cost/max(resource_ref,1e-12)
            reference=maximum/max(resource_ref,1e-12)
            self.last_budget_increment=cost-self.average_resource_budget*reference
            self.last_data_increment=data_cost/max(data_ref,1e-12)
            self.cost_virtual_queue=max(0.,self.cost_virtual_queue+self.last_budget_increment)
            self.cost_virtual_queue_peak=max(self.cost_virtual_queue_peak,self.cost_virtual_queue)
            self.budget_cost_sum+=cost;self.budget_reference_sum+=reference
            point={'time_ms':self.now,'queue':self.cost_virtual_queue,
                   'cost_sum':self.budget_cost_sum,'reference_sum':self.budget_reference_sum}
            if self.budget_trace and self.budget_trace[-1]['time_ms']==self.now:
                self.budget_trace[-1]=point
            else:self.budget_trace.append(point)
        self.node_queues.setdefault((stage.service,node),[]).append(
            Job(stage,ready,service.work_mb))

    def step(self,node):
        queue=self.cost_virtual_queue
        observation,reward,terminated,truncated,info=super().step(node)
        if self.average_resource_budget:
            # Raw +/-1 settlements equal twice SLA successes minus a fixed
            # request count over a full episode. All requests receive credit.
            reward=.5*reward-(queue/self.dpp_v)*self.last_budget_increment
            reward-=self.dpp_data_weight*self.last_data_increment
            info['cost_virtual_queue']=self.cost_virtual_queue
            info['budget_increment']=self.last_budget_increment
        return observation,reward,terminated,truncated,info

    def _path_hops(self, source, target, amount):
        key=source,target,float(amount)
        if key not in self._cost_paths:
            if source==target or amount==0:
                self._cost_paths[key]=0
            else:
                path=nx.shortest_path(self.network.graph,source,target,
                    weight=lambda a,b,d:amount*d['ms_per_mb']+d['propagation_ms'])
                self._cost_paths[key]=len(path)-1
        return self._cost_paths[key]

    def incremental_costs(self, stage, node):
        """Return (lambda*p_k, MB-hop) caused by one stage-node action."""
        req=stage.request;service=self.scenario.services[stage.service]
        resource=service.work_mb*self.resource_prices[node]
        parents=self.scenario.tasks[req.task].dependencies[stage.service]
        if parents:
            sources=[(req.finished[parent][0],self.scenario.services[parent].output_mb)
                     for parent in parents]
        else:
            amount=service.input_mb if service.input_mb is not None else self.scenario.tasks[req.task].input_mb
            sources=[(req.gateway,amount)]
        data=sum(amount*self._path_hops(source,node,amount) for source,amount in sources)
        return float(resource),float(data)

    def _cost_reference(self, task_name, gateway):
        """Action-independent upper bounds used only to normalize route costs."""
        key=task_name,gateway
        if key in self._cost_references:return self._cost_references[key]
        task=self.scenario.tasks[task_name]
        hosts={service:sorted(node for (name,node),count in self.fixed.counts.items()
                              if name==service and count)
               for service in task.dependencies}
        resource=sum(self.scenario.services[service].work_mb*
                     max(self.resource_prices[node] for node in hosts[service])
                     for service in task.dependencies)
        data=0.
        for service,parents in task.dependencies.items():
            if parents:
                for parent in parents:
                    amount=self.scenario.services[parent].output_mb
                    data+=amount*max(self._path_hops(source,target,amount)
                        for source in hosts[parent] for target in hosts[service])
            else:
                model=self.scenario.services[service]
                amount=model.input_mb if model.input_mb is not None else task.input_mb
                data+=amount*max(self._path_hops(gateway,target,amount)
                                  for target in hosts[service])
        self._cost_references[key]=float(resource),float(data)
        return self._cost_references[key]

    def request_costs(self, request):
        resource=float(self.request_resource_cost.get(request.id,0.))
        data=float(self.request_data_cost.get(request.id,0.))
        resource_ref,data_ref=self._cost_reference(request.task,request.gateway)
        return {'resource':resource,'data_mbhop':data,
                'resource_normalized':min(1.,resource/max(resource_ref,1e-12)),
                'data_normalized':min(1.,data/max(data_ref,1e-12)),
                'resource_reference':resource_ref,'data_reference_mbhop':data_ref}

    def _routable_stage(self):
        """Acquire initial state, then reuse delivered representations on refresh.

        No request waits for a newer in-flight reporting round. Reports update
        individual node caches as they arrive; running work continues throughout.
        """
        self.update_telemetry()
        stages=sorted(self.waiting,key=lambda s:(s.ready_ms,s.request.id,s.service))
        for stage in stages:
            acquisition=self.request_acquisition.setdefault(stage.request.id,{
                'ready_ms':stage.ready_ms,
                'required_sample_ms':None,
                'acquired_ms':None,'first_decision_ms':None})
            if acquisition['acquired_ms'] is None and (
                all(report is not None for report in self.received)):
                acquisition['acquired_ms']=self.now
        for stage in stages:
            if (self.request_acquisition[stage.request.id]['acquired_ms'] is not None
                and self.legal_nodes(stage)):
                return stage
        return None

    def _advance(self, now):
        # Reporting is periodic in physical time, including intervals with no
        # available routing decision; it is not triggered only by actions.
        self.update_telemetry()
        end=now+self.settings.slot_ms
        for key,queue in self.node_queues.items():
            if not queue:continue
            service=self.scenario.services[key[0]]
            pool=self.pools[key]
            # Only inputs arriving within this physical slot can start service.
            # The simulator may know transfer finish times; the actor sees counts.
            queue.sort(key=lambda j:(j.data_ready_ms,j.stage.request.id))
            while queue and queue[0].data_ready_ms<end:
                available=[i for i in pool if (not i.jobs if service.kind=='core'
                           else len(i.jobs)<self.settings.max_parallelism)]
                if not available:break
                instance=min(available,key=lambda i:(len(i.jobs),i.id))
                job=queue.pop(0)
                assert job.data_ready_ms<end
                instance.jobs.append(job)
                self.admitted_jobs+=1
        super()._advance(now)

    def update_telemetry(self):
        now=self.now;tr=self.resource
        if now==self.last_telemetry:return
        for delivery in self.channel.advance(now):
            node,tick,payload,_=unpack_report(delivery.packet)
            self.received[node]=(tick*tr.sample_ms,payload.copy())
            if self.codec.name == 'current':
                self.received_forecasts[node]=np.repeat(payload[None,:14],tr.horizon,axis=0)
            else:
                self.received_forecasts[node]=self.codec.forecast_wire(payload)
            reports=self.report_rounds.setdefault(tick,{})
            reports[node]=(tick*tr.sample_ms,payload.copy())
            if len(reports)==len(tr.nodes) and tick>self.completed_report_tick:
                self.completed_report_tick=tick
                self.report_completions.append({'sample_ms':tick*tr.sample_ms,
                    'available_ms':now,'last_packet_arrival_ms':delivery.arrival_ms,
                    'acquisition_duration_ms':now-tick*tr.sample_ms})
        if self.report_start_counters is None and now>=self.warmup_ms:
            self.report_start_counters=(self.channel.generated_bytes,self.channel.transmitted_bytes)
        if now+1e-8>=self.next_report:
            self.current_report_tick=round(now/tr.sample_ms)
            for node,history in enumerate(tr.window(now)):
                tick=round(now/tr.sample_ms)
                if self.record_training:
                    # Local training records are never part of a wire packet or
                    # controller state. Future labels are used by training only.
                    self.local_training_windows[node,tick]=(history.copy(),tr.future(now)[node].copy())
                payload=self.codec.encode_wire(history)
                packet=pack_report(node,tick,payload,
                                   0 if self.codec.name == 'current' else 11)
                self.channel.enqueue(node,packet,now+self.telemetry.encoding_ms)
            self.next_report=now+self.telemetry.report_ms
        self.forecast.fill(1.)
        for node,report in enumerate(self.received):
            if report is None:continue
            age=max(0,int((now-report[0])//tr.sample_ms))
            idx=np.minimum(age+np.arange(tr.horizon),tr.horizon-1)
            self.forecast[node]=self.received_forecasts[node][idx]
        self.forecast=np.clip(self.forecast,.15,1.5)
        self.observed_network.update(now,self.forecast)
        self.last_telemetry=now

    def backlog(self, stage, node):
        pool=self.pools.get((stage.service,node),[])
        if not pool:return float(self.settings.max_parallelism)
        jobs=sum(len(i.jobs) for i in pool)+len(self.node_queues.get((stage.service,node),[]))
        return float(jobs/len(pool))

    def downstream_pressure(self, remaining):
        """Known waiting work and reservations, scaled by deployed capacity.

        A waiting stage contributes to hosts nearest its known input locations.
        This is a causal locality estimate, not its eventual routing decision.
        No physical Job.data_ready or realized remaining work is inspected.
        """
        queued=Counter()
        for pending in self.waiting:
            if pending.service not in remaining:continue
            req=pending.request
            parents=self.scenario.tasks[req.task].dependencies[pending.service]
            sources=set(req.finished[p][0] for p in parents) if parents else {req.gateway}
            for source in sources:
                queued[pending.service,self.nearest_hosts[pending.service,source]]+=1./len(sources)
        pressure=[]
        for index,node in enumerate(self.node_ids):
            delay=0.
            for name in remaining:
                pool=self.pools.get((name,node),[])
                if not pool:continue
                service=self.scenario.services[name]
                jobs=sum(len(i.jobs) for i in pool)+len(self.node_queues.get((name,node),[]))+queued[name,node]
                if service.kind=='core':
                    per_job=ceil(service.mean_processing_ms/self.settings.slot_ms)*self.settings.slot_ms
                else:
                    availability=max(.15,float(self.forecast[index,:11,self.resource.service_index[name]].mean()))
                    per_job=service.mean_processing_ms/availability
                delay+=jobs*per_job/len(pool)
            pressure.append(np.log1p(delay/50.))
        return np.asarray(pressure,np.float32)

    def state(self):
        self.update_telemetry()
        stage=self._routable_stage();req=stage.request
        task=self.scenario.tasks[req.task]
        embeddings=np.zeros((len(self.node_ids),self.codec.latent_dim),np.float32)
        ages=np.zeros(len(self.node_ids),np.float32)
        missing=np.ones(len(self.node_ids),np.float32)
        for node,report in enumerate(self.received):
            if report is None:continue
            embeddings[node]=self.codec.decode_wire(report[1])
            ages[node]=(self.now-report[0])/100.
            missing[node]=0.
        if self.state_representation=='forecast':
            # Receiver restores the delivered Z into an age-aligned forecast.
            # No physical trace or source history enters this actor state.
            embeddings=self.forecast*(1.-missing[:,None,None])
        backlog=np.asarray([np.log1p(self.backlog(stage,node))/3.
                            for node in self.node_ids],np.float32)
        # Compact, known dispatch/completion ledger for the unfinished DAG.
        # The source compressor models resource/link dynamics, not these queues.
        remaining=set(task.dependencies)-set(req.finished)-{stage.service}
        downstream=self.downstream_pressure(remaining)
        # This is RECEIVED-forecast-based, never self.network.data_ready().
        service_model=self.scenario.services[stage.service]
        costs=[]
        for index,node in enumerate(self.node_ids):
            if not self.pools.get((stage.service,node)):
                costs.append(500.);continue
            transfer=max(0.,self.observed_network.data_ready(stage,node)-self.now)
            per_job=ceil(service_model.mean_processing_ms/self.settings.slot_ms)*self.settings.slot_ms
            if service_model.kind=='light':
                current=max(.15,float(self.forecast[index,0,self.resource.service_index[stage.service]]))
                offset=transfer+self.backlog(stage,node)*service_model.mean_processing_ms/current
                k=min(self.resource.horizon-1,int(offset/self.resource.sample_ms))
                rate=max(.15,float(self.forecast[index,k:k+3,self.resource.service_index[stage.service]].mean()))
                per_job=service_model.mean_processing_ms/rate
            costs.append(transfer+(self.backlog(stage,node)+1)*per_job)
        transfer=np.nan_to_num(np.asarray(costs,np.float32),posinf=500.)/50.
        action_costs=[self.incremental_costs(stage,node) for node in self.node_ids]
        resource_ref,data_ref=self._cost_reference(req.task,req.gateway)
        resource_cost=np.asarray([value[0]/max(resource_ref,1e-12)
                                  for value in action_costs],np.float32)
        data_cost=np.asarray([value[1]/max(data_ref,1e-12)
                              for value in action_costs],np.float32)
        services=sorted(self.scenario.services);tasks=sorted(self.scenario.tasks)
        service=np.zeros(len(services),np.float32);service[services.index(stage.service)]=1.
        task_type=np.zeros(len(tasks),np.float32);task_type[tasks.index(req.task)]=1.
        origin=np.zeros(len(self.node_ids),np.float32);origin[self.node_ids.index(req.gateway)]=1.
        slack=np.asarray([(task.deadline_ms-(self.now-req.arrival_ms))/50.,
                          (self.horizon_ms-self.now)/(self.horizon_ms-self.warmup_ms)],np.float32)
        self.age_samples.append(float(np.mean(ages[missing==0])*100.) if (missing==0).any() else None)
        self.missing_samples.append(float(missing.mean()))
        fields=[embeddings.reshape(-1),backlog,downstream,transfer]
        if self.include_route_costs:fields.extend((resource_cost,data_cost))
        state=np.concatenate([*fields,ages,missing,service,task_type,origin,slack]).astype(np.float32)
        if self.average_resource_budget:
            state=np.concatenate((state,np.asarray([self.cost_virtual_queue/self.dpp_v],np.float32)))
        if not np.isfinite(state).all():raise FloatingPointError('Nonfinite routing state')
        legal=set(self.legal_nodes(stage))
        mask=np.asarray([node in legal for node in self.node_ids],np.float32)
        return state,mask

    def learning_observation(self, state):
        """Replay histories only for reports that have actually been delivered.

        The optimizer re-encodes these local recorded windows to propagate task
        gradients into the source encoder. Actors use state(), never this API.
        """
        if self.state_representation=='forecast':return state,()
        records=tuple(None if report is None else self.local_training_windows[
            node,round(report[0]/self.resource.sample_ms)]
            for node,report in enumerate(self.received))
        if not self.end_to_end_forecast:return state,records
        return state,records,self.forecast_context()

    def forecast_context(self):
        """Known topology/queues for recomputing forecast features during learning."""
        stage=self._routable_stage();req=stage.request;task=self.scenario.tasks[req.task]
        model=self.scenario.services[stage.service];nodes=len(self.node_ids)
        parents=task.dependencies[stage.service]
        if parents:
            sources=[(*req.finished[p],self.scenario.services[p].output_mb) for p in parents]
        else:
            amount=model.input_mb if model.input_mb is not None else task.input_mb
            sources=[(req.gateway,req.uplink_ready_ms,amount)]
        # Task-global parent padding keeps mixed-stage minibatches stackable.
        parent_count=max(1,max(len(ps) for t in self.scenario.tasks.values() for ps in t.dependencies.values()))
        path_key=stage.service,tuple((src,amount) for src,_,amount in sources)
        if path_key not in self.training_paths:
            paths=np.zeros((nodes,parent_count,nodes-1,5),np.float32)
            paths[...,2]=1.
            amounts=np.zeros((nodes,parent_count),np.float32)
            for i,node in enumerate(self.node_ids):
                if not self.pools.get((stage.service,node)):continue
                for j,(source,_,amount) in enumerate(sources):
                    amounts[i,j]=amount
                    if source==node or amount==0:continue
                    path=self.observed_network.paths[source,node,amount]
                    for k,(a,b) in enumerate(zip(path[:-1],path[1:])):
                        edge=self.observed_network.graph[a][b]
                        paths[i,j,k]=[self.resource.node_index[a],9+self.resource.neighbors[a].index(b),
                                      edge['ms_per_mb'],edge['propagation_ms'],1.]
            self.training_paths[path_key]=paths,amounts
        paths,amounts=self.training_paths[path_key]
        starts=np.zeros((nodes,parent_count),np.float32)
        for j,(_,ready,_) in enumerate(sources):starts[:,j]=max(0.,ready-self.now)
        remaining=set(task.dependencies)-set(req.finished)-{stage.service}
        queued=Counter()
        for pending in self.waiting:
            if pending.service not in remaining:continue
            r=pending.request;ps=self.scenario.tasks[r.task].dependencies[pending.service]
            locations=set(r.finished[p][0] for p in ps) if ps else {r.gateway}
            for src in locations:queued[pending.service,self.nearest_hosts[pending.service,src]]+=1./len(locations)
        core=np.zeros(nodes,np.float32);light=np.zeros((nodes,9),np.float32)
        for i,node in enumerate(self.node_ids):
            for name in remaining:
                pool=self.pools.get((name,node),[])
                if not pool:continue
                m=self.scenario.services[name]
                jobs=sum(len(ins.jobs) for ins in pool)+len(self.node_queues.get((name,node),[]))+queued[name,node]
                if m.kind=='core':core[i]+=jobs*ceil(m.mean_processing_ms/self.settings.slot_ms)*self.settings.slot_ms/len(pool)
                else:light[i,self.resource.service_index[name]]+=jobs*m.mean_processing_ms/len(pool)
        return {'paths':paths,'amounts':amounts,'starts':starts,'phase':np.float32(self.now%self.resource.sample_ms),
                'age_bins':np.asarray([0 if r is None else int((self.now-r[0])//self.resource.sample_ms) for r in self.received],np.int64),
                'light':model.kind=='light','service_index':self.resource.service_index.get(stage.service,-1),
                'mean_processing_ms':model.mean_processing_ms,
                'core_processing_ms':ceil(model.mean_processing_ms/self.settings.slot_ms)*self.settings.slot_ms,
                'backlogs':np.asarray([self.backlog(stage,n) for n in self.node_ids],np.float32),
                'deployed':np.asarray([bool(self.pools.get((stage.service,n))) for n in self.node_ids]),
                'downstream_core_ms':core,'downstream_light_work':light}

    def greedy_node(self):
        self.update_telemetry()
        stage=self._routable_stage();service=self.scenario.services[stage.service]
        scores={}
        for node in self.legal_nodes(stage):
            rate=1.
            if service.kind=='light':
                i=self.resource.node_index[node];j=self.resource.service_index[stage.service]
                rate=float(self.forecast[i,0,j]) if self.codec.name == 'current' else float(
                    self.forecast[i,:11,j].mean())
            scores[node]=max(self.now,self.observed_network.data_ready(stage,node)) + (
                self.backlog(stage,node)+1)*service.mean_processing_ms/max(.15,rate)
        return min(scores,key=lambda n:(scores[n],n))

    def shortest_queue_node(self):
        """Join the shortest per-instance queue; current delay breaks ties."""
        self.update_telemetry()
        stage=self._routable_stage();service=self.scenario.services[stage.service]
        def score(node):
            rate=1.
            if service.kind=='light':
                rate=float(self.forecast[self.resource.node_index[node],0,
                                         self.resource.service_index[stage.service]])
            current_delay=max(0.,self.observed_network.data_ready(stage,node)-self.now)
            current_delay+=service.mean_processing_ms/max(.15,rate)
            return self.backlog(stage,node),current_delay,node
        return min(self.legal_nodes(stage),key=score)

    def diagnostics(self):
        def sla(requests):
            completed=[r for r in requests if r.completion_ms is not None]
            delays=[r.completion_ms-r.arrival_ms for r in completed]
            ontime=sum(r.completion_ms-r.arrival_ms<=self.scenario.tasks[r.task].deadline_ms+1e-9
                       for r in completed)
            return {'arrivals':len(requests),'ontime':ontime,'pending':len(requests)-len(completed),
                    'sla':ontime/len(requests) if requests else None,
                    'mean_completed_latency_ms':float(np.mean(delays)) if delays else None,
                    'p95_completed_latency_ms':float(np.percentile(delays,95)) if delays else None}
        ages=[a for a in self.age_samples if a is not None]
        waits=[a['acquired_ms']-a['ready_ms'] for a in self.request_acquisition.values()
               if a['acquired_ms'] is not None]
        request_costs=[self.request_costs(r) for r in self.requests]
        return {**sla(self.requests),'tasks':{t:sla([r for r in self.requests if r.task==t])
                    for t in sorted(self.scenario.tasks)},
            'resource_cost_total':float(sum(c['resource'] for c in request_costs)),
            'data_cost_total_mbhop':float(sum(c['data_mbhop'] for c in request_costs)),
            'mean_resource_cost_per_request':float(np.mean([c['resource'] for c in request_costs])) if request_costs else None,
            'mean_data_cost_mbhop_per_request':float(np.mean([c['data_mbhop'] for c in request_costs])) if request_costs else None,
            'mean_normalized_resource_cost':float(np.mean([c['resource_normalized'] for c in request_costs])) if request_costs else None,
            'mean_normalized_data_cost':float(np.mean([c['data_normalized'] for c in request_costs])) if request_costs else None,
            'resource_prices':self.resource_prices,
            'resource_price_model':'CPU/GPU capacity rank mapped to [3,10] cost units per work MB',
            'average_resource_budget':self.average_resource_budget or None,
            'budget_cost_sum':self.budget_cost_sum,
            'budget_reference_sum':self.budget_reference_sum,
            'budget_cost_ratio':self.budget_cost_sum/self.budget_reference_sum if self.budget_reference_sum else None,
            'cost_virtual_queue_final':self.cost_virtual_queue,
            'cost_virtual_queue_peak':self.cost_virtual_queue_peak,
            'cost_virtual_queue_trace':self.budget_trace,
            'arrival_sha256':self.arrival_hash.hexdigest(),'resource_sha256':self.resource.sha256,
            'report_generated_bytes':self.channel.generated_bytes,
            'report_transmitted_bytes':self.channel.transmitted_bytes,
            'report_generated_bytes_active_window':self.channel.generated_bytes-self.report_start_counters[0],
            'report_transmitted_bytes_active_window':self.channel.transmitted_bytes-self.report_start_counters[1],
            'report_bytes_per_packet':16+self.codec.payload_bytes,
            'mean_received_age_ms':float(np.mean(ages)) if ages else None,
            'mean_missing_node_fraction':float(np.mean(self.missing_samples)) if self.missing_samples else None,
            'decisions_horizon_ms':self.horizon_ms,
            'telemetry_warmup_ms':self.warmup_ms,
            'state_acquisition_protocol':'initial delivered state required; subsequent requests reuse node caches while periodic reports are in flight',
            'report_rounds_completed':self.report_completions,
            'mean_request_state_wait_ms':float(np.mean(waits)) if waits else None,
            'p95_request_state_wait_ms':float(np.percentile(waits,95)) if waits else None,
            'requests_without_acquired_state':sum(a['acquired_ms'] is None for a in self.request_acquisition.values()),
            'compute_admission':'node routing and input waiting do not reserve a compute instance; admit within data-arrival physical slot',
            'compute_admitted_jobs':self.admitted_jobs,
            'dynamics_profile':self.dynamics_profile,
            'azure_provenance':self.resource.provenance,
            'requests':[{'id':r.id,'user':r.user,'task':r.task,'gateway':r.gateway,
                         'arrival_ms':r.arrival_ms,'completion_ms':r.completion_ms,
                         'state_acquisition':self.request_acquisition.get(r.id),
                         'routing_costs':self.request_costs(r)} for r in self.requests]}
