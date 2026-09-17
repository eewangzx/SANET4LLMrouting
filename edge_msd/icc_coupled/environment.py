"""Original ICC execution and constraints with physically delayed observations.

Central orchestration knows its instance ledger, requests, and completion ACKs.
Only background service/link telemetry is charged to the narrow reporting link.
ACK/command traffic is assumed separate; no claim of zero total control traffic.
"""

from collections import Counter
from dataclasses import dataclass, replace
from math import floor

import networkx as nx
import numpy as np

from edge_msd.capacity import DelayModel
from edge_msd.model_routing.telemetry import NarrowbandChannel, pack_report, unpack_report
from edge_msd.models import Instance, Job
from edge_msd.network import Network
from edge_msd.simulation import Simulator

from .dynamics import ResourceTrace
from .fast_controller import FastController


@dataclass(frozen=True)
class TelemetrySettings:
    arrival_ms: int = 200
    control_ms: int = 1
    report_ms: int = 100
    report_bps: float = 64000
    propagation_ms: float = 1
    encoding_ms: float = .2
    dynamic: bool = True


class MultipliersNetwork(Network):
    """Fixed nominal paths, time-varying transfer rate; no future route choice."""

    def __init__(self, scenario, trace, physical=False):
        super().__init__(scenario, False)
        self.trace, self.physical, self.now = trace, physical, 0.
        self.predictions = np.ones((len(trace.nodes), trace.horizon, 14), np.float32)
        self.paths = {}
        self.ready_cache = {}

    def update(self, now, predictions=None):
        self.now = now
        if predictions is not None:
            self.predictions = predictions
        self._cache.clear()
        self.ready_cache.clear()

    def factor(self, source, target, when):
        i = self.trace.node_index[source]
        j = 9 + self.trace.neighbors[source].index(target)
        if self.physical:
            return float(self.trace.current(when)[i, j])
        # Received forecasts have already been shifted to the current absolute
        # sample bin. Integration also advances at absolute sample boundaries;
        # floor(when-now) would lag one bin when now is between boundaries.
        k = min(self.trace.horizon-1, max(0,
            int(when//self.trace.sample_ms)-int(self.now//self.trace.sample_ms)))
        return float(np.clip(self.predictions[i, k, j], .15, 1.5))

    def _delay(self, source, target, data_mb):
        return self._delay_at(source, target, data_mb, self.now)

    def _delay_at(self, source, target, data_mb, start_ms):
        key = source, target, data_mb
        if source == target or data_mb == 0:
            return 0.
        if key not in self.paths:
            self.paths[key] = nx.shortest_path(self.graph, source, target,
                weight=lambda a, b, d: data_mb*d['ms_per_mb']+d['propagation_ms'])
        path, cursor = self.paths[key], start_ms
        for a, b in zip(path[:-1], path[1:]):
            left = data_mb
            edge = self.graph[a][b]
            while left > 1e-10:
                rate = self.factor(a, b, cursor) / edge['ms_per_mb']
                boundary = (floor(cursor / self.trace.sample_ms)+1)*self.trace.sample_ms
                elapsed = min(left/rate, max(1e-8, boundary-cursor))
                left -= elapsed*rate
                cursor += elapsed
            cursor += edge['propagation_ms']
        return cursor-start_ms

    def data_ready(self, stage, target):
        key = stage.key, target
        if key not in self.ready_cache:
            # Preserve ICC's predecessor-finish-based communication abstraction.
            # Starting all transfers only at dispatch would introduce extra core
            # reservation time and change the baseline execution problem.
            if not self.physical:
                self.ready_cache[key] = super().data_ready(stage, target)
            else:
                req = stage.request
                parents = self.scenario.tasks[req.task].dependencies[stage.service]
                if not parents:
                    amount = self.scenario.services[stage.service].input_mb
                    if amount is None:
                        amount = self.scenario.tasks[req.task].input_mb
                    start = req.uplink_ready_ms
                    self.ready_cache[key] = start + self._delay_at(req.gateway, target, amount, start)
                else:
                    self.ready_cache[key] = max(req.finished[p][1] + self._delay_at(
                        req.finished[p][0], target, self.scenario.services[p].output_mb,
                        req.finished[p][1]) for p in parents)
        return self.ready_cache[key]


class InformedController(FastController):
    def __init__(self, *args, trace, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = trace
        self.resource_prediction = np.ones((len(trace.nodes), trace.horizon, 14), np.float32)
        self.scaled_models = {}
        self.service_scale = {}

    def set_prediction(self, prediction):
        self.resource_prediction = prediction
        # Mean availability over the next 50 ms; the same EC closure for every codec.
        rates = np.clip(prediction[:, :11, :9].mean(1), .15, 1.5)
        self.service_scale = {(s, n): round(float(rates[i, j])*20)/20
                              for i, n in enumerate(self.trace.nodes)
                              for j, s in enumerate(self.trace.services)}

    def processing_delay(self, service, node, parallelism):
        scale = max(.15, self.service_scale.get((service, node), 1.))
        key = service, scale
        if key not in self.scaled_models:
            original = self.scenario.services[service]
            scaled = replace(original, gamma_scale=original.gamma_scale*scale)
            self.scaled_models[key] = DelayModel({service: scaled}, self.delay_model.mode,
                self.settings.epsilon, self.settings.slot_ms,
                self.settings.ec_admission_window_ms)
        return self.scaled_models[key].delay(service, parallelism)


class CoupledSimulator(Simulator):
    def __init__(self, scenario, settings, telemetry=None, codec=None,
                 observation='periodic', core_counts=None, controller_method='proposed', policy=None):
        self.telemetry = telemetry or TelemetrySettings()
        if settings.wireless:
            raise ValueError("Current extension preserves the declared wired-only ICC ablation")
        if self.telemetry.arrival_ms > settings.duration_ms:
            raise ValueError("Arrival window exceeds the horizon")
        if observation not in ('periodic', 'instant', 'prior'):
            raise ValueError(observation)
        self.trace_resource = ResourceTrace(scenario, settings.duration_ms, settings.seed,
                                            self.telemetry.dynamic)
        self.instance_positions = {}
        self.rate_tables = {}
        self.estimated_ready = {}
        super().__init__(scenario, settings, controller_method, core_counts)
        self.network = MultipliersNetwork(scenario, self.trace_resource, physical=True)
        self.observed_network = MultipliersNetwork(scenario, self.trace_resource)
        self.controller = InformedController(scenario, self.observed_network, settings,
                            self.placement.counts, controller_method, trace=self.trace_resource)
        self.codec, self.observation = codec, observation
        self.policy = policy
        self.channel = NarrowbandChannel(self.telemetry.report_bps, self.telemetry.propagation_ms)
        self.received = [None for _ in self.trace_resource.nodes]
        self.received_forecasts = [None for _ in self.trace_resource.nodes]
        self.age_samples, self.missing_samples = [], []
        self.control_trace = []
        self.stage_routes = Counter()
        self._current_forecast = np.ones((len(self.received), self.trace_resource.horizon, 14), np.float32)

    def _new_instance(self, service, node):
        occupied = {self.instance_positions[i.id] for i in self.instances
                    if i.service == service and i.node == node}
        position = next(k for k in range(len(occupied)+1) if k not in occupied)
        super()._new_instance(service, node)
        self.instance_positions[self.instances[-1].id] = position

    def light_rate(self, instance, now):
        tr = self.trace_resource
        key = instance.service, instance.node
        if key not in self.rate_tables:
            service = self.scenario.services[instance.service]
            cap = min(int(c/d+1e-9) for c, d in zip(
                self.scenario.nodes[instance.node].resources, service.resources) if d > 0)
            rng = np.random.default_rng(np.random.SeedSequence([
                self.settings.seed, 802, tr.node_index[instance.node], tr.service_index[instance.service]]))
            shape = (round(self.settings.duration_ms/self.settings.slot_ms), cap)
            self.rate_tables[key] = rng.gamma(service.gamma_shape, service.gamma_scale, shape)
        tick = round(now/self.settings.slot_ms)
        gamma = self.rate_tables[key][tick, self.instance_positions[instance.id]]
        factor = tr.current(now)[tr.node_index[instance.node], tr.service_index[instance.service]]
        return float(gamma*factor)

    def core_candidate_key(self, stage, instance):
        return self.observed_network.data_ready(stage, instance.node), instance.node, instance.id

    def _telemetry(self, now):
        tr = self.trace_resource
        for delivery in self.channel.advance(now):
            node, tick, payload, _ = unpack_report(delivery.packet)
            self.received[node] = tick*tr.sample_ms, payload.copy()
            self.received_forecasts[node] = self.codec.forecast_wire(payload)
        if self.observation == 'periodic' and now % self.telemetry.report_ms < 1e-8:
            for node, history in enumerate(tr.window(now)):
                payload = self.codec.encode_wire(history)
                packet = pack_report(node, round(now/tr.sample_ms), payload, 11)
                self.channel.enqueue(node, packet, now+self.telemetry.encoding_ms)
        if self.observation == 'instant':
            forecast = np.repeat(tr.current(now)[:, None], tr.horizon, axis=1)
        else:
            forecast = np.ones_like(self._current_forecast)
            for n, report in enumerate(self.received):
                if report is None:
                    continue
                age = max(0, int((now-report[0])//tr.sample_ms))
                idx = np.minimum(age+np.arange(tr.horizon), tr.horizon-1)
                forecast[n] = self.received_forecasts[n][idx]
        self._current_forecast = np.clip(forecast, .15, 1.5)
        self.observed_network.update(now, self._current_forecast)
        self.controller.set_prediction(self._current_forecast)

    def _observed_instances(self):
        # Occupancy/ACKs come from the common orchestration ledger. No true in-flight
        # transfer finish, remaining work, or realized Gamma rate enters the controller.
        return [Instance(i.id, i.service, i.node, [Job(j.stage,
                    self.estimated_ready[j.stage.key], self.scenario.services[i.service].work_mb)
                    for j in i.jobs]) for i in self.instances]

    def _route(self, now, decision):
        old = {(j.stage.key) for i in self.instances for j in i.jobs}
        super()._route(now, decision)
        for instance in self.instances:
            for job in instance.jobs:
                if job.stage.key not in old:
                    self.estimated_ready[job.stage.key] = self.observed_network.data_ready(job.stage, instance.node)
                    self.stage_routes[instance.node] += 1

    def _maintenance(self):
        for i in self.instances:
            service = self.scenario.services[i.service]
            self.costs['core_maintenance' if service.kind == 'core' else 'light_maintenance'] += service.maintenance_cost
            if service.kind == 'light':
                self.costs['light_parallel'] += service.parallel_cost

    def accounting(self, now, terminal=False):
        violations = 0
        for req in self.requests:
            deadline = self.scenario.tasks[req.task].deadline_ms
            if req.completion_ms is None:
                violations += int(terminal or now-req.arrival_ms > deadline+1e-9)
            else:
                violations += int(req.completion_ms-req.arrival_ms > deadline+1e-9)
        return {'cost': sum(self.costs.values()), 'violations': violations,
                'arrivals': len(self.requests), 'report_bytes': self.channel.transmitted_bytes,
                'time_ms': now, 'terminal': terminal}

    def policy_observation(self, now):
        tr = self.trace_resource
        n = len(tr.nodes)
        latents = np.zeros((n, self.codec.latent_dim), np.float32)
        windows = np.zeros((n, tr.history, 16), np.float32)
        targets = np.zeros((n, tr.horizon, 14), np.float32)
        ages, present = np.zeros(n, np.float32), np.zeros(n, np.float32)
        for i, received in enumerate(self.received):
            if received is not None:
                sampled, payload = received
                latents[i] = self.codec.decode_wire(payload)
                ages[i], present[i] = now-sampled, 1
                windows[i] = tr.window(sampled)[i]
                targets[i] = tr.future(sampled)[i]
        public = np.zeros((n, 4), np.float32)
        for i, node in enumerate(tr.nodes):
            local = [a for a in self.instances if a.node == node]
            jobs = [j for a in local for j in a.jobs]
            public[i] = [sum(self.scenario.services[a.service].kind=='light' for a in local)/20,
                         len(jobs)/40,
                         sum(s.request.gateway==node for s in self.waiting)/40,
                         np.mean([self.controller.queue.get(j.stage.request.id, 1) for j in jobs])/100
                         if jobs else 0]
        global_state = [sum(r.task==name for r in self.active.values())/100
                        for name in sorted(self.scenario.tasks)]
        global_state.append((self.settings.duration_ms-now)/self.settings.duration_ms)
        obs = {'latents': latents, 'ages': ages, 'present': present, 'public': public,
               'global': np.asarray(global_state, np.float32), 'accounting': self.accounting(now)}
        return obs, {'windows': windows, 'targets': targets}

    def run(self):
        if self.has_run:
            raise RuntimeError("Create a new episode")
        self.has_run = True
        steps = round(self.settings.duration_ms/self.settings.slot_ms)
        for tick in range(steps):
            now = tick*self.settings.slot_ms
            self.network.update(now)
            self._telemetry(now)
            if now < self.telemetry.arrival_ms:
                self._generate(now)
            self._expose_ready(now)
            if now % self.telemetry.control_ms < 1e-8:
                if self.policy is not None:
                    observation, training_context = self.policy_observation(now)
                    eta_multiplier, cap = self.policy.act(observation, training_context)
                    busy_cap = max((len(i.jobs) for i in self.instances
                                    if self.scenario.services[i.service].kind=='light'), default=0)
                    self.controller.settings = replace(self.settings,
                        eta=self.settings.eta*eta_multiplier,
                        max_parallelism=max(busy_cap, min(cap, self.settings.max_parallelism)))
                instances = self._observed_instances()
                decision = self.controller.step(now, instances, self.waiting, self.active)
                self._deploy(decision)
                self._route(now, decision)
                self.control_trace.append({
                    'time_ms': now, 'light_instances': sum(decision.counts.values()),
                    'parallelism_sum': sum(decision.parallelism.values()),
                    'max_parallelism': max(decision.parallelism.values(), default=0),
                    'light_routes': len(decision.routes), 'active_tasks': len(self.active)})
                if self.observation == 'periodic':
                    self.age_samples.extend(now-r[0] for r in self.received if r is not None)
                    self.missing_samples.extend(r is None for r in self.received)
            else:
                self._maintenance()
            self._advance(now)
        self.channel.advance(self.settings.duration_ms)
        final_accounting = self.accounting(self.settings.duration_ms, terminal=True)
        if self.policy is not None:
            self.policy.finish({'accounting': final_accounting})
        result = self.results()
        result.update(
            accounting=final_accounting,
            observation=self.observation,
            codec=None if self.codec is None else getattr(self.codec, 'kind', type(self.codec).__name__),
            resource_sha256=self.trace_resource.sha256,
            gamma_protocol='independent keyed (seed,node,service,physical_slot,instance_position)',
            report_bytes=float(self.channel.transmitted_bytes),
            report_bps=float(self.channel.transmitted_bytes*8000/self.settings.duration_ms),
            report_generated_bytes=self.channel.generated_bytes,
            report_replacements=self.channel.replaced_reports,
            mean_state_age_ms=float(np.mean(self.age_samples)) if self.age_samples else None,
            missing_fraction=float(np.mean(self.missing_samples)) if self.missing_samples else None,
            core_counts={f'{s}@{n}':v for (s,n),v in self.placement.counts.items() if v},
            control_trace=self.control_trace, stage_routes=dict(self.stage_routes),
            task_sla='original task DAG end-to-end deadline; no answer-quality gate',
            acknowledgements='completion/occupancy ACKs and commands are common, separate from measured telemetry',
            constraints='original _deploy resource, busy-instance and per-instance parallelism validations')
        return result
