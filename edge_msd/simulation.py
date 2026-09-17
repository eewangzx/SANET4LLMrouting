"""Slotted controller with continuous transmission/processing within each slot.

At t: observe completions, generate arrivals, expose ready DAG stages, deploy and
route. During [t,t+dt]: advance instances. Successors are eligible at the next
control boundary. The horizon is fixed; unfinished requests remain censored.
"""

import hashlib
from collections import Counter
from dataclasses import asdict
from math import isfinite
from numbers import Integral

import numpy as np

from .config import Settings
from .controller import Controller
from .models import Instance, Job, Placement, Request, Scenario, Stage
from .network import Network
from .placement import PlacementResult, place_core, validate_core_budget, validate_resources


class Simulator:
    def __init__(
        self,
        scenario: Scenario,
        settings: Settings,
        method: str = "proposed",
        core_counts: Placement | None = None,
    ):
        scenario.validate()
        self.scenario, self.settings, self.method = scenario, settings, method
        self.network = Network(scenario, settings.wireless)
        self.placement = (
            place_core(scenario, self.network, settings, method)
            if core_counts is None
            else PlacementResult(
                core_counts, 0.0, sum(v > 0 for v in core_counts.values()), "supplied"
            )
        )
        if any(scenario.services[s].kind != "core" for s, _ in self.placement.counts):
            raise ValueError("core_counts must contain only core services")
        validate_resources(scenario, self.placement.counts)
        validate_core_budget(scenario, self.placement.counts)
        self.controller = Controller(
            scenario, self.network, settings, self.placement.counts, method
        )
        # Independent streams prevent policy-side sampling from changing demand.
        streams = np.random.SeedSequence(settings.seed).spawn(3)
        self.arrivals, self.channels, self.service_rng = [np.random.default_rng(s) for s in streams]
        self.instances: list[Instance] = []
        self.next_instance_id = 0
        for (service, node), count in sorted(self.placement.counts.items()):
            for _ in range(count):
                self._new_instance(service, node)
        self.active: dict[int, Request] = {}
        self.requests: list[Request] = []
        self.waiting: list[Stage] = []
        self.trace: list[dict] = []
        self.arrival_hash = hashlib.sha256()
        self.costs = {
            "core_deployment": sum(
                count * scenario.services[service].deployment_cost
                for (service, _), count in self.placement.counts.items()
            ),
            "core_maintenance": 0.0,
            "light_deployment": 0.0,
            "light_maintenance": 0.0,
            "light_parallel": 0.0,
        }
        self.peak_instances = Counter()
        self.has_run = False

    def _new_instance(self, service: str, node: str):
        self.instances.append(Instance(self.next_instance_id, service, node))
        self.next_instance_id += 1

    def add_request(
        self, user_name: str, task_name: str, arrival_ms: float, uplink_ms: float | None = None
    ) -> Request:
        """Inject a deterministic request for controlled examples or tests."""
        if not isfinite(arrival_ms) or not 0 <= arrival_ms < self.settings.duration_ms:
            raise ValueError("Request arrival must lie within the simulation horizon")
        user = next(u for u in self.scenario.users if u.name == user_name)
        task = self.scenario.tasks[task_name]
        if uplink_ms is None:
            uplink_ms = self.network.uplink_delay(user, task.input_mb, self.channels)
        if uplink_ms < 0 or np.isnan(uplink_ms):
            raise ValueError("Uplink delay must be nonnegative (infinity means unreachable)")
        req = Request(
            len(self.requests),
            user.name,
            task.name,
            arrival_ms,
            arrival_ms + uplink_ms,
            user.gateway,
        )
        self.requests.append(req)
        self.active[req.id] = req
        self.arrival_hash.update(
            f"{req.id}|{user.name}|{task.name}|{arrival_ms:.12g}|{uplink_ms:.12g}\n".encode()
        )
        return req

    def _generate(self, now: float):
        for user in sorted(self.scenario.users, key=lambda u: u.name):
            for task, rate in sorted(user.rates_per_second.items()):
                count = self.arrivals.poisson(rate * self.settings.slot_ms / 1000)
                for _ in range(count):
                    self.add_request(user.name, task, now)

    def _expose_ready(self, now: float):
        for req in self.active.values():
            if req.uplink_ready_ms > now:
                continue
            task = self.scenario.tasks[req.task]
            for service in task.order:
                if service in req.scheduled or service in req.finished:
                    continue
                parents = task.dependencies[service]
                if all(p in req.finished for p in parents):
                    ready = max((req.finished[p][1] for p in parents), default=req.uplink_ready_ms)
                    self.waiting.append(Stage(req, service, ready))
                    req.scheduled.add(service)

    def _deploy(self, decision):
        if any(self.scenario.services[s].kind != "light" for s, _ in decision.counts):
            raise ValueError("An online decision cannot change core placement")
        for key, cap in decision.parallelism.items():
            if (
                key not in decision.counts
                or isinstance(cap, bool)
                or not isinstance(cap, Integral)
                or not 0 <= cap <= self.settings.max_parallelism
            ):
                raise ValueError("Invalid parallelism limit")
        old = Counter(
            (i.service, i.node)
            for i in self.instances
            if self.scenario.services[i.service].kind == "light"
        )
        busy = Counter(
            (i.service, i.node)
            for i in self.instances
            if i.jobs and self.scenario.services[i.service].kind == "light"
        )
        if any(decision.counts.get(k, 0) < v for k, v in busy.items()):
            raise ValueError("A decision cannot remove a busy or reserved instance")
        for instance in self.instances:
            if self.scenario.services[instance.service].kind == "light" and len(instance.jobs) > (
                decision.parallelism.get((instance.service, instance.node), 0)
            ):
                raise ValueError("Parallelism cannot be lower than current occupancy")
        waiting = {stage.key: stage for stage in self.waiting}
        reserved = Counter()
        ordered = {}
        for key in decision.counts:
            ordered[key] = sorted(
                (i for i in self.instances if (i.service, i.node) == key and i.jobs),
                key=lambda i: i.id,
            )
        for stage_key, node in decision.routes.items():
            if stage_key not in waiting:
                raise ValueError("Route refers to a non-waiting stage")
            stage = waiting[stage_key]
            key = stage.service, node
            index = decision.instance_slots.get(stage_key)
            if (
                self.scenario.services[stage.service].kind != "light"
                or isinstance(index, bool)
                or not isinstance(index, Integral)
                or not 0 <= index < decision.counts.get(key, 0)
            ):
                raise ValueError("Route needs a valid light-instance slot")
            if not isfinite(self.network.data_ready(stage, node)):
                raise ValueError("Cannot route to an unreachable node")
            reserved[key, index] += 1
            occupied = len(ordered[key][index].jobs) if index < len(ordered[key]) else 0
            if occupied + reserved[key, index] > decision.parallelism.get(key, 0):
                raise ValueError("Routes exceed the instance parallelism limit")
        validate_resources(self.scenario, {**self.placement.counts, **decision.counts})
        # Remove idle instances first, then reuse retained ones before creating.
        for key in sorted(old.keys() | decision.counts.keys()):
            target = decision.counts.get(key, 0)
            remove = max(0, old[key] - target)
            idle = [i for i in self.instances if (i.service, i.node) == key and not i.jobs]
            removed_ids = {i.id for i in idle[:remove]}
            self.instances = [i for i in self.instances if i.id not in removed_ids]
            for _ in range(max(0, target - old[key])):
                self._new_instance(*key)
            service = self.scenario.services[key[0]]
            self.costs["light_deployment"] += max(0, target - old[key]) * service.deployment_cost
            self.costs["light_maintenance"] += target * service.maintenance_cost
            # Follow Eq. (7): parallel price per deployed instance, not per job.
            self.costs["light_parallel"] += target * service.parallel_cost
        self.costs["core_maintenance"] += sum(
            count * self.scenario.services[s].maintenance_cost
            for (s, _), count in self.placement.counts.items()
        )
        counts = Counter(i.service for i in self.instances)
        for service, count in counts.items():
            self.peak_instances[service] = max(self.peak_instances[service], count)

    def _route(self, now: float, decision):
        remaining = []
        # Freeze the same busy-first slot ordering used during evaluation.
        slots = {}
        for key in decision.counts:
            slots[key] = sorted(
                (i for i in self.instances if (i.service, i.node) == key),
                key=lambda i: (not bool(i.jobs), i.id),
            )
        for stage in sorted(self.waiting, key=lambda s: (s.ready_ms, s.request.id, s.service)):
            service = self.scenario.services[stage.service]
            candidates = [i for i in self.instances if i.service == stage.service]
            if service.kind == "core":
                candidates = [
                    i
                    for i in candidates
                    if not i.jobs and isfinite(self.network.data_ready(stage, i.node))
                ]
                candidates.sort(
                    key=lambda i: self.core_candidate_key(stage, i)
                )
                if self.method == "lbrr" and candidates:
                    # Round-robin core scheduling for the LBRR baseline as well.
                    candidates.sort(key=lambda i: i.id)
                    index = self.controller.robin[stage.service] % len(candidates)
                    candidates = candidates[index:] + candidates[:index]
                    self.controller.robin[stage.service] += 1
            else:
                target = decision.routes.get(stage.key)
                cap = decision.parallelism.get((stage.service, target), 0)
                if target is None:
                    candidates = []
                else:
                    index = decision.instance_slots[stage.key]
                    candidates = [slots[stage.service, target][index]]
                    if len(candidates[0].jobs) >= cap:
                        raise RuntimeError("Evaluated and executed instance assignment disagree")
            if not candidates:
                remaining.append(stage)
                continue
            instance = candidates[0]
            data_ready = max(now, self.network.data_ready(stage, instance.node))
            instance.jobs.append(Job(stage, data_ready, service.work_mb))
        self.waiting = remaining

    def _complete(self, instance: Instance, job: Job, when: float):
        req, service = job.stage.request, job.stage.service
        if service in req.finished:
            raise RuntimeError("A DAG stage completed twice")
        req.finished[service] = (instance.node, when)
        if self.settings.record_trace:
            self.trace.append(
                {
                    "request": req.id,
                    "task": req.task,
                    "service": service,
                    "instance": instance.id,
                    "node": instance.node,
                    "ready_ms": job.stage.ready_ms,
                    "data_ready_ms": job.data_ready_ms,
                    "start_ms": job.start_ms,
                    "finish_ms": when,
                }
            )
        if set(req.finished) == set(self.scenario.tasks[req.task].dependencies):
            req.completion_ms = max(v[1] for v in req.finished.values())
            del self.active[req.id]

    def _advance(self, now: float):
        end = now + self.settings.slot_ms
        for instance in sorted(self.instances, key=lambda i: i.id):
            if not instance.jobs:
                continue
            service = self.scenario.services[instance.service]
            if service.kind == "core":
                job = instance.jobs[0]
                if job.start_ms is None:
                    job.start_ms = max(now, job.data_ready_ms)
                    job.finish_ms = job.start_ms + service.processing_ms
                if job.finish_ms <= end + 1e-10:
                    self._complete(instance, job, job.finish_ms)
                    instance.jobs.clear()
            else:
                # Aggregate FIFO work-budget model: parallelism
                # limits admitted jobs; it does not multiply compute capacity.
                rate = self.light_rate(instance, now)
                cursor = now
                while instance.jobs and cursor < end and rate > 0:
                    job = instance.jobs[0]
                    cursor = max(cursor, job.data_ready_ms)
                    if cursor >= end:
                        break
                    if job.start_ms is None:
                        job.start_ms = cursor
                    duration = job.remaining_work / rate
                    if cursor + duration <= end + 1e-10:
                        cursor = min(end, cursor + duration)
                        self._complete(instance, job, cursor)
                        instance.jobs.pop(0)
                    else:
                        job.remaining_work -= rate * (end - cursor)
                        break

    def light_rate(self, instance: Instance, now: float) -> float:
        """Physical service rate; default retains the original IID Gamma stream."""
        service = self.scenario.services[instance.service]
        return float(self.service_rng.gamma(service.gamma_shape, service.gamma_scale))

    def core_candidate_key(self, stage: Stage, instance: Instance):
        """Routing estimate, separate from the physical transfer completion."""
        return self.network.data_ready(stage, instance.node), instance.node, instance.id

    def run(self) -> dict:
        """Execute the model once and return in-memory diagnostics; write no files."""
        if self.has_run:
            raise RuntimeError("Create a fresh Simulator for each run")
        self.has_run = True
        steps = round(self.settings.duration_ms / self.settings.slot_ms)
        for step in range(steps):
            now = step * self.settings.slot_ms
            self._generate(now)
            self._expose_ready(now)
            decision = self.controller.step(now, self.instances, self.waiting, self.active)
            self._deploy(decision)
            self._route(now, decision)
            self._advance(now)
        return self.results()

    def results(self) -> dict:
        def summarize(requests):
            complete = [r for r in requests if r.completion_ms is not None]
            delays = [r.completion_ms - r.arrival_ms for r in complete]
            ontime = sum(
                r.completion_ms - r.arrival_ms <= self.scenario.tasks[r.task].deadline_ms + 1e-9
                for r in complete
            )
            total = len(requests)
            return {
                "arrivals": total,
                "completed": len(complete),
                "ontime": ontime,
                "pending": total - len(complete),
                "completion_rate": len(complete) / total if total else None,
                "ontime_rate": ontime / total if total else None,
                "late_completed": len(complete) - ontime,
                "mean_completed_latency_ms": float(np.mean(delays)) if delays else None,
                "p95_completed_latency_ms": float(np.percentile(delays, 95)) if delays else None,
            }

        result = {
            "method": self.method,
            "settings": asdict(self.settings),
            "scenario_profile": self.scenario.profile,
            "overall": summarize(self.requests),
            "tasks": {
                name: summarize([r for r in self.requests if r.task == name])
                for name in sorted(self.scenario.tasks)
            },
            "costs": {
                **self.costs,
                "total": sum(self.costs.values()),
                "per_slot": sum(self.costs.values())
                / (self.settings.duration_ms / self.settings.slot_ms),
            },
            "core_placement": [
                {"service": s, "node": n, "count": count}
                for (s, n), count in sorted(self.placement.counts.items())
                if count
            ],
            "core_objective": self.placement.objective,
            "core_nonzero_pairs": self.placement.nonzero_pairs,
            "core_solver": self.placement.solver,
            "arrival_sha256": self.arrival_hash.hexdigest(),
            "peak_instances": dict(self.peak_instances),
            "greedy_limit_hits": self.controller.greedy_limit_hits,
        }
        if self.settings.record_trace:
            result["stage_trace"] = self.trace
        return result
