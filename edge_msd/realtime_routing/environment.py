"""Fixed-deployment, node-selection-only environment with ICC physical semantics.

Reuses ``edge_msd.simulation.Simulator`` for arrivals, DAG exposure, instance
advancement and completion, so the physics stays the upstream baseline. The only
behavioral change: *both* core and light ready stages are dispatched by an explicit
node action drawn from a legal-node mask, instead of the upstream core-greedy /
light-controller decision. Deployment is frozen; actions cannot change it.

Causality: legal nodes are derived from the fixed placement, static reachability
and the controller's own occupancy ledger. No true remaining work, future transfer
finish or un-arrived report is read to build a mask.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import replace
from math import isfinite

import numpy as np

from edge_msd.models import Job
from edge_msd.simulation import Simulator


class ArrivalHash:
    """A pickle-friendly incremental SHA-256 (hashlib objects cannot be pickled).

    Keeps the accumulated bytes so a restored checkpoint resumes with an identical
    running digest; ``hexdigest`` is memoized until the next ``update``.
    """

    def __init__(self):
        self._data = bytearray()
        self._digest = hashlib.sha256().hexdigest()

    def update(self, data: bytes):
        self._data += data
        self._digest = None

    def hexdigest(self) -> str:
        if self._digest is None:
            self._digest = hashlib.sha256(bytes(self._data)).hexdigest()
        return self._digest

    def __getstate__(self):
        return bytes(self._data)

    def __setstate__(self, data):
        self._data = bytearray(data)
        self._digest = None


class RoutingEnvironment(Simulator):
    """One decision = one ready DAG stage routed to one node.

    ``step(node)`` advances no physical time when other routable stages remain in the
    same slot (``info['elapsed_ms'] == 0``); it advances exactly ``slot_ms`` otherwise.
    """

    def __init__(self, scenario, settings, placement, seed=None, drain_ms=None, requests=None):
        self.fixed = placement
        if seed is not None:
            settings = replace(settings, seed=int(seed))
        self.drain_ms = drain_ms
        # Deterministic injection for controlled examples/tests; each item is
        # (user, task, arrival_ms, uplink_ms). Empty by default (pure Poisson).
        self.requests_spec = [tuple(item) for item in requests] if requests else []
        super().__init__(scenario, settings, "propavg", core_counts=placement.core)
        self.reset()

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.scenario.nodes))

    @property
    def horizon_ms(self) -> float:
        deadline = max(t.deadline_ms for t in self.scenario.tasks.values())
        if self.drain_ms is not None:
            return self.settings.duration_ms + max(float(self.drain_ms), deadline)
        return self.settings.duration_ms + deadline

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None):
        if seed is not None:
            self.settings = replace(self.settings, seed=int(seed))
        streams = np.random.SeedSequence(self.settings.seed).spawn(3)
        self.arrivals, self.channels, self.service_rng = [
            np.random.default_rng(s) for s in streams
        ]
        self.instances = []
        self.next_instance_id = 0
        for (service, node), count in sorted(self.fixed.counts.items()):
            for _ in range(count):
                self._new_instance(service, node)
        self.active = {}
        self.requests = []
        self.waiting = []
        self.trace = []
        self.peak_instances = Counter()
        self.arrival_hash = ArrivalHash()
        self.costs = self._fixed_costs()
        self.settled: set[int] = set()
        self.now = 0.0
        self.terminated = False
        self._reward_acc = 0.0
        self._generate(0.0)
        for user, task, arrival_ms, uplink_ms in self.requests_spec:
            self.add_request(user, task, arrival_ms, uplink_ms)
        self._expose_ready(0.0)
        self._advance_until_decision()
        return self.observe()

    def _fixed_costs(self) -> dict:
        def deployed(counts):
            return sum(
                count * (
                    self.scenario.services[s].deployment_cost
                    + self.scenario.services[s].maintenance_cost
                )
                for (s, _), count in counts.items()
            )

        # Fixed deployment: these costs do not depend on any routing action.
        return {
            "core_deployment": deployed(self.fixed.core),
            "light_deployment": deployed(self.fixed.light),
            "core_maintenance": 0.0,
            "light_maintenance": 0.0,
            "light_parallel": 0.0,
        }

    # ------------------------------------------------------------- core step
    def step(self, node: str):
        if self.terminated:
            raise RuntimeError("Episode already terminated")
        stage = self._routable_stage()
        if stage is None:
            raise RuntimeError("No routable stage (waiting for a physical event)")
        legal = self.legal_nodes(stage)
        if node not in legal:
            raise ValueError(f"Illegal node {node!r} for {stage.service}; legal={legal}")
        self.waiting = [s for s in self.waiting if s is not stage]
        start_now = self.now
        self._assign(stage, node)
        self._reward_acc += self._settle()
        self._advance_until_decision()
        reward = self._reward_acc
        self._reward_acc = 0.0
        info = {
            "elapsed_ms": self.now - start_now,
            "raw_sla_reward": reward,
            "episode_time_ms": self.now,
            "legal_mask": tuple(1 if n in legal else 0 for n in self.node_ids),
            "sla_counts": self.sla_counts(),
            "telemetry_bytes": 0,  # Stage A: no compressed reporting yet.
            "arrival_sha256": self.arrival_hash.hexdigest(),
            "placement_hash": self.fixed.placement_hash,
            "scenario_hash": self.fixed.scenario_hash,
        }
        return self.observe(), reward, self.terminated, False, info

    # ------------------------------------------------------------- internals
    def _routable_stage(self):
        """First ready stage in fixed order that has a legal node, else None.

        Mirrors upstream `_route`: a stage with no admissible node stays waiting and
        later stages are still processed in the same pass. When no stage is routable
        the environment advances physical time until an event frees capacity.
        """
        for stage in sorted(self.waiting, key=lambda s: (s.ready_ms, s.request.id, s.service)):
            if self.legal_nodes(stage):
                return stage
        return None

    def legal_nodes(self, stage) -> list[str]:
        """Nodes that can accept this ready stage: static, ledger-based, causal."""
        service = self.scenario.services[stage.service]
        out = []
        for node in self.node_ids:
            instances = [
                i for i in self.instances if i.service == stage.service and i.node == node
            ]
            if not instances:
                continue
            if not isfinite(self.network.data_ready(stage, node)):
                continue
            if service.kind == "core":
                if any(not i.jobs for i in instances):  # core runs one job at a time
                    out.append(node)
            elif any(len(i.jobs) < self.settings.max_parallelism for i in instances):
                out.append(node)
        return out

    def _assign(self, stage, node: str):
        service = self.scenario.services[stage.service]
        instances = [i for i in self.instances if i.service == stage.service and i.node == node]
        if not instances:
            raise ValueError(f"No instance of {stage.service} at {node}")
        if service.kind == "core":
            pool = [i for i in instances if not i.jobs]
        else:
            pool = [i for i in instances if len(i.jobs) < self.settings.max_parallelism]
        if not pool:
            raise ValueError(f"No free admission at {node} for {stage.service}")
        instance = min(pool, key=lambda i: (len(i.jobs), i.id))
        data_ready = max(self.now, self.network.data_ready(stage, node))
        instance.jobs.append(Job(stage, data_ready, service.work_mb))

    def _advance_until_decision(self):
        while self._routable_stage() is None:
            self._advance(self.now)
            self.now += self.settings.slot_ms
            self._reward_acc += self._settle()
            if self.now >= self.horizon_ms - 1e-9:
                self.terminated = True
                self._reward_acc += self._settle_terminal()
                return
            if self.now < self.settings.duration_ms - 1e-9:
                self._generate(self.now)
            self._expose_ready(self.now)

    def _settle(self) -> float:
        """Each request settles exactly once: +1 on-time, -1 first violation."""
        reward = 0.0
        for req in self.requests:
            if req.id in self.settled:
                continue
            deadline = self.scenario.tasks[req.task].deadline_ms
            if req.completion_ms is not None:
                ontime = req.completion_ms - req.arrival_ms <= deadline + 1e-9
                reward += 1.0 if ontime else -1.0
                self.settled.add(req.id)
            elif self.now - req.arrival_ms > deadline + 1e-9:
                reward += -1.0
                self.settled.add(req.id)
        return reward

    def _settle_terminal(self) -> float:
        reward = 0.0
        for req in self.requests:
            if req.id not in self.settled:
                reward += -1.0
                self.settled.add(req.id)
        return reward

    # ----------------------------------------------------------- observation
    def sla_counts(self) -> tuple[int, int, int]:
        ontime = late = pending = 0
        for req in self.requests:
            deadline = self.scenario.tasks[req.task].deadline_ms
            if req.completion_ms is None:
                pending += 1
            elif req.completion_ms - req.arrival_ms <= deadline + 1e-9:
                ontime += 1
            else:
                late += 1
        return (ontime, late, pending)

    def observe(self) -> dict:
        nodes = self.node_ids
        stage = self._routable_stage()
        legal = set(self.legal_nodes(stage)) if stage is not None else set()
        mask = np.array([1 if n in legal else 0 for n in nodes], dtype=np.int8)
        ledger = np.zeros(len(nodes), dtype=np.float32)
        queue = np.zeros(len(nodes), dtype=np.float32)
        for index, node in enumerate(nodes):
            local = [i for i in self.instances if i.node == node]
            capacity = len(local) * self.settings.max_parallelism
            ledger[index] = sum(len(i.jobs) for i in local) / capacity if capacity else 0.0
            queue[index] = sum(s.request.gateway == node for s in self.waiting)
        if stage is None:
            current = {
                "present": 0,
                "service": "",
                "task": "",
                "age_ms": 0.0,
                "deadline_ms": 0.0,
                "slack_ms": 0.0,
            }
        else:
            req = stage.request
            task = self.scenario.tasks[req.task]
            age = self.now - req.arrival_ms
            current = {
                "present": 1,
                "service": stage.service,
                "task": req.task,
                "age_ms": age,
                "deadline_ms": task.deadline_ms,
                "slack_ms": task.deadline_ms - age,
            }
        return {
            "nodes": nodes,
            "legal_mask": mask,
            "current": current,
            "ledger": ledger,
            "queue": queue,
            "time_ms": np.float32(self.now),
        }
