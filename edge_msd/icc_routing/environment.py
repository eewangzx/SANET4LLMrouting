"""Shared-link, shared-node routing simulator using an ICC Scenario dataset.

Models are new surrogate alternatives, not ICC microservice identities. Requests
retain the ICC task type, payload and deadline. Their work proxy is derived from
the sum of the task's service work times. This aggregate execution does not
reproduce the original distributed DAG. Quality labels are explicitly synthetic.
"""

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from edge_msd.icc_paper import load_scenario
from edge_msd.model_routing.telemetry import NarrowbandChannel, pack_report, unpack_report


@dataclass(frozen=True)
class Config:
    slot_ms: float = 5.0
    arrival_steps: int = 80
    drain_steps: int = 30
    history: int = 8
    horizon: int = 20
    load_multiplier: float = 0.05
    report_bps: float = 64000.0
    report_period_ms: float = 20.0
    report_min_ms: float = 10.0
    heartbeat_ms: float = 80.0
    event_threshold: float = 0.2
    propagation_ms: float = 1.0
    encoder_ms: float = 0.2
    quality_min: float = 0.8
    max_queue: int = 32
    dynamic: bool = True
    service_process: str = "gamma_per_slot"

    def __post_init__(self):
        for key in ("slot_ms", "report_bps", "report_period_ms", "report_min_ms", "heartbeat_ms"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"Invalid {key}")
        for key in ("arrival_steps", "drain_steps", "history", "horizon", "max_queue"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"Invalid {key}")
        if (
            not 0 < self.quality_min < 1
            or not np.isfinite(self.load_multiplier)
            or self.load_multiplier < 0
        ):
            raise ValueError("Invalid quality or load")
        if self.service_process not in ("gamma_per_slot", "sampled_inverse_rate"):
            raise ValueError("Invalid service process")


DEFAULT_MODELS = [
    {
        "name": "small",
        "work_multiplier": 0.65,
        "cost": 0.15,
        "resources_cpu_gpu_ram_vram": [4, 4, 8, 8],
        "quality": [0.91, 0.73, 0.88, 0.65],
    },
    {
        "name": "medium",
        "work_multiplier": 1.0,
        "cost": 0.40,
        "resources_cpu_gpu_ram_vram": [8, 8, 16, 16],
        "quality": [0.95, 0.88, 0.94, 0.83],
    },
    {
        "name": "large",
        "work_multiplier": 1.60,
        "cost": 1.0,
        "resources_cpu_gpu_ram_vram": [16, 16, 32, 32],
        "quality": [0.98, 0.96, 0.97, 0.94],
    },
]


@dataclass(frozen=True)
class Request:
    id: int
    tick: int
    user: str
    origin: int
    task: int
    input_mb: float
    output_mb: float
    deadline_ms: float


@dataclass
class Job:
    request: Request
    model: int
    node: int
    expected_work: float
    work_left: float
    output_mb: float
    quality: float
    estimated_left: float
    path: tuple
    phase: str = "upload"
    bytes_left_mb: float = 0.0
    ready_ms: float = 0.0
    finish_ms: float | None = None
    segments: list = field(default_factory=list)
    segment_index: int = 0


class RoutingEnv:
    n_nodes, features = 3, 16

    def __init__(self, dataset, config=None, seed=1, codec=None, reporting="event", models=None):
        self.config = config or Config()
        self.dataset = str(dataset)
        self.scenario = load_scenario(dataset)
        self.dataset_hash = hashlib.sha256(Path(dataset).read_bytes()).hexdigest()
        self.models = json.loads(json.dumps(models or DEFAULT_MODELS))
        if len(self.models) != 3:
            raise ValueError("This experiment declares three surrogate models")
        self.nodes = sorted(n for n in self.scenario.nodes if n.startswith("es"))
        if len(self.nodes) != 3:
            raise ValueError("The ICC background must have three edge servers")
        for node in self.nodes:
            demand = np.sum([m["resources_cpu_gpu_ram_vram"] for m in self.models], axis=0)
            if np.any(demand > self.scenario.nodes[node].resources):
                raise ValueError(f"Resident models do not fit on {node}")
        self.tasks = list(self.scenario.tasks.values())
        self.task_names = [t.name for t in self.tasks]
        self.service_names = sorted(self.scenario.services)
        self.service_index = {s: i for i, s in enumerate(self.service_names)}
        self.base_work = np.array(
            [
                sum(self.scenario.services[s].mean_processing_ms for s in t.dependencies)
                for t in self.tasks
            ]
        )
        cpu = np.array([self.scenario.nodes[n].resources[0] for n in self.nodes])
        self.node_speed = cpu / np.median(cpu)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(3))
        self.edges, self.edge_rate = [], []
        for a, b, gbps, delay in self.scenario.links:
            if a in self.nodes and b in self.nodes:
                i, j = self.nodes.index(a), self.nodes.index(b)
                self.graph.add_edge(i, j, weight=8 / gbps)
                self.edges.extend([(i, j), (j, i)])
                self.edge_rate.extend([gbps / 8, gbps / 8])
        self.edge_rate = np.array(self.edge_rate)
        self.edge_index = {e: i for i, e in enumerate(self.edges)}
        self.out_edges = [[self.edge_index[n, j] for j in range(3) if j != n] for n in range(3)]
        self.paths = {
            (i, j): tuple(zip(p[:-1], p[1:]))
            for i in range(3)
            for j in range(3)
            for p in [nx.shortest_path(self.graph, i, j, weight="weight")]
        }
        if reporting not in ("none", "instant", "periodic", "event"):
            raise ValueError("Invalid reporting mode")
        self.codec, self.reporting, self.seed = codec, reporting, seed
        self.reset()

    def reset(self):
        c = self.config
        self.tick, self.batch_index = 0, 0
        self.total_steps = c.arrival_steps + c.drain_steps
        self.compute = [deque() for _ in range(3)]
        self.links = [deque() for _ in self.edges]
        self.jobs, self.completed, self.rejected = [], [], []
        self.compute_used = np.zeros(3)
        self.link_used = np.zeros(len(self.edges))
        self.admissions, self.finished = np.zeros(3), np.zeros(3)
        self.link_transmitted = np.zeros(len(self.edges))
        self.cost = 0.0
        self.history = np.zeros((3, c.history, 16), np.float32)
        self.archive = {}
        self.reports = [None] * 3
        self.last_report_ms = np.full(3, -np.inf)
        self.last_report_state = np.zeros((3, 16))
        self.channel = NarrowbandChannel(c.report_bps, c.propagation_ms)
        self.age_sum, self.age_count, self.missing_count = 0.0, 0, 0
        self._generate()
        self._sample()
        return self.observe()

    @property
    def now_ms(self):
        return self.tick * self.config.slot_ms

    @property
    def request(self):
        batch = self.arrivals.get(self.tick, [])
        return batch[self.batch_index] if self.batch_index < len(batch) else None

    def expected_work(self, request, model):
        return float(self.base_work[request.task] * self.models[model]["work_multiplier"])

    def _generate(self):
        c = self.config
        arng, nrng, qrng = [
            np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(3)
        ]
        self.arrivals, self.outcomes = {}, {}
        self.all_requests = []
        for tick in range(c.arrival_steps):
            batch = []
            for user in sorted(self.scenario.users, key=lambda u: u.name):
                origin = self.nodes.index(user.gateway)
                for t, task in enumerate(self.tasks):
                    count = arng.poisson(
                        user.rates_per_second[task.name] * c.load_multiplier * c.slot_ms / 1000
                    )
                    terminal = next(s for s in task.dependencies if task.graph.out_degree(s) == 0)
                    output = self.scenario.services[terminal].output_mb
                    for _ in range(count):
                        r = Request(
                            len(self.all_requests),
                            tick,
                            user.name,
                            origin,
                            t,
                            task.input_mb,
                            output,
                            task.deadline_ms,
                        )
                        batch.append(r)
                        self.all_requests.append(r)
                        # Same candidate outcomes for every routing algorithm, including unchosen models.
                        for m, profile in enumerate(self.models):
                            work = float(self.base_work[t])
                            if c.service_process == "sampled_inverse_rate":
                                work = 0.0
                                for name in task.dependencies:
                                    s = self.scenario.services[name]
                                    work += (
                                        s.processing_ms
                                        if s.kind == "core"
                                        else s.work_mb / qrng.gamma(s.gamma_shape, s.gamma_scale)
                                    )
                            quality = float(
                                np.clip(qrng.normal(profile["quality"][t], 0.025), 0, 1)
                            )
                            self.outcomes[r.id, m] = (
                                work * profile["work_multiplier"],
                                output,
                                quality,
                            )
            if batch:
                self.arrivals[tick] = batch
        length = self.total_steps + c.horizon + 1
        # This correlated resource process is our extension, not ICC's iid assumption.
        dims = 3 + len(self.edges)
        periods = nrng.uniform(100, 220, dims)
        phases = nrng.uniform(0, 2 * np.pi, dims)
        state, trace = np.zeros(dims), []
        for tick in range(length):
            state = 0.85 * state + nrng.normal(0, 0.035, dims)
            values = np.clip(
                0.65 + 0.30 * np.sin(2 * np.pi * tick * c.slot_ms / periods + phases) + state,
                0.12,
                1.2,
            )
            trace.append(values if c.dynamic else np.ones(dims))
        trace = np.asarray(trace, dtype=np.float32)
        self.compute_capacity, self.link_capacity = trace[:, :3], trace[:, 3:]
        # ICC's Gamma variable is a per-slot processing RATE, not a once-drawn duration.
        gamma_rng = np.random.default_rng(np.random.SeedSequence(self.seed).spawn(4)[3])
        self.gamma_factors = np.ones((length, 3, len(self.service_names)), np.float32)
        if c.service_process == "gamma_per_slot":
            for s, name in enumerate(self.service_names):
                service = self.scenario.services[name]
                if service.kind == "light":
                    self.gamma_factors[:, :, s] = (
                        gamma_rng.gamma(service.gamma_shape, service.gamma_scale, (length, 3))
                        / service.mean_rate
                    )
        # Predictor targets: node compute and all its outgoing directed links.
        self.targets = np.stack(
            [
                np.column_stack(
                    (self.compute_capacity[:, n], self.link_capacity[:, self.out_edges[n]])
                )
                for n in range(3)
            ],
            axis=1,
        )
        h = hashlib.sha256(self.dataset_hash.encode())
        h.update(trace.tobytes())
        if c.service_process == "gamma_per_slot":
            h.update(self.gamma_factors.tobytes())
        h.update(repr([asdict(r) for r in self.all_requests]).encode())
        h.update(repr(self.outcomes).encode())
        self.trace_hash = h.hexdigest()

    def _local_state(self, n):
        c = self.config
        jobs = [j for j in self.jobs if j.node == n and j.finish_ms is None]
        compute = self.compute[n]
        out = self.out_edges[n]
        backlog = sum(j.bytes_left_mb for e in out for j in self.links[e])
        slack = [j.request.tick * c.slot_ms + j.request.deadline_ms - self.now_ms for j in jobs]
        mix = [sum(j.request.task == t for j in jobs) / max(1, len(jobs)) for t in range(4)]
        return np.array(
            [
                self.link_capacity[self.tick, out].mean(),
                self.compute_capacity[self.tick, n],
                len(compute) / 20,
                sum(j.estimated_left for j in compute) / 100,
                backlog / 10,
                *self.link_capacity[self.tick, out],
                self.compute_used[n],
                self.link_used[out].mean(),
                min(slack, default=100) / 100,
                self.admissions[n] / 10,
                self.finished[n] / 10,
                *mix,
            ],
            np.float32,
        )

    def _sample(self):
        self.history[:, :-1] = self.history[:, 1:].copy()
        self.history[:, -1] = np.stack([self._local_state(n) for n in range(3)])
        self.archive[self.tick] = self.history.copy()
        if self.reporting == "none":
            return
        c = self.config
        for n in range(3):
            elapsed = self.now_ms - self.last_report_ms[n]
            change = np.max(np.abs(self.history[n, -1, :7] - self.last_report_state[n, :7]))
            due = (
                elapsed >= c.report_period_ms
                if self.reporting == "periodic"
                else (
                    elapsed >= c.report_min_ms
                    and (change >= c.event_threshold or elapsed >= c.heartbeat_ms)
                )
            )
            if self.reporting == "instant":
                due = True
            if not due:
                continue
            payload = self.codec.encode(self.history[n])
            packet = pack_report(n, self.tick, payload, self.codec.codec_id)
            if self.reporting == "instant":
                self._receive(packet, self.now_ms)
            else:
                self.channel.enqueue(n, packet, self.now_ms + c.encoder_ms)
            self.last_report_ms[n] = self.now_ms
            self.last_report_state[n] = self.history[n, -1]

    def _receive(self, packet, arrival_ms):
        n, tick, payload, codec_id = unpack_report(packet)
        if codec_id != self.codec.codec_id or tick * self.config.slot_ms > arrival_ms + 1e-8:
            raise ValueError("Invalid report")
        if self.reports[n] is None or tick > self.reports[n]["tick"]:
            self.reports[n] = {"tick": tick, "payload": payload.copy(), "arrival_ms": arrival_ms}

    def observe(self):
        # Only bytes that actually arrived are passed to deployed decision rules.
        return {
            "request": self.request,
            "reports": [
                None if r is None else {**r, "payload": r["payload"].copy()} for r in self.reports
            ],
            "now_ms": self.now_ms,
            "slot_ms": self.config.slot_ms,
            "quality_min": self.config.quality_min,
            "models": self.models,
            "base_work": self.base_work.copy(),
            "node_speed": self.node_speed.copy(),
            "edges": self.edges,
            "edge_rate": self.edge_rate.copy(),
            "paths": self.paths,
            "horizon": self.config.horizon,
        }

    def training_windows(self):
        """Trainer-only source windows matching RECEIVED reports, never fresh hidden state."""
        windows = np.zeros_like(self.history)
        futures = np.zeros((3, self.config.horizon, 3), np.float32)
        for n, report in enumerate(self.reports):
            if report is not None:
                t = report["tick"]
                windows[n] = self.archive[t][n]
                futures[n] = self.targets[t + 1 : t + 1 + self.config.horizon, n]
        return windows, futures

    def _put_network(self, job, now):
        if not job.path:
            if job.phase == "upload":
                job.phase = "compute"
                self.compute[job.node].append(job)
                job.ready_ms = now
            else:
                job.phase, job.finish_ms = "done", now
                self.completed.append(job)
                self.finished[job.node] += 1
            return
        edge = job.path[0]
        job.path = job.path[1:]
        job.bytes_left_mb = job.request.input_mb if job.phase == "upload" else job.output_mb
        job.ready_ms = now
        self.links[self.edge_index[edge]].append(job)

    def _admit(self, action):
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 9:
            raise ValueError("Action must choose one of nine model/node pairs")
        r = self.request
        n, m = divmod(int(action), 3)
        count = sum(j.node == n and j.finish_ms is None for j in self.jobs)
        if count >= self.config.max_queue:
            self.rejected.append(r.id)
            return -2.0
        work, output, quality = self.outcomes[r.id, m]
        expected = self.expected_work(r, m)
        job = Job(r, m, n, expected, work, output, quality, expected, self.paths[r.origin, n])
        if self.config.service_process == "gamma_per_slot":
            job.segments = [
                [
                    self.service_index[s],
                    self.scenario.services[s].mean_processing_ms
                    * self.models[m]["work_multiplier"],
                ]
                for s in self.tasks[r.task].order
            ]
        self.jobs.append(job)
        self.admissions[n] += 1
        self.cost += self.models[m]["cost"]
        self._put_network(job, self.now_ms)
        return -0.08 * self.models[m]["cost"]

    def _advance(self):
        """Event sweep inside a slot; links and compute each have ONE shared capacity."""
        c = self.config
        now, end = self.now_ms, self.now_ms + c.slot_ms
        self.compute_used[:], self.link_used[:] = 0, 0
        completed_before = len(self.completed)
        while now < end - 1e-9:
            resources = []
            for n, queue in enumerate(self.compute):
                if queue:
                    job = queue[0]
                    factor = (
                        self.gamma_factors[self.tick, n, job.segments[job.segment_index][0]]
                        if job.segments
                        else 1.0
                    )
                    resources.append(
                        (
                            "compute",
                            n,
                            job,
                            max(
                                1e-9,
                                self.node_speed[n] * self.compute_capacity[self.tick, n] * factor,
                            ),
                        )
                    )
            for e, queue in enumerate(self.links):
                if queue:
                    resources.append(
                        ("link", e, queue[0], self.edge_rate[e] * self.link_capacity[self.tick, e])
                    )
            if not resources:
                break
            dt = end - now
            for kind, index, job, rate in resources:
                remaining = (
                    (job.segments[job.segment_index][1] if job.segments else job.work_left)
                    if kind == "compute"
                    else job.bytes_left_mb
                )
                dt = min(dt, max(0, remaining) / rate)
            if dt < 1e-12:
                dt = 0.0
            for kind, index, job, rate in resources:
                if kind == "compute":
                    job.work_left = max(0, job.work_left - dt * rate)
                    job.estimated_left = max(0, job.estimated_left - dt * rate)
                    if job.segments:
                        job.segments[job.segment_index][1] = max(
                            0, job.segments[job.segment_index][1] - dt * rate
                        )
                    self.compute_used[index] += dt / c.slot_ms
                else:
                    sent = min(job.bytes_left_mb, dt * rate)
                    job.bytes_left_mb -= sent
                    self.link_transmitted[index] += sent
                    self.link_used[index] += dt / c.slot_ms
            now += dt
            progress = False
            for kind, index, job, rate in resources:
                if (
                    kind == "compute"
                    and job.segments
                    and job.segments[job.segment_index][1] <= 1e-9
                ):
                    if job.segment_index + 1 < len(job.segments):
                        job.segment_index += 1
                        progress = True
                        continue
                    job.work_left = 0.0
                if kind == "compute" and job.work_left <= 1e-9:
                    self.compute[index].popleft()
                    job.phase = "download"
                    job.path = self.paths[job.node, job.request.origin]
                    self._put_network(job, now)
                    progress = True
                elif kind == "link" and job.bytes_left_mb <= 1e-9:
                    self.links[index].popleft()
                    self._put_network(job, now)
                    progress = True
            if dt == 0 and not progress:
                raise RuntimeError("Event sweep did not advance")
        reward = 0.0
        for job in self.completed[completed_before:]:
            good = (
                job.finish_ms - job.request.tick * c.slot_ms <= job.request.deadline_ms
                and job.quality >= c.quality_min
            )
            reward += 3.0 if good else -1.5
        reward -= 0.005 * sum(j.finish_ms is None for j in self.jobs)
        return reward

    def step(self, action):
        if self.tick >= self.total_steps:
            raise RuntimeError("Episode has ended")
        request = self.request
        if (request is None) != (action is None):
            raise ValueError("One action per arrived request is required")
        reward = 0.0
        if request is not None:
            reward += self._admit(action)
            for r in self.reports:
                if r is None:
                    self.missing_count += 1
                else:
                    self.age_sum += self.now_ms - r["tick"] * self.config.slot_ms
                    self.age_count += 1
            self.batch_index += 1
            if self.request is not None:
                return self.observe(), reward, False, {"elapsed_ms": 0.0}
        reward += self._advance()
        self.tick, self.batch_index = self.tick + 1, 0
        for delivery in self.channel.advance(self.now_ms):
            self._receive(delivery.packet, delivery.arrival_ms)
        done = self.tick == self.total_steps
        if done:
            reward -= 2 * sum(j.finish_ms is None for j in self.jobs)
        else:
            self._sample()
        self.admissions[:], self.finished[:] = 0, 0
        return self.observe(), reward, done, {"elapsed_ms": self.config.slot_ms}

    def results(self):
        c = self.config
        latencies = [j.finish_ms - j.request.tick * c.slot_ms for j in self.completed]
        ontime = [t <= j.request.deadline_ms for t, j in zip(latencies, self.completed)]
        good = sum(ok and j.quality >= c.quality_min for ok, j in zip(ontime, self.completed))
        pending = sum(j.finish_ms is None for j in self.jobs)
        total = len(self.all_requests)
        assert total == len(self.completed) + len(self.rejected) + pending
        return {
            "arrivals": total,
            "completed": len(self.completed),
            "pending": pending,
            "rejected": len(self.rejected),
            "successful": int(good),
            "success_rate": good / max(1, total),
            "ontime_rate": sum(ontime) / max(1, total),
            "p95_completed_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
            "cost_per_arrival": self.cost / max(1, total),
            "report_transmitted_bytes": self.channel.transmitted_bytes,
            "report_bps_actual": self.channel.transmitted_bytes * 8000 / max(1, self.now_ms),
            "mean_state_age_ms": self.age_sum / max(1, self.age_count),
            "missing_fraction": self.missing_count / max(1, self.missing_count + self.age_count),
            "trace_hash": self.trace_hash,
        }
