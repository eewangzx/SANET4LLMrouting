"""A causal, closed-loop model-routing environment built beside the ICC simulator.

One optional request per physical slot; one endpoint action per request. Six
fixed model/node pairs. FIFO, non-preemptive inference; independent full-duplex
request links and a shared telemetry link. This is not a token/batching engine.
"""

import hashlib
from dataclasses import asdict

import numpy as np

from edge_msd.models import Node, Scenario, Service, TaskType, User
from edge_msd.network import Network
from edge_msd.placement import validate_core_budget, validate_resources

from .telemetry import NarrowbandChannel, RawCodec, pack_report, unpack_report
from .types import (
    DEFAULT_PROFILES,
    EndpointQueues,
    Job,
    ReceivedReport,
    RequestView,
    RoutingConfig,
)

FEATURES = (
    "link_capacity_ratio",
    "compute_capacity_ratio",
    "outstanding_count_div8",
    "upload_backlog_mb_div005",
    "estimated_compute_backlog_ms_div2000",
    "compute_busy",
    "minimum_deadline_slack_div2000",
    "new_assignments",
    "easy_count_div8",
    "medium_count_div8",
    "hard_count_div8",
    "completed_last_slot",
    "link_utilization",
    "compute_utilization",
    "urgent_fraction",
    "service_time_ema_div1000",
)


def make_infrastructure(profiles):
    services = {
        p.name: Service(
            p.name, "core", (1, 1, 2, p.vram), 0.001, 0, 0, 0, processing_ms=p.service_ms
        )
        for p in profiles
    }
    # These one-stage tasks only supply the original ICC topology validator.
    # RoutingRequest alternatives are defined separately, never as a DAG of all models.
    tasks = {p.name: TaskType(p.name, {p.name: []}, 2000, 0.01) for p in profiles}
    nodes = {"gateway": Node("gateway", (1, 0, 1, 0))}
    placement, links = {}, []
    for n in range(6):
        name = f"edge{n}"
        nodes[name] = Node(name, (4, 1, 8, 8))
        placement[profiles[n % 3].name, name] = 1
        links.append(("gateway", name, 0.002, 2.0))  # 2 Mbit/s full-duplex data link
    user = User("source", (0, 0), "gateway", {}, 1, 1)
    scenario = Scenario(services, tasks, nodes, links, [user], profile="synthetic_model_routing")
    validate_resources(scenario, placement)
    validate_core_budget(scenario, placement)
    return scenario


class RoutingEnv:
    n_endpoints = 6
    n_features = len(FEATURES)

    def __init__(self, config=None, codec=None, reporting="periodic", profiles=DEFAULT_PROFILES):
        self.config = config or RoutingConfig()
        if reporting not in ("periodic", "event", "instant", "none"):
            raise ValueError("Unknown reporting mode")
        if len(profiles) != 3:
            raise ValueError("The initial six-endpoint scenario needs three model profiles")
        self.profiles, self.reporting = tuple(profiles), reporting
        self.codec = codec or RawCodec(self.config.history, self.n_features)
        self.scenario = make_infrastructure(self.profiles)
        self.network = Network(self.scenario)
        self.propagation = np.array(
            [self.network.delay("gateway", f"edge{n}", 0) for n in range(self.n_endpoints)]
        )
        self.base_mb_per_ms = np.full(self.n_endpoints, 0.002 / 8.0)
        self.reset()

    def reset(self):
        c = self.config
        self.tick = 0
        self.total_steps = c.arrival_steps + c.drain_steps
        self.queues = [EndpointQueues() for _ in range(self.n_endpoints)]
        self.history = np.zeros((6, c.history, self.n_features), dtype=np.float32)
        self.reports = [None] * 6
        self._embeddings = [None] * 6
        self.last_report_ms = np.full(6, -np.inf)
        self.last_report_state = np.zeros((6, self.n_features), dtype=np.float32)
        self.channel = NarrowbandChannel(c.report_bps, c.report_propagation_ms)
        self.completed, self.rejected, self.all_jobs = [], [], []
        self.cost = 0.0
        self.assignments = np.zeros(6)
        self.completed_last = np.zeros(6)
        self.link_util = np.zeros(6)
        self.compute_util = np.zeros(6)
        self.service_ema = np.zeros(6)
        self.age_sum = 0.0
        self.age_samples = 0
        self.missing_samples = 0
        self.observation_samples = 0
        self.encoder_calls = 0
        self._generate_exogenous_trace()
        self._sample_and_report()
        return self.observe()

    @property
    def now_ms(self):
        return self.tick * self.config.slot_ms

    @property
    def request(self):
        return self.requests_by_tick.get(self.tick)

    def _generate_exogenous_trace(self):
        """All counterfactual draws fixed before actions; policies share identical traces."""
        c = self.config
        a_rng, n_rng, p_rng = [
            np.random.default_rng(s) for s in np.random.SeedSequence(c.seed).spawn(3)
        ]
        self.requests_by_tick, self.outcomes = {}, {}
        rid = 0
        for t in range(c.arrival_steps):
            if a_rng.random() >= c.arrival_probability:
                continue
            difficulty = int(a_rng.integers(3))
            req = RequestView(
                rid,
                t * c.slot_ms,
                difficulty,
                int(a_rng.choice([512, 1024, 2048, 4096])),
                int(a_rng.choice([128, 256, 512])),
                float((900, 1400, 2000)[difficulty]),
            )
            self.requests_by_tick[t] = req
            for m, profile in enumerate(self.profiles):
                output = max(16, int(req.predicted_output_tokens * p_rng.lognormal(-0.02, 0.2)))
                quality = float(
                    np.clip(
                        p_rng.normal(
                            profile.quality_by_difficulty[difficulty], profile.quality_std
                        ),
                        0,
                        1,
                    )
                )
                work = profile.service_ms * self.work_scale(req.input_tokens, output)
                self.outcomes[rid, m] = (output, quality, work)
            rid += 1
        length = self.total_steps + c.prediction_steps + 1
        # Markov regimes + correlated ramps, with unpredictable innovations.
        self.capacity = np.empty((length, 6, 2), dtype=np.float32)
        regime = n_rng.integers(0, 2, size=(6, 2))
        level = np.where(regime, 1.0, 0.22).astype(float)
        for t in range(length):
            flips = n_rng.random((6, 2)) < (c.slot_ms / 1400.0)
            regime = np.where(flips, 1 - regime, regime)
            target = np.where(regime, 1.0, 0.22)
            level = np.clip(
                0.75 * level + 0.25 * target + n_rng.normal(0, 0.025, (6, 2)), 0.12, 1.2
            )
            self.capacity[t] = level
        h = hashlib.sha256()
        h.update(self.capacity.tobytes())
        for t, r in self.requests_by_tick.items():
            h.update(repr((t, asdict(r))).encode())
        h.update(repr(self.outcomes).encode())
        self.trace_sha256 = h.hexdigest()

    @staticmethod
    def work_scale(input_tokens, output_tokens):
        return 0.35 + 0.35 * input_tokens / 2048 + 0.3 * output_tokens / 256

    def expected_work(self, request, endpoint):
        return self.profiles[endpoint % 3].service_ms * self.work_scale(
            request.input_tokens, request.predicted_output_tokens
        )

    def _local_features(self, n):
        q = self.queues[n]
        jobs = q.jobs
        count = len(jobs)
        slack = [j.request.arrival_ms + j.request.deadline_ms - self.now_ms for j in jobs]
        # Predicted remaining work uses public profile and observed service time,
        # never the job's hidden output length / remaining_work_ms.
        estimated_work = 0.0
        for j in q.upload + q.compute:
            predicted = self.expected_work(j.request, n)
            if j.compute_start_ms is not None:
                elapsed = max(0, self.now_ms - j.compute_start_ms)
                predicted = max(0, predicted - elapsed * self.capacity[self.tick, n, 1])
            estimated_work += predicted
        x = np.array(
            [
                *self.capacity[self.tick, n],
                count / 8,
                sum(j.remaining_input_mb for j in q.upload) / 0.05,
                estimated_work / 2000,
                float(bool(q.compute)),
                min(slack, default=2000) / 2000,
                self.assignments[n],
                *[sum(j.request.difficulty == d for j in jobs) / 8 for d in range(3)],
                self.completed_last[n],
                self.link_util[n],
                self.compute_util[n],
                sum(s < 300 for s in slack) / max(1, count),
                self.service_ema[n] / 1000,
            ],
            dtype=np.float32,
        )
        return np.clip(x, -2, 5)

    def _sample_and_report(self):
        self.history[:, :-1] = self.history[:, 1:].copy()
        self.history[:, -1] = np.stack([self._local_features(n) for n in range(6)])
        if self.reporting == "none":
            return
        c = self.config
        for n in range(6):
            elapsed = self.now_ms - self.last_report_ms[n]
            change = float(np.max(np.abs(self.history[n, -1, :5] - self.last_report_state[n, :5])))
            due = (
                elapsed >= c.report_period_ms - 1e-8
                if self.reporting == "periodic"
                else elapsed >= c.report_min_ms - 1e-8
                and (change >= c.event_threshold or elapsed >= c.heartbeat_ms - 1e-8)
            )
            if self.reporting == "instant":
                due = True
            if not due:
                continue
            payload = self.codec.encode(self.history[n])
            packet = pack_report(n, self.tick, payload, self.codec.codec_id)
            self.encoder_calls += 1
            if self.reporting == "instant":
                self._receive(packet, self.now_ms)
            else:
                self.channel.enqueue(n, packet, self.now_ms + c.encoder_ms)
            self.last_report_ms[n] = self.now_ms
            self.last_report_state[n] = self.history[n, -1]

    def _receive(self, packet, arrival_ms):
        n, tick, payload, codec_id = unpack_report(packet)
        if codec_id != self.codec.codec_id or n >= 6:
            raise ValueError("Unexpected codec or endpoint")
        sample_ms = tick * self.config.slot_ms
        if sample_ms > arrival_ms + 1e-8:
            raise ValueError("A report cannot arrive before it was sampled")
        old = self.reports[n]
        if old is not None and old.sample_ms >= sample_ms:
            return
        self.reports[n] = ReceivedReport(
            sample_ms, arrival_ms, payload, self.codec.decode(payload), self.codec.predict(payload)
        )
        self._embeddings[n] = self.codec.embedding(payload)

    def observe(self):
        """Only copied received information and public request/model metadata."""
        dim = getattr(self.codec, "latent_dim", self.codec.payload_dim)
        latent = np.zeros((6, dim), dtype=np.float32)
        state = np.zeros((6, self.n_features), dtype=np.float32)
        state[:, :2] = 0.6  # same unobserved prior for all methods
        age = np.full(6, 5000.0, dtype=np.float32)
        missing = np.ones(6, dtype=np.float32)
        forecast = np.zeros((6, self.config.prediction_steps, 2), dtype=np.float32)
        has_forecast = np.zeros(6, dtype=np.float32)
        for n, report in enumerate(self.reports):
            if report is None:
                continue
            latent[n] = self._embeddings[n]
            state[n] = report.decoded[-1]
            age[n] = self.now_ms - report.sample_ms
            missing[n] = 0
            if report.forecast is not None:
                forecast[n] = report.forecast
                has_forecast[n] = 1
        req = self.request
        public = np.zeros((6, 5), dtype=np.float32)
        query = np.zeros(7, dtype=np.float32)
        if req is not None:
            query[req.difficulty] = 1
            query[3:] = (
                req.input_tokens / 4096,
                req.predicted_output_tokens / 512,
                req.deadline_ms / 2000,
                self.config.quality_threshold,
            )
            for n in range(6):
                profile = self.profiles[n % 3]
                public[n] = (
                    profile.quality_by_difficulty[req.difficulty],
                    self.expected_work(req, n) / req.deadline_ms,
                    req.input_mb / self.base_mb_per_ms[n] / req.deadline_ms,
                    profile.cost,
                    self.propagation[n] / req.deadline_ms,
                )
        return {
            "latent": latent,
            "state": state,
            "age_ms": age,
            "missing": missing,
            "forecast": forecast,
            "has_forecast": has_forecast,
            "public": public,
            "query": query,
            "request": req,
            "slot_ms": self.config.slot_ms,
            "now_ms": self.now_ms,
        }

    def _admit(self, request, action):
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 6:
            raise ValueError("Action must identify one of the six deployed endpoints")
        if len(self.queues[action].jobs) >= self.config.max_queue:
            self.rejected.append(request.id)
            return -2.0
        m = action % 3
        output, quality, work = self.outcomes[request.id, m]
        cost = self.profiles[m].cost * output / 256
        job = Job(request, action, request.input_mb, work, output * 4 / 1e6, quality, cost)
        self.queues[action].upload.append(job)
        self.all_jobs.append(job)
        self.assignments[action] += 1
        self.cost += cost
        return -0.08 * cost

    def _advance_jobs(self):
        start, end = self.now_ms, self.now_ms + self.config.slot_ms
        self.completed_last[:] = 0
        reward = 0.0
        for n, q in enumerate(self.queues):
            rate = self.base_mb_per_ms[n] * self.capacity[self.tick, n, 0]
            speed = self.capacity[self.tick, n, 1]
            cursor = start
            while q.upload and cursor < end - 1e-9:
                job = q.upload[0]
                dt = min(end - cursor, job.remaining_input_mb / rate)
                job.remaining_input_mb = max(0, job.remaining_input_mb - rate * dt)
                cursor += dt
                if job.remaining_input_mb <= 1e-12:
                    q.upload.pop(0)
                    job.ready_ms = cursor + self.propagation[n]
                    q.compute.append(job)
            self.link_util[n] = (cursor - start) / self.config.slot_ms
            cursor, used = start, 0.0
            while q.compute and cursor < end - 1e-9:
                job = q.compute[0]
                cursor = max(cursor, job.ready_ms)
                if cursor >= end:
                    break
                if job.compute_start_ms is None:
                    job.compute_start_ms = cursor
                dt = min(end - cursor, job.remaining_work_ms / speed)
                job.remaining_work_ms = max(0, job.remaining_work_ms - speed * dt)
                cursor += dt
                used += dt
                if job.remaining_work_ms <= 1e-8:
                    q.compute.pop(0)
                    job.compute_finish_ms = cursor
                    job.ready_ms = cursor + self.propagation[n]
                    q.download.append(job)
                    duration = cursor - job.compute_start_ms
                    self.service_ema[n] = 0.8 * self.service_ema[n] + 0.2 * duration
            self.compute_util[n] = used / self.config.slot_ms
            cursor = start
            while q.download and cursor < end - 1e-9:
                job = q.download[0]
                cursor = max(cursor, job.ready_ms)
                if cursor >= end:
                    break
                dt = min(end - cursor, job.remaining_output_mb / rate)
                job.remaining_output_mb = max(0, job.remaining_output_mb - rate * dt)
                cursor += dt
                if job.remaining_output_mb <= 1e-12:
                    q.download.pop(0)
                    job.finish_ms = cursor
                    self.completed.append(job)
                    self.completed_last[n] += 1
                    good = (
                        job.quality >= self.config.quality_threshold
                        and cursor - job.request.arrival_ms <= job.request.deadline_ms
                    )
                    reward += 3.0 if good else -1.5
        reward -= 0.01 * sum(len(q.jobs) for q in self.queues)
        return reward

    def step(self, action=None):
        if self.tick >= self.total_steps:
            raise RuntimeError("Episode already ended")
        request = self.request
        if request is None and action is not None:
            raise ValueError("No request is available at this step")
        if request is not None and action is None:
            raise ValueError("An arrived request needs an endpoint action")
        reward = 0.0
        self.assignments[:] = 0
        if request is not None:
            reward += self._admit(request, action)
        if self.tick < self.config.arrival_steps:
            for report in self.reports:
                self.observation_samples += 1
                if report is None:
                    self.missing_samples += 1
                else:
                    self.age_sum += self.now_ms - report.sample_ms
                    self.age_samples += 1
        reward += self._advance_jobs()
        self.tick += 1
        for delivery in self.channel.advance(self.now_ms):
            self._receive(delivery.packet, delivery.arrival_ms)
        done = self.tick == self.total_steps
        if done:
            reward -= 2.0 * sum(len(q.jobs) for q in self.queues)
        else:
            self._sample_and_report()
        return self.observe(), float(reward), done, {}

    def results(self):
        total = len(self.requests_by_tick)
        latency = [j.finish_ms - j.request.arrival_ms for j in self.completed]
        ontime = int(sum(t <= j.request.deadline_ms for t, j in zip(latency, self.completed)))
        successful = int(
            sum(
                t <= j.request.deadline_ms and j.quality >= self.config.quality_threshold
                for t, j in zip(latency, self.completed)
            )
        )
        pending = sum(len(q.jobs) for q in self.queues)
        assert len(self.completed) + len(self.rejected) + pending == total
        return {
            "synthetic_profiles": True,
            "trace_sha256": self.trace_sha256,
            "arrivals": total,
            "completed": len(self.completed),
            "rejected": len(self.rejected),
            "pending": pending,
            "successful": successful,
            "ontime": ontime,
            "success_rate": successful / total if total else 0.0,
            "ontime_rate": ontime / total if total else 0.0,
            "mean_quality_completed": float(np.mean([j.quality for j in self.completed]))
            if self.completed
            else None,
            "p95_completed_latency_ms": float(np.percentile(latency, 95)) if latency else None,
            "cost_per_arrival": self.cost / total if total else 0.0,
            "report_generated_bytes": self.channel.generated_bytes,
            "report_transmitted_bytes": self.channel.transmitted_bytes,
            "report_transmitted_bps": self.channel.transmitted_bytes * 8000 / max(1, self.now_ms),
            "report_replacements": self.channel.replaced_reports,
            "mean_observation_age_ms": self.age_sum / max(1, self.age_samples),
            "missing_observation_fraction": self.missing_samples / max(1, self.observation_samples),
            "encoder_calls": self.encoder_calls,
        }
