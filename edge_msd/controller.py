"""Online light deployment, routing and parallelism through one shared interface.

Proposed and PropAvg differ only in DelayModel. LBRR and GA use the same
simulator and resource accounting, so changing a baseline cannot change physics.
"""

from collections import Counter, defaultdict
from math import isfinite

import numpy as np

from .capacity import DelayModel
from .config import Settings
from .models import Decision, Instance, Placement, Request, Scenario, Stage
from .network import Network
from .placement import resource_usage, validate_core_budget, validate_resources


class Controller:
    def __init__(
        self,
        scenario: Scenario,
        network: Network,
        settings: Settings,
        core_counts: Placement,
        method: str,
    ):
        if method not in ("proposed", "propavg", "lbrr", "ga"):
            raise ValueError(f"Unknown method: {method}")
        scenario.validate()
        validate_resources(scenario, core_counts)
        validate_core_budget(scenario, core_counts)
        if any(scenario.services[s].kind != "core" for s, _ in core_counts):
            raise ValueError("core_counts must contain only core services")
        self.scenario, self.network, self.settings, self.method = (
            scenario,
            network,
            settings,
            method,
        )
        self.delay_model = DelayModel(
            scenario.services,
            "effective" if method == "proposed" else "average",
            settings.epsilon,
            settings.slot_ms,
            settings.ec_admission_window_ms,
        )
        self.core_usage = resource_usage(scenario, core_counts)
        self.queue: dict[int, float] = {}
        self.rng = np.random.default_rng(np.random.SeedSequence([settings.seed, 3]))
        self.robin = defaultdict(int)
        self.greedy_limit_hits = 0

    def step(
        self,
        now: float,
        instances: list[Instance],
        waiting: list[Stage],
        active: dict[int, Request],
    ) -> Decision:
        # Eq. (18): one update per active task, even when two DAG branches wait.
        if not isfinite(now) or now < 0:
            raise ValueError("Control time must be finite and nonnegative")
        if hasattr(self, "now") and now <= self.now:
            raise ValueError("Controller.step must be called at strictly increasing times")
        if len({s.key for s in waiting}) != len(waiting):
            raise ValueError("A waiting stage cannot occur twice")
        self.queue = {
            rid: max(
                self.settings.queue_floor,
                self.queue.get(rid, self.settings.queue_floor)
                + now
                - req.arrival_ms
                - self.scenario.tasks[req.task].deadline_ms,
            )
            for rid, req in active.items()
            if req.arrival_ms <= now
        }
        light_instances = [
            i for i in instances if self.scenario.services[i.service].kind == "light"
        ]
        self.previous = Counter((i.service, i.node) for i in light_instances)
        self.busy = defaultdict(list)
        self.busy_ready = defaultdict(list)
        for instance in sorted(light_instances, key=lambda i: i.id):
            if instance.jobs:  # Includes reserved instances receiving data in transit.
                self.busy[instance.service, instance.node].append(len(instance.jobs))
                self.busy_ready[instance.service, instance.node].append(
                    max(job.data_ready_ms for job in instance.jobs)
                )
        self.base = {key: len(loads) for key, loads in self.busy.items()}
        self.waiting = sorted(
            (s for s in waiting if self.scenario.services[s.service].kind == "light"),
            key=lambda s: (s.ready_ms, s.request.id, s.service),
        )
        self.now = now
        if not self.waiting:
            return self.evaluate(self.base)[1]
        if self.method == "lbrr":
            return self._load_balance()
        if self.method == "ga":
            return self._genetic()
        return self._greedy()

    def feasible(self, counts: Placement) -> bool:
        if any(self.scenario.services[s].kind != "light" for s, _ in counts):
            return False
        if any(counts.get(key, 0) < count for key, count in self.base.items()):
            return False
        usage = resource_usage(self.scenario, counts)
        return all(
            np.all(
                usage[n] + self.core_usage[n] <= np.asarray(self.scenario.nodes[n].resources) + 1e-8
            )
            for n in self.scenario.nodes
        )

    def slot_cost(self, counts: Placement) -> float:
        return sum(
            self.scenario.services[s].deployment_cost * max(0, count - self.previous[s, n])
            + (self.scenario.services[s].maintenance_cost + self.scenario.services[s].parallel_cost)
            * count
            for (s, n), count in counts.items()
        )

    def evaluate(self, counts: Placement, round_robin: bool = False) -> tuple[float, Decision]:
        """Route each waiting stage without mutating queues or deployment state.

        Candidate loads include running jobs. Every unassigned stage incurs a
        finite deferral penalty; this prevents inf-inf and avoids rewarding a
        candidate that silently drops difficult requests from its objective.
        """
        if not self.feasible(counts):
            raise ValueError("Candidate must preserve busy instances and obey resource capacity")
        loads = {
            key: list(self.busy[key]) + [0] * (count - len(self.busy[key]))
            for key, count in counts.items()
            if count > 0
        }
        ready = {
            key: list(self.busy_ready[key]) + [self.now] * (count - len(self.busy[key]))
            for key, count in counts.items()
            if count > 0
        }
        routes, predictions, assignments = {}, {}, {}
        objective = self.settings.eta * self.slot_cost(counts)
        local_robin = self.robin.copy()
        for stage in self.waiting:
            task = self.scenario.tasks[stage.request.task]
            defer = self.settings.deferral_multiplier * task.deadline_ms + self.settings.slot_ms
            candidates = []
            for (service, node), slots in sorted(loads.items()):
                if service != stage.service:
                    continue
                index = min(range(len(slots)), key=lambda i: (slots[i], i))
                parallel = slots[index] + 1
                if parallel > self.settings.max_parallelism:
                    continue
                processing = self.processing_delay(service, node, parallel)
                arrival = self.network.data_ready(stage, node)
                delay = (
                    max(0.0, arrival - self.now, ready[service, node][index] - self.now)
                    + processing
                )
                if isfinite(delay):
                    candidates.append((delay, node, index))
            if candidates:
                if round_robin:
                    candidates.sort(key=lambda c: c[1])
                    selected = candidates[local_robin[stage.service] % len(candidates)]
                    local_robin[stage.service] += 1
                else:
                    selected = min(candidates)
                delay, node, index = selected
                if round_robin or delay < defer:
                    loads[stage.service, node][index] += 1
                    ready[stage.service, node][index] = max(
                        ready[stage.service, node][index], self.network.data_ready(stage, node)
                    )
                    routes[stage.key] = node
                    assignments[stage.key] = index
                else:
                    delay = defer
            else:
                delay = defer
        # Earlier assignments must be re-scored after later jobs change their
        # shared instance's occupancy or input-ready time. These are conservative
        # batch proxies, not per-job exact FIFO completion predictions.
        for stage in self.waiting:
            task = self.scenario.tasks[stage.request.task]
            delay = self.settings.deferral_multiplier * task.deadline_ms + self.settings.slot_ms
            if stage.key in routes:
                key = stage.service, routes[stage.key]
                index = assignments[stage.key]
                delay = max(0, ready[key][index] - self.now) + self.processing_delay(
                    stage.service, key[1], loads[key][index]
                )
                predictions[stage.key] = delay
            if self.method == "ga":
                predicted_e2e = self.now - stage.request.arrival_ms + delay
                # Finite fitness: cost + per-stage predicted deadline excess.
                objective += task.priority * (delay + max(0, predicted_e2e - task.deadline_ms))
            else:
                objective += task.priority * self.queue[stage.request.id] * delay
        parallelism = {key: max(slots, default=0) for key, slots in loads.items()}
        decision = Decision(dict(counts), parallelism, routes, predictions, assignments)
        if round_robin:
            self.robin = local_robin
        return objective, decision

    def processing_delay(self, service: str, node: str, parallelism: int) -> float:
        """Scheduling estimate; extensions may use a node's received telemetry."""
        return self.delay_model.delay(service, parallelism)

    def _greedy(self) -> Decision:
        counts = dict(self.base)
        objective, decision = self.evaluate(counts)
        keys = [
            (s, n)
            for s in sorted({s.service for s in self.waiting})
            for n in sorted(self.scenario.nodes)
        ]
        for _ in range(self.settings.max_greedy_steps):
            best = None
            for key in keys:
                candidate = counts.copy()
                candidate[key] = candidate.get(key, 0) + 1
                if not self.feasible(candidate):
                    continue
                value, trial = self.evaluate(candidate)
                if value < objective - 1e-9 and (best is None or value < best[0] - 1e-9):
                    best = value, candidate, trial
            if best is None:
                return decision
            # Re-evaluate all marginal gains after each accepted deployment.
            objective, counts, decision = best
        self.greedy_limit_hits += 1
        return decision

    def _load_balance(self) -> Decision:
        counts = dict(self.base)
        needed = Counter(s.service for s in self.waiting)
        for service, waiting_count in sorted(needed.items()):
            occupied = sum(sum(v) for (s, _), v in self.busy.items() if s == service)
            slots = sum(v for (s, _), v in counts.items() if s == service)
            free = slots * self.settings.max_parallelism - occupied
            while free < waiting_count:
                used = resource_usage(self.scenario, counts)
                candidates = []
                for node in self.scenario.nodes:
                    trial = counts.copy()
                    trial[service, node] = trial.get((service, node), 0) + 1
                    if self.feasible(trial):
                        utilization = np.mean(
                            (used[node] + self.core_usage[node])
                            / np.maximum(self.scenario.nodes[node].resources, 1)
                        )
                        candidates.append((float(utilization), node))
                if not candidates:
                    break
                node = min(candidates)[1]
                counts[service, node] = counts.get((service, node), 0) + 1
                free += self.settings.max_parallelism
        return self.evaluate(counts, round_robin=True)[1]

    def _genetic(self) -> Decision:
        """Feasible integer populations; elite retention and bounded generation.

        Shared busy-instance lower bounds eliminate the old invalid randint
        ranges and empty-population/infinite-crossover failure modes.
        """
        keys = [
            (s, n)
            for s in sorted({s.service for s in self.waiting})
            for n in sorted(self.scenario.nodes)
        ]

        def repair(candidate):
            result = self.base.copy()
            for index in self.rng.permutation(len(keys)):
                key = keys[index]
                extra = max(0, candidate.get(key, 0) - self.base.get(key, 0))
                for _ in range(extra):
                    trial = result.copy()
                    trial[key] = trial.get(key, 0) + 1
                    if self.feasible(trial):
                        result = trial
                    else:
                        break
            return result

        population = [dict(self.base)]
        while len(population) < self.settings.ga_population:
            candidate = {
                key: self.base.get(key, 0)
                + int(self.rng.integers(0, self.settings.ga_max_instances + 1))
                for key in keys
            }
            population.append(repair(candidate))
        for _ in range(self.settings.ga_generations):
            population.sort(key=lambda p: self.evaluate(p)[0])
            elites = population[: max(2, self.settings.ga_population // 2)]
            new_population = [p.copy() for p in elites]
            while len(new_population) < self.settings.ga_population:
                a, b = self.rng.choice(len(elites), size=2, replace=False)
                point = int(self.rng.integers(len(keys) + 1))
                child = {
                    key: elites[a if index < point else b].get(key, 0)
                    for index, key in enumerate(keys)
                }
                for key in keys:
                    if self.rng.random() < self.settings.ga_mutation:
                        child[key] = self.base.get(key, 0) + int(
                            self.rng.integers(0, self.settings.ga_max_instances + 1)
                        )
                new_population.append(repair(child))
            population = new_population
        return min((self.evaluate(p) for p in population), key=lambda item: item[0])[1]
