"""Equivalent per-control-step caches for the original ICC controller.

No scheduling period, candidate set, objective or physical model is changed.
Caches expire before each step, after which the original controller constructs
the current busy-instance lower bounds and waiting-stage state as usual.
"""

from math import isfinite

import numpy as np

from edge_msd.controller import Controller
from edge_msd.models import Decision


class FastController(Controller):
    def __init__(self, scenario, network, settings, core_counts, method):
        super().__init__(scenario, network, settings, core_counts, method)
        self._node_names = tuple(scenario.nodes)
        self._node_index = {name: index for index, name in enumerate(self._node_names)}
        self._light_keys = tuple(
            (service, node)
            for service in scenario.services
            if scenario.services[service].kind == "light"
            for node in self._node_names
        )
        self._key_index = {key: index for index, key in enumerate(self._light_keys)}
        # A count vector times this constant matrix gives CPU/GPU/RAM/VRAM use
        # at every node. Resources and static core placement are immutable for
        # the lifetime of a controller; dynamic *rates* are handled separately.
        self._resource_matrix = np.zeros((len(self._light_keys), 4 * len(self._node_names)))
        for index, (service, node) in enumerate(self._light_keys):
            offset = 4 * self._node_index[node]
            self._resource_matrix[index, offset : offset + 4] = scenario.services[service].resources
        self._remaining_capacity = np.concatenate(
            [
                np.asarray(scenario.nodes[node].resources, dtype=float) - self.core_usage[node]
                for node in self._node_names
            ]
        )
        self._feasible_cache = {}
        self._evaluation_cache = {}
        self._processing_cache = {}
        self._stage_ready_cache = {}
        self._stage_rows = None
        self.cache_statistics = {
            "feasible_hits": 0,
            "feasible_misses": 0,
            "evaluation_hits": 0,
            "evaluation_misses": 0,
        }

    @staticmethod
    def _counts_key(counts):
        return tuple(sorted(counts.items()))

    @staticmethod
    def _copy_decision(decision):
        # All keys and values in these mappings are immutable scalars/tuples.
        # Copy each mapping so a caller cannot mutate the cached decision.
        return Decision(
            dict(decision.counts),
            dict(decision.parallelism),
            dict(decision.routes),
            dict(decision.predicted_ms),
            dict(decision.instance_slots),
        )

    def step(self, now, instances, waiting, active):
        self._feasible_cache.clear()
        self._evaluation_cache.clear()
        self._processing_cache.clear()
        self._stage_ready_cache.clear()
        self._stage_rows = None
        return super().step(now, instances, waiting, active)

    def feasible(self, counts):
        # Keep original rejection order, including busy lower bounds. Validate
        # counts before caching: bool(1) and int(1) otherwise share a cache key.
        if any(self.scenario.services[service].kind != "light" for service, _ in counts):
            return False
        if any(counts.get(key, 0) < count for key, count in self.base.items()):
            return False
        for key, count in counts.items():
            if key not in self._key_index:
                return super().feasible(counts)
            if (
                isinstance(count, (bool, np.bool_))
                or not isinstance(count, (int, np.integer))
                or count < 0
            ):
                raise ValueError("Instance counts must be nonnegative integers")
        key = self._counts_key(counts)
        if key in self._feasible_cache:
            self.cache_statistics["feasible_hits"] += 1
            return self._feasible_cache[key]
        vector = np.zeros(len(self._light_keys))
        for item, count in counts.items():
            vector[self._key_index[item]] = count
        feasible = bool(np.all(vector @ self._resource_matrix <= self._remaining_capacity + 1e-8))
        self._feasible_cache[key] = feasible
        self.cache_statistics["feasible_misses"] += 1
        return feasible

    def evaluate(self, counts, round_robin=False):
        if round_robin:
            # The original round-robin evaluator updates its routing pointer.
            return super().evaluate(counts, round_robin=True)
        # Validation still runs on a cache hit so an invalid int/bool alias
        # cannot retrieve a previously evaluated valid candidate.
        if not self.feasible(counts):
            raise ValueError("Candidate must preserve busy instances and obey resource capacity")
        return self._evaluate_feasible(counts)

    def _evaluate_feasible(self, counts):
        """Evaluate a candidate whose resource and busy bounds are checked."""
        key = self._counts_key(counts)
        if key in self._evaluation_cache:
            self.cache_statistics["evaluation_hits"] += 1
            objective, decision = self._evaluation_cache[key]
            return objective, self._copy_decision(decision)
        objective, decision = self._evaluate_candidate(counts)
        self._evaluation_cache[key] = objective, self._copy_decision(decision)
        self.cache_statistics["evaluation_misses"] += 1
        return objective, decision

    def _greedy(self):
        # Preserve polymorphism for callers that deliberately replace evaluate.
        if type(self).evaluate is not FastController.evaluate:
            return super()._greedy()
        counts = dict(self.base)
        objective, decision = self.evaluate(counts)
        vector = np.zeros(len(self._light_keys))
        for key, count in counts.items():
            vector[self._key_index[key]] = count
        used = (vector @ self._resource_matrix).reshape(-1, 4)
        capacity = self._remaining_capacity.reshape(-1, 4)
        keys = [
            (service, node)
            for service in sorted({stage.service for stage in self.waiting})
            for node in sorted(self.scenario.nodes)
        ]
        demands = {service: np.asarray(self.scenario.services[service].resources)
                   for service in {key[0] for key in keys}}
        for _ in range(self.settings.max_greedy_steps):
            best = None
            for key in keys:
                node_index = self._node_index[key[1]]
                trial_used = used[node_index]+demands[key[0]]
                if not np.all(trial_used <= capacity[node_index]+1e-8):
                    continue
                # One nonnegative integer addition to a feasible deployment
                # preserves every busy lower bound and all other nodes' use.
                candidate = counts.copy()
                candidate[key] = candidate.get(key, 0)+1
                value, trial = self._evaluate_feasible(candidate)
                if value < objective-1e-9 and (best is None or value < best[0]-1e-9):
                    best = value, candidate, trial, key, trial_used
            if best is None:
                return decision
            objective, counts, decision, selected, trial_used = best
            used[self._node_index[selected[1]]] = trial_used
        self.greedy_limit_hits += 1
        return decision

    def _processing(self, service, node, parallelism):
        key = service, node, parallelism
        if key not in self._processing_cache:
            self._processing_cache[key] = self.processing_delay(service, node, parallelism)
        return self._processing_cache[key]

    def _stage_ready(self, stage, stage_key, node):
        key = stage_key, node
        if key not in self._stage_ready_cache:
            self._stage_ready_cache[key] = self.network.data_ready(stage, node)
        return self._stage_ready_cache[key]

    def _evaluate_candidate(self, counts):
        """Original non-round-robin evaluator with equivalent indexed lookups.

        Requests, observed resource predictions and settings are fixed within
        one controller step. Grouping by service removes scans of unrelated
        instances; the final shared-occupancy rescore and objective order are
        unchanged. Counts retain their input order for cost accounting.
        """
        loads = {
            key: list(self.busy[key]) + [0] * (count-len(self.busy[key]))
            for key, count in counts.items() if count > 0
        }
        ready = {
            key: list(self.busy_ready[key]) + [self.now] * (count-len(self.busy[key]))
            for key, count in counts.items() if count > 0
        }
        by_service = {}
        for key in sorted(loads):
            by_service.setdefault(key[0], []).append(key)
        if self._stage_rows is None:
            self._stage_rows = []
            for stage in self.waiting:
                task = self.scenario.tasks[stage.request.task]
                defer = self.settings.deferral_multiplier*task.deadline_ms + self.settings.slot_ms
                self._stage_rows.append((stage, stage.key, task, defer))
        routes, predictions, assignments = {}, {}, {}
        objective = self.settings.eta*self.slot_cost(counts)
        for stage, stage_key, task, defer in self._stage_rows:
            selected = None
            for key in by_service.get(stage.service, ()):
                slots = loads[key]
                # The first minimum is the same (occupancy, index) tie-break.
                index = slots.index(min(slots))
                parallel = slots[index]+1
                if parallel > self.settings.max_parallelism:
                    continue
                node = key[1]
                processing = self._processing(stage.service, node, parallel)
                arrival = self._stage_ready(stage, stage_key, node)
                delay = max(0.0, arrival-self.now, ready[key][index]-self.now)+processing
                if isfinite(delay):
                    candidate = delay, node, index
                    if selected is None or candidate < selected:
                        selected = candidate
            if selected is not None:
                delay, node, index = selected
                if delay < defer:
                    key = stage.service, node
                    loads[key][index] += 1
                    ready[key][index] = max(ready[key][index], self._stage_ready(stage, stage_key, node))
                    routes[stage_key] = node
                    assignments[stage_key] = index
        # Earlier requests must see the final occupancy chosen for the entire
        # batch, including later requests sharing their instance.
        for stage, stage_key, task, defer in self._stage_rows:
            delay = defer
            if stage_key in routes:
                key = stage.service, routes[stage_key]
                index = assignments[stage_key]
                delay = max(0, ready[key][index]-self.now) + self._processing(
                    stage.service, key[1], loads[key][index])
                predictions[stage_key] = delay
            if self.method == "ga":
                predicted_e2e = self.now-stage.request.arrival_ms+delay
                objective += task.priority*(delay+max(0, predicted_e2e-task.deadline_ms))
            else:
                objective += task.priority*self.queue[stage.request.id]*delay
        parallelism = {key: max(slots, default=0) for key, slots in loads.items()}
        return objective, Decision(dict(counts), parallelism, routes, predictions, assignments)
