"""Physical transmission and DAG data readiness: equations (1), (2), (4)."""

from collections import Counter
from math import dist, isfinite, log, log1p, pi

import networkx as nx

from .models import Scenario, Stage, User


class Network:
    def __init__(self, scenario: Scenario, wireless: bool = False):
        scenario.validate()
        if wireless and scenario.radio is None:
            raise ValueError("Wireless operation requires explicit Scenario.radio parameters")
        self.scenario = scenario
        self.wireless = wireless
        self.graph = nx.Graph()
        self.graph.add_nodes_from(scenario.nodes)
        for a, b, gbps, propagation_ms in scenario.links:
            self.graph.add_edge(a, b, ms_per_mb=8.0 / gbps, propagation_ms=propagation_ms)
        self.user_counts = Counter(u.gateway for u in scenario.users)
        self._cache = {}

    def delay(self, source: str, target: str, data_mb: float) -> float:
        if not isfinite(data_mb) or data_mb < 0:
            raise ValueError("Transfer size must be finite and nonnegative")
        key = source, target, data_mb
        if key not in self._cache:
            self._cache[key] = self._delay(source, target, data_mb)
        return self._cache[key]

    def _delay(self, source: str, target: str, data_mb: float) -> float:
        if source not in self.graph or target not in self.graph:
            raise ValueError("Unknown network node")
        if source == target:
            return 0.0
        try:
            return nx.shortest_path_length(
                self.graph,
                source,
                target,
                weight=lambda a, b, d: data_mb * d["ms_per_mb"] + d["propagation_ms"],
            )
        except nx.NetworkXNoPath:
            return float("inf")

    def uplink_delay(self, user: User, input_mb: float, rng=None) -> float:
        """Nakagami power gain; static scoring uses mean gain when rng is None.

        Gateway bandwidth is specified in Hz and shared equally among users.
        Noise is total power over the allocated bandwidth, not a density.
        """
        if not isfinite(input_mb) or input_mb < 0:
            raise ValueError("Input size must be finite and nonnegative")
        if not self.wireless or input_mb == 0:
            return 0.0
        node = self.scenario.nodes[user.gateway]
        if node.uplink_bandwidth_hz <= 0 or node.position is None:
            return float("inf")
        gain = (
            user.omega
            if rng is None
            else float(rng.gamma(user.nakagami_m, user.omega / user.nakagami_m))
        )
        radio = self.scenario.radio
        distance = max(radio.minimum_distance_m, dist(user.position, node.position))
        path_gain = (3e8 / radio.carrier_hz / (4 * pi * distance)) ** 2
        snr = 10 ** ((radio.transmit_dbm - radio.noise_dbm) / 10) * gain * path_gain
        rate_gbps = (
            node.uplink_bandwidth_hz / 1e9 / self.user_counts[user.gateway] * log1p(snr) / log(2)
        )
        return 8 * input_mb / rate_gbps if rate_gbps > 0 else float("inf")

    def data_ready(self, stage: Stage, target: str) -> float:
        """All predecessors must have finished AND delivered their outputs."""
        req = stage.request
        parents = self.scenario.tasks[req.task].dependencies[stage.service]
        if not parents:
            input_mb = self.scenario.services[stage.service].input_mb
            if input_mb is None:
                input_mb = self.scenario.tasks[req.task].input_mb
            return req.uplink_ready_ms + self.delay(req.gateway, target, input_mb)
        return max(
            req.finished[p][1]
            + self.delay(req.finished[p][0], target, self.scenario.services[p].output_mb)
            for p in parents
        )
