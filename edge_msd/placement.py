"""Static core placement: network-aware scores, integer capacity and diversity.

SciPy's bundled HiGHS solver avoids the old hard-coded GLPK/CBC paths.
"""

from dataclasses import dataclass
from math import ceil, floor, isfinite

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .config import Settings
from .models import Placement, Scenario
from .network import Network


def resource_usage(scenario: Scenario, counts: Placement) -> dict[str, np.ndarray]:
    usage = {name: np.zeros(4) for name in scenario.nodes}
    for (service, node), count in counts.items():
        if node not in scenario.nodes or service not in scenario.services:
            raise ValueError(f"Unknown placement key: {(service, node)}")
        if (
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            or count < 0
        ):
            raise ValueError("Instance counts must be nonnegative integers")
        usage[node] += count * np.asarray(scenario.services[service].resources)
    return usage


def validate_resources(scenario: Scenario, counts: Placement):
    for node, used in resource_usage(scenario, counts).items():
        if np.any(used > np.asarray(scenario.nodes[node].resources) + 1e-8):
            raise ValueError(f"Resource capacity exceeded at {node}: {used}")


def validate_core_budget(scenario: Scenario, counts: Placement):
    for node, used in resource_usage(scenario, counts).items():
        if np.any(used > np.asarray(scenario.nodes[node].core_budget) + 1e-8):
            raise ValueError(f"Core placement consumes reserved light resources at {node}")


def core_requirements(scenario: Scenario) -> dict[str, int]:
    """Mean concurrent demand lambda * processing time; at least one per service.

    Unlike the old D=ones override, this bound responds to the configured load.
    Queue stability requires headroom in practice; this is only a mean bound.
    """
    return {
        name: max(
            1,
            ceil(
                sum(
                    user.rates_per_second.get(task.name, 0)
                    / 1000
                    * scenario.services[name].processing_ms
                    for user in scenario.users
                    for task in scenario.tasks.values()
                    if name in task.dependencies
                )
            ),
        )
        for name in scenario.core
    }


def qos_scores(
    scenario: Scenario, network: Network, settings: Settings
) -> dict[tuple[str, str], float]:
    """Eqs. (4), (15): optimistic mean-delay DAG embedding, then load allocation.

    Predecessors may choose different feasible nodes before their join. Resource
    contention is excluded from this statistical score and enforced by the MILP.
    """
    names = sorted(scenario.nodes)
    load = {(s, n): 0.0 for s in scenario.core for n in names}
    urgency = dict.fromkeys(load, 0.0)
    for user in scenario.users:
        for task in scenario.tasks.values():
            rate = user.rates_per_second.get(task.name, 0) * settings.slot_ms / 1000
            if rate == 0:
                continue
            prefix, finish = {}, {}
            for service_name in task.order:
                service = scenario.services[service_name]
                parents = task.dependencies[service_name]
                for node in names:
                    feasible = all(
                        a <= b
                        for a, b in zip(
                            service.resources,
                            scenario.nodes[node].core_budget
                            if service.kind == "core"
                            else scenario.nodes[node].resources,
                        )
                    )
                    if not feasible:
                        prefix[service_name, node] = finish[service_name, node] = float("inf")
                        continue
                    if not parents:
                        root_input = task.input_mb if service.input_mb is None else service.input_mb
                        before = network.uplink_delay(user, task.input_mb) + network.delay(
                            user.gateway, node, root_input
                        )
                    else:
                        before = max(
                            min(
                                finish[parent, src]
                                + network.delay(src, node, scenario.services[parent].output_mb)
                                for src in names
                            )
                            for parent in parents
                        )
                    prefix[service_name, node] = before
                    finish[service_name, node] = before + service.mean_processing_ms
            for service_name in set(task.dependencies) & set(scenario.core):
                finite = [n for n in names if isfinite(prefix[service_name, n])]
                if not finite:
                    raise ValueError(f"No reachable feasible location for {service_name}")
                minimum = min(prefix[service_name, n] for n in finite)
                weights = {
                    n: np.exp(-settings.decay * (prefix[service_name, n] - minimum)) for n in finite
                }
                total = sum(weights.values())
                suffix = sum(
                    scenario.services[s].mean_processing_ms
                    for s in nx.descendants(task.graph, service_name)
                )
                # A terminal core has no descendants: use one slot as reference budget.
                suffix = max(suffix, settings.slot_ms)
                for node in finite:
                    load[service_name, node] += rate * weights[node] / total
                    ratio = (task.deadline_ms - finish[service_name, node]) / suffix
                    urgency[service_name, node] += max(ratio, settings.urgency_floor)
    return {key: load[key] * urgency[key] for key in load}


@dataclass
class PlacementResult:
    counts: Placement
    objective: float
    nonzero_pairs: int
    solver: str


def place_core(
    scenario: Scenario, network: Network, settings: Settings, method: str = "proposed"
) -> PlacementResult:
    scenario.validate()
    if method not in ("proposed", "propavg", "lbrr", "ga"):
        raise ValueError(f"Unknown method: {method}")
    if method in ("lbrr", "ga"):
        return _least_loaded_core(scenario)
    scores = qos_scores(scenario, network, settings)
    keys = sorted(scores)
    count = len(keys)
    if count == 0:
        return PlacementResult({}, 0.0, 0, "no-core-services")
    # Tight, resource-derived big-M for x <= M*b; x >= b enforces integrality.
    upper = np.array(
        [
            min(
                floor(cap / demand + 1e-10)
                for cap, demand in zip(
                    scenario.nodes[n].core_budget, scenario.services[s].resources
                )
                if demand > 0
            )
            for s, n in keys
        ]
    )
    cost = np.array(
        [
            scenario.services[s].deployment_cost
            + scenario.services[s].maintenance_cost
            - settings.alpha * scores[s, n]
            for s, n in keys
        ]
    )
    c = np.r_[cost, np.zeros(count)]
    rows, lower, limits = [], [], []

    def add(coefficients, low=-np.inf, high=np.inf):
        rows.append(coefficients)
        lower.append(low)
        limits.append(high)

    for node in scenario.nodes:
        for resource in range(4):
            row = np.zeros(2 * count)
            for i, (s, n) in enumerate(keys):
                if n == node:
                    row[i] = scenario.services[s].resources[resource]
            add(row, high=scenario.nodes[node].core_budget[resource])
    for service, required in core_requirements(scenario).items():
        row = np.zeros(2 * count)
        for i, (s, _) in enumerate(keys):
            row[i] = s == service
        add(row, low=required)
    for i in range(count):
        row = np.zeros(2 * count)
        row[i], row[count + i] = 1, -upper[i]
        add(row, high=0)
        row = np.zeros(2 * count)
        row[i], row[count + i] = 1, -1
        add(row, low=0)

    def solve(kappa):
        row = np.r_[np.zeros(count), np.ones(count)]
        constraint = LinearConstraint(np.array(rows + [row]), lower + [kappa], limits + [np.inf])
        return milp(
            c,
            integrality=np.ones(2 * count),
            bounds=Bounds(np.zeros(2 * count), np.r_[upper, np.minimum(upper, 1)]),
            constraints=constraint,
            options={"time_limit": 60.0, "mip_rel_gap": 0.0},
        )

    best = solve(settings.min_core_pairs or 0)
    if not best.success:
        raise ValueError(f"Core placement not solved to optimality: {best.message}")
    if settings.min_core_pairs is None:
        # Preserve the old search for maximal diversity near the best objective,
        # but make the tolerance correct for both positive and negative optima.
        threshold = best.fun + (1 - settings.diversity_retention) * abs(best.fun) + 1e-8
        lo, hi = int(np.count_nonzero(best.x[:count] > 0.5)), int(np.count_nonzero(upper))
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = solve(mid)
            if candidate.success and candidate.fun <= threshold:
                best, lo = candidate, mid + 1
            elif candidate.status in (0, 2):
                hi = mid - 1
            else:
                raise RuntimeError(f"Diversity optimization did not finish: {candidate.message}")
    counts = {key: round(best.x[i]) for i, key in enumerate(keys)}
    validate_resources(scenario, counts)
    validate_core_budget(scenario, counts)
    return PlacementResult(
        counts, float(best.fun), sum(v > 0 for v in counts.values()), "scipy-highs"
    )


def _least_loaded_core(scenario: Scenario) -> PlacementResult:
    counts = {(s, n): 0 for s in scenario.core for n in scenario.nodes}
    usage = resource_usage(scenario, counts)
    # Big services first avoids avoidable packing failures in a greedy baseline.
    for s in sorted(scenario.core, key=lambda s: (-sum(scenario.services[s].resources), s)):
        demands = np.asarray(scenario.services[s].resources)
        for _ in range(core_requirements(scenario)[s]):
            candidates = [
                n
                for n in scenario.nodes
                if np.all(usage[n] + demands <= np.asarray(scenario.nodes[n].core_budget) + 1e-8)
            ]
            if not candidates:
                raise ValueError(f"Least-loaded core placement has insufficient resources for {s}")
            node = min(
                candidates,
                key=lambda n: (
                    float(np.mean(usage[n] / np.maximum(scenario.nodes[n].core_budget, 1))),
                    n,
                ),
            )
            counts[s, node] += 1
            usage[node] += demands
    objective = sum(
        v * (scenario.services[s].deployment_cost + scenario.services[s].maintenance_cost)
        for (s, _), v in counts.items()
    )
    return PlacementResult(counts, objective, sum(v > 0 for v in counts.values()), "least-loaded")
