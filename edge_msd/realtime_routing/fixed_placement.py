"""Fixed two-tier deployment shared by every method.

The core tier is solved once (upstream MILP, or the least-loaded baseline). The
light tier is packed into the remaining four-dimensional budget by a deterministic
least-loaded rule. One placement is validated against all four resource dimensions
and then frozen; routing methods may not change it (node selection only).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from edge_msd.config import Settings
from edge_msd.models import Placement, Scenario
from edge_msd.network import Network
from edge_msd.placement import (
    place_core,
    resource_usage,
    validate_core_budget,
    validate_resources,
)

SCHEMA = "edge-msd-fixed-placement-v1"


@dataclass(frozen=True)
class FixedPlacement:
    core: dict  # (service, node) -> count, core services only
    light: dict  # (service, node) -> count, light services only
    core_solver: str
    scenario_hash: str
    placement_hash: str

    @property
    def counts(self) -> Placement:
        """Combined (service, node) -> count across both tiers."""
        merged = dict(self.core)
        for key, value in self.light.items():
            merged[key] = merged.get(key, 0) + value
        return merged


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def scenario_hash(scenario: Scenario) -> str:
    """Content hash of the exogenous scenario (services, tasks, nodes, links, users)."""
    payload = {
        "services": [
            {**asdict(s), "kind": s.kind} for s in sorted(scenario.services.values(), key=lambda x: x.name)
        ],
        "tasks": [
            {
                "name": t.name,
                "dependencies": t.dependencies,
                "deadline_ms": t.deadline_ms,
                "input_mb": t.input_mb,
                "priority": t.priority,
            }
            for t in sorted(scenario.tasks.values(), key=lambda x: x.name)
        ],
        "nodes": [asdict(n) for n in sorted(scenario.nodes.values(), key=lambda x: x.name)],
        "links": [list(link) for link in scenario.links],
        "users": [asdict(u) for u in sorted(scenario.users, key=lambda x: x.name)],
        "profile": scenario.profile,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _key_pairs(counts: Placement) -> list[list]:
    return [[s, n, c] for (s, n), c in sorted(counts.items()) if c]


def placement_hash(core: Placement, light: Placement) -> str:
    return hashlib.sha256(
        _canonical({"core": _key_pairs(core), "light": _key_pairs(light)}).encode()
    ).hexdigest()


def _pack_light(
    scenario: Scenario, core: Placement, per_service: int
) -> Placement:
    """Deterministically pack light instances into the post-core resource budget."""
    if not isinstance(per_service, int) or isinstance(per_service, bool) or per_service < 1:
        raise ValueError("light_per_service must be a positive integer")
    counts = {(s, n): 0 for s in scenario.light for n in scenario.nodes}
    usage = resource_usage(scenario, core)
    for service_name in sorted(scenario.light):
        demand = np.asarray(scenario.services[service_name].resources)
        placed = 0
        for _ in range(per_service):
            candidates = {}
            for node in scenario.nodes:
                cap = np.asarray(scenario.nodes[node].resources)
                if np.all(usage[node] + demand <= cap + 1e-8):
                    utilization = float(np.mean(usage[node] / np.maximum(cap, 1.0)))
                    candidates[node] = (utilization, node)
            if not candidates:
                break
            node = min(candidates, key=candidates.get)
            counts[service_name, node] += 1
            usage[node] = usage[node] + demand
            placed += 1
        if placed == 0:
            raise ValueError(f"no node can host light service {service_name} after core placement")
    return {key: value for key, value in counts.items() if value}


def build_fixed_placement(
    scenario: Scenario,
    settings: Settings,
    core_method: str = "proposed",
    light_per_service: int = 1,
) -> FixedPlacement:
    """Solve/pack and validate the frozen two-tier deployment for one scenario."""
    scenario.validate()
    network = Network(scenario, settings.wireless)
    result = place_core(scenario, network, settings, core_method)
    core = {key: value for key, value in result.counts.items() if value}
    light = _pack_light(scenario, core, light_per_service)
    validate_resources(scenario, {**core, **light})
    validate_core_budget(scenario, core)
    if any(scenario.services[s].kind != "core" for s, _ in core):
        raise ValueError("core placement must contain only core services")
    if any(scenario.services[s].kind != "light" for s, _ in light):
        raise ValueError("light placement must contain only light services")
    return FixedPlacement(
        core=core,
        light=light,
        core_solver=result.solver,
        scenario_hash=scenario_hash(scenario),
        placement_hash=placement_hash(core, light),
    )


def save_placement(placement: FixedPlacement, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "core_solver": placement.core_solver,
        "scenario_hash": placement.scenario_hash,
        "placement_hash": placement.placement_hash,
        "core": _key_pairs(placement.core),
        "light": _key_pairs(placement.light),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def load_placement(path) -> FixedPlacement:
    data = json.loads(Path(path).read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError("Unsupported fixed-placement schema")
    core = {(s, n): int(c) for s, n, c in data["core"]}
    light = {(s, n): int(c) for s, n, c in data["light"]}
    if placement_hash(core, light) != data["placement_hash"]:
        raise ValueError("Fixed-placement file is corrupt (hash mismatch)")
    return FixedPlacement(
        core=core,
        light=light,
        core_solver=data["core_solver"],
        scenario_hash=data["scenario_hash"],
        placement_hash=data["placement_hash"],
    )
