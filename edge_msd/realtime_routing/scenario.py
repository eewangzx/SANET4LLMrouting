"""ICC scenario loading for the realtime-routing rebuild.

Parameters are the published Table I ranges reconstructed by ``edge_msd.icc_paper``.
This module adds an explicit, in-memory construction path plus unit/DAG validation;
it does not invent parameter values. The provenance of every parameter lives in
``edge_msd.icc_paper.TABLE_I`` / ``PAPER`` and in ``configs/parameter_sources.csv``.
"""

from __future__ import annotations

from pathlib import Path

from edge_msd.icc_paper import CODE_RESOURCE_ORDER, generate_dataset
from edge_msd.models import Node, Radio, Scenario, Service, TaskType, User

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "icc_paper" / "scenario_2026.json"


def scenario_from_dataset(data: dict, load_multiplier: float = 1.0) -> Scenario:
    """Build a Scenario from an in-memory ``edge-msd-icc-paper-v1`` dataset.

    Mirrors ``edge_msd.icc_paper.load_scenario`` but avoids a round-trip through
    disk so callers can sample a fresh, deterministic instance by seed.
    """
    if not isinstance(load_multiplier, (int, float)) or load_multiplier != load_multiplier:
        raise ValueError("load_multiplier must be a finite number")
    if load_multiplier < 0:
        raise ValueError("load_multiplier must be nonnegative")
    if (
        data.get("schema") != "edge-msd-icc-paper-v1"
        or data.get("resource_order") != CODE_RESOURCE_ORDER
    ):
        raise ValueError("Unsupported schema or resource order")
    raw = data["scenario"]
    users = []
    for raw_user in raw["users"]:
        user = dict(raw_user)
        user["rates_per_second"] = {
            k: v * load_multiplier for k, v in user["rates_per_second"].items()
        }
        users.append(User(**user))
    scenario = Scenario(
        {s["name"]: Service(**s) for s in raw["services"]},
        {t["name"]: TaskType(**t) for t in raw["tasks"]},
        {n["name"]: Node(**n) for n in raw["nodes"]},
        [tuple(link) for link in raw["links"]],
        users,
        raw["profile"],
        Radio(**raw["radio"]) if raw.get("radio") else None,
    )
    scenario.validate()
    _validate_units(scenario)
    return scenario


def build_scenario(seed: int = 2026, load_multiplier: float = 1.0) -> Scenario:
    """Sample one deterministic Table I instance and validate it.

    Deterministic in ``seed``: the same seed yields the same scenario, so every
    method in a comparison shares identical exogenous parameters.
    """
    return scenario_from_dataset(generate_dataset(seed), load_multiplier)


def load_paper_scenario(path=None, load_multiplier: float = 1.0) -> Scenario:
    """Load the persisted Table I instance (default: data/icc_paper/scenario_2026.json)."""
    from edge_msd.icc_paper import load_scenario

    return load_scenario(Path(path) if path else DEFAULT_DATASET, load_multiplier)


def _validate_units(scenario: Scenario) -> None:
    """Fail loudly on unit inconsistencies the paper's ranges do not permit."""
    for name, service in scenario.services.items():
        if service.kind == "core" and not service.processing_ms > 0:
            raise ValueError(f"core service {name} needs a positive processing_ms")
        if service.kind == "light":
            if not service.work_mb > 0 or not service.gamma_scale > 0 or not service.gamma_shape > 0:
                raise ValueError(f"light service {name} needs positive work/Gamma parameters")
    for name, task in scenario.tasks.items():
        if task.deadline_ms <= 0:
            raise ValueError(f"task {name} needs a positive deadline_ms")
