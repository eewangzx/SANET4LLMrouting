"""ICC Table I / Figure 1 scenario reconstruction, with explicit provenance.

These are newly sampled instances of the published parameter ranges, NOT the
authors' original experimental files. The original paper does not supply LLM
answer-quality labels. No such labels are invented by this module.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .models import Node, Radio, Scenario, Service, TaskType, User

PAPER = {
    "file": "2026ICC_ZJ_Final_v1.pdf",
    "title": "Modular Foundation Model Inference at the Edge: Network-Aware Microservice Optimization",
    "parameter_source": "Table I, PDF page 6",
    "dag_source": "Figure 1, PDF page 2",
    "topology_source": "Figure 2, PDF page 2",
    "data_kind": "published-parameter reconstruction; not author-original samples",
}

# Keep the source order visible. Domain objects use a DIFFERENT resource order.
PAPER_RESOURCE_ORDER = ["CPU", "RAM", "GPU", "VRAM"]
CODE_RESOURCE_ORDER = ["CPU", "GPU", "RAM", "VRAM"]
RESOURCE_RANGES = {
    "core": [[2, 16], [1, 4], [4, 32], [4, 32]],
    "light": [[0.5, 2], [0, 0.5], [0.25, 4], [0, 1]],
    "ed": [[1, 64], [1, 32], [0, 64], [0, 64]],
    "es": [[128, 256], [64, 128], [1024, 2048], [256, 512]],
}
TABLE_I = {
    "resource_ranges_in_paper_order": RESOURCE_RANGES,
    "core_work_mb": [2, 16],
    "core_output_mb": [0.1, 1],
    "core_rate_mb_per_ms": [8, 32],
    "light_work_mb": [0.5, 2],
    "light_output_mb": [0.25, 1.5],
    "light_gamma_shape": [1, 2],
    "light_gamma_scale_mb_per_ms": [1, 20],
    "core_costs_deploy_maintain_parallel": [20, 4, 0],
    "light_costs_deploy_maintain_parallel": [4, 1, 0.5],
    "arrival_mean_per_ms_per_user_task": [0.15, 1.5],
    "task_deadline_ms": [50, 100],
    "task_input_mb": [0.5, 4],
    "nakagami_m": [1.5, 3],
    "nakagami_omega": [0.5, 1],
    "wired_mb_per_ms": [0.1, 1],
    "epsilon": 0.2,
    "load_multipliers": [1, 1.5, 2],
}

SERVICE_NAMES = {
    1: "text_preprocess",
    2: "image_preprocess",
    3: "audio_preprocess",
    4: "text_encoder",
    5: "image_encoder",
    6: "audio_encoder",
    7: "compression",
    8: "projection_c",
    9: "projection_l",
    10: "multimodal_fusion",
    11: "reasoning",
    12: "postprocess_qa",
    13: "postprocess_crossmodal",
    14: "postprocess_classification",
    15: "postprocess_driving",
}
CORE_IDS = {4, 5, 6, 8, 10, 11}
FIGURE_1_DAGS = {
    "open_domain_qa": {1: [], 4: [1], 8: [4], 11: [8], 12: [11]},
    "cross_modal_understanding": {1: [], 2: [], 4: [1], 5: [2], 8: [4, 5], 10: [8], 13: [10]},
    "classification": {2: [], 5: [2], 9: [5], 11: [9], 14: [11]},
    "multimodal_driving": {2: [], 3: [], 5: [2], 6: [3], 7: [6], 8: [5, 7], 10: [8], 15: [10]},
}


def _sample(rng, bounds):
    return float(rng.uniform(*bounds))


def _resources(rng, kind):
    values = [_sample(rng, b) for b in RESOURCE_RANGES[kind]]
    return [values[i] for i in (0, 2, 1, 3)]


def generate_dataset(seed=2026):
    """Sample once; save this exact instance and reuse it across algorithms."""
    rng = np.random.default_rng(seed)
    services, sampled = [], {}
    for i, name in SERVICE_NAMES.items():
        kind = "core" if i in CORE_IDS else "light"
        work = _sample(rng, TABLE_I[f"{kind}_work_mb"])
        entry = dict(
            name=name,
            kind=kind,
            resources=_resources(rng, kind),
            output_mb=_sample(rng, TABLE_I[f"{kind}_output_mb"]),
            deployment_cost=TABLE_I[f"{kind}_costs_deploy_maintain_parallel"][0],
            maintenance_cost=TABLE_I[f"{kind}_costs_deploy_maintain_parallel"][1],
            parallel_cost=TABLE_I[f"{kind}_costs_deploy_maintain_parallel"][2],
            work_mb=work,
        )
        if kind == "core":
            rate = _sample(rng, TABLE_I["core_rate_mb_per_ms"])
            entry["processing_ms"] = work / rate
            sampled[name] = {"paper_id": i, "work_mb": work, "rate_mb_per_ms": rate}
        else:
            entry["gamma_shape"] = _sample(rng, TABLE_I["light_gamma_shape"])
            entry["gamma_scale"] = _sample(rng, TABLE_I["light_gamma_scale_mb_per_ms"])
            sampled[name] = {"paper_id": i, "work_mb": work}
        services.append(entry)
    tasks = [
        dict(
            name=name,
            dependencies={
                SERVICE_NAMES[s]: [SERVICE_NAMES[p] for p in parents] for s, parents in dag.items()
            },
            deadline_ms=_sample(rng, TABLE_I["task_deadline_ms"]),
            input_mb=_sample(rng, TABLE_I["task_input_mb"]),
        )
        for name, dag in FIGURE_1_DAGS.items()
    ]
    # Figure 2 contains seven EDs and three ESs. Metric coordinates are absent.
    positions = {
        "es1": [0, 100],
        "es2": [100, 0],
        "es3": [200, 100],
        "ed1": [-20, 140],
        "ed2": [70, 150],
        "ed3": [-30, 40],
        "ed4": [70, -30],
        "ed5": [160, -20],
        "ed6": [230, 40],
        "ed7": [240, 140],
    }
    nodes = [
        dict(name=name, resources=_resources(rng, name[:2]), position=position)
        for name, position in positions.items()
    ]
    attachment = {
        "ed1": "es1",
        "ed2": "es1",
        "ed3": "es1",
        "ed4": "es2",
        "ed5": "es2",
        "ed6": "es3",
        "ed7": "es3",
    }
    pairs = [("es1", "es2"), ("es1", "es3"), ("es2", "es3"), *attachment.items()]
    links = [(a, b, _sample(rng, TABLE_I["wired_mb_per_ms"]) * 8, 0.0) for a, b in pairs]
    users = []
    for ed, es in attachment.items():
        users.append(
            dict(
                name="user_" + ed,
                position=positions[ed],
                gateway=es,
                rates_per_second={
                    t["name"]: 1000 * _sample(rng, TABLE_I["arrival_mean_per_ms_per_user_task"])
                    for t in tasks
                },
                nakagami_m=_sample(rng, TABLE_I["nakagami_m"]),
                omega=_sample(rng, TABLE_I["nakagami_omega"]),
            )
        )
    return {
        "schema": "edge-msd-icc-paper-v1",
        "source": PAPER,
        "seed": seed,
        "table_i": TABLE_I,
        "paper_resource_order": PAPER_RESOURCE_ORDER,
        "resource_order": CODE_RESOURCE_ORDER,
        "sampled_service_quantities": sampled,
        "scenario": {
            "services": services,
            "tasks": tasks,
            "nodes": nodes,
            "links": links,
            "users": users,
            "profile": "icc_paper_reconstruction",
        },
        "assumptions_not_specified_numerically_by_paper": {
            "range_sampling": "independent continuous uniform; paper specifies ranges only",
            "node_counts": "7 ED + 3 ES read from Figure 2",
            "topology": "three connected ESs plus explicit ED attachments; illustrative reconstruction",
            "positions": "illustrative coordinates, not measured or author-original",
            "users": "one user per ED, each generating all four task types",
            "gateway": "associated ES represents entry into the wired edge network",
            "wired_propagation_ms": 0.0,
            "wireless_default": "disabled; Table I lacks allocated bandwidth and complete radio settings",
            "gamma_parameterization": "shape and scale of processing RATE, MB/ms",
            "nakagami_parameterization": "stored m and Omega; not treated as Gbps despite table heading",
            "root_payload": "full task input delivered to each root; modality split unspecified",
            "missing": [
                "original run seeds",
                "sampled original scenarios",
                "original traces",
                "simulation horizon and step",
                "radio configuration",
                "answer-quality labels",
            ],
        },
    }


def save_dataset(dataset, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dataset, indent=2) + "\n")


def load_scenario(path, load_multiplier=1.0):
    """Import a persisted instance into the original ICC Scenario API."""
    if not np.isfinite(load_multiplier) or load_multiplier < 0:
        raise ValueError("Load multiplier must be finite and nonnegative")
    data = json.loads(Path(path).read_text())
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
    return scenario


def export_scenario(scenario):
    """Round-trip helper for explicit replacements from an author-supplied file."""
    return {
        "services": [asdict(s) for s in scenario.services.values()],
        "tasks": [
            dict(
                name=t.name,
                dependencies=t.dependencies,
                deadline_ms=t.deadline_ms,
                input_mb=t.input_mb,
                priority=t.priority,
            )
            for t in scenario.tasks.values()
        ],
        "nodes": [asdict(n) for n in scenario.nodes.values()],
        "links": scenario.links,
        "users": [asdict(u) for u in scenario.users],
        "profile": scenario.profile,
        "radio": asdict(scenario.radio) if scenario.radio else None,
    }
