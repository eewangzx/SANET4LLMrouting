"""Atomic, traceable checkpoints for the fixed-deployment environment.

Stage A stores the full environment state (instances, requests, DAG progress,
settlement ledger and all three RNG streams) together with configuration hashes
so a fresh process can reproduce the exact continuation. Later stages append
policy/optimizer/replay fields only when they exist; a weights-only file is never
presented as a complete resume.
"""

from __future__ import annotations

import base64
import hashlib
import json
import pickle
from collections import Counter
from dataclasses import asdict
from pathlib import Path

SCHEMA = "edge-msd-realtime-routing-ckpt-v1"

_ENV_FIELDS = (
    "now",
    "terminated",
    "next_instance_id",
    "instances",
    "active",
    "requests",
    "waiting",
    "costs",
    "trace",
)


def _capture(env) -> dict:
    return {
        "fields": {name: getattr(env, name) for name in _ENV_FIELDS},
        "settled": sorted(env.settled),
        "reward_acc": env._reward_acc,
        "peak_instances": dict(env.peak_instances),
        "arrival_hash": env.arrival_hash,
        "rng": {
            "arrivals": env.arrivals.bit_generator.state,
            "channels": env.channels.bit_generator.state,
            "service": env.service_rng.bit_generator.state,
        },
    }


def save_checkpoint(path, env, extra: dict | None = None) -> dict:
    state = _capture(env)
    blob = base64.b64encode(pickle.dumps((state, extra or {}))).decode()
    payload = {
        "schema": SCHEMA,
        "placement_hash": env.fixed.placement_hash,
        "scenario_hash": env.fixed.scenario_hash,
        "settings": asdict(env.settings),
        "horizon_ms": env.horizon_ms,
        "now": env.now,
        "blob_sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "state": blob,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)
    return payload


def load_checkpoint(path) -> tuple[dict, dict, dict]:
    data = json.loads(Path(path).read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError("Unsupported checkpoint schema")
    if hashlib.sha256(data["state"].encode()).hexdigest() != data["blob_sha256"]:
        raise ValueError("Checkpoint payload is corrupt (hash mismatch)")
    state, extra = pickle.loads(base64.b64decode(data["state"]))
    return data, state, extra


def restore_environment(env, data: dict, state: dict) -> None:
    """Restore ``env`` in place after validating configuration compatibility."""
    if data["scenario_hash"] != env.fixed.scenario_hash:
        raise ValueError("Checkpoint scenario hash does not match this environment")
    if data["placement_hash"] != env.fixed.placement_hash:
        raise ValueError("Checkpoint placement hash does not match this environment")
    if asdict(env.settings) != data["settings"]:
        raise ValueError("Checkpoint settings do not match this environment")
    for name, value in state["fields"].items():
        setattr(env, name, value)
    env.settled = set(state["settled"])
    env._reward_acc = state["reward_acc"]
    env.peak_instances = Counter(state["peak_instances"])
    env.arrival_hash = state["arrival_hash"]
    env.arrivals.bit_generator.state = state["rng"]["arrivals"]
    env.channels.bit_generator.state = state["rng"]["channels"]
    env.service_rng.bit_generator.state = state["rng"]["service"]
