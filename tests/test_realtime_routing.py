"""Stage A tests for the realtime-routing rebuild.

These are model-correctness and anti-leakage tests, not algorithmic-gain claims.
They exercise: (i) the fixed deployment under all four resource dimensions,
(ii) one shared node-action interface for core and light stages, (iii) the inherited
ICC physical semantics (fork/join, FIFO work budget, reserved-instance occupancy),
(iv) one-time reward settlement and the full arrival denominator, and (v) exact
checkpoint continuation.
"""

from dataclasses import replace

import pytest

from edge_msd.config import Settings
from edge_msd.models import Node, Scenario, Service, TaskType, User
from edge_msd.placement import validate_resources
from edge_msd.realtime_routing import checkpoint as ckpt
from edge_msd.realtime_routing.environment import RoutingEnvironment
from edge_msd.realtime_routing.fixed_placement import (
    FixedPlacement,
    build_fixed_placement,
    placement_hash,
    scenario_hash,
)
from edge_msd.realtime_routing.scenario import build_scenario


def controls(**changes):
    base = {
        "duration_ms": 20,
        "slot_ms": 1.0,
        "ec_admission_window_ms": 1,
        "max_parallelism": 4,
    }
    base.update(changes)
    return Settings(**base)


class UnitRate:
    """One MB/ms per physical instance, independent of admitted parallelism."""

    def gamma(self, shape, scale):
        return 1.0


def fixed(scenario, core=None, light=None):
    core, light = dict(core or {}), dict(light or {})
    return FixedPlacement(
        core=core,
        light=light,
        core_solver="test",
        scenario_hash=scenario_hash(scenario),
        placement_hash=placement_hash(core, light),
    )


def drive(env, first=True, limit=100000):
    """Route every ready stage to the first (or last) legal node; return rewards."""
    rewards = []
    for _ in range(limit):
        if env.terminated:
            break
        obs = env.observe()
        legal = [i for i, v in enumerate(obs["legal_mask"]) if v]
        assert legal, "a non-terminated step must expose a legal node"
        index = legal[0] if first else legal[-1]
        _, reward, _, _, _ = env.step(env.node_ids[index])
        rewards.append(reward)
    return rewards


def rollout(env, steps, first=True):
    """Record a deterministic continuation for equality comparison."""
    out = []
    for _ in range(steps):
        if env.terminated:
            break
        obs = env.observe()
        legal = [i for i, v in enumerate(obs["legal_mask"]) if v]
        index = legal[0] if first else legal[-1]
        node = env.node_ids[index]
        _, reward, _, _, info = env.step(node)
        out.append((node, round(reward, 9), round(info["episode_time_ms"], 9), info["sla_counts"]))
    return out


# --------------------------------------------------------------------- fixtures
def core_fork_scenario():
    services = {
        name: Service(name, "core", (1, 0, 1, 0), output, 1, 0.1, 0, processing_ms=duration)
        for name, duration, output in [("left", 1, 1), ("right", 4, 0.25), ("join", 2, 0)]
    }
    return Scenario(
        services,
        {"fork": TaskType("fork", {"left": [], "right": [], "join": ["left", "right"]}, 11, 0)},
        {name: Node(name, (4, 0, 4, 0)) for name in ("a", "b", "c")},
        [("a", "c", 1, 0), ("b", "c", 1, 0)],
        [User("user", (0, 0), "a", {"fork": 0}, 2, 1)],
    )


def mixed_scenario():
    services = {
        "pre": Service("pre", "light", (1, 0, 1, 0), 0, 1, 0.1, 0.1, work_mb=1,
                       gamma_shape=2, gamma_scale=0.5),
        "core": Service("core", "core", (1, 0, 1, 0), 0.1, 1, 0.1, 0, processing_ms=1),
    }
    task = TaskType("pipeline", {"pre": [], "core": ["pre"]}, 30, 0)
    nodes = {name: Node(name, (8, 0, 8, 0)) for name in ("n0", "n1", "n2")}
    return Scenario(
        services, {"pipeline": task}, nodes,
        [("n0", "n1", 1, 0), ("n1", "n2", 1, 0)],
        [User("u", (0, 0), "n0", {"pipeline": 0}, 2, 1)],
    )


def serial_core_scenario():
    service = Service("core", "core", (1, 0, 1, 0), 0.1, 1, 0.1, 0, processing_ms=5)
    task = TaskType("job", {"core": []}, 100, 0)
    nodes = {name: Node(name, (4, 0, 4, 0)) for name in ("n0", "n1")}
    return Scenario(
        {"core": service}, {"job": task}, nodes, [("n0", "n1", 1, 0)],
        [User("u", (0, 0), "n0", {"job": 0}, 2, 1)],
    )


# ------------------------------------------------------------------ deployment
def test_fixed_placement_obeys_all_four_resource_dimensions():
    scenario = build_scenario(2026)
    settings = controls(duration_ms=50)
    placement = build_fixed_placement(scenario, settings, "proposed", 1)
    validate_resources(scenario, placement.counts)  # must not raise
    assert placement.core and placement.light
    assert all(scenario.services[s].kind == "core" for s, _ in placement.core)
    assert all(scenario.services[s].kind == "light" for s, _ in placement.light)
    key = sorted(placement.light)[0]
    over = dict(placement.counts)
    over[key] += 10_000
    with pytest.raises(ValueError):
        validate_resources(scenario, over)


def test_illegal_action_is_rejected_and_does_not_mutate_state():
    scenario = core_fork_scenario()
    settings = controls(duration_ms=20)
    env = RoutingEnvironment(
        scenario, settings,
        fixed(scenario, core={("left", "a"): 1, ("right", "b"): 1, ("join", "c"): 1}),
        requests=[("user", "fork", 0.0, 0.0)], drain_ms=5,
    )
    before = len(env.waiting)
    with pytest.raises(ValueError, match="Illegal node"):
        env.step("c")  # no left instance at c
    assert len(env.waiting) == before


# ------------------------------------------------------- shared node interface
def test_core_and_light_stages_share_one_node_action_interface():
    scenario = mixed_scenario()
    settings = controls(duration_ms=20)
    env = RoutingEnvironment(
        scenario, settings,
        fixed(scenario, core={("core", "n1"): 1}, light={("pre", "n0"): 1}),
        requests=[("u", "pipeline", 0.0, 0.0)], drain_ms=10,
    )
    routed = []
    while not env.terminated:
        obs = env.observe()
        routed.append(obs["current"]["service"])
        legal = [i for i, v in enumerate(obs["legal_mask"]) if v]
        env.step(env.node_ids[legal[0]])
    assert "pre" in routed and "core" in routed


# ----------------------------------------------------------- physical semantics
def test_fork_join_matches_hand_calculation():
    scenario = core_fork_scenario()
    settings = controls(duration_ms=15, record_trace=True)
    env = RoutingEnvironment(
        scenario, settings,
        fixed(scenario, core={("left", "a"): 1, ("right", "b"): 1, ("join", "c"): 1}),
        requests=[("user", "fork", 0.0, 0.0)], drain_ms=5,
    )
    rewards = drive(env)
    trace = {stage["service"]: stage for stage in env.trace}
    assert trace["left"]["finish_ms"] == 1
    assert trace["right"]["finish_ms"] == 4
    # The early parent's larger output arrives last: 1 + 8 = 9 ms.
    assert trace["join"]["data_ready_ms"] == 9
    assert trace["join"]["finish_ms"] == 11
    assert env.sla_counts() == (1, 0, 0)
    assert sum(rewards) == pytest.approx(1.0)


def test_parallelism_admits_but_does_not_multiply_capacity():
    # One light service, one instance, admission 2, unit rate: two jobs serialize.
    service = Service("light", "light", (1, 0, 1, 0), 0, 1, 0.1, 0.1,
                      work_mb=1, gamma_shape=2, gamma_scale=0.5)
    task = TaskType("t", {"light": []}, 100, 0)
    scenario = Scenario(
        {"light": service}, {"t": task}, {n: Node(n, (8, 0, 8, 0)) for n in ("n0", "n1")},
        [("n0", "n1", 1, 0)], [User("u", (0, 0), "n0", {"t": 0}, 2, 1)],
    )
    settings = controls(duration_ms=10, max_parallelism=2, record_trace=True)
    env = RoutingEnvironment(
        scenario, settings, fixed(scenario, light={("light", "n0"): 1}),
        requests=[("u", "t", 0.0, 0.0), ("u", "t", 0.0, 0.0)], drain_ms=5,
    )
    env.service_rng = UnitRate()
    drive(env)
    finishes = sorted(s["finish_ms"] for s in env.trace)
    assert finishes == [1, 2]  # serialized FIFO, NOT [1, 1]


def test_reserved_instance_is_occupied_while_input_is_in_transit():
    # Root input crosses one hop (n0 -> n1), so data_ready is in the future; the
    # instance is nevertheless reserved the moment the stage is dispatched, and a
    # busy core instance is no longer offered to the same service.
    service = Service("core", "core", (1, 0, 1, 0), 0.1, 1, 0.1, 0,
                      processing_ms=5, input_mb=2.0)
    task = TaskType("job", {"core": []}, 100, 0)
    nodes = {name: Node(name, (4, 0, 4, 0)) for name in ("n0", "n1")}
    scenario = Scenario(
        {"core": service}, {"job": task}, nodes, [("n0", "n1", 1, 0)],
        [User("u", (0, 0), "n0", {"job": 0}, 2, 1)],
    )
    settings = controls(duration_ms=30)
    env = RoutingEnvironment(
        scenario, settings, fixed(scenario, core={("core", "n1"): 1}),
        requests=[("u", "job", 0.0, 0.0)], drain_ms=10,
    )
    stage = env._routable_stage()
    assert env.legal_nodes(stage) == ["n1"]
    env._assign(stage, "n1")  # 2 MB over 1 Gbps = 16 ms of input transfer
    instance = env.instances[0]
    assert len(instance.jobs) == 1
    assert instance.jobs[0].data_ready_ms > env.now
    assert "n1" not in env.legal_nodes(stage)  # a reserved instance is not offered


def test_core_instances_serialize_jobs():
    scenario = serial_core_scenario()  # one core instance, 5 ms each
    settings = controls(duration_ms=30, record_trace=True)
    env = RoutingEnvironment(
        scenario, settings, fixed(scenario, core={("core", "n0"): 1}),
        requests=[("u", "job", 0.0, 0.0), ("u", "job", 0.0, 0.0)], drain_ms=10,
    )
    drive(env)
    finishes = sorted(stage["finish_ms"] for stage in env.trace)
    assert finishes == [5, 10]  # one instance: the second job waits for the first


# ----------------------------------------------------------- reward and metric
def test_reward_settles_once_and_uses_full_arrival_denominator():
    service = Service("light", "light", (1, 0, 1, 0), 0, 1, 0.1, 0.1,
                      work_mb=1, gamma_shape=2, gamma_scale=0.5)
    task = TaskType("t", {"light": []}, 1, 0)  # deadline 1 ms: the second job is late
    scenario = Scenario(
        {"light": service}, {"t": task}, {"n0": Node("n0", (8, 0, 8, 0))}, [],
        [User("u", (0, 0), "n0", {"t": 0}, 2, 1)],
    )
    settings = controls(duration_ms=10, max_parallelism=2, record_trace=True)
    env = RoutingEnvironment(
        scenario, settings, fixed(scenario, light={("light", "n0"): 1}),
        requests=[("u", "t", 0.0, 0.0), ("u", "t", 0.0, 0.0)], drain_ms=5,
    )
    env.service_rng = UnitRate()
    rewards = drive(env)
    ontime, late, pending = env.sla_counts()
    assert (ontime, late, pending) == (1, 1, 0)
    assert len(env.requests) == 2  # arrivals include the late one
    # sum reward == 2*ontime - arrivals when every request settles exactly once
    assert sum(rewards) == pytest.approx(2 * ontime - len(env.requests))


def test_policy_does_not_change_exogenous_arrivals():
    scenario = build_scenario(2026)
    settings = controls(duration_ms=40)
    placement = build_fixed_placement(scenario, settings, "proposed", 1)
    left = RoutingEnvironment(scenario, settings, placement, seed=7, drain_ms=20)
    right = RoutingEnvironment(scenario, settings, placement, seed=7, drain_ms=20)
    drive(left, first=True)
    drive(right, first=False)
    assert left.arrival_hash.hexdigest() == right.arrival_hash.hexdigest()
    assert len(left.requests) == len(right.requests)


# --------------------------------------------------------------- checkpointing
def chain_scenario():
    services = {
        name: Service(name, "core", (1, 0, 1, 0), 0.5, 1, 0.1, 0, processing_ms=2)
        for name in ("s1", "s2", "s3")
    }
    task = TaskType("chain", {"s1": [], "s2": ["s1"], "s3": ["s2"]}, 200, 0)
    return Scenario(
        services, {"chain": task}, {n: Node(n, (8, 0, 8, 0)) for n in ("n0", "n1")},
        [("n0", "n1", 1, 0)], [User("u", (0, 0), "n0", {"chain": 0}, 2, 1)],
    )


def test_checkpoint_restores_exact_continuation(tmp_path):
    scenario = chain_scenario()
    settings = controls(duration_ms=40, record_trace=True)
    placement = fixed(
        scenario, core={("s1", "n0"): 1, ("s2", "n0"): 1, ("s3", "n1"): 1}
    )
    requests = [("u", "chain", 0.0, 0.0) for _ in range(3)]
    env = RoutingEnvironment(scenario, settings, placement, seed=7, drain_ms=20,
                             requests=requests)
    rollout(env, 5)
    path = tmp_path / "ckpt.json"
    ckpt.save_checkpoint(path, env)
    data, state, extra = ckpt.load_checkpoint(path)
    resumed = RoutingEnvironment(scenario, settings, placement, seed=7, drain_ms=20,
                                 requests=requests)
    ckpt.restore_environment(resumed, data, state)
    assert rollout(env, 40) == rollout(resumed, 40)


def test_checkpoint_rejects_mismatched_configuration(tmp_path):
    scenario = chain_scenario()
    settings = controls(duration_ms=40)
    placement = fixed(scenario, core={("s1", "n0"): 1, ("s2", "n0"): 1, ("s3", "n1"): 1})
    env = RoutingEnvironment(scenario, settings, placement, seed=7, drain_ms=20,
                             requests=[("u", "chain", 0.0, 0.0)])
    path = tmp_path / "ckpt.json"
    ckpt.save_checkpoint(path, env)
    data, state, _ = ckpt.load_checkpoint(path)
    other = RoutingEnvironment(
        scenario, replace(settings, slot_ms=2.0), placement, seed=7, drain_ms=20,
        requests=[("u", "chain", 0.0, 0.0)],
    )
    with pytest.raises(ValueError, match="settings"):
        ckpt.restore_environment(other, data, state)
