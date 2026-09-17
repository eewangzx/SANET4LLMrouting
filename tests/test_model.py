import math
from dataclasses import replace

import numpy as np
import pytest

from edge_msd.capacity import DelayModel, effective_capacity
from edge_msd.config import Settings
from edge_msd.models import Node, Radio, Request, Scenario, Service, Stage, TaskType, User
from edge_msd.network import Network
from examples.minimal import make_scenario


def light():
    return Service("l", "light", (1, 0, 1, 0), 0, 1, 1, 1, work_mb=1, gamma_shape=2, gamma_scale=5)


def test_dag_validation_and_copy():
    with pytest.raises(ValueError, match="Cyclic"):
        TaskType("cycle", {"a": ["b"], "b": ["a"]}, 10, 1)
    with pytest.raises(ValueError, match="Unknown"):
        TaskType("missing", {"a": ["b"]}, 10, 1)
    source = {"a": [], "b": ["a"]}
    task = TaskType("ok", source, 10, 1)
    source["b"].clear()
    assert task.dependencies["b"] == ["a"]


@pytest.mark.parametrize(
    "change",
    [
        {"epsilon": 0},
        {"duration_ms": 0},
        {"slot_ms": 0.3},
        {"slot_ms": float("nan")},
        {"eta": -1},
        {"ec_admission_window_ms": 0},
        {"max_parallelism": 1.5},
        {"ga_population": 2},
        {"queue_floor": 0},
    ],
)
def test_invalid_settings(change):
    with pytest.raises(ValueError):
        Settings(**change)


def test_scenario_rejects_mismatched_keys_and_duplicate_links():
    scenario = make_scenario()
    scenario.nodes["wrong"] = scenario.nodes.pop("edge")
    with pytest.raises(ValueError, match="keys"):
        scenario.validate()
    scenario = make_scenario()
    scenario.links.append(("server", "edge", 2, 0))
    with pytest.raises(ValueError, match="Duplicate"):
        scenario.validate()


def test_multihop_units_disconnection_and_invalid_size():
    scenario = make_scenario()
    scenario.nodes["third"] = Node("third", (1, 0, 1, 0))
    scenario.nodes["isolated"] = Node("isolated", (1, 0, 1, 0))
    scenario.links.append(("server", "third", 2, 3))
    network = Network(scenario)
    assert network.delay("edge", "edge", 1) == 0
    assert network.delay("edge", "third", 1) == 8 + 4 + 3
    assert network.delay("edge", "isolated", 1) == math.inf
    with pytest.raises(ValueError):
        network.delay("edge", "server", -1)


def test_root_defaults_to_task_input_not_zero():
    scenario = make_scenario()
    req = Request(0, "user", "pipeline", 0, 2, "edge")
    stage = Stage(req, "prepare", 2)
    assert Network(scenario).data_ready(stage, "server") == pytest.approx(2.8)
    scenario.services["prepare"] = replace(scenario.services["prepare"], input_mb=0.25)
    assert Network(scenario).data_ready(stage, "server") == 4


def test_wireless_has_explicit_parameters_and_hz_conversion():
    scenario = make_scenario()
    with pytest.raises(ValueError, match="explicit"):
        Network(scenario, wireless=True)
    scenario.nodes["edge"] = Node("edge", (6, 0, 6, 0), (0, 0), 1e9)
    scenario.radio = Radio(0, 0, 3e8 / (4 * math.pi))
    # At distance 1, unit mean gain and equal transmit/noise power give SNR=1.
    assert Network(scenario, True).uplink_delay(scenario.users[0], 1) == pytest.approx(8)
    scenario.users.append(User("second", (0, 0), "edge", {"pipeline": 0}, 2, 1))
    assert Network(scenario, True).uplink_delay(scenario.users[0], 1) == pytest.approx(16)


def test_effective_capacity_gamma_laplace_transform():
    service = light()
    theta, dt = 0.2, 2.0
    expected = 2 * math.log(1 + theta * 5 * dt) / (theta * dt)
    assert effective_capacity(service, theta, dt) == pytest.approx(expected)
    rates = np.random.default_rng(4).gamma(2, 5, 200_000)
    empirical = -math.log(np.mean(np.exp(-theta * rates * dt))) / (theta * dt)
    assert empirical == pytest.approx(expected, rel=0.01)
    assert effective_capacity(service, 0) == 10
    assert effective_capacity(service, 1) < effective_capacity(service, 0.1) < 10


def test_proxy_requires_explicit_window_and_respects_epsilon():
    service = light()
    with pytest.raises(ValueError, match="explicit"):
        DelayModel({"l": service}, "effective", 0.2, 1)
    robust = DelayModel({"l": service}, "effective", 0.2, 1, 1)
    strict = DelayModel({"l": service}, "effective", 0.05, 1, 1)
    average = DelayModel({"l": service}, "average", 0.2, 1)
    assert strict.delay("l", 9) > robust.delay("l", 9) > average.delay("l", 9)
    assert robust.delay("l", 10) == math.inf
    assert math.isfinite(average.delay("l", 10))
    assert robust.delay("l", 0) == 0
    with pytest.raises(ValueError):
        robust.delay("l", 1.5)


def test_ec_window_is_independent_of_rate_sampling_slot():
    service = light()
    one = DelayModel({"l": service}, "effective", 0.2, 1, 1)
    two = DelayModel({"l": service}, "effective", 0.2, 1, 2)
    assert one.delay("l", 10) == math.inf
    assert math.isfinite(two.delay("l", 10))


def branching_scenario():
    services = {
        name: Service(name, "core", (1, 0, 1, 0), 1, 20, 4, 0, processing_ms=duration)
        for name, duration in (("a", 1), ("b", 2), ("join", 1))
    }
    task = TaskType("fork", {"a": [], "b": [], "join": ["a", "b"]}, 30, 0)
    nodes = {n: Node(n, (4, 0, 4, 0)) for n in ("n0", "n1", "n2")}
    user = User("u", (0, 0), "n0", {"fork": 0}, 2, 1)
    return Scenario(
        services, {"fork": task}, nodes, [("n0", "n1", 1, 0), ("n1", "n2", 1, 0)], [user]
    )
