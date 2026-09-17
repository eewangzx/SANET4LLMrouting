"""Physical invariants inherited by the ICC telemetry extension.

These tests exercise the original Simulator directly. They deliberately avoid
the earlier single-node LLM surrogate and do not assert algorithmic gains.
"""

from dataclasses import replace

import pytest

from edge_msd.config import Settings
from edge_msd.models import Decision, Job, Node, Scenario, Service, Stage, TaskType, User
from edge_msd.simulation import Simulator


def controls(**changes):
    return Settings(duration_ms=10, ec_admission_window_ms=1, **changes)


def light_scenario():
    service = Service(
        "light", "light", (1, 1, 1, 1), 0, 1, 0.1, 0.1,
        work_mb=1, gamma_shape=2, gamma_scale=0.5,
    )
    return Scenario(
        {"light": service},
        {"task": TaskType("task", {"light": []}, 5, 0)},
        {"edge": Node("edge", (4, 4, 4, 4))},
        [],
        [User("user", (0, 0), "edge", {"task": 0}, 2, 1)],
    )


class UnitRate:
    """One MB/ms per physical instance, independent of admitted parallelism."""

    def gamma(self, shape, scale):
        return 1.0


@pytest.mark.parametrize("bottleneck", range(4), ids=["cpu", "gpu", "ram", "vram"])
def test_core_and_light_instances_share_every_resource_capacity(bottleneck):
    scenario = light_scenario()
    requirements = [1, 1, 1, 1]
    requirements[bottleneck] = 2
    scenario.services["light"] = replace(
        scenario.services["light"], resources=tuple(requirements)
    )
    scenario.services["core"] = Service(
        "core", "core", (1, 1, 1, 1), 0, 1, 0.1, 0, processing_ms=1
    )
    sim = Simulator(scenario, controls(), core_counts={("core", "edge"): 1})
    # One core plus one light instance fits; two light instances violate only
    # the selected resource. A rejected decision must not partially deploy.
    sim._deploy(Decision({("light", "edge"): 1}, {("light", "edge"): 0}, {}))
    before = [(instance.id, instance.service) for instance in sim.instances]
    costs = sim.costs.copy()
    with pytest.raises(ValueError, match="capacity"):
        sim._deploy(Decision({("light", "edge"): 2}, {("light", "edge"): 0}, {}))
    assert [(instance.id, instance.service) for instance in sim.instances] == before
    assert sim.costs == costs


@pytest.mark.parametrize("data_ready", [0, 9], ids=["computing", "input_in_transit"])
def test_busy_or_reserved_instance_and_occupancy_are_preserved(data_ready):
    sim = Simulator(light_scenario(), controls(), core_counts={})
    sim._new_instance("light", "edge")
    for _ in range(2):
        request = sim.add_request("user", "task", 0, uplink_ms=0)
        sim.instances[0].jobs.append(Job(Stage(request, "light", 0), data_ready, 1))
    with pytest.raises(ValueError, match="busy or reserved"):
        sim._deploy(Decision({}, {}, {}))
    with pytest.raises(ValueError, match="current occupancy"):
        sim._deploy(Decision({("light", "edge"): 1}, {("light", "edge"): 1}, {}))
    sim._deploy(Decision({("light", "edge"): 1}, {("light", "edge"): 2}, {}))
    assert len(sim.instances[0].jobs) == 2


def test_deployment_removes_idle_instance_before_busy_instance():
    sim = Simulator(light_scenario(), controls(), core_counts={})
    sim._new_instance("light", "edge")
    sim._new_instance("light", "edge")
    busy = sim.instances[1]
    request = sim.add_request("user", "task", 0, uplink_ms=0)
    busy.jobs.append(Job(Stage(request, "light", 0), 0, 1))
    sim._deploy(Decision({("light", "edge"): 1}, {("light", "edge"): 1}, {}))
    assert [instance.id for instance in sim.instances] == [busy.id]


@pytest.mark.parametrize("instance_count", [1, 2])
def test_parallelism_admits_work_but_only_instances_add_compute_capacity(instance_count):
    sim = Simulator(light_scenario(), controls(), core_counts={})
    sim.service_rng = UnitRate()
    requests = [sim.add_request("user", "task", 0, uplink_ms=0) for _ in range(2)]
    sim._expose_ready(0)
    decision = Decision(
        {("light", "edge"): instance_count},
        {("light", "edge"): 2 if instance_count == 1 else 1},
        {stage.key: "edge" for stage in sim.waiting},
        instance_slots={stage.key: index % instance_count for index, stage in enumerate(sim.waiting)},
    )
    sim._deploy(decision)
    sim._route(0, decision)
    sim._advance(0)
    assert sum(request.completion_ms is not None for request in requests) == instance_count
    if instance_count == 1:
        assert requests[0].completion_ms == 1
        assert requests[1].completion_ms is None
        sim._advance(1)
        assert requests[1].completion_ms == 2
    else:
        assert [request.completion_ms for request in requests] == [1, 1]


def test_routes_cannot_exceed_deployed_instance_times_parallelism():
    sim = Simulator(light_scenario(), controls(), core_counts={})
    for _ in range(2):
        sim.add_request("user", "task", 0, uplink_ms=0)
    sim._expose_ready(0)
    decision = Decision(
        {("light", "edge"): 1},
        {("light", "edge"): 1},
        {stage.key: "edge" for stage in sim.waiting},
        instance_slots={stage.key: 0 for stage in sim.waiting},
    )
    with pytest.raises(ValueError, match="parallelism limit"):
        sim._deploy(decision)
    assert not sim.instances


def fork_scenario():
    services = {
        name: Service(
            name, "core", (1, 0, 1, 0), output, 1, 0.1, 0, processing_ms=duration
        )
        for name, duration, output in [("left", 1, 1), ("right", 4, 0.25), ("join", 2, 0)]
    }
    return Scenario(
        services,
        {"fork": TaskType("fork", {"left": [], "right": [], "join": ["left", "right"]}, 11, 0)},
        {name: Node(name, (4, 0, 4, 0)) for name in ("a", "b", "c")},
        [("a", "c", 1, 0), ("b", "c", 1, 0)],
        [User("user", (0, 0), "a", {"fork": 0}, 2, 1)],
    )


def test_dag_join_waits_for_all_parent_completions_and_all_data_transfers():
    settings = Settings(duration_ms=15, ec_admission_window_ms=1, record_trace=True)
    sim = Simulator(
        fork_scenario(), settings,
        core_counts={("left", "a"): 1, ("right", "b"): 1, ("join", "c"): 1},
    )
    sim.add_request("user", "fork", 0, uplink_ms=0)
    result = sim.run()
    trace = {stage["service"]: stage for stage in result["stage_trace"]}
    assert trace["left"]["finish_ms"] == 1
    assert trace["right"]["finish_ms"] == 4
    assert trace["join"]["ready_ms"] == 4
    # The early parent's larger output arrives last: 1 + 8 = 9 ms.
    assert trace["join"]["data_ready_ms"] == 9
    assert trace["join"]["start_ms"] == 9
    assert trace["join"]["finish_ms"] == 11
    assert result["overall"]["ontime_rate"] == 1


def test_sla_counts_deadline_equality_and_all_arrivals_including_pending():
    scenario = light_scenario()
    scenario.tasks["task"].deadline_ms = 1
    sim = Simulator(scenario, controls(), core_counts={})
    sim.service_rng = UnitRate()
    requests = [sim.add_request("user", "task", 0, uplink_ms=0) for _ in range(3)]
    # Deliver two requests through real compute, leave one queued at the horizon.
    sim._new_instance("light", "edge")
    sim.instances[0].jobs = [Job(Stage(request, "light", 0), 0, 1) for request in requests[:2]]
    sim._advance(0)
    sim._advance(1)
    result = sim.results()["overall"]
    assert result["arrivals"] == 3
    assert result["completed"] == 2
    assert result["ontime"] == 1
    assert result["late_completed"] == 1
    assert result["pending"] == 1
    assert result["ontime_rate"] == pytest.approx(1 / 3)
    assert result["completion_rate"] == pytest.approx(2 / 3)
