from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from edge_msd.config import Settings
from edge_msd.controller import Controller
from edge_msd.models import Decision, Instance, Job, Request, Stage
from edge_msd.network import Network
from edge_msd.placement import (
    core_requirements,
    place_core,
    validate_core_budget,
    validate_resources,
)
from edge_msd.simulation import Simulator
from examples.minimal import make_scenario
from tests.test_model import branching_scenario

CORE_COUNTS = {("a", "n0"): 1, ("b", "n1"): 1, ("join", "n2"): 1}


def settings(**kwargs):
    return Settings(ec_admission_window_ms=1, **kwargs)


def test_join_waits_for_both_branches_and_transfers():
    sim = Simulator(
        branching_scenario(), settings(duration_ms=25, record_trace=True), core_counts=CORE_COUNTS
    )
    sim.add_request("u", "fork", 0, uplink_ms=0)
    result = sim.run()
    trace = {stage["service"]: stage for stage in result["stage_trace"]}
    assert trace["a"]["finish_ms"] == 1
    assert trace["b"]["finish_ms"] == 2
    assert trace["join"]["start_ms"] == 17
    assert trace["join"]["finish_ms"] == 18
    assert result["overall"]["ontime"] == 1
    assert result["costs"]["total"] == 3 * 20 + 25 * 3 * 4


def test_horizon_does_not_drop_pending():
    sim = Simulator(branching_scenario(), settings(duration_ms=10), core_counts=CORE_COUNTS)
    sim.add_request("u", "fork", 0, uplink_ms=0)
    result = sim.run()
    assert result["overall"]["pending"] == 1
    assert result["overall"]["mean_completed_latency_ms"] is None
    with pytest.raises(RuntimeError, match="fresh"):
        sim.run()


def test_zero_traffic_and_future_arrival():
    sim = Simulator(make_scenario(), settings(duration_ms=5))
    assert sim.run()["overall"]["ontime_rate"] is None
    sim = Simulator(make_scenario(), settings(duration_ms=5))
    req = sim.add_request("user", "pipeline", 4, uplink_ms=0)
    sim._expose_ready(0)
    sim.controller.step(0, sim.instances, sim.waiting, sim.active)
    assert not sim.waiting and req.id not in sim.controller.queue


def test_core_milp_matches_small_exhaustive_search():
    scenario = branching_scenario()
    controls = settings(alpha=0, min_core_pairs=0)
    result = place_core(scenario, Network(scenario), controls)
    # Each service costs 24 per instance and requires one. Enumerate binary
    # placements; any extra count >1 can only increase this positive objective.
    keys = [(s, n) for s in scenario.core for n in scenario.nodes]
    feasible_costs = []
    for bits in product((0, 1), repeat=len(keys)):
        counts = dict(zip(keys, bits))
        if any(
            sum(v for (s, _), v in counts.items() if s == name) < required
            for name, required in core_requirements(scenario).items()
        ):
            continue
        try:
            validate_resources(scenario, counts)
        except ValueError:
            continue
        feasible_costs.append(24 * sum(bits))
    assert result.objective == min(feasible_costs) == 72


def test_core_diversity_capacity_and_invalid_method():
    scenario = branching_scenario()
    controls = settings(min_core_pairs=4)
    result = place_core(scenario, Network(scenario), controls)
    assert result.nonzero_pairs >= 4
    validate_resources(scenario, result.counts)
    with pytest.raises(ValueError, match="not solved"):
        place_core(scenario, Network(scenario), replace(controls, min_core_pairs=10))
    with pytest.raises(ValueError, match="Unknown method"):
        place_core(scenario, Network(scenario), controls, "typo")


def test_requirements_scale_with_rate_and_processing_time():
    scenario = branching_scenario()
    scenario.users[0].rates_per_second["fork"] = 1200
    assert core_requirements(scenario) == {"a": 2, "b": 3, "join": 2}


def test_online_cannot_remove_busy_or_overwrite_core():
    sim = Simulator(make_scenario(), settings(duration_ms=10))
    req = sim.add_request("user", "pipeline", 0, uplink_ms=0)
    sim._new_instance("prepare", "edge")
    sim.instances[-1].jobs.append(Job(Stage(req, "prepare", 0), 2, 1))
    with pytest.raises(ValueError, match="busy"):
        sim._deploy(Decision({}, {}, {}))
    with pytest.raises(ValueError, match="core"):
        sim._deploy(Decision({("infer", "edge"): 1}, {}, {}))


def test_virtual_queue_updates_once_per_request_and_control_time():
    sim = Simulator(branching_scenario(), settings(duration_ms=100), core_counts=CORE_COUNTS)
    req = sim.add_request("u", "fork", 0, uplink_ms=0)
    sim._expose_ready(0)
    assert len(sim.waiting) == 2
    sim.controller.step(0, sim.instances, sim.waiting, sim.active)
    sim.controller.step(50, sim.instances, sim.waiting, sim.active)
    assert sim.controller.queue[req.id] == 21
    with pytest.raises(ValueError, match="increasing"):
        sim.controller.step(50, sim.instances, sim.waiting, sim.active)


def test_predictions_use_final_shared_load_and_wait_for_busy_input():
    scenario = make_scenario()
    scenario.services["prepare"] = replace(scenario.services["prepare"], gamma_scale=0.5)
    ctl = Controller(scenario, Network(scenario), settings(), {}, "propavg")
    active = {i: Request(i, "user", "pipeline", 0, 0, "edge") for i in range(3)}
    waiting = [Stage(active[i], "prepare", 0) for i in (1, 2)]
    job = Job(Stage(active[0], "prepare", 0), 7, 1)
    ctl.step(0, [Instance(0, "prepare", "edge", [job])], waiting, active)
    value, decision = ctl.evaluate({("prepare", "edge"): 1})
    # Rate mean=1, final occupancy=3, input wait=7. Both predictions must be 10,
    # not the earlier partial-load estimate of 9 for the first waiting stage.
    assert list(decision.predicted_ms.values()) == [10, 10]
    assert value == pytest.approx(ctl.slot_cost(decision.counts) + 20)


def test_route_executes_the_evaluated_instance_slot():
    sim = Simulator(make_scenario(), settings(), core_counts={})
    sim._new_instance("prepare", "edge")
    sim._new_instance("prepare", "edge")
    sim.add_request("user", "pipeline", 0, uplink_ms=0)
    sim._expose_ready(0)
    stage = sim.waiting[0]
    decision = Decision(
        {("prepare", "edge"): 2},
        {("prepare", "edge"): 1},
        {stage.key: "edge"},
        instance_slots={stage.key: 1},
    )
    expected_id = sim.instances[1].id
    sim._deploy(decision)
    sim._route(0, decision)
    assert next(i.id for i in sim.instances if i.jobs) == expected_id


def test_invalid_route_is_rejected_before_deployment():
    sim = Simulator(make_scenario(), settings(), core_counts={})
    req = sim.add_request("user", "pipeline", 0, uplink_ms=0)
    sim._expose_ready(0)
    decision = Decision(
        {("prepare", "edge"): 1},
        {("prepare", "edge"): 1},
        {(req.id, "prepare"): "edge"},
        instance_slots={(req.id, "prepare"): 2},
    )
    with pytest.raises(ValueError, match="slot"):
        sim._deploy(decision)
    assert sim.instances == []


def test_light_compute_spends_real_time_and_costs_match():
    sim = Simulator(make_scenario(), settings(duration_ms=10, record_trace=True))

    class FixedRate:
        def gamma(self, shape, scale):
            return 1.0

    sim.service_rng = FixedRate()
    sim.add_request("user", "pipeline", 0, uplink_ms=0)
    result = sim.run()
    trace = next(t for t in result["stage_trace"] if t["service"] == "prepare")
    assert trace["finish_ms"] - trace["start_ms"] == pytest.approx(1)
    assert result["costs"]["light_deployment"] == 1
    assert result["costs"]["light_maintenance"] == pytest.approx(0.1)
    assert result["costs"]["light_parallel"] == pytest.approx(0.1)


def test_all_methods_share_demand_and_do_not_reset_global_rng():
    controls = settings(duration_ms=100, ga_population=4, ga_generations=2)
    np.random.seed(12)
    expected = np.random.random(3)
    np.random.seed(12)
    results = [
        Simulator(make_scenario(100), controls, method).run()
        for method in ("proposed", "propavg", "lbrr", "ga")
    ]
    assert np.array_equal(expected, np.random.random(3))
    assert len({r["arrival_sha256"] for r in results}) == 1
    assert results[0] == Simulator(make_scenario(100), controls).run()
    assert results[0]["core_placement"] == results[1]["core_placement"]
    for result in results:
        counts = result["overall"]
        assert 0 < counts["completed"] <= counts["arrivals"]
        assert counts["completed"] + counts["pending"] == counts["arrivals"]


@pytest.mark.parametrize("method", ["proposed", "propavg", "lbrr", "ga"])
def test_core_stage_respects_explicit_light_reserve(method):
    scenario = make_scenario(100)
    result = place_core(scenario, Network(scenario), settings(), method)
    validate_core_budget(scenario, result.counts)
    assert all(count <= 2 for count in result.counts.values())
    with pytest.raises(ValueError, match="reserved"):
        Simulator(scenario, settings(), core_counts={("infer", "edge"): 3})


def test_greedy_result_has_no_improving_single_instance_addition():
    scenario = make_scenario()
    active = {i: Request(i, "user", "pipeline", 0, 0, "edge") for i in range(6)}
    waiting = [Stage(req, "prepare", 0) for req in active.values()]
    ctl = Controller(scenario, Network(scenario), settings(), {}, "proposed")
    decision = ctl.step(0, [], waiting, active)
    final_value = ctl.evaluate(decision.counts)[0]
    assert ctl.greedy_limit_hits == 0
    assert len(decision.routes) == 6
    for node in scenario.nodes:
        counts = decision.counts.copy()
        key = "prepare", node
        counts[key] = counts.get(key, 0) + 1
        if ctl.feasible(counts):
            assert ctl.evaluate(counts)[0] >= final_value - 1e-9
