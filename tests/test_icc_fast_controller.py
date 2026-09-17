"""Cache acceleration must preserve original ICC decisions and trajectories."""

import copy

import pytest

from edge_msd.config import Settings
from edge_msd.controller import Controller
from edge_msd.icc_coupled.fast_controller import FastController
from edge_msd.models import Instance, Job, Request, Stage
from edge_msd.network import Network
from edge_msd.simulation import Simulator
from examples.minimal import make_scenario


@pytest.mark.parametrize("method", ["proposed", "propavg", "lbrr", "ga"])
def test_fast_controller_matches_original_decisions_and_complete_trajectories(method):
    settings = Settings(
        duration_ms=30, seed=13, ec_admission_window_ms=1, record_trace=True,
        ga_population=4, ga_generations=2,
    )
    simulations = [
        Simulator(make_scenario(800), settings, method, core_counts={("infer", "edge"): 2})
        for _ in range(2)
    ]
    fast = simulations[1]
    fast.controller = FastController(
        fast.scenario, fast.network, settings, fast.placement.counts, method
    )
    decisions = [[], []]
    for index, simulation in enumerate(simulations):
        original_step = simulation.controller.step

        def record_step(*args, _index=index, _step=original_step, **kwargs):
            decision = _step(*args, **kwargs)
            decisions[_index].append(copy.deepcopy(decision))
            return decision

        simulation.controller.step = record_step
    original_result, fast_result = [simulation.run() for simulation in simulations]
    assert original_result["overall"]["arrivals"] > 0
    assert decisions[0] == decisions[1]
    assert original_result == fast_result


def make_controller(kind=FastController):
    scenario = make_scenario()
    return kind(
        scenario, Network(scenario), Settings(ec_admission_window_ms=1), {}, "proposed"
    )


def test_cached_decisions_are_independent_mutable_copies():
    controller = make_controller()
    request = Request(0, "user", "pipeline", 0, 0, "edge")
    controller.step(0, [], [Stage(request, "prepare", 0)], {0: request})
    counts = {("prepare", "edge"): 1}
    objective, first = controller.evaluate(counts)
    saved = copy.deepcopy(first)
    first.counts.clear()
    first.parallelism.clear()
    first.routes.clear()
    first.predicted_ms.clear()
    first.instance_slots.clear()
    assert controller.evaluate(counts) == (objective, saved)
    assert controller.cache_statistics["evaluation_hits"] > 0


def test_busy_lower_bound_is_recomputed_after_every_control_step():
    fast = make_controller()
    original = make_controller(Controller)
    request = Request(0, "user", "pipeline", 0, 0, "edge")
    instances = [Instance(0, "prepare", "edge", [Job(Stage(request, "prepare", 0), 0, 1)])]
    for controller in (original, fast):
        controller.step(0, [], [], {})
        assert controller.feasible({})
        controller.step(1, instances, [], {0: request})
        assert not controller.feasible({})
    assert fast.evaluate({("prepare", "edge"): 1}) == original.evaluate({("prepare", "edge"): 1})


def test_invalid_bool_count_cannot_alias_valid_cached_integer_count():
    controller = make_controller()
    controller.step(0, [], [], {})
    controller.evaluate({("prepare", "edge"): 1})
    with pytest.raises(ValueError, match="nonnegative integers"):
        controller.evaluate({("prepare", "edge"): True})


def test_round_robin_evaluation_keeps_pointer_side_effects():
    request = Request(0, "user", "pipeline", 0, 0, "edge")
    controllers = [make_controller(Controller), make_controller()]
    for controller in controllers:
        controller.step(0, [], [Stage(request, "prepare", 0)], {0: request})
    counts = {("prepare", "edge"): 1, ("prepare", "server"): 1}
    for _ in range(3):
        expected = controllers[0].evaluate(counts, round_robin=True)
        actual = controllers[1].evaluate(counts, round_robin=True)
        assert actual == expected
        assert dict(controllers[0].robin) == dict(controllers[1].robin)
