"""Restored ICC timing must match the original simulator under static resources."""

import numbers
from dataclasses import replace

import pytest

from edge_msd.config import Settings
from edge_msd.icc_coupled.dynamics import ResourceTrace
from edge_msd.icc_coupled.environment import CoupledSimulator, TelemetrySettings
from edge_msd.models import TaskType
from edge_msd.simulation import Simulator
from tests.test_icc_coupled_causality import scenario


class OriginalWithPairedGamma(Simulator):
    """Only bound arrivals and key IID samples; keep all original execution."""

    def __init__(self, scenario, settings, *, arrival_ms, core_counts):
        self.arrival_end_ms = arrival_ms
        self.trace_resource = ResourceTrace(scenario, settings.duration_ms, settings.seed, False)
        self.rate_tables = {}
        self.instance_positions = {}
        super().__init__(scenario, settings, core_counts=core_counts)

    def _new_instance(self, service, node):
        occupied = {self.instance_positions[instance.id] for instance in self.instances
                    if instance.service == service and instance.node == node}
        position = next(index for index in range(len(occupied)+1) if index not in occupied)
        super()._new_instance(service, node)
        self.instance_positions[self.instances[-1].id] = position

    light_rate = CoupledSimulator.light_rate

    def _generate(self, now):
        if now < self.arrival_end_ms:
            super()._generate(now)


def assert_results_equal(actual, expected):
    """All baseline fields match; allow only roundoff from rate integration."""
    if isinstance(expected, dict):
        assert set(expected) <= set(actual)
        for key in expected:
            assert_results_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert_results_equal(a, e)
    elif isinstance(expected, numbers.Real) and not isinstance(expected, (bool, int)):
        assert actual == pytest.approx(expected, abs=1e-9, rel=1e-12)
    else:
        assert actual == expected


@pytest.mark.parametrize("seed", [17, 29])
def test_original_static_execution_matches_coupled_full_trace_with_paired_gamma(seed):
    source = scenario()
    for name in source.light:
        source.services[name] = replace(source.services[name], gamma_scale=2)
    source.tasks["task"] = TaskType("task", {"s0": [], "core": ["s0"], "s1": ["core"]}, 40, 0)
    source.users[0].rates_per_second["task"] = 1500
    settings = Settings(duration_ms=60, seed=seed, ec_admission_window_ms=1, record_trace=True)
    counts = {("core", "b"): 1}
    original = OriginalWithPairedGamma(source, settings, arrival_ms=10, core_counts=counts)
    coupled = CoupledSimulator(source, settings,
        TelemetrySettings(arrival_ms=10, control_ms=1, dynamic=False),
        observation="prior", core_counts=counts)
    expected, actual = original.run(), coupled.run()
    assert expected["overall"]["arrivals"] > 0
    assert expected["overall"]["completed"] > 0
    assert_results_equal(actual, expected)
