"""Causal telemetry and accounting checks for the restored ICC simulator."""

import numpy as np
import pytest

from edge_msd.config import Settings
from edge_msd.icc_coupled.environment import CoupledSimulator, TelemetrySettings
from edge_msd.models import Job, Node, Scenario, Service, Stage, TaskType, User


def scenario():
    services = {
        f"s{i}": Service(f"s{i}", "light", (1, 0, 1, 0), 1, 1, .1, .1,
                        work_mb=1, gamma_shape=2, gamma_scale=.5)
        for i in range(9)
    }
    services["core"] = Service("core", "core", (1, 0, 1, 0), 0, 2, .2, 0,
                                processing_ms=.1)
    return Scenario(services, {"task": TaskType("task", {"s0": []}, 5, 0)},
                    {n: Node(n, (10, 0, 10, 0)) for n in ("a", "b")},
                    [("a", "b", 1, 0)],
                    [User("user", (0, 0), "a", {"task": 0}, 2, 1)])


def simulator(*, observation="prior", codec=None, control_ms=1, core_counts=None):
    return CoupledSimulator(scenario(),
        Settings(duration_ms=20, slot_ms=1, seed=17, ec_admission_window_ms=1),
        TelemetrySettings(arrival_ms=10, control_ms=control_ms, dynamic=False), codec,
        observation, {} if core_counts is None else core_counts)


def test_instant_reference_uses_current_state_not_future_trace():
    left, right = simulator(observation="instant"), simulator(observation="instant")
    current = right.trace_resource.index(0)
    right.trace_resource.values[current+1:, :, 9:14] *= .25
    for sim in (left, right):
        sim._telemetry(0)
        sim.network.update(0)
    np.testing.assert_array_equal(left._current_forecast, right._current_forecast)
    assert left.observed_network.delay("a", "b", 2) == right.observed_network.delay("a", "b", 2)
    # Ground-truth propagation may integrate the future exogenous process;
    # that exact finish must remain on the physical side of the interface.
    assert left.network.delay("a", "b", 2) < right.network.delay("a", "b", 2)


def test_forecast_bins_follow_absolute_sample_boundaries_and_clamp_the_past():
    sim = simulator()
    prediction = np.ones((2, 61, 14), dtype=np.float32)
    prediction[:, :, 9] = .2 + .01*np.arange(61)[None, :]
    sim.observed_network.update(6, prediction)
    for when, index in [(0, 0), (5, 0), (6, 0), (9.99, 0), (10, 1), (15, 2), (500, 60)]:
        assert sim.observed_network.factor("a", "b", when) == pytest.approx(prediction[0, index, 9])


def test_observed_instances_hide_true_future_transfer_finish_and_remaining_work():
    sim = simulator()
    req = sim.add_request("user", "task", 0, uplink_ms=0)
    sim._new_instance("s0", "a")
    job = Job(Stage(req, "s0", 0), 100, .2)
    sim.instances[0].jobs.append(job)
    sim.estimated_ready[job.stage.key] = 7
    observed = sim._observed_instances()[0].jobs[0]
    assert observed.data_ready_ms == 7
    assert observed.remaining_work == 1
    assert observed.finish_ms is None
    job.data_ready_ms, job.remaining_work, job.finish_ms = 1000, .9, 2000
    assert sim._observed_instances()[0].jobs[0] == observed


def test_controller_decision_is_invariant_to_hidden_transfer_finish_perturbation():
    simulations = [simulator(), simulator()]
    decisions = []
    for index, sim in enumerate(simulations):
        busy_request = sim.add_request("user", "task", 0, uplink_ms=0)
        busy_request.scheduled.add("s0")
        waiting_request = sim.add_request("user", "task", 0, uplink_ms=0)
        sim._new_instance("s0", "a")
        job = Job(Stage(busy_request, "s0", 0), 10+1000*index, .1+.8*index)
        sim.instances[0].jobs.append(job)
        sim.estimated_ready[job.stage.key] = 7
        sim._telemetry(0)
        sim._expose_ready(0)
        assert sim.waiting[0].request is waiting_request
        decisions.append(sim.controller.step(0, sim._observed_instances(), sim.waiting, sim.active))
    assert decisions[0] == decisions[1]


def test_past_transfer_readiness_cannot_depend_on_future_channel_values():
    simulations = [simulator(), simulator()]
    readiness = []
    for index, sim in enumerate(simulations):
        sim.scenario.tasks["task"].input_mb = 2
        request = sim.add_request("user", "task", 0, uplink_ms=0)
        stage = Stage(request, "s0", 0)
        sim.network.update(50)
        if index:
            sim.trace_resource.values[sim.trace_resource.index(50):, :, 9:14] *= .25
        readiness.append(sim.network.data_ready(stage, "b"))
    # Original ICC starts root forwarding at uplink readiness (0 ms), so this
    # 2 MB / 1 Gbit/s transfer finished at 16 ms in both physical histories.
    # Channel changes at/after 50 ms cannot alter the already finished transfer.
    assert readiness == pytest.approx([16, 16])


class FixedWireCodec:
    kind = "causality_test"

    def encode_wire(self, history):
        return np.full(4, .33, dtype=np.float32)

    def forecast_wire(self, payload):
        assert payload.shape == (4,)
        return np.full((61, 14), payload[0], dtype=np.float32)


def test_undelivered_reports_cannot_change_the_controller_prior():
    sim = simulator(observation="periodic", codec=FixedWireCodec())

    def no_future(*_args, **_kwargs):
        raise AssertionError("Receiver must not query labels or future state")

    sim.trace_resource.future = no_future
    for now in range(6):
        sim._telemetry(now)
        assert all(item is None for item in sim.received)
        np.testing.assert_array_equal(sim._current_forecast, np.ones_like(sim._current_forecast))
    # 32 B / 64 kbit/s = 4 ms, plus 0.2 ms encoding and 1 ms propagation.
    sim._telemetry(6)
    assert sim.received[0] is not None
    assert sim.received[1] is None
    np.testing.assert_allclose(sim._current_forecast[0], .33)
    np.testing.assert_array_equal(sim._current_forecast[1], np.ones_like(sim._current_forecast[1]))


@pytest.mark.parametrize("control_ms", [1, 5])
def test_maintenance_is_charged_per_physical_slot_and_core_placement_is_fixed(control_ms):
    sim = simulator(control_ms=control_ms, core_counts={("core", "a"): 1})
    req = sim.add_request("user", "task", 0, uplink_ms=0)
    req.scheduled.add("s0")
    sim._new_instance("s0", "a")
    job = Job(Stage(req, "s0", 0), 1000, 1)
    sim.instances[-1].jobs.append(job)
    sim.estimated_ready[job.stage.key] = 1000
    result = sim.run()
    assert result["costs"]["core_maintenance"] == pytest.approx(20*.2)
    assert result["costs"]["light_maintenance"] == pytest.approx(20*.1)
    assert result["costs"]["light_parallel"] == pytest.approx(20*.1)
    assert result["core_counts"] == {"core@a": 1}
    assert sum(instance.service == "core" for instance in sim.instances) == 1


def test_keyed_gamma_is_invariant_to_unrelated_instance_creation_and_query_order():
    left, right = simulator(), simulator()
    left._new_instance("s0", "a")
    right._new_instance("s1", "b")
    right._new_instance("s0", "a")
    expected = [left.light_rate(left.instances[0], tick) for tick in range(3)]
    _ = right.light_rate(right.instances[0], 2)
    actual = [right.light_rate(right.instances[1], tick) for tick in range(3)]
    assert actual == expected
    # Separate physical instance positions retain independent Gamma draws.
    right._new_instance("s0", "a")
    assert right.light_rate(right.instances[-1], 0) != actual[0]
