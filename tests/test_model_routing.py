"""Behavioral checks for new physics, causal telemetry and RL return accounting."""

from dataclasses import replace

import numpy as np
import pytest

from edge_msd.model_routing.environment import RoutingEnv
from edge_msd.model_routing.routers import HeuristicRouter
from edge_msd.model_routing.telemetry import NarrowbandChannel, pack_report, unpack_report
from edge_msd.model_routing.types import Job, RequestView, RoutingConfig


def small_config(**kwargs):
    return RoutingConfig(arrival_steps=20, drain_steps=30, **kwargs)


def test_serialized_size_and_no_early_delivery():
    packet = pack_report(2, 4, np.arange(16), 2)
    assert len(packet) == 16 + 16 * 4
    channel = NarrowbandChannel(8000, propagation_ms=5)
    channel.enqueue(2, packet, 0)
    assert channel.advance(79) == []
    assert channel.transmitted_bytes == pytest.approx(79)
    assert channel.advance(80) == []
    (delivery,) = channel.advance(85)
    assert delivery.finish_ms == 80
    assert delivery.arrival_ms == 85
    assert unpack_report(delivery.packet)[1] == 4


def test_latest_unsent_replaces_old_but_not_packet_in_service():
    ch = NarrowbandChannel(8000)
    packets = [pack_report(0, tick, np.zeros(16), 2) for tick in range(3)]
    ch.enqueue(0, packets[0], 0)
    assert not ch.advance(10)
    ch.enqueue(0, packets[1], 10)
    ch.enqueue(0, packets[2], 10)
    delivered = ch.advance(160)
    assert [unpack_report(d.packet)[1] for d in delivered] == [0, 2]
    assert ch.replaced_reports == 1
    assert ch.transmitted_bytes == pytest.approx(160)


def test_shared_bandwidth_not_multiplied_by_number_of_nodes():
    ch = NarrowbandChannel(8000)
    for node in range(6):
        ch.enqueue(node, pack_report(node, 0, np.zeros(16), 2), 0)
    delivered = ch.advance(160)
    assert len(delivered) == 2
    assert ch.transmitted_bytes == pytest.approx(160)


def test_observations_do_not_reveal_unreported_state_or_future_outcomes():
    env = RoutingEnv(small_config(report_bps=8000), reporting="periodic")
    before = env.observe()
    assert before["missing"].all()
    env.capacity[:] = 0.12
    env.outcomes = {key: (99999, 0.0, 99999) for key in env.outcomes}
    after = env.observe()
    np.testing.assert_array_equal(before["state"], after["state"])
    np.testing.assert_array_equal(before["public"], after["public"])
    np.testing.assert_array_equal(before["latent"], after["latent"])
    after["state"][:] = 999
    assert env.observe()["state"].max() < 999


def test_older_report_cannot_overwrite_newer_sample():
    env = RoutingEnv(small_config(), reporting="none")
    fresh = np.full(env.codec.payload_dim, 0.9, dtype=np.float32)
    old = np.full(env.codec.payload_dim, 0.2, dtype=np.float32)
    env._receive(pack_report(0, 2, fresh, env.codec.codec_id), 100)
    env._receive(pack_report(0, 1, old, env.codec.codec_id), 120)
    assert env.reports[0].sample_ms == 100
    np.testing.assert_allclose(env.reports[0].payload, fresh)


def test_capacity_changes_during_execution_change_completion_time():
    env = RoutingEnv(small_config(arrival_probability=0), reporting="none")
    req = RequestView(99, 0, 0, 0, 0, 1000)
    job = Job(req, 0, 0, 100, 0, 0.9, 0, ready_ms=0)
    env.queues[0].compute.append(job)
    env.capacity[:, 0, 1] = 0.5
    env.capacity[0, 0, 1] = 1
    env.step()
    assert job.remaining_work_ms == pytest.approx(50)
    env.step()
    assert job.remaining_work_ms == pytest.approx(25)
    assert job.compute_finish_ms is None
    env.step()
    assert job.compute_finish_ms == pytest.approx(150)


def test_one_model_per_request_and_accounting_with_pending():
    env = RoutingEnv(small_config(arrival_probability=1), reporting="none")
    env.capacity[:] = 0.12
    for _ in range(env.total_steps):
        env.step(2 if env.request is not None else None)
    result = env.results()
    assert len({j.request.id for j in env.all_jobs}) == len(env.all_jobs)
    assert all(j.endpoint == 2 for j in env.all_jobs)
    assert result["completed"] + result["pending"] + result["rejected"] == 20
    assert result["pending"] > 0


def test_external_randomness_identical_across_policies_and_report_modes():
    config = small_config(arrival_probability=0.8)
    first = RoutingEnv(config, reporting="none")
    second = RoutingEnv(replace(config, report_bps=8000), reporting="event")
    for _ in range(first.total_steps):
        first.step(0 if first.request is not None else None)
        second.step(5 if second.request is not None else None)
    assert first.trace_sha256 == second.trace_sha256
    assert first.outcomes == second.outcomes


def test_rejected_invalid_action_leaves_environment_unchanged():
    env = RoutingEnv(small_config(arrival_probability=1), reporting="none")
    with pytest.raises(ValueError, match="six"):
        env.step(6)
    assert env.tick == 0
    assert env.all_jobs == []
    assert env.cost == 0


def test_no_traffic_episode_runs_without_division_by_zero():
    env = RoutingEnv(small_config(arrival_probability=0), reporting="event")
    router = HeuristicRouter()
    for _ in range(env.total_steps):
        env.step(router.select(env.observe()))
    assert env.results()["arrivals"] == 0
    assert env.results()["success_rate"] == 0


def test_gae_does_not_bootstrap_through_terminal_boundary():
    pytest.importorskip("torch")
    from edge_msd.model_routing.learning import compute_gae

    advantages, returns = compute_gae([1, 2], [0, 0], [True, False], 100, gamma=0.9, lam=1)
    assert advantages[0] == 1
    assert returns[1] == pytest.approx(92)


def test_autoencoder_packet_and_raw_central_embedding_match():
    torch = pytest.importorskip("torch")
    from edge_msd.model_routing.learning import AECodec, RawAECodec, StateAutoencoder

    torch.set_num_threads(1)
    model = StateAutoencoder()
    history = np.random.default_rng(0).normal(size=(8, 16)).astype(np.float32)
    ae, raw = AECodec(model), RawAECodec(model)
    ae_packet = pack_report(0, 0, ae.encode(history), ae.codec_id)
    raw_packet = pack_report(0, 0, raw.encode(history), raw.codec_id)
    assert len(ae_packet) == 80 and len(raw_packet) == 528
    np.testing.assert_allclose(
        ae.embedding(unpack_report(ae_packet)[2]),
        raw.embedding(unpack_report(raw_packet)[2]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        ae.predict(unpack_report(ae_packet)[2]),
        raw.predict(unpack_report(raw_packet)[2]),
        atol=1e-6,
    )


def test_quality_guard_uses_public_predictions_and_handles_no_eligible_model():
    torch = pytest.importorskip("torch")
    from edge_msd.model_routing.learning import ActorCritic

    policy = ActorCritic(48, quality_guard=True, separate_value=True)
    x = torch.zeros(1, 6, 48)
    x[..., -1] = 0.8
    x[0, :, -12] = torch.tensor([0.5, 0.7, 0.9, 0.5, 0.7, 0.9])
    logits, value = policy(x)
    probabilities = logits.softmax(-1)
    assert probabilities[0, [0, 1, 3, 4]].sum() == 0
    assert float(probabilities[0, [2, 5]].sum().detach()) == pytest.approx(1)
    value.sum().backward()
    assert all(p.grad is None for p in policy.body.parameters())
    x[..., -12] = 0.1
    logits, _ = policy(x)
    assert torch.isfinite(logits).all()
