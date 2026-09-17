from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from edge_msd.icc_routing.environment import Config, RoutingEnv
from edge_msd.icc_routing.learning import (
    Codec,
    Heuristic,
    SemanticNet,
    advantages,
    public_features,
    report_inputs,
    window_stats,
)
from edge_msd.model_routing.telemetry import pack_report

DATA = Path(__file__).resolve().parents[1] / "data/icc_paper/scenario_2026.json"


def config(**kwargs):
    return replace(Config(arrival_steps=8, drain_steps=10), **kwargs)


def test_icc_fields_poisson_batches_and_conservation():
    env = RoutingEnv(DATA, config(), 219, Codec("raw"), "instant")
    assert any(len(batch) > 1 for batch in env.arrivals.values())
    for r in env.all_requests:
        original = env.tasks[r.task]
        assert r.input_mb == original.input_mb
        assert r.deadline_ms == original.deadline_ms
        assert env.nodes[r.origin] == next(
            u.gateway for u in env.scenario.users if u.name == r.user
        )
    assert env.nodes == ["es1", "es2", "es3"]
    ticks = []
    router = Heuristic(Codec("raw"))
    while env.tick < env.total_steps:
        ticks.append(env.tick)
        env.step(router.select(env.observe()))
    assert len(ticks) > env.total_steps  # zero-time decisions retain every Poisson arrival
    r = env.results()
    assert r["arrivals"] == r["completed"] + r["pending"] + r["rejected"]


def test_reports_are_causal_and_shared_wire_budget():
    torch.manual_seed(1)
    model = SemanticNet()
    c = config(report_bps=32000, encoder_ms=0, propagation_ms=0)
    env = RoutingEnv(DATA, c, 221, Codec("semantic", model, 8), "event")
    before = env.observe()
    assert all(r is None for r in before["reports"])
    # 16-byte header + 8 floats = 48 bytes, not a dense 16-dimensional payload.
    assert len(pack_report(0, 0, env.codec.encode(env.history[0]), env.codec.codec_id)) == 48
    for _ in range(2):
        current = env.tick
        while env.tick == current:
            env.step(0 if env.request is not None else None)
    assert env.channel.transmitted_bytes <= 32000 * env.now_ms / 8000 + 1e-8
    assert all(r is None for r in env.reports)  # first packet takes 12 ms
    current = env.tick
    while env.tick == current:
        env.step(0 if env.request is not None else None)
    assert sum(r is not None for r in env.reports) == 1
    stored = env.reports[0]
    assert stored["tick"] == 0
    packet = pack_report(0, 0, np.zeros(8), env.codec.codec_id)
    env._receive(packet, env.now_ms)
    np.testing.assert_array_equal(env.reports[0]["payload"], stored["payload"])


def test_received_payload_and_training_window_produce_same_policy():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    for mode, codec_mode, k in (("semantic", "semantic", 8), ("stats", "stats", 16)):
        model = SemanticNet(mode=mode)
        codec = Codec(codec_mode, model, k)
        env = RoutingEnv(DATA, config(), 222, codec, "instant")
        obs = env.observe()
        z, age, present, dims = report_inputs(obs, codec)
        windows, _ = env.training_windows()
        public = public_features(obs)
        args = [torch.as_tensor(x)[None] for x in (age, present, public)]
        a, av = model.forward_z(torch.as_tensor(z)[None], *args)
        b, bv = model(torch.as_tensor(windows)[None], torch.as_tensor(dims)[None], *args)
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(av, bv, atol=1e-5, rtol=1e-5)


def test_hidden_capacity_changes_do_not_change_unreported_observations():
    model = SemanticNet()
    env = RoutingEnv(DATA, config(), 223, Codec("semantic", model), "none")
    obs = env.observe()
    env.compute_capacity[:] = 0.12
    env.link_capacity[:] = 1.2
    other = env.observe()
    assert obs["reports"] == other["reports"]
    np.testing.assert_array_equal(public_features(obs), public_features(other))
    windows, targets = env.training_windows()
    assert not np.any(windows) and not np.any(targets)


def test_resource_sharing_and_policy_independent_trace():
    c = config(load_multiplier=0.1)
    a = RoutingEnv(DATA, c, 224, Codec("raw"), "none")
    b = RoutingEnv(DATA, c, 224, Codec("latest"), "event")
    assert a.trace_hash == b.trace_hash
    while a.tick < a.total_steps:
        a.step(8 if a.request is not None else None)
    # Every directed link is shared across all requests/models, never multiplied by replica count.
    budget = (a.link_capacity[: a.total_steps] * a.edge_rate[None] * c.slot_ms).sum(0)
    assert np.all(a.link_transmitted <= budget + 1e-8)
    assert np.all(a.compute_used <= 1 + 1e-8)
    assert np.all(a.link_used <= 1 + 1e-8)


def test_actor_loss_reaches_encoder_but_critic_does_not():
    torch.manual_seed(3)
    model = SemanticNet()
    windows = torch.rand(2, 3, 8, 16)
    k = torch.full((2, 3), 8)
    public = torch.rand(2, 9, 15)
    public[..., 14] = 0  # all candidates eligible
    args = (windows, k, torch.zeros(2, 3), torch.ones(2, 3), public)
    logits, value = model(*args)
    value.sum().backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    model.zero_grad(set_to_none=True)
    logits, _ = model(*args)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 8])).backward()
    assert sum(p.grad.abs().sum() for p in model.encoder.parameters()).item() > 0


def test_zero_duration_gae_and_window_statistics():
    rows = [
        dict(reward=1.0, value=0.0, elapsed=0.0, done=False),
        dict(reward=2.0, value=0.0, elapsed=5.0, done=True),
    ]
    adv, returns = advantages(rows, 5, gamma=0.9, lam=0.8)
    np.testing.assert_allclose(adv, [3, 2])
    np.testing.assert_array_equal(adv, returns)
    x = np.random.default_rng(1).normal(size=(8, 16)).astype(np.float32)
    np.testing.assert_allclose(window_stats(x), window_stats(torch.as_tensor(x)).numpy(), atol=1e-6)


def test_zero_load_and_invalid_action():
    env = RoutingEnv(DATA, config(load_multiplier=0), 220, Codec("raw"), "none")
    while env.tick < env.total_steps:
        env.step(None)
    assert env.results()["arrivals"] == 0
    env = RoutingEnv(DATA, config(), 220, Codec("raw"), "none")
    while env.request is None:
        env.step(None)
    with pytest.raises(ValueError, match="nine"):
        env.step(9)
    assert not env.jobs


def test_gamma_rate_changes_during_an_already_running_job():
    env = RoutingEnv(DATA, config(), 225, Codec("raw"), "none")
    while env.request is None:
        env.step(None)
    r = env.request
    env.compute_capacity[:] = 1
    env.gamma_factors[:] = 1
    env._admit(r.origin * 3)  # Local small model; no network transfer.
    job = env.jobs[0]
    assert job.segments
    first_service = job.segments[0][0]
    env.gamma_factors[env.tick, r.origin, first_service] = 1e-5
    env._advance()
    assert job.finish_ms is None
    assert job.segment_index == 0
    env.tick += 1
    env._advance()
    assert job.finish_ms is not None
    assert job.finish_ms <= env.now_ms + env.config.slot_ms


def test_periodic_sweep_changes_only_reporting_not_exogenous_workload():
    hashes = []
    for period in (20, 50, 100, 200):
        c = config(report_period_ms=period)
        codec = Codec("latest")
        env = RoutingEnv(DATA, c, 231, codec, "periodic")
        router = Heuristic(codec)
        hashes.append(env.trace_hash)
        while env.tick < env.total_steps:
            env.step(router.select(env.observe()))
        samples = int(np.ceil(env.total_steps * c.slot_ms / period))
        assert env.channel.generated_bytes == samples * 3 * 80
        assert env.channel.transmitted_bytes <= env.now_ms * c.report_bps / 8000 + 1e-8
    assert len(set(hashes)) == 1


def test_no_explicit_forecast_policy_does_not_use_predictor_weights():
    torch.manual_seed(8)
    model = SemanticNet(use_forecast=False)
    z = torch.rand(2, 3, 16)
    public = torch.rand(2, 9, 15)
    public[..., 14] = 0
    args = (z, torch.full((2, 3), 50.0), torch.ones(2, 3), public)
    before, _ = model.forward_z(*args)
    model.use_forecast = True
    predictive_before, _ = model.forward_z(*args)
    with torch.no_grad():
        for parameter in model.predictor.parameters():
            parameter.add_(5)
    predictive_after, _ = model.forward_z(*args)
    assert not torch.allclose(predictive_before, predictive_after)
    model.use_forecast = False
    after, _ = model.forward_z(*args)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    after.sum().backward()
    assert all(p.grad is None for p in model.predictor.parameters())


@pytest.mark.parametrize("dimension", (4, 8))
def test_true_low_dimensional_bottleneck_matches_wire_and_training_inputs(dimension):
    torch.manual_seed(9)
    model = SemanticNet(latent_dim=dimension)
    assert model.encoder[-1].out_features == dimension
    assert model.decoder[0].in_features == dimension
    assert model.predictor[0].in_features == dimension
    codec = Codec("semantic", model, dimension)
    env = RoutingEnv(DATA, config(), 241, codec, "instant")
    obs = env.observe()
    windows, _ = env.training_windows()
    z, age, present, dims = report_inputs(obs, codec)
    assert np.all(dims == dimension)
    assert not np.any(z[:, dimension:])
    for n, report in enumerate(obs["reports"]):
        assert len(report["payload"]) == dimension
        packet = pack_report(n, report["tick"], report["payload"], codec.codec_id)
        assert len(packet) == 16 + dimension * 4
    args = [torch.as_tensor(x)[None] for x in (age, present, public_features(obs))]
    wire_logits, _ = model.forward_z(torch.as_tensor(z)[None], *args)
    train_logits, _ = model(torch.as_tensor(windows)[None], torch.as_tensor(dims)[None], *args)
    torch.testing.assert_close(wire_logits, train_logits, atol=1e-5, rtol=1e-5)
    torch.nn.functional.cross_entropy(train_logits, torch.tensor([0])).backward()
    assert model.encoder[-1].weight.grad.shape[0] == dimension
    assert model.encoder[-1].weight.grad.abs().sum() > 0
