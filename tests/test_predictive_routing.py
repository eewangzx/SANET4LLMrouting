from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from edge_msd.icc_routing.environment import Config, RoutingEnv
from edge_msd.icc_routing.learning import tensors
from edge_msd.icc_routing.predictive import PredictiveCodec, PredictiveNet, ReportingEnv
from edge_msd.icc_routing.predictive_experiment import collect_rollout, distribution_diagnostics
from edge_msd.model_routing.telemetry import pack_report, unpack_report

DATA = Path(__file__).resolve().parents[1] / "data/icc_paper/scenario_2026.json"


@pytest.mark.parametrize("selection,dimension", [("dense", 4), ("importance", 8), ("random", 8), ("stats", 16), ("adaptive", 8)])
def test_exact_wire_budget_and_causal_ppo_recomputation(selection, dimension):
    torch.set_num_threads(1)
    torch.manual_seed(6)
    model = PredictiveNet(selection=selection, dimension=dimension)
    codec = PredictiveCodec(model, stochastic=selection == "adaptive")
    cfg = replace(Config(arrival_steps=8, drain_steps=5, horizon=60),
                  report_period_ms=20, encoder_ms=0, propagation_ms=0)
    env = RoutingEnv(DATA, cfg, 260, codec, "periodic")
    payload = codec.encode(env.history[0])
    packet = pack_report(0, 0, payload, codec.codec_id)
    assert len(packet) in ((24, 28, 32) if selection == "adaptive" else (80 if selection == "stats" else 32,))
    reconstructed = codec.latent(unpack_report(packet)[2])
    with torch.no_grad():
        direct = model.encode(torch.as_tensor(env.history[0])[None], torch.tensor([len(payload)]))[0].numpy()
    np.testing.assert_array_equal(reconstructed, direct)
    # Remove the extra test serialization event; the environment starts afresh.
    codec.events.clear()
    env = RoutingEnv(DATA, cfg, 260, codec, "periodic")
    rows = collect_rollout(env, model, codec)
    diagnostics = distribution_diagnostics(model, tensors(rows), cfg.slot_ms)
    assert diagnostics["max_logprob_difference"] < 1e-5
    assert diagnostics.get("budget_logprob_difference", 0) < 1e-5
    assert env.channel.transmitted_bytes <= cfg.report_bps * env.now_ms / 8000 + 1e-8


def gradient_sum(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


def test_prediction_and_policy_gradients_reach_importance_and_encoder():
    torch.manual_seed(12)
    model = PredictiveNet(selection="importance", dimension=8)
    windows = torch.rand(3, 3, 8, 16)
    public = torch.rand(3, 9, 15)
    public[..., 14] = 0
    args = (windows, torch.full((3, 3), 4), torch.zeros(3, 3), torch.ones(3, 3), public)
    logits, _ = model(*args)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 4, 8])).backward()
    assert gradient_sum(model.encoder) > 0
    assert gradient_sum(model.importance) > 0
    model.zero_grad(set_to_none=True)
    z = model.encode(windows)
    model.forecast(z).square().mean().backward()
    assert gradient_sum(model.encoder) > 0
    assert gradient_sum(model.importance) > 0
    model.zero_grad(set_to_none=True)
    _, value = model(*args)
    value.square().mean().backward()
    assert gradient_sum(model.encoder) == gradient_sum(model.importance) == 0


def test_prediction_head_is_not_an_actor_input_and_no_future_is_observed():
    model = PredictiveNet(selection="importance", dimension=8)
    cfg = Config(arrival_steps=8, horizon=60)
    env = RoutingEnv(DATA, cfg, 261, PredictiveCodec(model), "instant")
    windows, _ = env.training_windows()
    x = torch.as_tensor(windows)[None]
    z_before = model.encode(x).detach()
    public = torch.rand(1, 9, 15)
    public[..., 14] = 0
    args = (torch.zeros(1, 3), torch.ones(1, 3), public)
    logits_before, _ = model.forward_z(z_before, *args)
    env.targets[:] = 10000
    with torch.no_grad():
        for p in model.predictor.parameters():
            p.fill_(100)
    torch.testing.assert_close(model.encode(x), z_before)
    logits_after, _ = model.forward_z(z_before, *args)
    torch.testing.assert_close(logits_before, logits_after)


def test_sparse_metadata_is_checked():
    codec = PredictiveCodec(PredictiveNet(selection="importance", dimension=8))
    for mask in (1, 256, 7.5):
        with pytest.raises(ValueError):
            codec.latent(np.array([mask, 0.1, 0.2, 0.3], np.float32))


def test_bandwidth_reward_charges_actual_transmission_including_header():
    torch.manual_seed(3)
    model = PredictiveNet(selection="dense", dimension=4)
    cfg = Config(arrival_steps=5, drain_steps=5, horizon=60)
    free = ReportingEnv(DATA, cfg, 262, PredictiveCodec(model), "periodic", wire_price=0)
    paid = ReportingEnv(DATA, cfg, 262, PredictiveCodec(model), "periodic", wire_price=1)
    difference = 0
    while paid.tick < paid.total_steps:
        action = 0 if paid.request is not None else None
        _, rf, _, _ = free.step(action)
        _, rp, _, _ = paid.step(action)
        difference += rf - rp
    assert difference == pytest.approx(paid.channel.transmitted_bytes / 32)
    assert paid.trace_hash == free.trace_hash
