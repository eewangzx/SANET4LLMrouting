"""Bounded callback tests; no ICC training sweep is launched here."""

import random

import numpy as np
import pytest
import torch

from edge_msd.icc_coupled.codecs import CoupledCodec
from edge_msd.icc_coupled.ppo import HierarchicalPPO, interval_reward, train_policy


def observation(policy, x, time=0, cost=0, violations=0, report_bytes=0):
    codec = policy.codec
    latent = np.stack([codec.decode_wire(codec.encode_wire(w)) for w in x])
    return {"latents": latent, "ages": np.full(len(x), 5, np.float32),
            "present": np.ones(len(x), np.float32), "public": np.zeros((len(x), 4)),
            "global": np.array([0.3, 0.2, 0.1, 0.1, 1 - time / 20]),
            "accounting": {"cost": cost, "violations": violations,
                           "report_bytes": report_bytes, "time_ms": time}}


class MockSimulator:
    def __init__(self, policy, seed):
        self.policy, self.seed = policy, seed

    def run(self):
        rng = np.random.default_rng(self.seed)
        x = rng.uniform(0.3, 1.1, (2, 8, 16)).astype(np.float32)
        y = np.repeat(x[:, -1:, :14], self.policy.codec.horizon, axis=1)
        cost, violations = 0.0, 0
        for step in range(3):
            obs = observation(self.policy, x, time=step * 5, cost=cost, violations=violations,
                              report_bytes=step * 32)
            eta, cap = self.policy.act(obs, {"windows": x, "targets": y})
            cost += 100 / eta + cap
            violations += int(cap == 2)
        metrics = {"accounting": {"cost": cost, "violations": violations,
                                  "report_bytes": 96, "time_ms": 20},
                   "overall": {"arrivals": 3, "ontime_rate": 1 - violations / 3}}
        self.policy.finish(metrics)
        return metrics


def test_interval_cost_and_elapsed_terminal_mc():
    a = {"cost": 0, "violations": 0, "report_bytes": 0, "time_ms": 0}
    b = {"cost": 100, "violations": 1, "report_bytes": 32, "time_ms": 5}
    assert interval_reward(a, b) == pytest.approx(-3.01)
    with pytest.raises(ValueError):
        interval_reward(b, a)
    policy = HierarchicalPPO(CoupledCodec("dense4", horizon=3), stochastic=False)
    x = np.ones((2, 8, 16), np.float32)
    policy.act(observation(policy, x))
    policy.act(observation(policy, x, time=5, cost=100, violations=1, report_bytes=32))
    policy.finish({"accounting": {"cost": 200, "violations": 1, "report_bytes": 64,
                                   "time_ms": 15}})
    ep = policy.episode()
    assert ep["transitions"][0]["discount"] == pytest.approx(0.995 ** 5)
    assert ep["discounted_return"] == pytest.approx(-3.01 + 0.995 ** 5 * -1.01)
    assert ep["transitions"][-1]["mc_return"] == pytest.approx(-1.01 * policy.reward_scale)
    assert ep["optimization_discounted_return"] == pytest.approx(
        ep["discounted_return"] * policy.reward_scale)


def test_actor_ignores_source_windows_and_future_labels():
    policy = HierarchicalPPO(CoupledCodec("importance", horizon=3), joint=True, stochastic=False)
    x = np.full((2, 8, 16), 0.7, np.float32)
    obs = observation(policy, x)
    a = policy.act(obs, {"windows": x, "targets": np.zeros((2, 3, 14), np.float32)})
    policy.reset_episode(stochastic=False)
    b = policy.act(obs, {"windows": x * 100, "targets": np.ones((2, 3, 14), np.float32) * 100})
    assert a == b


def test_critic_does_not_backpropagate_to_encoder():
    policy = HierarchicalPPO(CoupledCodec("importance", horizon=3), joint=True)
    x = torch.rand(4, 8, 16)
    z = policy.codec.representation(x)[0].reshape(2, 2, -1)
    _, value = policy(z, torch.zeros(2, 2), torch.ones(2, 2),
                      torch.zeros(2, 2, 4), torch.zeros(2, 5))
    value.sum().backward()
    assert all(p.grad is None for p in policy.codec.parameters())


@pytest.mark.parametrize("joint", [False, True])
def test_mock_training_replay_freeze_and_save(tmp_path, joint):
    codec = CoupledCodec("importance", horizon=3)
    original = codec.encoding_hash()
    policy, logs = train_policy(MockSimulator, codec, tmp_path, seed=7, updates=2,
                                episodes_per_update=2, validation_seeds=(6000,),
                                validation_every=2, joint=joint, minibatch_size=4)
    assert codec.encoding_hash() == original
    if joint:
        assert policy.codec.encoding_hash() != original
    else:
        assert policy.codec.encoding_hash() == original
    assert all(r["replay_logprob_max_error"] < 1e-4 for r in logs["updates"])
    assert {r["deployment"] for r in logs["validation"][0]["episodes"]} == {"greedy", "sampled"}
    assert len(logs["validation"]) == 2
    loaded = HierarchicalPPO.load(tmp_path / "last.pt")
    assert loaded.codec.encoding_hash() == policy.codec.encoding_hash()
    x = np.ones((2, 8, 16), np.float32)
    policy.reset_episode(stochastic=False)
    assert loaded.act(observation(loaded, x)) == policy.act(observation(policy, x))


def _assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for first, second in zip(a, b, strict=True):
            _assert_nested_equal(first, second)
    else:
        assert a == b


@pytest.mark.parametrize("joint", [False, True])
def test_resume_matches_continuous_training_and_does_not_repeat_seeds(tmp_path, joint):
    torch.manual_seed(27)
    codec = CoupledCodec("importance", horizon=3)
    settings = dict(seed=7, episodes_per_update=2, validation_seeds=(6000,),
                    validation_every=3, joint=joint, minibatch_size=4,
                    train_seed_base=4100, run_signature={"mock": 1})
    continuous, continuous_logs = train_policy(MockSimulator, codec, tmp_path / "continuous",
                                              updates=4, **settings)
    train_policy(MockSimulator, codec, tmp_path / "split", updates=2, **settings)
    seeds = []

    def factory(policy, episode_seed):
        seeds.append(episode_seed)
        return MockSimulator(policy, episode_seed)

    # Embedded, checkpoint-aligned history must suffice even if the JSON was lost.
    (tmp_path / "split" / "training.json").unlink()
    resumed, resumed_logs = train_policy(factory, None, tmp_path / "resumed", updates=4,
        resume_path=tmp_path / "split" / "last.pt", **settings)
    assert [seed for seed in seeds if seed < 6000] == [4104, 4105, 4106, 4107]
    assert resumed.update_number == 4
    assert (tmp_path / "resumed" / "best.pt").exists()
    assert resumed_logs["best_checkpoint_available"]
    _assert_nested_equal(continuous.state_dict(), resumed.state_dict())
    _assert_nested_equal(continuous.optimizer.state_dict(), resumed.optimizer.state_dict())
    _assert_nested_equal(continuous_logs["updates"], resumed_logs["updates"])
    with pytest.raises(ValueError, match="configuration differs"):
        train_policy(factory, None, tmp_path / "invalid", updates=5,
                     resume_path=tmp_path / "resumed" / "last.pt",
                     **{**settings, "episodes_per_update": 3})
    with pytest.raises(ValueError, match="target total"):
        train_policy(factory, None, tmp_path / "invalid", updates=3,
                     resume_path=tmp_path / "resumed" / "last.pt", **settings)
    with pytest.raises(ValueError, match="configuration differs"):
        train_policy(factory, None, tmp_path / "invalid", updates=5, reward_scale=1.0,
                     resume_path=tmp_path / "resumed" / "last.pt", **settings)


def test_checkpoint_restores_rng_and_rejects_legacy_exact_resume(tmp_path):
    policy = HierarchicalPPO(CoupledCodec("dense4", horizon=3), seed=19)
    policy.save(tmp_path / "initial.pt")
    expected = (torch.rand(3), np.random.rand(3), random.random(),
                torch.rand(3, generator=policy.action_rng))
    torch.manual_seed(500)
    np.random.seed(501)
    random.seed(502)
    restored = HierarchicalPPO.load(tmp_path / "initial.pt", resume_optimizer=True)
    assert torch.equal(torch.rand(3), expected[0])
    np.testing.assert_array_equal(np.random.rand(3), expected[1])
    assert random.random() == expected[2]
    assert torch.equal(torch.rand(3, generator=restored.action_rng), expected[3])
    assert restored.optimizer is None  # Exactly uninitialized at update zero.
    legacy = torch.load(tmp_path / "initial.pt", weights_only=True)
    legacy["format_version"] = 1
    legacy.pop("optimizer_state")
    legacy.pop("rng_state")
    legacy["policy_config"].pop("reward_scale")
    legacy["policy_config"].pop("gae_lambda_unit")
    torch.save(legacy, tmp_path / "legacy.pt")
    old = HierarchicalPPO.load(tmp_path / "legacy.pt")
    assert old.update_number == 0
    assert old.reward_scale == 1.0 and old.gae_lambda_unit == "decision"
    with pytest.raises(ValueError, match="weights-only"):
        HierarchicalPPO.load(tmp_path / "legacy.pt", resume_optimizer=True)


def test_fixed_scaling_preserves_physical_objective_and_gae_uses_elapsed_ms():
    x = np.ones((2, 8, 16), np.float32)
    policy = HierarchicalPPO(CoupledCodec("dense4", horizon=3), stochastic=False)
    # Zero critic makes the two-step GAE attenuation analytically checkable.
    for parameter in policy.critic.parameters():
        parameter.data.zero_()
    policy.act(observation(policy, x))
    policy.act(observation(policy, x, time=10, cost=100))
    policy.finish({"accounting": {"cost": 200, "violations": 1, "report_bytes": 32,
                                   "time_ms": 30}})
    ep = policy.episode()
    assert ep["reward"] == pytest.approx(-4.01)
    assert ep["transitions"][0]["advantage"] == pytest.approx(
        -0.001 + (0.995 * 0.995) ** 10 * -0.00301)
    assert ep["transitions"][0]["mc_return"] == pytest.approx(
        ep["transitions"][0]["physical_mc_return"] / 1000)
