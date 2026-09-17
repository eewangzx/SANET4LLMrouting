"""Policies consume observations only; no reference to RoutingEnv is passed."""

import numpy as np


def effective_rates(observation, predictive=False):
    rates = np.clip(observation["state"][:, :2], 0.12, 1.2).copy()
    if predictive:
        for n in range(6):
            if not observation["has_forecast"][n]:
                continue
            age_step = int(observation["age_ms"][n] / observation["slot_ms"])
            horizon = observation["forecast"].shape[1]
            first = min(max(age_step - 1, 0), horizon - 1)
            last = min(first + 5, horizon)
            # Forecasts refer to sample time, never pretend an old packet is current.
            rates[n] = observation["forecast"][n, first:last].mean(axis=0)
    return np.clip(rates, 0.12, 1.2)


def routing_scores(observation, predictive=False, query_only=False):
    if observation["request"] is None:
        return np.zeros(6, dtype=np.float32)
    req, public = observation["request"], observation["public"]
    state = observation["state"].copy()
    rates = effective_rates(observation, predictive)
    if query_only:
        state[:] = 0
        rates[:] = 0.6
    link, compute = rates[:, 0], rates[:, 1]
    # Known 2 Mbit/s per-endpoint request link, distinct from telemetry channel.
    upload = (np.maximum(0, state[:, 3]) * 0.05 + req.input_mb) / (0.00025 * link)
    work = np.maximum(0, state[:, 4]) * 2000 + public[:, 1] * req.deadline_ms
    response = req.predicted_output_tokens * 4 / 1e6 / (0.00025 * link)
    predicted_latency = upload + work / compute + response + 4
    quality_probability = 1 / (
        1 + np.exp(np.clip(-(public[:, 0] - observation["query"][-1]) / 0.025, -30, 30))
    )
    deadline_probability = 1 / (
        1 + np.exp(np.clip((predicted_latency / req.deadline_ms - 1) * 5, -30, 30))
    )
    return (
        3 * quality_probability * deadline_probability
        - 0.12 * public[:, 3]
        - 0.1 * predicted_latency / req.deadline_ms
    ).astype(np.float32)


class HeuristicRouter:
    def __init__(self, predictive=False, query_only=False):
        self.predictive, self.query_only = predictive, query_only
        self.counter = 0

    def select(self, observation):
        if observation["request"] is None:
            return None
        scores = routing_scores(observation, self.predictive, self.query_only)
        best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-6))
        action = int(best[self.counter % len(best)])
        self.counter += 1
        return action


def policy_features(observation, predictive=True):
    rates = effective_rates(observation, predictive)
    age = np.minimum(observation["age_ms"] / 2000, 5)[:, None]
    missing = observation["missing"][:, None]
    query = np.repeat(observation["query"][None, :], 6, axis=0)
    # Decoded quantities are deterministically derived at the receiver from the
    # received latent. No fresh/hidden queue features are appended here.
    return np.concatenate(
        (
            observation["latent"],
            observation["state"],
            rates,
            age,
            missing,
            observation["public"],
            query,
        ),
        axis=1,
    ).astype(np.float32)
