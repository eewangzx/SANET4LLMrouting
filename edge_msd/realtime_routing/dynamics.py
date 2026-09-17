"""Time-correlated exogenous factors: a declared synthetic extension of ICC.

The published ICC model draws light-service Gamma rates independently across slots,
so no history can predict a future rate (I(Z;Y_future) = 0). This module adds an
AR(1) multiplicative factor per (node, service) so predictive telemetry has a signal
to learn. The process is identical for every method and its parameters are fixed
before any algorithm comparison.
"""

from __future__ import annotations

import numpy as np

DEFAULTS = {"rho": 0.9, "sigma": 0.35, "low": 0.5, "high": 1.5}


class CorrelatedFactorTrace:
    """Pre-generated AR(1) factors indexed by (node, service, physical slot).

    Pre-generation (not on-the-fly sampling) keeps the exogenous trajectory
    independent of any policy, so two methods with the same seed see identical
    factors regardless of their call order.
    """

    def __init__(self, nodes, services, n_slots, seed,
                 rho=DEFAULTS["rho"], sigma=DEFAULTS["sigma"],
                 low=DEFAULTS["low"], high=DEFAULTS["high"]):
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must be in [0, 1)")
        if not sigma > 0:
            raise ValueError("sigma must be positive")
        if not 0 < low < 1 < high:
            raise ValueError("clipping bounds must satisfy 0 < low < 1 < high")
        n_slots = int(n_slots)
        if n_slots < 2:
            raise ValueError("need at least two slots")
        self.nodes = tuple(sorted(nodes))
        self.services = tuple(sorted(services))
        self.n_slots = n_slots
        self.rho, self.sigma, self.low, self.high = rho, sigma, low, high
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9001]))
        self.table = {}
        for node in self.nodes:
            for service in self.services:
                white = rng.normal(0.0, 1.0, n_slots)
                walk = np.empty(n_slots)
                walk[0] = 0.0
                step = np.sqrt(1.0 - rho * rho)
                for t in range(1, n_slots):
                    walk[t] = rho * walk[t - 1] + step * white[t]
                factor = np.exp(sigma * walk)
                factor /= factor.mean()  # unit mean before clipping
                self.table[node, service] = np.clip(factor, low, high)

    def factor(self, slot, node, service) -> float:
        t = min(max(int(slot), 0), self.n_slots - 1)
        return float(self.table[node, service][t])

    def future(self, slot, node, service, horizon) -> np.ndarray:
        return np.array(
            [self.factor(slot + h, node, service) for h in range(1, horizon + 1)],
            dtype=np.float32,
        )

    def observation(self, slot, node, service) -> np.ndarray:
        """Node-local observable: the realized factor is measured with light noise."""
        return np.float32(self.factor(slot, node, service))
