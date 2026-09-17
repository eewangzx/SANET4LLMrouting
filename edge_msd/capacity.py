"""Gamma effective capacity and the explicit parallelism-to-delay closure.

The paper leaves g_{m,epsilon}(y) implicit. The caller must specify an admission
window to use a stationary-load proxy. A finite FIFO occupancy is NOT an arrival
rate. This closure is a scheduling approximation, not an end-to-end guarantee.
"""

from math import isfinite, log, log1p
from numbers import Integral

from scipy.optimize import brentq

from .models import Service


def effective_capacity(service: Service, theta: float, slot_ms: float = 1.0) -> float:
    """Eq. (20) for independent Gamma rates held constant within each slot."""
    if not isfinite(theta) or not isfinite(slot_ms) or theta < 0 or slot_ms <= 0:
        raise ValueError("theta must be nonnegative and slot_ms positive")
    if service.kind != "light":
        raise ValueError("Gamma effective capacity applies only to light services")
    if theta == 0:
        return service.mean_rate
    return service.gamma_shape * log1p(theta * service.gamma_scale * slot_ms) / (theta * slot_ms)


class DelayModel:
    def __init__(
        self,
        services: dict[str, Service],
        mode: str,
        epsilon: float,
        slot_ms: float,
        admission_window_ms: float | None = None,
    ):
        if (
            mode not in ("effective", "average")
            or not 0 < epsilon < 1
            or not isfinite(slot_ms)
            or slot_ms <= 0
        ):
            raise ValueError("Invalid delay model")
        if (
            mode == "effective"
            and any(s.kind == "light" for s in services.values())
            and (
                admission_window_ms is None
                or not isfinite(admission_window_ms)
                or admission_window_ms <= 0
            )
        ):
            raise ValueError("Effective mode requires an explicit admission_window_ms")
        self.services, self.mode, self.epsilon, self.slot_ms = services, mode, epsilon, slot_ms
        self.admission_window_ms = admission_window_ms
        self._cache = {}

    def delay(self, name: str, parallelism: int) -> float:
        if (
            isinstance(parallelism, bool)
            or not isinstance(parallelism, Integral)
            or parallelism < 0
        ):
            raise ValueError("parallelism must be a nonnegative integer")
        if self.services[name].kind != "light":
            raise ValueError("DelayModel applies only to light services")
        key = name, parallelism
        if key not in self._cache:
            self._cache[key] = self._compute_delay(name, parallelism)
        return self._cache[key]

    def _compute_delay(self, name: str, parallelism: int) -> float:
        if parallelism < 0:
            raise ValueError("parallelism must be nonnegative")
        if parallelism == 0:
            return 0.0
        service = self.services[name]
        mean_delay = parallelism * service.work_mb / service.mean_rate
        if self.mode == "average":
            return max(self.slot_ms, mean_delay)
        load_rate = parallelism * service.work_mb / self.admission_window_ms
        if load_rate >= service.mean_rate:
            return float("inf")  # No positive QoS exponent for an unstable load.
        # Solve in dimensionless x = theta * beta * dt. Scaling theta directly
        # makes an absolute root tolerance depend on the selected work units.
        ratio = load_rate / service.mean_rate

        def residual(x):
            return (log1p(x) / x if x else 1.0) - ratio

        upper = 1.0
        while residual(upper) > 0:
            upper *= 2
            if not isfinite(upper):
                raise ValueError("Effective-capacity proxy is outside numerical range")
        x = brentq(residual, 0.0, upper, xtol=5e-324)
        theta = x / (service.gamma_scale * self.slot_ms)
        tail_delay = log(load_rate / (service.mean_rate * self.epsilon)) / (theta * load_rate)
        return max(self.slot_ms, mean_delay, tail_delay)
