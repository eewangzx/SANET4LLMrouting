"""Algorithm controls only. All topology, service and traffic data are caller-owned."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Settings:
    duration_ms: int = 1000
    slot_ms: float = 1.0
    seed: int = 0
    wireless: bool = False
    record_trace: bool = False
    epsilon: float = 0.2
    # REQUIRED for Proposed with light services. y jobs per this interval is
    # the declared stationary proxy, not an inferred property of a finite queue.
    ec_admission_window_ms: float | None = None
    alpha: float = 3.0
    eta: float = 1.0
    queue_floor: float = 1.0
    decay: float = 0.01
    urgency_floor: float = -1.0
    diversity_retention: float = 0.95
    min_core_pairs: int | None = None
    max_parallelism: int = 20
    max_greedy_steps: int = 1000
    deferral_multiplier: float = 2.0
    ga_population: int = 10
    ga_generations: int = 30
    ga_mutation: float = 0.1
    ga_max_instances: int = 3

    def __post_init__(self):
        integers = (
            self.duration_ms,
            self.seed,
            self.max_parallelism,
            self.max_greedy_steps,
            self.ga_population,
            self.ga_generations,
            self.ga_max_instances,
        )
        if any(type(x) is not int for x in integers):
            raise ValueError("Duration, seed and count settings must be integers")
        if self.seed < 0 or self.duration_ms <= 0:
            raise ValueError("Seed must be nonnegative; duration must be positive")
        if not isfinite(self.slot_ms) or self.slot_ms <= 0:
            raise ValueError("slot_ms must be finite and positive")
        if abs(self.duration_ms / self.slot_ms - round(self.duration_ms / self.slot_ms)) > 1e-9:
            raise ValueError("duration_ms must be an integer multiple of slot_ms")
        if any(not isfinite(x) or x < 0 for x in (self.alpha, self.eta, self.decay)):
            raise ValueError("Objective weights must be finite and nonnegative")
        if not isfinite(self.queue_floor) or self.queue_floor <= 0:
            raise ValueError("queue_floor must be finite and positive")
        if not isfinite(self.deferral_multiplier) or self.deferral_multiplier <= 0:
            raise ValueError("deferral_multiplier must be finite and positive")
        if not isfinite(self.urgency_floor):
            raise ValueError("urgency_floor must be finite")
        if not 0 < self.epsilon < 1 or not 0 < self.diversity_retention <= 1:
            raise ValueError("epsilon must be in (0,1); retention in (0,1]")
        if self.ec_admission_window_ms is not None and (
            not isfinite(self.ec_admission_window_ms) or self.ec_admission_window_ms <= 0
        ):
            raise ValueError("ec_admission_window_ms must be finite and positive")
        if self.max_parallelism < 1 or self.max_greedy_steps < 1 or self.ga_population < 4:
            raise ValueError("Invalid parallelism/iteration/population limit")
        if self.ga_generations < 1 or self.ga_max_instances < 1 or not 0 <= self.ga_mutation <= 1:
            raise ValueError("Invalid GA settings")
        if self.min_core_pairs is not None and (
            type(self.min_core_pairs) is not int or self.min_core_pairs < 0
        ):
            raise ValueError("min_core_pairs must be a nonnegative integer")
