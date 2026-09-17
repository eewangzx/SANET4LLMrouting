"""Public configuration and private simulation records (ms, decimal MB)."""

from dataclasses import dataclass, field
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class RoutingConfig:
    seed: int = 1
    slot_ms: float = 50.0
    arrival_steps: int = 400
    drain_steps: int = 120
    arrival_probability: float = 0.5
    history: int = 8
    prediction_steps: int = 20
    report_period_ms: float = 200.0
    report_min_ms: float = 100.0
    heartbeat_ms: float = 800.0
    event_threshold: float = 0.2
    report_bps: float = 64000.0
    report_propagation_ms: float = 5.0
    encoder_ms: float = 0.2
    quality_threshold: float = 0.8
    max_queue: int = 20

    def __post_init__(self):
        positive = (
            self.slot_ms,
            self.report_period_ms,
            self.report_min_ms,
            self.heartbeat_ms,
            self.report_bps,
        )
        if any(not isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Time intervals and reporting rate must be positive and finite")
        if not 0 <= self.arrival_probability <= 1 or not 0 <= self.quality_threshold <= 1:
            raise ValueError("Probabilities must be in [0, 1]")
        if any(
            not isinstance(v, int) or v <= 0
            for v in (
                self.arrival_steps,
                self.drain_steps,
                self.history,
                self.prediction_steps,
                self.max_queue,
            )
        ):
            raise ValueError("Step counts and queue size must be positive integers")
        if self.report_min_ms > self.heartbeat_ms:
            raise ValueError("Minimum report interval cannot exceed heartbeat")
        if any(
            not isfinite(v) or v < 0
            for v in (self.encoder_ms, self.report_propagation_ms, self.event_threshold)
        ):
            raise ValueError("Costs, propagation and trigger threshold must be nonnegative")


@dataclass(frozen=True)
class ModelProfile:
    name: str
    service_ms: float
    cost: float
    quality_by_difficulty: tuple[float, float, float]
    quality_std: float = 0.025
    vram: float = 2.0

    def __post_init__(self):
        if not self.name or self.service_ms <= 0 or self.vram <= 0:
            raise ValueError("Profile needs a name and positive runtime/resource demand")
        if len(self.quality_by_difficulty) != 3 or any(
            not isfinite(q) or not 0 <= q <= 1 for q in self.quality_by_difficulty
        ):
            raise ValueError("Exactly three finite quality means in [0, 1] are required")
        if any(
            not isfinite(v) or v < 0
            for v in (self.service_ms, self.cost, self.quality_std, self.vram)
        ):
            raise ValueError("Invalid model profile")


DEFAULT_PROFILES = (
    ModelProfile("small", 90.0, 0.15, (0.91, 0.72, 0.50), vram=2.0),
    ModelProfile("medium", 180.0, 0.40, (0.94, 0.89, 0.74), vram=4.0),
    ModelProfile("large", 350.0, 1.00, (0.97, 0.94, 0.90), vram=7.0),
)


@dataclass(frozen=True)
class RequestView:
    id: int
    arrival_ms: float
    difficulty: int
    input_tokens: int
    predicted_output_tokens: int
    deadline_ms: float

    @property
    def input_mb(self):
        return self.input_tokens * 4 / 1e6


@dataclass
class Job:
    request: RequestView
    endpoint: int
    remaining_input_mb: float
    remaining_work_ms: float
    remaining_output_mb: float
    quality: float
    cost: float
    ready_ms: float = float("inf")
    compute_start_ms: float | None = None
    compute_finish_ms: float | None = None
    finish_ms: float | None = None


@dataclass
class EndpointQueues:
    upload: list[Job] = field(default_factory=list)
    compute: list[Job] = field(default_factory=list)
    download: list[Job] = field(default_factory=list)

    @property
    def jobs(self):
        return self.upload + self.compute + self.download


@dataclass(frozen=True)
class ReceivedReport:
    sample_ms: float
    received_ms: float
    payload: np.ndarray
    decoded: np.ndarray
    forecast: np.ndarray | None
