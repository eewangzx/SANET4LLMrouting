"""Domain objects and units shared by the controller and simulator.

Time: ms. Data/work: decimal MB. Wired bandwidth: Gbps.
Resource vectors always use (CPU, GPU, RAM, VRAM). No paper dataset is loaded.
"""

from dataclasses import dataclass, field
from math import isfinite

import networkx as nx

Resources = tuple[float, float, float, float]
Placement = dict[tuple[str, str], int]  # (service, node) -> instance count
StageKey = tuple[int, str]  # (request ID, service)


@dataclass(frozen=True)
class Service:
    name: str
    kind: str
    resources: Resources
    output_mb: float
    deployment_cost: float
    maintenance_cost: float
    parallel_cost: float
    processing_ms: float = 0.0
    work_mb: float = 0.0
    input_mb: float | None = None  # Root forwarding: None uses the task's input size.
    gamma_shape: float = 1.0
    gamma_scale: float = 1.0  # Rate distribution, MB/ms; NOT a time distribution.

    def __post_init__(self):
        if self.kind not in ("core", "light"):
            raise ValueError(f"Unknown service kind: {self.kind}")
        values = (
            *self.resources,
            self.output_mb,
            self.deployment_cost,
            self.maintenance_cost,
            self.parallel_cost,
        )
        if len(self.resources) != 4 or any(not isfinite(x) or x < 0 for x in values):
            raise ValueError(f"Invalid resources/data/costs for {self.name}")
        if not any(self.resources):
            raise ValueError("At least one resource demand must be positive")
        if self.input_mb is not None and (not isfinite(self.input_mb) or self.input_mb < 0):
            raise ValueError("input_mb must be nonnegative or None")
        positive = (
            (self.processing_ms,)
            if self.kind == "core"
            else (self.work_mb, self.gamma_shape, self.gamma_scale)
        )
        if any(not isfinite(x) or x <= 0 for x in positive):
            raise ValueError(f"Invalid processing parameters for {self.name}")

    @property
    def mean_rate(self) -> float:
        return self.gamma_shape * self.gamma_scale

    @property
    def mean_processing_ms(self) -> float:
        """Mean-rate approximation a/E[f], not E[a/f]."""
        return self.processing_ms if self.kind == "core" else self.work_mb / self.mean_rate


@dataclass
class TaskType:
    name: str
    dependencies: dict[str, list[str]]
    deadline_ms: float
    input_mb: float
    priority: float = 1.0
    graph: nx.DiGraph = field(init=False, repr=False)
    order: list[str] = field(init=False)

    def __post_init__(self):
        self.dependencies = {name: list(parents) for name, parents in self.dependencies.items()}
        if any(not isfinite(x) or x <= 0 for x in (self.deadline_ms, self.priority)):
            raise ValueError("Deadline and priority must be positive")
        if not isfinite(self.input_mb) or self.input_mb < 0 or not self.dependencies:
            raise ValueError("Invalid task input or empty DAG")
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.dependencies)
        for name, parents in self.dependencies.items():
            if any(parent not in self.dependencies for parent in parents):
                raise ValueError(f"Unknown predecessor in {self.name}/{name}")
            self.graph.add_edges_from((parent, name) for parent in parents)
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError(f"Cyclic service dependencies in {self.name}")
        self.order = list(nx.lexicographical_topological_sort(self.graph))


@dataclass(frozen=True)
class Node:
    name: str
    resources: Resources
    position: tuple[float, float] | None = None
    uplink_bandwidth_hz: float = 0.0
    light_reserve: Resources = (0.0, 0.0, 0.0, 0.0)

    @property
    def core_budget(self) -> Resources:
        return tuple(cap - reserve for cap, reserve in zip(self.resources, self.light_reserve))


@dataclass(frozen=True)
class Radio:
    """Explicit free-space radio assumptions; no hidden transmit/noise constants."""

    transmit_dbm: float
    noise_dbm: float  # Total noise power over one user's allocated bandwidth.
    carrier_hz: float
    minimum_distance_m: float = 1.0

    def __post_init__(self):
        if not all(isfinite(x) for x in (self.transmit_dbm, self.noise_dbm)):
            raise ValueError("Radio powers must be finite")
        if any(not isfinite(x) or x <= 0 for x in (self.carrier_hz, self.minimum_distance_m)):
            raise ValueError("Carrier frequency and minimum distance must be positive")


@dataclass(frozen=True)
class User:
    name: str
    position: tuple[float, float]
    gateway: str
    rates_per_second: dict[str, float]
    nakagami_m: float
    omega: float


@dataclass
class Scenario:
    services: dict[str, Service]
    tasks: dict[str, TaskType]
    nodes: dict[str, Node]
    links: list[tuple[str, str, float, float]]  # endpoints, Gbps, propagation ms
    users: list[User]
    profile: str = "custom"
    radio: Radio | None = None

    def validate(self):
        if not self.nodes or not self.services or not self.tasks:
            raise ValueError("A scenario needs nodes, services and tasks")
        for objects in (self.nodes, self.services, self.tasks):
            if any(not key or key != value.name for key, value in objects.items()):
                raise ValueError("Scenario dictionary keys must match nonempty object names")
        for node in self.nodes.values():
            if len(node.resources) != 4 or any(not isfinite(x) or x < 0 for x in node.resources):
                raise ValueError(f"Invalid node resources: {node.name}")
            if len(node.light_reserve) != 4 or any(
                not isfinite(r) or not 0 <= r <= cap
                for r, cap in zip(node.light_reserve, node.resources)
            ):
                raise ValueError("Light reserve must fit within physical resources")
            if not isfinite(node.uplink_bandwidth_hz) or node.uplink_bandwidth_hz < 0:
                raise ValueError(f"Invalid uplink bandwidth: {node.name}")
            if node.position is not None and (
                len(node.position) != 2 or any(not isfinite(x) for x in node.position)
            ):
                raise ValueError("Node positions must be finite 2D coordinates")
        for task in self.tasks.values():
            if not set(task.dependencies) <= self.services.keys():
                raise ValueError(f"Unknown service in {task.name}")
        seen_links = set()
        for a, b, bandwidth, propagation in self.links:
            if a not in self.nodes or b not in self.nodes or a == b:
                raise ValueError("Invalid link endpoint")
            if (
                not isfinite(bandwidth)
                or bandwidth <= 0
                or not isfinite(propagation)
                or propagation < 0
            ):
                raise ValueError("Invalid link bandwidth/propagation")
            pair = frozenset((a, b))
            if pair in seen_links:
                raise ValueError("Duplicate undirected link")
            seen_links.add(pair)
        if len({u.name for u in self.users}) != len(self.users):
            raise ValueError("Duplicate user IDs")
        for user in self.users:
            if len(user.position) != 2 or any(not isfinite(x) for x in user.position):
                raise ValueError("User positions must be finite 2D coordinates")
            if (
                user.gateway not in self.nodes
                or not user.rates_per_second.keys() <= self.tasks.keys()
            ):
                raise ValueError(f"Invalid gateway/task reference for {user.name}")
            if any(not isfinite(x) or x < 0 for x in user.rates_per_second.values()):
                raise ValueError("Arrival rates must be finite and nonnegative")
            if any(not isfinite(x) or x <= 0 for x in (user.nakagami_m, user.omega)):
                raise ValueError("Invalid fading parameters")

    @property
    def core(self) -> list[str]:
        return sorted(s.name for s in self.services.values() if s.kind == "core")

    @property
    def light(self) -> list[str]:
        return sorted(s.name for s in self.services.values() if s.kind == "light")


@dataclass
class Request:
    id: int
    user: str
    task: str
    arrival_ms: float
    uplink_ready_ms: float
    gateway: str
    finished: dict[str, tuple[str, float]] = field(default_factory=dict)
    scheduled: set[str] = field(default_factory=set)
    completion_ms: float | None = None


@dataclass
class Stage:
    request: Request
    service: str
    ready_ms: float

    @property
    def key(self) -> StageKey:
        return self.request.id, self.service


@dataclass
class Job:
    stage: Stage
    data_ready_ms: float
    remaining_work: float
    start_ms: float | None = None
    finish_ms: float | None = None


@dataclass
class Instance:
    id: int
    service: str
    node: str
    jobs: list[Job] = field(default_factory=list)


@dataclass
class Decision:
    counts: Placement
    parallelism: Placement
    routes: dict[StageKey, str]
    predicted_ms: dict[StageKey, float] = field(default_factory=dict)
    instance_slots: dict[StageKey, int] = field(default_factory=dict)
    # Slot index within (service,node): busy instances by ID, then idle by ID.
