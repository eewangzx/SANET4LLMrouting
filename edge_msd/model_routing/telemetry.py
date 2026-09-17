"""Byte serialization, causal report delivery and a shared narrow uplink.

Non-preemptive FIFO service, at most one replaceable unsent report per node.
Control and request data use distinct links. All codecs pay the same header.
"""

import heapq
import struct
from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite

import numpy as np

HEADER = struct.Struct("<4sBBHII")  # magic, version, codec, node, sample tick, float count


def pack_report(node: int, tick: int, values: np.ndarray, codec_id: int) -> bytes:
    values = np.asarray(values, dtype="<f4").reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("Report payload must be finite")
    return HEADER.pack(b"EMSD", 1, codec_id, node, tick, len(values)) + values.tobytes()


def unpack_report(packet: bytes):
    if len(packet) < HEADER.size:
        raise ValueError("Truncated report")
    magic, version, codec, node, tick, count = HEADER.unpack_from(packet)
    if magic != b"EMSD" or version != 1 or len(packet) != HEADER.size + count * 4:
        raise ValueError("Invalid report header or length")
    values = np.frombuffer(packet, dtype="<f4", offset=HEADER.size).copy()
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite received payload")
    return node, tick, values, codec


@dataclass(frozen=True)
class Delivery:
    packet: bytes
    finish_ms: float
    arrival_ms: float


class NarrowbandChannel:
    def __init__(self, rate_bps: float, propagation_ms: float = 0.0):
        if not isfinite(rate_bps) or rate_bps <= 0:
            raise ValueError("rate_bps must be positive")
        if not isfinite(propagation_ms) or propagation_ms < 0:
            raise ValueError("propagation_ms must be nonnegative")
        self.rate_bps = rate_bps
        self.propagation_ms = propagation_ms
        self.pending = OrderedDict()
        self.active = None
        self.deliveries = []
        self.sequence = 0
        self.now_ms = 0.0
        self.generated_bytes = 0
        self.transmitted_bytes = 0.0
        self.replaced_reports = 0

    def enqueue(self, node: int, packet: bytes, ready_ms: float):
        if not isfinite(ready_ms) or ready_ms < self.now_ms - 1e-8:
            raise ValueError("Cannot enqueue a report in the past")
        decoded_node, *_ = unpack_report(packet)
        if decoded_node != node:
            raise ValueError("Packet node does not match queue key")
        self.generated_bytes += len(packet)
        if node in self.pending:
            self.replaced_reports += 1
        # Replacement retains FIFO position; a packet in service is never replaced.
        self.pending[node] = (packet, ready_ms)

    def advance(self, until_ms: float):
        if not isfinite(until_ms) or until_ms < self.now_ms - 1e-8:
            raise ValueError("Channel clock must be monotone")
        cursor = self.now_ms
        while cursor < until_ms - 1e-10:
            if self.active is None:
                if not self.pending:
                    break
                ready_nodes = [n for n, (_, r) in self.pending.items() if r <= cursor + 1e-9]
                if not ready_nodes:
                    cursor = max(cursor, min(r for _, r in self.pending.values()))
                    if cursor >= until_ms:
                        break
                    ready_nodes = [n for n, (_, r) in self.pending.items() if r <= cursor + 1e-9]
                packet, _ = self.pending.pop(ready_nodes[0])
                self.active = [packet, float(len(packet))]
            packet, remaining = self.active
            dt = min(until_ms - cursor, remaining * 8000 / self.rate_bps)
            sent = min(remaining, dt * self.rate_bps / 8000)
            self.transmitted_bytes += sent
            self.active[1] -= sent
            cursor += dt
            if self.active[1] <= 1e-8:
                self.sequence += 1
                item = Delivery(packet, cursor, cursor + self.propagation_ms)
                heapq.heappush(self.deliveries, (item.arrival_ms, self.sequence, item))
                self.active = None
        self.now_ms = until_ms
        received = []
        while self.deliveries and self.deliveries[0][0] <= until_ms + 1e-9:
            received.append(heapq.heappop(self.deliveries)[2])
        return received


class RawCodec:
    codec_id = 0

    def __init__(self, history: int, features: int):
        self.history, self.features = history, features
        self.payload_dim = history * features

    def encode(self, history):
        return np.asarray(history, dtype=np.float32).reshape(-1)

    def decode(self, payload):
        return payload.reshape(self.history, self.features).copy()

    def predict(self, payload):
        return None

    def embedding(self, payload):
        return payload.copy()


class StatsCodec:
    """Strong small-message baseline: send current measurements directly."""

    codec_id = 1

    def __init__(self, history: int, features: int):
        self.history, self.features = history, features
        self.payload_dim = features

    def encode(self, history):
        return np.asarray(history[-1], dtype=np.float32).copy()

    def decode(self, payload):
        return np.repeat(payload[None, :], self.history, axis=0)

    def predict(self, payload):
        return None

    def embedding(self, payload):
        return payload.copy()
