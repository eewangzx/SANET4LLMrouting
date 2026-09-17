"""Explicit temporally correlated extension of ICC service/link environments.

Resource reservations and core execution are unchanged. Light-service rate
modulation represents background CPU/GPU interference, not extra capacity from
parallelism. Independent Gamma innovations remain per physical instance slot.
"""

import hashlib
from math import ceil

import numpy as np


class ResourceTrace:
    sample_ms = 5.0
    history = 8
    horizon = 61

    def __init__(self, scenario, duration_ms, seed, dynamic=True, profile='icc',
                 azure_path=None,azure_split='train',azure_case='busy_transitions'):
        if profile not in ('icc', 'driving', 'azure'):
            raise ValueError('Unknown resource dynamics profile')
        self.scenario, self.seed = scenario, seed
        self.nodes, self.services = sorted(scenario.nodes), scenario.light
        self.node_index = {n: i for i, n in enumerate(self.nodes)}
        self.service_index = {s: i for i, s in enumerate(self.services)}
        if len(self.services) != 9:
            raise ValueError("This ICC feature schema requires its nine light services")
        self.neighbors = {n: sorted(b if a == n else a for a, b, *_ in scenario.links
                                    if n in (a, b)) for n in self.nodes}
        if max(map(len, self.neighbors.values())) > 5:
            raise ValueError("Feature schema supports at most five outgoing neighbors")
        self.offset = self.history - 1
        steps = ceil(duration_ms / self.sample_ms) + self.offset + self.horizon + 2
        rng = np.random.default_rng(np.random.SeedSequence([seed, 801]))
        n = len(self.nodes)
        values = np.ones((steps, n, 14), np.float32)
        t = np.arange(steps) * self.sample_ms
        self.provenance=None
        if dynamic and profile=='azure':
            if azure_path is None:raise ValueError('Azure dynamics require an aggregated trace asset')
            from edge_msd.icc_coupled.azure_trace import azure_multipliers
            values,self.provenance=azure_multipliers(scenario,self.nodes,self.neighbors,
                                                   steps,seed,azure_path,azure_split,azure_case)
        elif dynamic:
            pressure = np.empty((steps, n, 2))
            for i in range(n):
                for r in range(2):
                    period = rng.uniform(140, 200) if profile == 'driving' else rng.uniform(120, 300)
                    phase = rng.uniform(0, 2*np.pi)
                    residual = np.zeros(steps)
                    for k in range(1, steps):
                        residual[k] = .88*residual[k-1] + rng.normal(0, .012 if profile == 'driving' else .025)
                    pressure[:, i, r] = np.clip(.5 + (.45 if profile == 'driving' else .38)*np.sin(2*np.pi*t/period + phase)
                                                + residual, 0, 1)
            for s, name in enumerate(self.services):
                resources = np.asarray(scenario.services[name].resources[:2])
                weight = resources / max(resources.sum(), 1e-9)
                jitter = rng.normal(0, .025, (steps, n))
                values[:, :, s] = np.clip(1 - (.85 if profile == 'driving' else .75)*(pressure*weight).sum(-1) + jitter,
                                          .15, 1.1)
            for i, node in enumerate(self.nodes):
                for j, _ in enumerate(self.neighbors[node]):
                    period, phase = rng.uniform(100, 260), rng.uniform(0, 2*np.pi)
                    noise = np.zeros(steps)
                    for k in range(1, steps):
                        noise[k] = .85*noise[k-1] + rng.normal(0, .035)
                    if profile == 'driving':
                        # Alternating egress paths follow a shared site regime.
                        # The regime is observable only through causal history;
                        # phases and periods are random across training/test seeds.
                        wave=(pressure[:,i,0]-.5)/.45
                        values[:,i,9+j]=np.clip(.58+.42*((-1)**j)*wave+.3*noise,.15,1.)
                    else:
                        values[:, i, 9+j] = np.clip(.68 + .3*np.sin(2*np.pi*t/period+phase)
                                                    + noise, .15, 1.1)
        self.values = values
        features = np.ones((steps, n, 16), np.float32)
        features[:, :, :14] = values
        for k in range(steps):
            window = values[max(0, k-self.history+1):k+1]
            features[k, :, 14] = window[:, :, :9].mean((0, 2))
            features[k, :, 15] = window[:, :, 9:14].mean((0, 2))
        self.features = features
        self.sha256 = hashlib.sha256(values.tobytes()).hexdigest()

    def index(self, when):
        return min(len(self.values)-1, max(0, int(when // self.sample_ms) + self.offset))

    def window(self, when):
        k = self.index(when)
        return self.features[k-self.history+1:k+1].transpose(1, 0, 2).copy()

    def current(self, when):
        return self.values[self.index(when)]

    def future(self, when):
        k = self.index(when)
        return self.values[k:k+self.horizon].transpose(1, 0, 2).copy()


def training_arrays(scenario, seeds, duration_ms=1600, stride=4):
    xs, ys, hashes = [], [], []
    for seed in seeds:
        trace = ResourceTrace(scenario, duration_ms, seed)
        hashes.append(trace.sha256)
        for when in np.arange(0, duration_ms, trace.sample_ms*stride):
            xs.extend(trace.window(when))
            ys.extend(trace.future(when))
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), hashes
