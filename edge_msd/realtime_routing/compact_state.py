"""Compact routing state: frozen predictive embedding and current-stage context.

Uses the existing controller's immediate dispatch/completion ledger assumption.
Transfer estimates follow the unchanged fixed-link ICC execution model. No future
service sample or uncompleted predecessor timestamp is used.
"""
from __future__ import annotations

import numpy as np


def build_compact_state(env, embeddings, services_sorted, tasks_sorted):
    stage = env._routable_stage()
    if stage is None:
        raise ValueError('Compact state is defined at routing decisions only')
    task = env.scenario.tasks[stage.request.task]
    service = np.zeros(len(services_sorted), dtype=np.float32)
    service[services_sorted.index(stage.service)] = 1.
    task_type = np.zeros(len(tasks_sorted), dtype=np.float32)
    task_type[tasks_sorted.index(task.name)] = 1.
    ledger = env.observe()['ledger']
    # All parent stages have completed before a stage is routable. Locations and
    # timestamps below belong to past completion records, not future predictions.
    transfer = np.asarray([
        max(0., env.network.data_ready(stage, node) - env.now)
        for node in env.node_ids
    ], dtype=np.float32)
    transfer = np.nan_to_num(transfer, posinf=500., neginf=0.) / 50.
    slack = (task.deadline_ms - (env.now-stage.request.arrival_ms)) / 50.
    # Omit aggregate ingress waiting counts, constant present flag, duplicate
    # age/deadline, and mask from Q input. Legal masks still constrain acting and
    # Double-DQN targets through the existing agent and rollout implementation.
    return np.concatenate([embeddings.reshape(-1), ledger, transfer, service,
                           task_type, np.asarray([slack], np.float32)]).astype(np.float32)
