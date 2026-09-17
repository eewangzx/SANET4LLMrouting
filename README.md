# SANET4LLMrouting

SANet-inspired predictive compression and reinforcement learning for routing DAG-based services across fixed light/heavy edge nodes.

This repository contains the current implementation, scenario/deployment configurations, dependency definitions and tests. It is a code snapshot of source commit `7c16b5ea96ee857e662faa1c3910d2c61b67474f`. Research PDFs, pretrained weights, simulation outputs and their earlier Git history are not included.

## Setup

```bash
python -m pip install -e '.[routing,reports,dev]'
```

Python 3.11 or newer is required. Experiment checkpoints and the optional Azure trace asset must be supplied separately through command-line paths.

## Main entry points

- `scripts/run_joint_routing.py`: joint predictive compression/importance selection and masked Double DQN; PPO is also available for the existing request-utility formulation.
- `scripts/train_separate_forecast.py`: independent compressor/predictor training for the decoded-state benchmark.
- `scripts/prepare_azure_trace.py`: prepare the Azure background-workload trace.
- `configs/fixed_placement_routing_main.json`: fixed deployment for routing.
- `configs/routing_resource_emphasis_20260917.json`: latest resource-budget experiment settings.

```bash
python scripts/run_joint_routing.py --help
```

The current average-budget mode uses chronological decisions across requests, a resource-cost virtual queue, physical-time discounting and jointly trained prediction/compression. Its internal queue target is 0.87; the confirmatory service-contract ratio is 0.90. Execution slots, resource sampling and telemetry reporting retain separate time scales. The actor reads delivered reports and the dispatch ledger; source histories and future labels are training-only.

The original source repository and experiment artifacts remain on Fedora and in the local workspace. This upload does not change an experiment or launch a simulation.
