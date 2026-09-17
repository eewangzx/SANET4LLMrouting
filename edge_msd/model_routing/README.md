# Narrowband edge-model routing prototype

This is a runnable extension of `edge-msd-core`. **The bundled models, quality
scores, runtime profiles and traffic are synthetic. Results validate a mechanism
and its implementation; they are not evidence of performance on actual LLMs.**

The original ICC simulator remains available through `examples.minimal`.

## Run

From the repository root, install the ordinary ICC dependencies and CPU PyTorch:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pytest -q
.venv/bin/python -m edge_msd.model_routing.experiment --output runs/my_experiment
```

Defaults train an AE for 20 epochs, a separate frozen-latent predictor for 20
epochs, and PPO for 16,384 physical slots. PPO first imitates a causal heuristic
to initialize the policy. Training includes both raw and AE reporting and several
channel capacities. The final checkpoint, not a test-selected checkpoint, is evaluated.

The first pilot used a shared actor/critic representation and 8 warm-start epochs;
its PPO underperformed, and the original results/checkpoint are retained. The
revised default uses separate policy/value networks, 40 warm-start epochs, and
a mask based on **public predicted quality**, with a fallback if no model passes.
This mask is an engineering prior, not a guarantee about actual answer quality.
`--legacy-policy` reproduces the initial architecture/training setup. Checkpoints
record architecture flags, so old checkpoints retain their original behavior.

Evaluate the same trained models on more independent workload traces:

```bash
.venv/bin/python -m edge_msd.model_routing.experiment \
  --checkpoint runs/my_experiment/checkpoint.pt \
  --output runs/additional_evaluation \
  --eval-seeds 104 105 106 107 108 \
  --bandwidths 32000 64000 128000 256000
```

When loading a checkpoint, its simulation configuration and profiles are reused.
Use a fresh output directory. Seeds 1000–99999 are reserved for training/validation.
The CLI saves CSV, JSON, checkpoints, a Markdown report, and optional PNG/PDF plots
when matplotlib is available. The plots explicitly identify synthetic results.

## What is implemented

- Six fixed endpoints: two replicas each of small/medium/large model profiles.
  Models can answer the same requests with different quality, service work and cost.
- Reuse of ICC `Scenario`, `Node`, `Service`, `Network` propagation/topology and
  placement resource checks. Alternative models are not encoded as mandatory DAG stages.
- Independent exogenous random streams; request arrivals, capacity traces and
  **all** request/model counterfactual outcomes are fixed before policy actions.
- 50 ms physical slots. Network/compute capacity changes during active transfers
  and inference. Remaining input bytes, work and output bytes determine completion.
- FIFO, non-preemptive inference. Each request is routed once. Completed, pending
  and rejected requests are separately counted; no free migration of running work.
- A shared finite-rate reporting channel. Reports are packed with a real 16-byte
  header and float32 payload. An in-service packet cannot be preempted. Each node
  has at most one replaceable unsent report; replacing it does not reorder its FIFO slot.
- Causal receive cache: no fresh state before the report arrives; stale arrivals
  cannot overwrite newer samples. Online policies receive copied observations only.
- Local history of 8 samples × 16 features = 128 floats per node. Raw packet:
  528 bytes. Default 16-dimensional AE packet: 80 bytes. The current-statistics
  baseline also sends 80 bytes. Feature dimensions come from actual simulated
  measurements/history, not repetition to manufacture a compression advantage.
- Periodic and fixed-rule event reporting. Event trigger uses locally observed
  rate/queue changes, a minimum interval and a heartbeat. It does not inspect a
  hidden central state, and the trigger itself is not learned.
- Deterministic AE reconstruction pretraining, followed by a predictor on frozen
  latent vectors. Prediction targets are future link/compute availability over
  20 slots. Forecasts are indexed relative to the report's **sample time**.
- Categorical PPO with a shared endpoint scoring network and pooled value head.
  It reads latent, receiver-decoded features, report age, forecast, public model
  information and current request features. No true outcome labels enter the observation.

The predictor is a small MLP on a latent vector of the local history. It is not
a GRU or an uncertainty-calibrated model. There is no claim of strict QoS guarantees.

## Reward and clocks

An admission incurs `-0.08 * inference_cost`. A request earns `+3` when both its
realized quality and deadline are satisfied, otherwise `-1.5` on completion.
Each physical slot incurs `-0.01 * outstanding_requests`; rejection costs `-2`.
At the end of the finite drain interval each remaining request costs `-2` and is
reported as pending. Rewards are available to the trainer after events occur.

There is at most one arrival per physical slot, sampled by a Bernoulli process;
this differs from the ICC simulator's Poisson batch arrivals. Each PPO transition
is one fixed 50 ms slot, including slots without an arrival. Policy loss and
entropy apply only at request arrivals; the critic also learns from other slots.
Discount is 0.99 per slot, with GAE lambda 0.95 and terminal masking.

Request links are independent full-duplex gateway-to-node links with 2 ms
propagation per direction. They are separate from the shared narrow reporting
link. Thus this experiment investigates state-reporting constraints, not the
effect of telemetry competing with request payload on the same link.

Encoding delay is an explicit configured constant (0.2 ms), not a hardware
measurement. Decoder/predictor CPU overhead and router time do not yet advance
the physical simulation clock. Actual router time is measured separately.

## Comparison methods

| Name | Observation / decision rule |
|---|---|
| `query_only` | Public request/model profile, no state reports |
| `full_state_heuristic` | Instantaneous full state, heuristic; information-rich reference |
| `raw_periodic` | Complete history every 200 ms, causal heuristic |
| `raw_budget_periodic` | Raw reporting period increased to fit approximately 80% offered load |
| `stats_periodic`, `stats_event` | Latest 16 statistics; strong equal-packet-size baseline |
| `ae_periodic`, `ae_event` | AE reports; same reactive heuristic |
| `ae_periodic_predict`, `ae_event_predict` | Same AE, forecast-aware heuristic |
| `ae_event_ppo` | AE + event reporting + forecast + learned PPO |
| `raw_budget_ppo` | Raw reports, same frozen PPO; encoder runs centrally after receipt |
| `full_state_ppo` | Instant raw state, same frozen PPO; information-rich reference |

All constrained methods use the same physical channel capacity and buffering
rules. **Actual average reporting bitrates can differ**; inspect them in results.
Instant-state references do not pay communication costs and are not optimal oracles.
Per-seed trace hashes must match across all methods or the experiment aborts.
The plotted error bars describe workload-seed variation, not multiple training seeds.

## Files

| File | Role |
|---|---|
| `types.py` | Profiles, configuration and domain records |
| `environment.py` | ICC infrastructure reuse and new routing physics |
| `telemetry.py` | Wire format, uplink queue, raw/statistics codecs |
| `routers.py` | Causal heuristics and policy feature assembly |
| `learning.py` | AE, predictor, warm start and PPO |
| `experiment.py` | Training, paired evaluation and output artifacts |

## Before using results in a paper

1. Replace synthetic quality/runtime assumptions with query–model measurements
   and device profiles. `profiles.json` shows the current scalar schema;
   `--profiles file.json` can replace these parameters, but does not turn the
   parametric FIFO simulator into a token-level LLM serving engine.
2. Determine whether the 16-statistic baseline is already sufficient. If it wins,
   do not claim a learned-compression gain. Investigate additional decision-relevant
   workload information and task-driven representation training only with evidence.
3. Train multiple independent policy seeds and test held-out network/workload
   regimes. Tune reporting thresholds/periods on validation traces under comparable
   actual communication budgets. Do not tune on the final evaluation seeds.
4. Calibrate encoder, predictor and scheduler overhead, prediction uncertainty,
   runtime/response-size errors and real endpoint resource constraints.
5. Add actual semantic/decision-driven encoding or reporting as the proposed
   mechanism. This prototype implements **AE + prediction + PPO baselines**, not
   a new SANet algorithm or an established novelty/performance claim.

Related-work motivation: Zhuge (SIGCOMM 2022), Martini (ICNP 2020), Decima
(SIGCOMM 2019), Mortise (NSDI 2026), and QoS-aware edge LLM routing all cover parts
of the problem. The extension needs evidence of a specific benefit from limited,
delayed, decision-relevant state updates, beyond combining standard components.
