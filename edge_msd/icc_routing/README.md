# ICC-background, SANet-inspired model routing

This runnable experiment imports `data/icc_paper/scenario_2026.json`. That file
reconstructs published ICC parameters; it is not the author's original samples.
The earlier `model_routing` toy environment and its results are retained separately.

## What comes from ICC

- All seven users' four task-type arrival rates; actual Poisson batches are kept.
- Per-task input MB, output MB from the terminal service, and 50–100 ms deadlines.
- User-to-ES association, three ES resource vectors, and ES-to-ES wired capacities.
- Per-task service workloads, deterministic core processing, and light-service
  Gamma **rate** parameters. Units and resource order are imported by `icc_paper.py`.

## Explicit extensions

- Three surrogate model alternatives reside at every ES. Their quality matrix,
  cost, resource footprint and work multipliers are in `surrogate_models.json`.
  They are not identified with the different-function core microservices.
- An inference request follows a serial aggregate of its task's service work at
  the chosen ES. This is a work proxy, not the original distributed/fork-join DAG.
  Each ES has one FIFO compute executor shared across the three resident models.
- Light-service rate factors are drawn independently per physical slot, node and
  service, using Gamma rate / its mean. A running request can change speed in the
  next slot; a single Gamma sample is NOT frozen into its entire service time.
- Node baseline speed is proportional to the imported CPU resource relative to
  the median ES. This calibration is an assumption, not a hardware measurement.
- Correlated sinusoidal availability plus AR noise changes compute and directed
  link capacities. This adds temporal structure to ICC's independent-state model.
  The predictor does not see the latent phases, periods or future noise online.
- Requests enter at their associated ES, with wireless access excluded. Each
  directed wired link has one shared FIFO capacity across all models and requests.
  Local requests incur zero inter-ES payload transfer. Replies return to the origin.
- Default load is explicitly `0.05 * ICC arrival rates` because this model has
  three shared inference executors instead of ICC's multiple microservice instances.
  `--loads 1.0` evaluates the unscaled rates. No per-slot arrivals are discarded.

## Learned representation and routing

Each node samples 16 local features every 5 ms and retains eight samples. A
two-layer temporal CNN maps this window to 16 latent channels. A bandwidth-
conditioned prefix carries 4, 8 or 16 channels. The wire packet has a real 16-byte
header plus float32 payload: 32, 48 or 80 bytes, versus 528 bytes for the full window.
No dense masked vector or uncounted feature-index array is transmitted.

The receiver predicts 20 steps (100 ms) of node availability and outgoing-link
availability. The policy sees the received latent, receiver-decoded features,
the whole forecast trajectory aligned to the sample timestamp, information age,
missing-report indicators, and public request/model information.

Training compares three policies with the same actor architecture and PPO budget:

1. `window_stats_ppo`: 16 hand-designed window features; learned forecast and actor.
2. `ae_ppo`: reconstruction-pretrained CNN encoder/decoder frozen during PPO.
3. `semantic_*_ppo`: the same initialization, then encoder, predictor and actor
   receive joint task training. The loss is PPO policy + value regression +
   0.5 forecast MSE + 0.1 auxiliary reconstruction MSE - 0.01 entropy.

Value-regression features are detached to prevent the critic from dominating
encoder learning. Routing-policy and prediction gradients reach the encoder.
This is SANet-inspired task-driven compression, not a reproduction of SANet's
full multi-agent partition/sharing or dynamic multi-objective weighting algorithm.

The environment rewards joint deadline/quality success (+3), penalizes a completed
failure (-1.5), charges 0.08 times model cost and a per-slot backlog penalty, and
penalizes rejected/pending requests. Trainer rewards are scaled by 1/10. PPO uses
gamma=0.99 and lambda=0.95 per physical slot. Multiple actions within a Poisson
batch take zero simulated time and use discount=1, not an extra physical slot.

The trainer uses source windows that generated RECEIVED reports to backpropagate
through encoding. Future capacity labels enter training losses only. Inference
uses actual received wire values. Model weights are held fixed during each full
episode; updates happen between episodes, avoiding mixed encoder versions in flight.

## Run

From the repo root after installing the ordinary dependencies and CPU torch:

```bash
python -m edge_msd.icc_routing.experiment \
  --output runs/my_icc_semantic --epochs 20 --updates 10 \
  --eval-seeds 211 212 213 --loads 0.05
```

Reuse the delivered checkpoint without retraining:

```bash
python -m edge_msd.icc_routing.experiment \
  --checkpoint runs/icc_semantic_gamma/checkpoint.pt \
  --output runs/my_icc_replay --eval-seeds 211 212 213
```

Check higher loads or a longer observation horizon:

```bash
python -m edge_msd.icc_routing.experiment \
  --checkpoint runs/icc_semantic_gamma/checkpoint.pt \
  --output runs/my_icc_stress --loads 0.1 1.0 --bandwidths 64000 \
  --eval-seeds 214 --methods latest_event window_stats_ppo ae_ppo \
  semantic_fixed16_ppo semantic_adaptive_ppo instant_full_heuristic
```

`--arrival-steps 400` extends arrivals from 400 ms to 2000 ms. `--static` fixes
the added link/compute availability process while retaining per-slot Gamma rates.
Model outputs remain synthetic. The quality threshold is 0.8; actual quality
samples are private and only public quality predictions enter decisions.

## Comparisons and limits

The suite includes full periodic reports, slower bandwidth-budgeted full reports,
latest snapshots, window statistics, ordinary AE, joint semantic encoding,
periodic/event reporting, and full-state information references. Reactive versus
trajectory heuristics use the same respective encoder to isolate forecast usage.
Adaptive prefix length is a declared capacity-based rule, not a learned bit allocator.
Event triggers are also fixed rules, not a learned scheduling policy.

Primary semantic-fixed16 / AE / statistics packets all carry 16 floats. Adaptive
packets can be smaller. Actual transmitted bitrate and report age are reported;
equal physical capacity does not imply equal average bitrate at high capacities.

All methods for a test seed share the exact arrivals, candidate quality outcomes,
availability and per-slot Gamma-rate traces; hashes are checked. Test seeds are
disjoint from pretraining (1000–1005), validation (2000–2001), warm start (3000–3001)
and PPO (4000 onward). Only one learning seed is used in this first run.

Encoder delay is configured at 0.2 ms, not measured. Router CPU time is recorded
but decoder/router CPU cost does not advance the simulation clock. No TTFT,
token-level batching or strict SLA guarantee is implemented. Short experiments
start with empty state caches; the longer-horizon check measures sensitivity to this.

The first diagnostic run used a once-per-request inverse-rate approximation,
which created excessive head-of-line blocking. Those artifacts are retained as
`icc_semantic_routing`, `icc_semantic_ablations`, `icc_semantic_stress`, and
`icc_semantic_longer`. The corrected per-slot Gamma run is `icc_semantic_gamma`.
Old checkpoints retain their old service-process interpretation for replay.
