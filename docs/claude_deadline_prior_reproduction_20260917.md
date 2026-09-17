# Deadline-aware prior reproduction, 2026-09-17

The supplied Claude patch was applied without changing its numerical decision
rule in commit `e361de9449d2f135048a16999a3512fd363c2a75`. Later wording changes
describe the deadline classification as a heuristic rather than a guarantee.

## Decision rule

Let `d_k` be received-forecast stage latency divided by 50 ms, `b_k` be
`log1p(jobs per instance)/3`, and `B` be remaining deadline minus the sum of
other unfinished stages' processing-work estimates, divided by 50 ms.

The routing prior is `P_k=d_k` when `B >= min_deployed(d_k)`, and
`P_k=-b_k+0.02*d_k` otherwise. The policy maximizes its learned action logit
minus `2*P_k`; the action residual is bounded by `0.25*tanh` in this run.
With zero-initialized output heads and no optimizer updates, this is a fixed
prediction-driven routing mechanism. It preferentially isolates predicted
deadline-risky work on more congested nodes, protecting less congested nodes.

This does not change the DQN update algorithm. Differentiable replay recomputes
the forecast-derived prior, preserving RL gradients into the codec. The hard
risk threshold itself is not differentiable. The work estimate sums parallel
and in-flight unfinished stages and is not a critical-path lower bound.

## Reproduction settings

- Original mixed ICC tasks, `data/icc_paper/scenario_2026.json`, load 0.8.
- Fixed placement `configs/fixed_placement_routing_main.json`.
- 200 ms warmup, 120 ms arrival window, 150 ms drain, 1 ms physical slot.
- Reports every 100 ms over 64 kbit/s; previously delivered reports are reused.
- Test seeds 91000, 91001, 91002, matching the supplied probe.
- Existing `runs/icc_coupled_models/importance.pt`, SHA-256
  `9bb0c764c2ae7d1c2297317ccab83ada61966f125472d07fe014eec6c421f105`.
- Supplied scripts instead name `importance_pretrained_icc.pt`; that checkpoint
  was not included. Their probe also approximates remaining work differently
  from the integrated patch. The small residual result difference cannot be
  uniquely attributed to model weights.
- No latency, resource-cost, data-cost, or average-budget objective enabled.
- Zero RL updates for the following table; this is prior gain, not RL gain.

## Results

| Metric | LOM | Deadline-aware prior |
|---|---:|---:|
| Seed 91000 SLA | 0.699751861 | 0.800248139 |
| Seed 91001 SLA | 0.724266226 | 0.793716412 |
| Seed 91002 SLA | 0.759778912 | 0.874149660 |
| Mean of three seed SLAs | 72.7932% | 82.2705% |
| Pooled SLA, 7189 requests | 72.7639% | 82.2228% |
| On-time requests | 5231 | 5911 |
| SLA failures, including pending | 1958 | 1278 |
| Pending at end of drain | 0 | 371 |
| Mean completed latency | 47.9366 ms | 47.3842 ms |
| P95 completed latency | 81.7022 ms | 112.8462 ms |
| Transmitted report bytes in active window, mean per trace | 2160 B | 960 B |

The pooled gain is 9.4589 percentage points, or 12.9994% relative SLA gain;
SLA failures fall by 34.7293%. The LOM per-seed numbers exactly match the supplied
results. Supplied deadline-aware numbers average 82.2776%, close to the reproduced
82.2705%.

The mechanism trades completion of predicted-risky work for more on-time work.
It does not improve all latency metrics: completed P95 is worse and pending
backlog rises. This finite-burst experiment does not establish continuous-load
queue stability or long-term cost-budget feasibility. Pending requests count as
SLA failures, and completed latency summaries exclude them. No cost improvement
or additional RL contribution should be inferred from this table.

## Artifacts

- Local: `/Users/wangzixin/output/icc-review/claude_repro_20260917/`.
- Fedora LOM: `runs/routing_icc_lom_20260917_claude_repro/`.
- HPC4 prior: `runs/routing_icc_deadline_prior_20260917_claude_repro/`.
- Four-episode joint-training check: HPC4 job 1879585,
  `runs/routing_icc_deadline_rl4_20260917/`, supplied `train_resumable.py`.

These tests intentionally reuse the supplied test seeds for reproduction and
are not a new blind evaluation.

## Four-episode joint-training check

The supplied `train_resumable.py` was also run for four episodes with 500 DQN
updates per episode, learning rate `1e-4`, exploration starting at 0.1, and
validation seeds 71000--71002. Validation selected the zero-update checkpoint.

| Checkpoint | Updates | Validation SLA |
|---|---:|---:|
| Initial prior | 0 | 83.88% |
| Episode 4 | 2000 | 74.89% |

The task gradient reached the codec: mean codec RL gradient norm rose from
0.0268 in episode 1 to 0.1654 in episode 4, while mean Q-network gradient norm
rose from 0.507 to 0.826. The degradation therefore is not evidence of a dead
gradient path. The non-budget learner assigns terminal utility separately to
each request trajectory, so it does not directly credit one request's routing
action for preserving capacity for later requests. The reproduced 82.2705%
test SLA is the selected initial prior and must not be reported as learned RL
gain.
