# Additional training and routing benchmarks

The user requested more training, a Random benchmark, a further traditional
benchmark, an Azure trace experiment, and joint use of Fedora and HPC4.

The existing `driving-poc-20260916` result is preserved. Continue its joint model
for 48 episodes on new training seeds 72000--72047, with the same deployment,
physics, resource profile, traffic intensity, SLA, learning rate, and bounded Q
parameterization. Exploration epsilon is 0.05. Evaluate without exploration on
validation seeds 71000--71002 before training and every eight episodes. Select
the highest validation SLA, with mean completed latency breaking ties; include
the initialization checkpoint as a candidate. Evaluate the selected checkpoint
on the original held-out seeds 70000--70002. Save both last and selected models.

Additional policies use raw delivered current-state reports and the existing
dispatch/completion ledger. Random selects uniformly among legal deployed nodes,
using a separate policy RNG that cannot perturb exogenous randomness. Shortest
Queue minimizes the outstanding current-service jobs per deployed instance;
estimated current transfer and processing delay break ties. Neither policy uses
the proposed predictor. All four policies retain the common telemetry warmup,
cache reuse, report channel, fixed placement and task SLA accounting.

Training diagnostics include exploration-enabled SLA, independent deterministic
validation SLA, joint and task-only compressor/importance gradient norms, and
pre-clipping gradient norm. A higher exploration-enabled SLA is not itself
evidence of a better evaluation policy or convergence.

Each run records the exact source commit, initialization hash, seeds and model
selection. HPC4 code synchronization is explicitly authorized; old runs remain.

## Azure Code trace experiment

Use the complete official Azure Code one-week CSV, rather than the earlier
partial correlation sample. Aggregate ordered request counts and context token
counts into five-second bins. Fit feature-wise p5/p95 normalization only on the
first four days, clip normalized pressure to [0,1], and preserve chronology:
first four days training, fifth day validation, final two days testing. Output
token counts are not used, since they are unknown on request arrival.

Map five real seconds to five simulated milliseconds (1000x time acceleration).
Thus the 35ms source history spans 35 real seconds; the 100ms reporting period
spans 100 real seconds; the 300ms forecast spans 300 real seconds. The physical
simulator retains 1ms slots and the original driving DAG/deadline. This is an
explicitly time-scaled, trace-driven simulation, not measured Azure networking.

The initial all-week uniform background sample was too easy: its three test
Greedy traces all met SLA, and Random was close to 99%. Preserve that source and
partial outputs as a superseded setting. Select a stated busy transition
condition instead: continuous segment mean normalized pressure >=0.45 and
maximum range of either pressure feature >=0.45; candidate starts are 10 bins
apart. Select these candidates from each split using workload statistics only,
without policy performance criteria; sample eligible segments uniformly for
each node. Forecast labels and histories remain strictly within that split.
This selected-condition experiment is not an all-week average performance claim.

The two pressure features modulate light-service
availability using the existing two resource weights and stochastic innovations.
Link multipliers decrease with the mean request pressure of their two endpoints.
No sinusoidal regime or complementary link multiplier from the synthetic driving
profile is retained. Task arrivals, instance counts, nominal rates, bandwidth,
reporting rules and mean service models stay identical to the driving setup.

Continue the preserved driving joint initialization for 48 episodes, with
training seeds 82000--82047, validation seeds 81000--81002, and test seeds
83000--83002. Apply the same validation checkpoint selection as above, and run
Proposed, Greedy, Random and Shortest Queue on identical paired traces. Record
raw and aggregated data hashes, split, source segment offsets and normalization.

HPC4's base PyTorch 2.11 CUDA13 cannot initialize CUDA with its 570.211.01 driver.
Use the existing `fnochannel` environment (PyTorch 2.10 CUDA12.8) for all four
HPC policies, rather than installing a new stack or allocating a GPU to CPU work.

## Completed driving continuation

Run: `runs/routing_driving_more_20260916_235258`, training source commit
`63f14f3cbbff8af0bc018ce1dbe7722dacf9b110`. The 48-episode continuation completed
in 608.6 seconds on Fedora. Independent validation rose from 94.99% to 98.80%;
validation selected episode 48. The three original test traces contain 2,989
arrivals for every method, with matching arrival/resource hashes.

| Policy | Test SLA | Failures | Mean completed latency | P95 |
|---|---:|---:|---:|---:|
| Proposed | 99.83% | 5 | 36.91ms | 65.25ms |
| Greedy | 95.92% | 122 | 43.30ms | 80.35ms |
| Random | 52.26% | 1427 | 83.63ms | 139.37ms |
| Shortest Queue | 77.89% | 661 | 61.55ms | 108.37ms |

Random leaves one request pending at the finite evaluation horizon; it is
included in the SLA-failure count. All other requests finish. Compared with the
preserved earlier Proposed model, failures decrease from seven to five, mean
latency from 38.46ms to 36.91ms, and P95 from 68.61ms to 65.25ms. Relative to
Greedy, SLA increases by 3.91 percentage points, mean latency decreases 14.76%,
P95 decreases 18.80%, and failures decrease 95.90%.

The mean pre-clipping gradient norm is 0.7412 and the maximum is 5.3377; no update
clips at threshold 10. Norms remain nonzero, so do not claim a proven optimum or
strict convergence. Task gradients reach both compression and importance layers.
Actually transmitted active-window reporting bytes remain 1,280 versus 3,120
per trace, a 58.97% saving against Greedy.

Random receives the common experiment monitoring traffic but its action ignores
measurement values. Its reported telemetry bytes describe the common monitoring
protocol, not an intrinsic communication requirement of random routing. The
communication-saving claim is against the measurement-based Greedy controller.

The final 48-episode validation result was still the highest observed, and Fedora
became idle while HPC4 was training Azure. Continue the validation-selected model
for another 48 episodes (training seeds 72400--72447), bringing additional
driving training to 96 episodes. The preceding model remains an eligible initial
validation candidate. This decision uses the rising validation curve and
available runtime, rather than a test-selected checkpoint. Preserve the complete
48-episode result and include both stages in the final training plot.


## Small PPO comparison

Add masked categorical PPO on the same delivered 161-dimensional state, with
64-wide actor/critic, LayerNorm, lr=3e-5, clipping ratio 0.2, four epochs of
current-episode batches of 256, temperature 0.25 and entropy weight 0.001.
The actor retains the identical forecast-stage-cost prior and bounded 0.25
learned residual. Returns are whole-request terminal SLA success with gamma=1,
lambda=1; normalize advantages over the rollout. No persistent replay or DQN
bootstrap targets are used. Compression and importance remain jointly trained
with the task surrogate/value loss and the same forecast/budget auxiliary.
Log policy/value loss, entropy, approximate KL, policy clipping fraction, and
pre-clipping gradient norms including task-only compressor/importance gradients.

Run 48 episodes on driving and Azure busy transitions, using the same first-stage
train/validation/test seeds and physics. Initialize compression from the retained
original joint driving checkpoint, but initialize PPO actor/value heads fresh.
DQN continuation retains its previously trained Q network. This compares useful
end-to-end implementations and runtime; it is not an initialization-controlled
claim about universal algorithm superiority. Select checkpoints only by validation
SLA and completed-latency tie-break. Keep existing DQN results and jobs.


## Completed 96-episode driving continuation

The second 48-episode stage completed in 608.1s. Highest validation SLA is 99.53%
at global additional episode 56 (stage episode 8); this validation-selected
checkpoint obtains 99.60% test SLA, 12/2989 failures, 37.13ms mean and 67.34ms P95.
The preceding 48-episode result remains intact at 99.83%, five failures,
36.91ms mean and 65.25ms P95. Further training therefore improved validation but
did not improve this small held-out test set; preserve and report both results.
Do not silently replace the validation-selected 96-stage checkpoint using test
scores. Pre-clipping gradient maximum across both stages is 5.3377, with no
clipping. Curves stabilize at high SLA with fluctuations; gradient monitoring
does not establish an optimal-convergence guarantee. No more DQN driving
continuation is justified by these measurements for the current deadline.
Result directory: runs/routing_driving_more96_20260917_000933.


## Completed Azure busy-transition DQN run

Root runs/routing_azure_20260917_000437, scientific source 025b491,
48 episodes, 48,000 joint updates, 981.9s including validation/test. Validation
selects episode 8 at 73.44% versus initial 71.45%. On all 3,025 held-out arrivals,
Proposed achieves 46.58% (1409 ontime), Greedy 42.58% (1288), Shortest Queue
29.29% (886), Random 11.97% (362). Proposed improves SLA by 4.00pp; all three
traces improve. Completed mean latency is 86.11ms versus 90.16ms Greedy,
P95 168.37ms versus 174.05ms. Proposed/Greedy have 138/147 pending at the horizon;
all pending count as SLA failures. This is a demanding selected background
condition with absolute SLA below 50%, rather than a universally feasible load
regime. Report wire stays 1280/3120B. Gradient max 10.7317, three of 48,000 updates
clipped (0.00625%). Task-only gradients reach compression and importance.
Future trace labels are training-only and all paired exogenous hashes match.


## Completed driving PPO comparison

Root runs/routing_driving_ppo_20260917_001925, source a5779eef1e5c90dcc95abdf0cfd70aa9a0d4f40e.
All 48 episodes complete with 6,148 joint updates in 388.5s. Validation selects
episode 0 at 95.39%; subsequent validation values drop to 81--92%. Thus the
selected model is the original inherited predictor with fresh zero-headed
forecast-cost actor, not a PPO-improved trained policy. It nevertheless obtains
99.83% test SLA (five failures /2989), 37.14ms mean and 67.21ms P95. DQN +48 has
identical SLA but lower latency (36.91ms, 65.25ms) and improves validation from
94.99% to98.80%. This PPO configuration did not yield training improvement;
do not present its fallback test success as evidence of PPO learning. Preserve
last.pt and the entire validation/gradient history. Pre-clipping norm maximum
4.4433, with no clipping and nonzero task gradients into compression/importance.


## Completed Azure PPO and final comparison

Azure PPO root runs/routing_azure_ppo_20260917_001928, source a5779ee,
48 episodes/6168 updates/674.3s, gradient maximum4.0898 and no clipping.
Validation again selects episode0 (71.25%); no later PPO policy improves it.
Test SLA is48.60%,1470/3025 ontime,83.46ms completed mean,156.39ms P95,
145 pending counted as failures. This fallback is higher than the trained
validation-selected DQN test46.58%, but is not a PPO-learning gain. Both PPO
plans inherit the original DQN-trained compressor and forecast-cost prior.
Keep DQN as the current main learned algorithm; no further algorithm sweep.

Final report and exportable SLA/validation figures:
runs/routing_algorithm_comparison_20260917/COMPARISON.md.
Complete outputs, source snapshots, last/best models, raw tests, and PDF figures
are copied to /Users/wangzixin/output/icc-review. Code has been synchronized to
Fedora and HPC4 while preserving all older results. Random and Shortest Queue
are completed and all algorithms share verified exogenous trace/arrival hashes.


## End-to-end forecast-cost repair

The prior optimizer re-encoded delivered source histories but retained stored
forecast-derived stage costs and downstream pressures. Consequently routing
losses did not train the forecast head, and PPO likelihood ratios did not include
the changed decoder/cost computation when joint prediction optimization changed
it. Add optional --end-to-end-forecast: each training observation records only
known static paths, parent input locality/size/readiness, queue ledger and received
report ages. During optimization, recompute the same integrated piecewise-rate
transfer estimate, shifted light execution rate and remaining-DAG pressure in
Torch from the current codec. Gradients now traverse the entire prediction/cost
path. Labels remain exclusively auxiliary-training targets; no physical future
realizations enter this recomputation. Actor state size and serialized reporting
protocol stay identical. Physical simulator, scenario, deadlines, load and
baseline policies remain unchanged.

Verify numerical equality against the existing estimator across all eight driving
stages and multiple physical sample phases, independence from future labels, and
nonzero task-only gradients into encoder, importance selector and predictor.
Then train complete driving/Azure cases under the existing validation-selection
protocol. Preserve all previously completed runs and checkpoints.


## Bounded SLA/delay objective

The repaired Azure PPO episode8 improves validation SLA73.44% to74.83%, but
completed P95 worsens151.75 to155.66ms and pending27 to50. Repaired DQN episode8
improves all three validation figures (SLA74.24%,mean64.26ms,P95149.73ms), yet
binary terminal SLA alone does not explicitly reward shorter failed-request
latency. Add one fixed small-weight delay preference, not a parameter sweep:
u=(I{L<=D}+0.02*(1-min(L/(3D),1)^2))/1.02 for completed requests; u=0 for pending.
Per-request success utility remains at least0.98 above every failure; all targets
stay in[0,1]. The delay term particularly differentiates late requests below3D.
DQN probability-shaped output is now an expected bounded utility, rather than a
pure success probability. Gamma1 and full same-request DAG credit remain intact.
No physical deadline, load, capacity, telemetry or baseline setting changes.

Expose --latency-weight0.02 and --selection-metricp95 (validation SLA first,
completed P95 then mean break ties). Record per-episode mean terminal utility.
Run a complete joint DQN training plan in each existing scenario, preserving the
four running binary-objective repaired plans. Future labels remain auxiliary
training-only. Pending requests receive no optimistic censored latency reward.
