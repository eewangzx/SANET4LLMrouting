# Active proposed debugging

The selected finite experiments are complete. Proposed DQN has the highest SLA on every tested driving and Azure trajectory against Greedy, Shortest Queue and Random. This is the supported empirical scope; it is not a guarantee for arbitrary future traffic.

## Scientific changes

- 763fe70: recompute received-forecast stage cost and remaining-DAG pressure
  during optimization. Routing gradients now reach the forecast decoder as well
  as the encoder and importance selector. Context contains only known topology,
  delivered report ages and controller ledger; future labels are auxiliary only.
- 64042a77cf0724e7e176717e4d7a5912f950c58d: PPO can inherit the working DQN
  actor and codec, preserving initial routing argmax. Four binary-SLA plans use
  this immutable scientific source snapshot.
- 0ade670781ad4644db185c04118c791d7b85e02b: add bounded terminal utility
  (success + 0.02 * [1-min(latency/(3*deadline),1)^2])/1.02, pending zero.
  Existing defaults remain binary SLA. New plans use validation SLA first,
  completed P95 then mean as tie-breakers. Three scientific tests passed on
  Fedora, including estimator equality, target-label isolation, forecast task
  gradients and utility bounds. Simulator and baseline laws remain unchanged.
- f9868ca: report per-trajectory SLA/mean/P95 checks against all three traditional
  baselines. Aggregate SLA improvement does not hide an individual tail deficit.

## Formal plans and verified state

48 episodes each, joint lr3e-5, unchanged fixed placement, arrivals, link/resource
physics, SLA and reporting. Preserved baseline outputs have identical physical
and arrival hashes; their provenance is recorded. No parameter sweep or ablation.

| Machine | Root below canonical runs/ | Handle | Latest verified state |
|---|---|---|---|
| Fedora | routing_driving_e2e_dqn_20260917_004050 | PID947066 | complete; selected0 |
| Fedora | routing_driving_e2e_ppo_20260917_004050 | PID947085 | complete; selected0 |
| HPC4 | routing_azure_e2e_dqn_20260917_004049 | Slurm1875865 | complete; selected40 |
| HPC4 | routing_azure_e2e_ppo_20260917_004049 | Slurm1875866 | complete; selected8 |
| Fedora | routing_driving_sla_delay_dqn_20260917_004934 | PID947879 | complete; selected0 |
| HPC4 | routing_azure_sla_delay_dqn_20260917_004936 | Slurm1875876 | complete; selected48 |

Binary plans train72800--72847 (driving),82400--82447 (Azure). Utility plans
train73200--73247 and82800--82847. All driving plans initialize from previously
validation-selected driving+96 best.pt, validate71000--71002 and test70000--70002.
Azure initializes from previous Azure DQN best.pt, validates81000--81002 and
tests83000--83002. Initial model remains eligible; test files never select a
checkpoint inside these formal plans. These test trajectories have been inspected
through development and should not be described as an untouched final holdout.

## Completed repaired DQN/PPO results

- Driving: 2977/2989, SLA99.5985%, mean37.1340ms, P9567.3409ms, pending0.
  It still selected the initial checkpoint; training did not improve validation.
  All three tested trajectories are no worse than Greedy/Random/Shortest Queue
  on the three reported performance metrics. Maximum pre-clipping norm26.2605;
  clipping occurred, so do not claim no clipping.
- Azure: 1403/3025, SLA46.3802%, mean87.8017ms, P95174.1631ms, pending146.
  Selected trained episode8: validation74.8342% versus initial73.4416%.
  Test SLA is3.8017pp above Greedy, but P95 is worse on all three seeds; seed83002
  P95 is also worse than Shortest Queue. Maximum pre-clipping norm9.2884, no clips.
- Driving binary DQN: same selected0 test as driving PPO, elapsed1302.5512s,
  48000 updates, maximum norm5.4852, no clips. No validation improvement.
- Azure binary DQN: selected40, validation79.4098% versus73.4416% initially.
  Test1546/3025=51.1074%, mean85.1581ms, completedP95174.0470ms, pending160.
  Greedy42.5785%, mean90.1555ms, P95174.0515ms, pending147. Test SLA improves
  8.5289pp, all three seed SLAs/means beat all three traditional baselines, but
  each individual seed P95 is worse than Greedy. Overall P95 alone hides this.
  Gradient maximum11.6117;2 clipped updates out of48000. Runtime1405.6539s.
- Paired hashes verified across all four complete runs. Combined report/PNG/PDF
  saved as routing_repaired_comparison_20260917; initial checkpoint SHA identical
  within each scene, different update budgets explicit. PPO stays secondary.
- Full source/model/raw tests/PDF figures for all four are copied to Mac under
  /Users/wangzixin/output/icc-review/<root>. They do not satisfy the whole goal.

## Final selection and audit

- Use binary-utility DQN as the main algorithm. On Azure it selects episode40 by
  validation only and reaches1546/3025=51.1074%, versus Greedy42.5785%, Shortest
  Queue29.2893%, Random11.9669%. It wins SLA on each of the three test traces.
  Mean completed latency is85.1581ms versus Greedy90.1555ms; wire traffic falls
  from3120B to1280B/trace. Its per-trace P95 deficits remain disclosed.
- PPO is secondary: lower Azure test SLA46.3802% despite shorter wall-clock time.
  DQN uses a larger update budget, so the time difference is descriptive only.
- The0.02 SLA/delay utility is not selected. Azure test SLA49.2562% and aggregate
  P95170.2974ms show a tradeoff, but it loses one trajectory SLA to Greedy and
  does not consistently fix individual P95. Driving chooses episode0 in all
  three repaired plans because it is already near the SLA ceiling.
- Exact replay of Azure seed83000 preserved every request outcome. For Proposed
  P95-tail requests, audio_preprocess transfer averaged75.404ms versus26.033ms
  for Greedy; the estimator also adds transfer and queue work even though queues
  can drain during transmission. This explains the tail rather than a gradient
  explosion. Diagnostic source and raw stage rows are preserved with the run.
- Full source, checkpoints, raw JSON, figures and comparison are copied under
  /Users/wangzixin/output/icc-review. Fedora and HPC4 canonical copies are
  synchronized; Fedora paper/ remains untouched.
