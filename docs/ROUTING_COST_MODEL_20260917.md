# Routing cost model, 2026-09-17

Scientific implementation commit: `14980e2bc01ae8c99c2db2a9b19104ec65c3418c`.

## Source and adaptation

Wang et al., *Service Routing in Multi-Tier Edge Computing: A Matching Game Approach*, IEEE JSAC 41(3), 2023, DOI 10.1109/JSAC.2022.3229378, defines a task computation requirement lambda_ij and node unit price p_k. Its Eq. (3), printed p. 838, is p_ijk = lambda_ij p_k. Experiment settings on printed p. 842 use allocated computation resources in [0.5, 5] GHz, unit prices in [3, 10], and 20 repetitions.

The paper motivates expensive nearby lower-tier resources versus cheaper remote higher-tier resources. It also describes high-cost nodes with shorter queues or transmission delays. It does not specify a complete per-node price assignment or sampling distribution. Its Greedy chooses a utility including cost; our existing traditional baseline chooses estimated stage completion time and is therefore latency-greedy, not a reproduction of that baseline. The source experiment says every request meets its deadline, whereas our dynamic ICC routing experiment measures deadline failures.

For our DAG, the per-request resource charge is:

\[
C_j^{res}=\sum_s w_s p_{n_{j,s}},
\]

where w_s is the existing scalar computation workload `service.work_mb`. It has work-MB units, consistent with the simulator's work/rate model; it is not the business payload size. The price has normalized cost units per work-MB, not dollars or measured energy.

Our explicit premium-compute pricing assumption separately normalizes CPU and GPU capacities, averages their normalized values, then min-max maps that score to [3, 10]. The interval comes from the source; this capacity-price mapping is our assumption. Prices stay fixed throughout all traces and policies:

| Node | p_k |
|---|---:|
| ED1 | 3.1702 |
| ED2 | 3.2704 |
| ED3 | 3.0994 |
| ED4 | 3.0000 |
| ED5 | 3.7793 |
| ED6 | 3.5845 |
| ED7 | 3.1206 |
| ES1 | 8.0206 |
| ES2 | 8.9069 |
| ES3 | 10.0000 |

Business communication cost is the sum of transferred payload MB times hops on the same fixed nominal shortest-latency paths used by the simulator. It includes gateway-to-root and predecessor-to-stage transfers. It excludes telemetry and does not add a new physical delay.

Both costs are charged once per stage-node assignment. Completed requests have complete DAG costs; pending requests can have only partial dispatched-work costs and are assigned zero terminal utility. Raw cost averages therefore must be read with the completion counts, especially in Azure.

## Learning objective

Resource cost is divided by the task's maximum additive legal-node resource charge. MB-hop is divided by an action-independent sum of maximum legal transfer costs for that task and gateway. These are upper bounds, so normalized costs lie in [0, 1]. Both incremental candidate costs are included in the actor state using known topology, prices and completed predecessor locations only.

For the first cost run, resource_weight=0.20 and data_weight=0.05:

\[
U_j=\frac{\mathbf1[T_j\le D_j]+0.20(1-\widetilde C_j^{res})+0.05(1-\widetilde C_j^{data})}{1.25}.
\]

This affine transform has the same ranking as SLA success minus the weighted normalized costs. It preserves [0, 1] targets for the sigmoid DQN/BCE critic and PPO critic. The sum of non-SLA weights stays below one, so an individual completed successful request has higher utility than any completed failure. This is a soft aggregate objective, not an automatic hard SLA guarantee. Validation selects mean objective utility, with SLA/cost/latency tie-breaks; test data are not used for selection. Episode zero is excluded.

## Completed driving run

Fedora: `/home/eric/icc_deploy/icc_sanet_project/runs/routing_driving_resource_cost_20260917_130300`.

Local: `/Users/wangzixin/output/icc-review/routing_driving_resource_cost_20260917_130300`.

Same placement, Poisson load 0.8, 200 ms warmup, 240 ms arrival window, 150 ms drain, 1 ms physics slots, 5 ms resource samples, 100 ms reporting period and shared 64 kbit/s telemetry channel. Compression/importance/predictor and DQN are jointly trainable. Fresh Q, prior joint codec, 48 episodes/48,000 updates; validation selects episode 24. Training took 629.04 seconds. Unclipped gradient maximum 1.518, no clipping; task gradient into codec was nonzero on at least 99.9% of updates.

| Method | SLA | Resource cost/request | MB-hop/request | Mean latency | P95 latency | Mean utility |
|---|---:|---:|---:|---:|---:|---:|
| Proposed | 99.26397% | 497.535 | 3.56852 | 37.8449 ms | 66.6386 ms | 0.840747 |
| Latency-greedy | 95.91837% | 494.747 | 3.57899 | 43.2983 ms | 80.3534 ms | 0.814761 |
| Random | 52.25828% | 430.527 | 9.62432 | 83.6287 ms | 139.3722 ms | 0.468095 |
| Shortest queue | 77.88558% | 464.012 | 7.79735 | 61.5454 ms | 108.3676 ms | 0.668262 |

Proposed has 22 failures versus Greedy's 122, improves SLA by 3.3456 percentage points and utility by 3.1893% relative. Resource cost rises 0.5636%, MB-hop falls 0.2927%. This run supports better SLA and latency at almost unchanged cost; it does not support significant resource-cost savings. All 2,989 requests complete for Proposed and Greedy. Random has one pending request.

## Completed Azure-background run

HPC4: `/home/eewangzx/icc_sanet_project/runs/routing_azure_resource_cost_20260917_130651`, Slurm job 1878056, completed. Local: `/Users/wangzixin/output/icc-review/routing_azure_resource_cost_20260917_130651`.

The same fixed driving DAG and Poisson arrivals are retained. Azure trace drives background link/resource dynamics, not the foreground driving requests. The trace maps 5 real seconds to 5 simulation milliseconds and uses a chronological 4/1/2-day train/validation/test split. Joint end-to-end forecast training has 48 episodes/48,000 updates; validation selects episode 48. Training, validation and testing took 1472.23 seconds. Unclipped gradient maximum 4.878, no clipping; task gradient into the codec was nonzero on every update.

| Method | SLA | Dispatched resource cost/arrival | MB-hop/arrival | Pending | Mean utility |
|---|---:|---:|---:|---:|---:|
| Proposed | 44.79339% | 458.414 | 3.84792 | 178 | 0.409523 |
| Latency-greedy | 42.57851% | 499.659 | 3.30666 | 147 | 0.381720 |
| Random | 11.96694% | 407.147 | 9.38871 | 407 | 0.139099 |
| Shortest queue | 29.28926% | 458.825 | 7.36129 | 186 | 0.275386 |

Across 3,025 requests, Proposed has 1,355 on time versus Greedy's 1,288, improves SLA by 2.2149 percentage points and utility by 7.2837% relative. Dispatched-work resource cost falls 8.2544%, while MB-hop rises 16.3686%. The extra pending requests mean that partial-DAG accounting must not be confused with full-request cost savings. Completed-request mean resource costs are 469.457 versus 512.129; on the 2,797 identical requests that both policies complete, they are 469.579 versus 512.234, a reduction of 8.3274%. This conditional statistic removes partial-DAG accounting but does not establish a full-cost bound for every arriving request.

The first two test traces have slightly lower Proposed SLA than Greedy; the third improves from 44.08% to 52.52%. Every per-trace completed-request P95 is higher than Greedy. Pooled completed-request P95 is 170.17 versus 174.05 ms; the mixture and differing completion counts explain why this pooled comparison must not be presented as uniformly improved tail latency.

Both run directories contain original JSON, models, summary.json, RESULTS.md and PDF/PNG plots. The output index is `/Users/wangzixin/output/icc-review/CURRENT_RESULTS_20260917.md`. The compatibility follow-up `948249e` preserves the cost-enabled state numerics while leaving legacy zero-cost states unchanged; each method's manifest records exact source hashes.
