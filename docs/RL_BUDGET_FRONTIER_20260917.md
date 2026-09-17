# Learned routing under a binding average resource budget

The paper objective is maximum full-DAG deadline success subject to a long-run
average normalized resource charge. A request's deadline success and the
system's average cost constraint are different accounting levels. A high-SLA
policy exceeding the cost budget is an infeasible reference, not a winner over
a feasible learned policy. Comparisons use the SLA-cost frontier and feasible
SLA at the same budget; raw SLA alone is not the model-ranking objective.

## Policy roles

- Proposed: joint predictive codec and unrestricted Double DQN. Delivered Z,
  received predicted stage latency, the dispatch/completion ledger, incremental
  route costs and the cost virtual queue are inputs. No fixed action prior is
  added to the Q output. Codec/predictor/importance remain jointly trainable.
  Each candidate's Z, queue, forecast latency, charge and report status are
  explicitly aligned into a shared small action-scoring head conditioned on
  global state and static node identity. This improves the routing inductive
  structure without fixing a preference for latency or price.
- PDH (`predictive_deadline`): Claude's fixed deadline-aware predictive rule,
  with no Q network and no cost-aware routing. It is an independent baseline,
  never an episode-zero Proposed result.
- PM-DPP (`predictive_budget`): the same predictive information and virtual
  budget queue, but a current-request smooth deadline-success proxy minus the
  instantaneous shadow-priced resource charge. The proxy transition width is
  10 ms and its stage utility is divided by the known DAG stage count. No
  learned action value estimates future cross-request effects.
- LOM: latest delivered raw current state, no predictor and no cost objective.

Both predictive baselines can load `--joint-init best.pt` to use exactly the
selected Proposed codec while discarding all learned routing parameters.
This is a benchmark of learned sequential decisions with matched predictive
information, not a separate codec ablation.

## Credit and selection

Average-budget DQN uses chronological decisions across requests. Its reward
contains all SLA settlements between decisions and the Lyapunov shadow-priced
stage charge. Replay aggregates at least 10 ms of physical time before
bootstrapping, because three node choices often occur in the same millisecond
and cannot expose the ensuing queue/deadline effects. The budget queue is
uncapped; only its actor feature is transformed as log1p(Q/V) to preserve
information at a bounded scale under input LayerNorm.
The complete SLA-plus-shadow-priced-cost replay reward is divided by expected
arrivals in that credit window. This common positive scaling changes neither
the relative business-objective weights nor the optimal routing policy, and
prevents accumulated multi-request targets from reaching thousands.

Episode zero is diagnostic only. Only checkpoints after actual RL updates can
be selected. Validation first checks average-budget feasibility, then maximizes
SLA (and the explicitly configured data objective, if nonzero). Train,
validation and test seeds are distinct. Finite-trace budget violations and
pending requests are reported explicitly rather than hidden by raw SLA.

Average-budget PPO uses the same system rewards and budget queue. Its shared
candidate actor normalizes each node's latent input and learns unrestricted
logits; its critic regresses unbounded system returns with smooth-L1 loss.
GAE discounts by elapsed physical time, with a trace time of one quarter of
the return time. The entire business reward uses the same common scaling as
DQN. PPO has no fixed predictive action prior.

## Why sequential learning can help

Routing changes future per-instance backlog, DAG locality and the budget
shadow price. A forecast-informed policy can therefore prefer an action with
a smaller immediate deadline proxy when it preserves future capacity or
expensive-node budget for more valuable requests. The class of sequential
policies contains myopic policies, so its optimal attainable value is weakly
better; strict gain needs an active tradeoff and inter-request coupling. This
does not prove that a finite DQN run reaches that optimum. Actual RL value is
established by its feasible SLA-cost frontier relative to PM-DPP and PDH.

## Selected experiment

Fixed ICC deployment and task DAGs, load 0.5, Azure busy-transition dynamics,
240 ms arrival window, 200 ms common warmup, 250 ms drain, 100 ms reporting at
64 kbps. Nominal core demand is below aggregate deployed core capacity.
Pricing-only normalized lower bound is 0.56676; existing PDH costs are about
0.827. A core-flow capacity LP gives the stronger optimistic lower bound
0.75068 at load 0.5: every task-stage flow equals offered arrival rate, and
each core-host flow is at most fixed instance count / slot-rounded execution
time. Light capacities, transmission and deadlines are relaxed. This is a
necessary bound for stable complete-flow service, not a deadline guarantee.
`scripts/check_budget_capacity.py` reproduces the calculation. Budget 0.70 was
stopped and replaced before use in paper results.
The selected budget points are 0.76, 0.79 and 0.82, with no fixed
resource/data/latency reward weights in the main two-objective experiment.
The pricing-only lower bound does not establish deadline feasibility at that
minimum: concentrating all work on cheap nodes can create queues.

Produce three figures: SLA-cost frontier, deadline success by budget, and
realized normalized resource charge by budget. Mark infeasible points.
