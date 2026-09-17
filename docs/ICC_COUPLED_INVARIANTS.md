# ICC 耦合模型接入审计

当前最终状态：下文原模型检查和加速检查之后，已进一步通过动态通信因果性及
原 Simulator 静态等价门槛。最终结果见文末“动态接入最终检查”。中间发现的
dispatch-start 版本只保留作工程诊断，不作为算法性能结果。

本次检查对象是 `edge_msd/simulation.py`、`controller.py`、`capacity.py`、
`network.py` 和用户提供的 ICC 论文，不是先前的单节点 LLM 路由代理。

## 应继承的语义

- 原 Eq. (8)：CPU、GPU、RAM、VRAM 四个维度，core 和 light 实例共同消耗节点容量。
- 原 Eq. (10)–(12)：core 实例每次只承载一个 stage；light 部署数乘并行度必须覆盖
  已路由及保留的占用。已执行或等待输入传输的实例都不能被撤销。
- 并行度是每实例可接纳任务数，不能把一个实例的计算能力乘以并行度。原运行器以
  FIFO aggregate work budget 实现这个闭合；新增实例才增加计算能力和资源成本。
- 原 Eq. (4)–(5)：完整 DAG 的依赖和跨节点输出传输决定完成时刻；fork-join 等待
  所有父服务及全部父输出到达。
- 原 Eq. (9)：只有端到端 deadline，不含先前代理新增的 LLM 回答质量门槛。
- 实验按时完成率采用所有到达请求为分母，包含晚完成和期末仍未完成请求。
  短期接入测试必须报告 censoring，不能只在完成请求中计算 SLA。
- 原 Eq. (7) 的并行成本在现有代码中按每部署实例计价，并非乘以并行度；本次不改。

`tests/test_icc_coupled_invariants.py` 直接对原 Simulator 检验以上物理约束，
共 12 个测试通过，ruff 检查通过。没有修改原生产文件，也不主张算法收益。

## 动态资源的最小接入点

1. `Simulator.light_rate(instance, now)`：默认保留原 Gamma 抽样；新模式使用
   按时间、节点、服务索引的预生成外生轨迹，避免策略改变 RNG 消耗顺序。
2. `Controller.processing_delay(service, node, parallelism)`：节点相关的预测
   服务率进入同一个平均时延或 EC 映射，替代仅按 service 查表的接口。
3. `Simulator.core_candidate_key(stage, instance)`：core 路由评分也应基于控制器
   实际收到的资源信息，不能仍用真实瞬时网络排序。
4. 物理 Network 和控制观测 Network 分离；状态变化后清空各自 delay cache。
   实际执行用物理值，部署与路由评分只读已送达报文推得的值。

需要特别防止 `Controller.step` 从真实 `instance.jobs[*].data_ready_ms` 得到尚未
完成传输的真实未来到达时刻。控制端应使用派发时按观测网络估计的 ready 时刻，
或只使用已确认的完成事件，不能直接复制物理端未来时间戳。

原 Network 在派发时把传输折算为 `Job.data_ready_ms`，不是逐时隙共享链路队列。
继续使用该机制必须声明其为传输开始时确定时延的近似；不能宣称已模拟传输期间
每一次 channel 变化。跨时间相关服务率下，原独立 Gamma EC 公式也只是条件近似，
不能由此声称新场景具有严格的端到端违约概率保证。

## 小规模性能检查

实测原 10 节点、9 类 light 服务、4 个 DAG，`load=.2`，20 个 1ms slot：

| 项目 | 时间或次数 |
|---|---:|
| core MILP 初始化 | 3.14 s |
| cProfile 下整个运行 | 2.96 s |
| Controller.step 累计 | 2.89 s |
| evaluate | 3,579 次 / 2.07 s 累计 |
| feasible | 8,939 次 / 1.35 s 累计 |
| resource_usage | 0.455 s 累计 |
| DelayModel.delay | 98,480 次 / 0.336 s 累计，主要为缓存查询 |

各函数累计时间有包含关系，不应相加。该样例只有 93 个到达、33 个完成，故只用于
性能分析，不能用其 35.5% 按时完成率判断算法能力。`max_greedy_steps=100` 未触顶。

主要瓶颈是重复的候选部署评价与容量检查，而不是 EC 求根。优先复用固定 core
placement，并在每个控制时刻缓存同一 counts 的可行性和 stage/node 数据就绪时间。
改变调度周期需要作为明确实验参数；不能为了提速无声改变 DAG 的调度粒度。

先完成 100–300ms 到达窗口、至少覆盖最大 deadline 的 drain、2 个配对种子和少量
方法的接入验证，再扩大统计实验。负载和到达轨迹对比时必须保持各方法一致。

## 已落地的等价加速

新增 `edge_msd/icc_coupled/fast_controller.py`，预构建资源矩阵并在每个控制时刻
缓存部署可行性和纯候选评价。round-robin 的带副作用评价不缓存；每次返回独立的
Decision 字典，避免调用方修改缓存。没有改变控制周期或 greedy 候选集合。

`tests/test_icc_fast_controller.py` 增加 8 项测试。四种原方法在同种子小场景中的
逐 slot 决策、完整 stage trace、SLA 和成本均与原 Controller 完全一致；另检查
busy 状态更新后的缓存失效、非法 bool 数量、返回值副本和 round-robin 副作用。

10 节点、30ms、load=.2、seed109 的两次配对实测：

| 重复 | 原 Controller | FastController | 加速比 |
|---|---:|---:|---:|
| 0 | 3.018 s | 1.993 s | 1.51× |
| 1 | 3.160 s | 2.059 s | 1.53× |

四次运行的完整 results 字典完全一致，每次 140 arrivals、63 completed；core
placement 在计时前固定。该 greedy 样例没有重复 evaluate 候选，实测收益主要来自
资源矩阵和 feasible 缓存。原始记录位于
`runs/icc_coupled_fast_controller_profile.json`。20 项新增测试全部通过。

## 动态接入最终检查

`tests/test_icc_coupled_causality.py` 和 `tests/test_icc_coupled_equivalence.py`
共 10 项测试通过，覆盖：

- instant 参考仅使用当前值，改变未来轨迹不改变控制器得到的预测；
- 控制器的实例副本不泄露真实未来输入到达时间、剩余工作量或完成时刻；
- 改变这些隐藏物理信息不会改变同一观测下的控制决策；
- 32B 报文实际传输完成并经过传播延迟前，控制器保持 prior；
- 1ms 和 5ms 控制周期均按每 1ms 物理时隙收取维护费用；core 部署不变；
- keyed Gamma 不随无关实例创建或查询顺序改变，同位置样本配对；
- 已于过去完成的传输，不受未来 channel 变化影响；
- 两个小场景种子下，静态 CoupledSimulator 与使用相同 keyed Gamma 的原
  Simulator 完整结果与轨迹一致。

最后一项又在真实 ICC 参数重建场景复核：seed8500、load=.2、40ms 到达窗口、
100ms drain、固定相同 core placement、control/physical/EC window 均 1ms。
198 个请求和 1,239 条 stage event 的全部原结果字段一致；均按时完成，平均时延
16.5503628817ms。这是实现等价性验证，不是新算法增益。
记录在 `runs/icc_coupled_restored_equivalence_v1.json`。

动态网络最终沿固定 nominal 路径积分真实资源轨迹，积分起点是原 ICC 的
uplink-ready 或各前驱完成时刻。控制器只用已收到报文推得的预测。修复过程曾发现
两个中间版本问题：在 dispatch 时才开始转发改变了原 baseline 的 core 预约时间；
恢复原就绪公式后，积分若仍从当前时刻开始，会令过去传输依赖未来 channel。
最终版本同时保留原时间语义并修复积分锚点，以上因果性测试防止回归。

`runs/icc_coupled_timing_8200` 和早期 `icc_coupled_smoke`、`icc_coupled_eval_8200`
等目录属于被发现执行语义偏离的中间工程诊断。其低 SLA 不应解释为原 ICC 问题
不可行，也不应与修复后的实验混合聚合。alpha 离线扫描仅为保留的参数审计，
没有据此更改主实验的 alpha=3。

## 最后一次有界性能改进

FastController 进一步按服务索引候选节点，按控制时刻缓存 processing delay 和
data-ready，并对 greedy 的单实例增量只核查被改变节点的资源使用。仍保留原
候选顺序、tie-break、最终共享占用重评分、目标与控制周期。

load=.4、seed8800、importance、20ms 到达＋50ms drain 的配对测量中，原 evaluator
耗时 2.310s，优化后 1.344s，即 **1.72× 加速**。70 个逐 slot 完整 Decision 严格
相同，全部结果字段和 stage trace 一致。36 项相关测试与 ruff 检查通过。

没有达到 2×，因此停止进一步优化，不声称更高加速。
原文件快照为 `runs/icc_coupled_fast_controller_before_v2.py`，最终测量记录为
`runs/icc_coupled_fast_controller_v2_benchmark.json`。已运行的 Python 进程不会
自动载入磁盘代码改动。
