# 预测接入的有界审计

检查范围：InformedController 的 Gamma 缩放、EC 映射、50ms 聚合、0.05 量化、
预测时间索引、节点与链路顺序。不更改 SLA、负载、控制周期或原 ICC EC 代理。

发现并修复了一个确定的时间索引错误：接收预测已对齐当前绝对 5ms 采样区间，
而原 observed Network.factor 用 `floor((when-now)/5)` 索引。now=6ms 时，
10ms 的链路积分仍取原 5ms 区间的预测。现在使用两个绝对区间索引之差，保留
过去时间 clamp 到 0 和预测末端截断。新增边界测试，因果性与静态等价测试共
11 项通过。此前结果保留作修复前诊断，不与修复后的结果混合。

没有发现 Gamma scale、节点/服务映射或传输倍率方向的其他确定实现错误。
0.5、1、1.5 三个常数倍率下，九类服务的 processing_delay 均随可用率增加
而单调不增；倍率为 1 时回到原 ICC 映射。

更准确的状态不保证固定启发式表现更好。目前仍有如下明确的代理局限：

- EC 使用 `parallelism * work / 1ms` 作为持续到达负载，不是当前有限队列
  的精确清空时间。预测资源变差可能让本来能在几毫秒内完成的有限 job 被评为
  无穷时延，导致延后调度或额外部署。
- audio_preprocess 的单 job 临界倍率约 0.56675。倍率 0.5 时 p=1 就返回
  无穷，而倍率 1 时为 1ms。seed8500 的 270ms 轨迹上，约 34.4% 的节点—时间
  位置在当前值量化后触发这一 p=1 拒绝条件；这不是请求失败率。
- 0.05 量化会跨越硬阈值：预测 0.574 被量化为 0.55，audio p=1 为无穷；
  0.576 被量化为 0.60，对应 8.26ms。因此很小的预测误差可能改变候选集合。
- 先平均未来 0–50ms 的 11 个采样值再套平稳 Gamma EC，是启发式闭合。
  实际时变过程有时间相关性，这不提供新的端到端违约概率保证。
- instant 参考把当前值延伸到全部未来，并不掌握未来真实轨迹；raw/stats16
  也经过学习预测头，而非把报文里的当前量无损直通至控制器。

上述代理局限没有在本次审计中被修改。常数倍率检查及量化示例保存在
`runs/icc_coupled_forecast_integration_audit.json`。

固定模式的正式对照必须与 PPO 的选择目标一致：在 seed8400 上比较九种固定
eta/parallelism 模式的 **physical-unit discounted return**，gamma=.995 每物理
毫秒，并以 interval 起点计权；选择后仅在 seed8500/8501 测试。总奖励、SLA、
完整成本另行报告。正式脚本和记录为
`edge_msd/icc_coupled/fixed_modes.py` 与
`runs/icc_coupled_fixed_modes_v2_discounted`。早期按 undiscounted total reward
选择的结果单独留在 `runs/icc_coupled_fixed_modes_v2`，不作为匹配的最佳固定模式。
