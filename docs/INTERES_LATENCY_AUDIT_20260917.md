# 跨边缘延迟调整与状态过时诊断

本轮是用户要求的远节点传输延迟调整，以及并行 DQN/PPO 训练。保留此前 `routing-selected-results-20260917` 的选定结果；这轮没有显示足够比较增益，不替代旧场景。

仿真源码提交 `e63e73fc7f71c6f72e814849f3702388c6fe69be`。三个跨 ES 链路的基准速率统一为 6.425834431 Gbit/s；传播延迟原本就是 0。部署 72 core/54 light、load=0.8、驾驶 DAG/SLA、100 ms 上报周期与共享 64 kbit/s 上报信道不变。

| 方法 | 测试 SLA | 已完成平均延迟 | 已完成 P95 | 未完成 |
|---|---:|---:|---:|---:|
| DQN | 72.60% | 67.08 ms | 148.19 ms | 91 |
| PPO | 75.27% | 65.69 ms | 144.82 ms | 97 |
| Greedy | 74.98% | 67.16 ms | 141.64 ms | 45 |
| Shortest Queue | 69.52% | 69.70 ms | 148.54 ms | 101 |
| Random | 54.84% | 85.87 ms | 178.26 ms | 202 |

三条测试共 3025 请求。全部到达进入 SLA 分母，未完成计失败；平均/P95 仅统计已完成请求。压缩上报 1280 B/轨迹，原始上报 3120 B/轨迹。

DQN/PPO 从相同已有 DQN 联合 checkpoint 继续训练 48 轮，训练 seed 84400…84447，验证 81000…81002，测试 83000…83002。本轮 checkpoint 仅由验证 SLA/平均延迟选择；开发期间已检查相同测试 seed，不能称为未接触 holdout。

DQN 选第 8 轮，验证 99.60%，48,000 更新、940.58 秒，裁剪前梯度最大 13.169，9 次裁剪。PPO 选第 40 轮，验证 99.77%，6,188 更新、742.00 秒，梯度最大 8.342，无裁剪。两种算法压缩器和重要性层的 RL 任务梯度均非零。梯度没有失控，但不能据此证明最优收敛，也不能将验证成绩作为测试成绩。

## 状态过时是否真实出现

Greedy 重放逐请求与原结果一致。固定决策时队列，用候选输入就绪时的实际外部链路/服务轨迹重算 Greedy 代价：全部 light 阶段 2085/12024（17.34%）发生排序反转，audio_preprocess 为 1048/3025（34.64%）。这些 light 阶段只有两个合法节点，因此排序反转就是报告认为最好的节点成为最差。

这项诊断比较的是过时报告下的估计和输入就绪时的实际外部状态，涵盖报告老化及传输期间变化；它没有单独分离“决策后”的变化，也不是改选另一个节点后的完整反事实执行。仿真资源/链路每 5 ms 更新，计算实例处理每 1 ms 更新。

## 文件位置

- 本地汇总：`/Users/wangzixin/output/icc-review/routing_homogeneous_comparison_20260917/COMPARISON.md`。
- Fedora DQN：`/home/eric/icc_deploy/icc_sanet_project/runs/routing_azure_homogeneous_interes_dqn_20260917_104208`。
- HPC4 PPO：`/home/eewangzx/icc_sanet_project/runs/routing_azure_homogeneous_interes_ppo_20260917_104244`。
- 两套完整实验均已保存到本地 `/Users/wangzixin/output/icc-review/` 的同名目录，包括源码、模型、原始 JSON/CSV 与 PDF/PNG 图。

此前仅将跨边缘速率统一乘 2 的中间尝试已停止：Fedora `routing_azure_fast_interes_dqn_20260917_102954`、HPC4 `routing_azure_fast_interes_ppo_20260917_103000`，保留已有输出，不作为完成的正式训练结果。

Azure 使用真实请求/输入 token 轨迹映射背景压力；链路/资源状态仍为模型，前台业务仍为 ICC 驾驶 DAG。这轮提示：减少远端传输代价也会帮助贪心，状态会过时这一事实本身不足以保证学习策略取得大增益。
