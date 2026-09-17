# 恢复预测状态并独立训练的正式基线

该基线先单独训练压缩预测器，再冻结编码器、重要性层和预测头。线上仍上报32 B的Z；控制器将Z恢复为每节点61×14、按报告年龄对齐的预测序列，并与可观测队列、任务及剩余SLA一同输入两层、宽度64的DQN。恢复发生在控制器本地，不增加通信字节。

初版只训练预测器60轮，而且模型选择允许第0轮。第0轮并非随机路由：Q的value/advantage输出头被置零，策略只剩预测代价先验，因此它是确定性预测贪心，不能称为训练后的DQN。初版结果保留为诊断，不再作为正式独立RL基线。

正式重跑将预测器增加到600轮（34,200次梯度更新），DQN仍为48个episode/48,000次更新。第0轮只记录参考指标、禁止参与选择；被测checkpoint必须至少完成8个episode的RL更新。预测器仅使用训练轨迹历史与未来资源标签，以独立验证MSE选择checkpoint；它不使用SLA奖励。DQN阶段预测器完全冻结。

| 场景/方法 | SLA | 已完成平均 | 已完成P95 | 未完成 | 上报量/轨迹 |
|---|---:|---:|---:|---:|---:|
| 驾驶 Joint Proposed | **99.60%** | **37.13 ms** | **67.34 ms** | 0 | 1280 B |
| 驾驶 Decoded + trained DQN | 98.70% | 39.92 ms | 69.33 ms | 0 | 1280 B |
| 驾驶 Greedy | 95.92% | 43.30 ms | 80.35 ms | 0 | 3120 B |
| Azure Joint Proposed | **51.11%** | **85.16 ms** | **174.05 ms** | 160 | 1280 B |
| Azure Decoded + trained DQN | 46.68% | 88.14 ms | 180.72 ms | 205 | 1280 B |
| Azure Greedy | 42.58% | 90.16 ms | 174.05 ms | 147 | 3120 B |

驾驶独立基线相对Greedy提高2.78pp、比Joint低0.90pp。Azure相对Greedy提高4.10pp、比Joint低4.43pp。Joint Proposed在两套场景均领先真正训练后的独立DQN基线。

驾驶预测器验证MSE从0.135827降至0.018361，选择第522/600轮，耗时91.40s；DQN选择第8轮，验证SLA 96.99%，测试2950/2989按时完成。Azure预测器从0.208831降至0.006903，选择第196/600轮，耗时91.97s；DQN选择第48轮，验证SLA 69.03%，测试1412/3025按时完成。

驾驶DQN梯度最大8.924、无裁剪；Azure最大13.509、22/48,000次裁剪。两套codec参数变化严格为0。训练、验证、测试seed分离；逐seed到达和资源hash与Joint/Greedy基线一致。

## 成本边界

ICC场景仍包含部署、维护和light并行成本：core为20/4/0，light为4/1/0.5。但当前研究固定部署，初始化时计算的总成本对所有路由动作相同，因此不进入路由reward。当前优化目标是完整DAG是否在deadline内完成，不包含可由路由改变的货币、能耗或独立可靠性成本。不能声称当前路由实验复现了ICC的SLA—部署成本权衡。

若保持纯路由问题，需要另行定义动作相关的边际运行成本，例如跨站流量费用、节点执行价格或由并发使用造成的light parallel cost；固定部署成本不能直接加入reward，因为它只会给所有动作增加相同常数。

完整结果：

- `/Users/wangzixin/output/icc-review/routing_driving_decoded_trained_20260917_120939/RESULTS.md`
- `/Users/wangzixin/output/icc-review/routing_azure_decoded_trained_20260917_120942/RESULTS.md`

旧的60轮/允许第0轮目录保留为诊断：`routing_driving_decoded_separate_20260917_110502`和`routing_azure_decoded_separate_20260917_110501`。
