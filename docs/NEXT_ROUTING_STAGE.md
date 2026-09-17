# 下一阶段：固定容量与延迟遥测下的预测路由

当前已完成主实验的源码基线：`21fddcfd2eec8811332d2aa7bc2daff0385bec3e`；标签 `joint-routing-baseline-20260916`。
下一阶段分支：`routing-next`。旧实验结果在 `runs/request_joint_20260916_195426`。

## 研究定义

物理网络状态持续变化，源节点本地采样，但控制器只收到周期性的延迟遥测。压缩器从本地历史提取未来相关浅层表征，重要性选择限制报文大小，RL 用完整请求的 SLA 结果联合训练。动作仍为一个 ready DAG 阶段选择执行节点。

物理推进为 1 ms，资源采样为 5 ms，上报为 100 ms；这些是不同的时间尺度。用户已明确：允许新请求复用已送达的预测表征。先完成初始状态获取，周期更新在后台进行；报文按实际到达时间逐节点更新缓存，不强迫请求等待最新一整轮。新流量开始前双方共同预热遥测 200 ms，排除无初始状态的冷启动影响。业务任务不在预热期生成，负载和 120 ms 实际到达窗口不变。

## 已准备的下一阶段修改

1. `configs/fixed_placement_routing_main.json` 是双方共享的固定参考部署，初始化后不再更改。Core 共 47 个：image_encoder 10、text_encoder 7、projection_c 10、multimodal_fusion 7、audio_encoder 6、reasoning 7。这是按当前平均工作负载及离散执行上限留基本余量的起点，不是 SLA 保证。所有 CPU/GPU/RAM/VRAM 约束已按现有模型核对。
2. Light 总数保持 27：9 类服务各 3 个，每个服务在 es1/es2/es3 各放 1 个。关键前端不再只有 es2 一个候选节点，为路由提供真实选择；每种方法使用同一布局。
3. 主脚本增加 `--report-ms` 和 `--report-bps`，默认仍为 100 ms 和 64 kbit/s。显式配置遥测预算，物理推进和逐阶段路由保持现有定义。
4. 保留小 Double DQN、输入 LayerNorm、lr=3e-5、同请求 DAG 轨迹、0/1 终端 SLA 目标，以及任务损失与预测辅助损失的联合优化。

## 已完成的容量调整主对比

固定上述布局、load=0.4、arrival=120 ms、drain=150 ms、report=100 ms、channel=64 kbit/s。训练 24 轮，每轮 1000 次更新。两方法各自独立运行；proposed 与传统 greedy 共享测试 seeds `56000,56001,56002`。该轮已经完成：联合框架 SLA 99.59%，传统贪心 100%；容量和全链本地实例使阈值饱和，没有证明 SLA 增益。输出目录为 `runs/routing_main_20260916_204411`。

命令模板（把 RUN_DIR 设置为本次唯一输出目录）：

```bash
RUN_DIR=runs/routing_main_next_YYYYMMDD_HHMMSS
.venv/bin/python -u scripts/run_joint_routing.py --bench proposed --output "$RUN_DIR/proposed" --placement configs/fixed_placement_routing_main.json --report-ms 100 --report-bps 64000 --episodes 24 --updates-per-episode 1000 --test-seeds 56000,56001,56002
.venv/bin/python -u scripts/run_joint_routing.py --bench greedy --output "$RUN_DIR/greedy" --placement configs/fixed_placement_routing_main.json --report-ms 100 --report-bps 64000 --test-seeds 56000,56001,56002
```

报告总体/分任务 SLA、合并请求平均/P95 时延、实际遥测字节、状态年龄，以及联合训练梯度。以配置冻结后的主结果为准，不把旧结果的事后分组当成新布局的验证。

## Git 与多机器维护

源码、配置、参考资料、初始化 importance.pt 与其数据 provenance 纳入 Git。`.venv`、缓存及实验输出忽略；每次运行的 last.pt、日志和请求结果留在 runs 中。

每个科学修改单独提交，用提交号识别版本。Fedora 作为主仓库；HPC 现有 Reasonix 工作保存在独立分支/目录，先比较 diff，再选择性合并。可以通过 git bundle 在两台机器间传递已提交版本。

```bash
git status --short
git diff
git add scripts/run_joint_routing.py configs/fixed_placement_routing_main.json
git commit -m "routing: describe the actual scientific change"
git log --oneline -5
```

## 当前主对比：先获取状态、再路由，负载加倍

保留相同固定部署（47 core、27 light），将 load 从 0.4 提至 0.8；不再次增加实例。平均请求率由约 9.94 / ms 提至 19.89 / ms。所有请求的到达、状态等待、传输、服务队列和执行都计入原始 SLA 截止时间。

上报周期仍为 100 ms，共享信道仍为 64 kbit/s。含头部报文 proposed 为 32 B，传统当前状态 greedy 为 80 B。10 节点单轮序列化为 40 / 100 ms，加相同 0.2 ms 编码、1 ms 传播和 1 ms 仿真量化；完整一轮状态实际分别约 42 / 102 ms 后全部可用，但不用于逐请求强制阻塞。初始状态在 200 ms 共同预热期间送达；之后每个包一到就更新对应节点缓存，请求继续复用最近送达的表征和预测。序列化延迟通过缓存状态年龄、预测的有效时间和刷新速度影响决策，不在持续业务阶段强制添加一个请求等待常数。若允许缓存复用，不能同时声称每个请求都必须额外等待一轮上报。实际字节分别报告总物理窗口和扣除预热的 270 ms 活跃窗口。

目的节点在控制器选定前未知，所以该阶段的数据转发从选择之后开始。沿用 ICC 的数据量、静态最短路径、时变速率积分、服务执行、Gamma light 工作预算和完整 DAG SLA。该时序修正双方相同。上报和业务数据仍为独立信道；这里研究获取状态造成的控制等待，未宣称两类流量在业务链路上的额外带宽竞争。

load=0.8 下 core 的平均阶段需求约 image 14.92、text 10.38、projection_c 14.60、fusion 9.63、audio 4.22、reasoning 10.26 / ms；当前 1 ms 离散执行的容量上界为 10、7、10、7、3、7 / ms。这是平均过载的有限突发实验，不能表述成稳定无限时域负载。新增容量对算法不免费等同于降低负载；该轮工作量严格加倍。

训练继续小 Double DQN、LayerNorm、lr=3e-5、联合任务与预测损失，16 轮 × 1000 更新，监测裁剪前梯度及任务损失传到压缩/重要性层的梯度。只跑完整 proposed 对传统 greedy，测试 seeds `58000,58001,58002`；不添加消融。双方共享实际到达、外生动态、部署、信道、获取协议和物理窗口。

## SANet 方法与版本命名

当前运行入口 `scripts/run_joint_routing.py` 加载 `CoupledCodec(kind=importance)`：本地 8×16 历史由小 MLP 映射为 8 维 latent，学习重要性，hard top-3 加位置掩码上报；控制器按掩码补零恢复 8 维稀疏表征，预测头估计未来资源和链路，Q 网络读取表征及紧凑上下文。任务 TD/BCE、未来预测 MSE 和重要性 soft budget 联合更新。对应 SANet V-D 的任务相关重要性筛选与联合训练哲学，不复现完整 MoPS/CNN 解压网络，不将 SANet 本身称为 IB。

旧版 `PredictiveIBCompressor` 的 GRU/高斯 latent 与 KL 损失仍在仓库，但当前入口未使用，因此本轮不能标记为 IB 方法。接收端的补零恢复不是学习复原原始完整高维历史；预测头进行未来状态预测，没有引入高维原始输入重建目标。

`runs/routing_acquisition_20260916_210337` 使用了逐请求等待最新整轮的额外假设，用户澄清缓存复用后已停止并标记 SUPERSEDED，不作为论文主结果。保留数据仅用于追溯。

## 控制器等待队列补正

旧 remaining-DAG 摘要使用每个实例池的最小 job 数，漏掉控制器 waiting 列表中的已知 ready 阶段。压缩器只预测后台算力/链路，不能替代此队列信息。当前摘要将 ready 阶段按已知输入位置映射到最近可执行节点，多输入分摊；累加等待量与已派发 job 数，按实例数归一化。Core 使用 1 ms 控制边界下的离散服务时间，light 使用收到的未来 50 ms 平均可用率估计工作压力，然后 log1p 缩放。只替换原 10 维摘要，整体 state 仍为 161 维。这个压力是可获得账本信息的近似，不读取真实剩余工作或物理未来传输时间。

Greedy 的节点分数、backlog 方法、合法 mask、物理模型和遥测协议不使用这个 state 字段。因此复用 `runs/routing_cached_20260916_211052/greedy` 完成的三条结果（提交 `b66f7a6`），保留原 manifest/命令/提交出处，不冒充新代码重跑。新 Proposed 重新训练 16 轮，使用同一部署、背景资源 seed、到达 seed、时间线和窗口。报告逐条检查配对到达与资源哈希。`routing_cached_20260916_211052/proposed` 的训练已停止，不作为最终 Proposed 结果。
