# 使用 ICC 的实验设定：数据来源与接入状态

## 结论

ICC 的实验设定可以复用。论文 Section IV 明确说明每次实验从给定范围采样参数。
用户提供的仓库是独立于数据的模拟器，`examples/minimal.py` 明确写明不是论文数据。
本次检查 main 分支、远端分支/标签与 GitHub Releases：main 为
`682f41857fa5dd7a35be4757567f44e83a847207`，无其他分支/标签或 release 数据附件。
因此当前取得的是论文公开的生成设定，并非作者实验时保存的随机实例或 trace。

已经新增可序列化的数据生成器和原 ICC 模拟器的数据导入接口。数据类型标记为
`published-parameter reconstruction`，不是作者原始数据，也不是实际 LLM 测量。

## 原样转录的公开信息

来源：`2026ICC_ZJ_Final_v1.pdf`，Table I（PDF 第 6 页）、Figure 1/2（第 2 页）。

| 字段 | ICC 公开设定 |
|---|---|
| 任务与微服务 | Figure 1 的四类 DAG，6 个 core MS、9 个 light MS |
| 节点类型 | ED / ES；示意图含 7 ED、3 ES |
| 资源范围 | 按 Table I 转录 CPU、RAM、GPU、VRAM |
| core 工作量、处理率 | 2–16 MB，8–32 MB/ms；处理时间由工作量除以处理率计算 |
| light 工作量、处理率 | 0.5–2 MB；Gamma，shape 范围 1–2，scale 范围 1–20 |
| 请求到达 | Poisson，均值参数范围 0.15–1.5 /ms，按用户和任务类型 |
| 任务截止时间、输入大小 | 50–100 ms，0.5–4 MB |
| 有线链路 | 0.1–1.0 MB/ms |
| Nakagami 参数 | m 范围 1.5–3，Omega 范围 0.5–1；已保存但有线检查不使用 |
| 服务成本 | core: 20/4/0；light: 4/1/0.5；部署/维护/并行 |
| EC 参数、负载对照 | epsilon=0.2；1/1.5/2 倍负载，可由命令行选择 |

特别处理了两个单位陷阱：论文资源顺序是 `(CPU, RAM, GPU, VRAM)`，仓库是
`(CPU, GPU, RAM, VRAM)`；论文链路 MB/ms 转成仓库 Gbps 必须乘以 8。
请求到达参数从 /ms 转成仓库 /s 必须乘以 1000，不能使用 Bernoulli 到达替代 Poisson。

## 必须显式补充、不能冒充论文原值的内容

1. 论文给出范围但没有完整指定范围内采样规则；当前使用独立连续均匀采样，seed=2026。
2. 拓扑边按示意图构造，坐标为说明性坐标；有线传播时延取 0 ms。
3. 每个 ED 对应一个用户，各自产生四类任务；关联 ES 作为有线网络入口。
4. Figure 1 多模态根节点没有给出输入载荷分拆比例；遵循现有模拟器，根节点各接收完整输入。
5. Table I 对 gamma 的表头/单位不够清晰；没有把 Nakagami 随机值直接当作 Gbps。
   无线带宽及发射/噪声等完整配置未给出，所以本轮为显式的有线接入检查。
6. 检查时长 120 ms、时隙 1 ms、请求随机种子 109；这些不是已知的作者原实验值。
7. 除 epsilon 外的控制器设置使用当前仓库默认值，EC admission window 显式设置为 1 ms。

因此接入检查中的完成率/成本不能与论文图 3/4 的数字直接比较。

## 运行与文件

```bash
python -m examples.icc_paper --output runs/my_icc_check \
  --duration-ms 120 --methods proposed propavg lbrr --load 1.0
```

仅生成并保存场景：

```bash
python -m examples.icc_paper --output runs/my_icc_data --generate-only
```

再次读取同一份场景，而不重新采样参数：

```bash
python -m examples.icc_paper \
  --dataset data/icc_paper/scenario_2026.json \
  --output runs/my_icc_replay --methods proposed propavg lbrr
```

通过 Python 使用同一数据对象：

```python
from edge_msd.icc_paper import load_scenario

scenario = load_scenario("data/icc_paper/scenario_2026.json", load_multiplier=1.0)
# scenario 包含具体的 services、tasks、nodes、links、users。
```

- `edge_msd/icc_paper.py`：来源、范围、Figure 1 DAG、采样、JSON 导入。
- `examples/icc_paper.py`：原 ICC 模拟器的可复跑实验入口。
- `data/icc_paper/scenario_2026.json`：固定的已采样实例，可加载复用。
- `runs/icc_paper_check/`：场景、显式请求 trace、三种方法结果、配置和来源。
- `tests/test_icc_paper.py`：资源轴/单位、DAG、序列化、固定随机种子与负载缩放检查。

## 与 SANet / LLM 路由工作的关系

此次接通的是 ICC 原模拟器的数据层。早先 `runs/narrowband_*` 的 AE/PPO 结果仍然使用
旧的六节点合成环境，不能改标签称作 ICC 数据上的结果。本次也没有把 ICC 的六类核心
微服务当成可以互相替换的六个 LLM；它们在 Figure 1 中有不同功能。

新路由任务可以复用这份场景的资源、网络、请求类型与到达设定。改为备选 LLM 后，
需要显式定义模型部署、模型—请求适配关系、质量标签及运行时映射；ICC 本身没有
提供候选 LLM 的质量矩阵。50–100 ms / 0.5–4 MB 是 ICC 多模态 DAG 设定，也不能
不经验证地解释成真实 LLM 的 token 长度与完成时限。

要忠实复现作者原实验，还需要作者导出原 `Scenario` 数据或场景生成脚本、随机种子、
仿真时长/时隙、无线配置。取得后可以替换此 JSON 的对应字段，保留数据来源与版本。
