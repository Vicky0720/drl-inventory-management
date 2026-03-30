# IE5604 Project 说明

## 项目简介

本项目对应 IE5604 Final Project 中的 `Project 1: DRL for Inventory Management`。

研究对象是一个经典的 `lost-sales inventory replenishment` 问题，目标是：

- 正确模拟库存系统动态
- 实现 benchmark heuristic policies
- 实现三类课程要求的 DRL 方法
  - Q-learning: `DQN`
  - Actor-Critic: `AC_agent`
  - Policy Gradient: `PG_agent`
- 在统一指标下比较 heuristic 与 DRL 的表现
- 尝试利用 heuristic / inventory structure 改进 DRL

当前仓库已经完成从环境、heuristic、DRL 到实验汇总的基础框架。

## 项目结构

### 核心代码

- `project1_part_a.py`
  - 环境定义
  - heuristic 策略
  - 评估函数
  - heuristic 参数搜索

- `project1_part_b.py`
  - `Replaybuffer`
  - `QNetwork`
  - `Policy`
  - `ValueNetwork`
  - `DQN_agent`
  - `AC_agent`

- `project1_part_c.py`
  - `PG_agent`（REINFORCE）
  - 单组实验入口
  - 多算法训练汇总
  - 结果整理与绘图接口
  - heuristic-guided improvement 入口

### Notebook

- `project1_notebook.ipynb`
  - 课程要求的主 notebook
  - 已接入上述 `.py` 模块
  - 可以顺序运行

### 检查脚本

- `sanity_check_part_a.py`
- `sanity_check_part_b.py`
- `sanity_check_part_c.py`

这些脚本用于快速验证：

- A 部分是否正常
- B 部分是否能训练
- C 部分是否能跑通统一实验

### 项目文档

- `REPORT_TEMPLATE.md`
  - 报告结构模板

- `memory.md`
  - 当前进度与上下文记忆

## 当前已完成内容

### A 部分：环境、heuristic、评估

已完成：

- `Lostsale` 环境
  - `reset`
  - `sample_demand`
  - `step`
  - `rollout`
  - `generate_buffers`
  - `buffer_targets`

- heuristic 策略
  - `Basestock`
  - `Cappedbasestock`
  - `Constantorder`
  - `Myopic1`

- 统一评估函数
  - `discounted_cost`
  - `average_discounted_cost`
  - `evaluate_policy`

- heuristic 参数搜索
  - `search_basestock`
  - `search_capped_basestock`
  - `search_constant_order`
  - `evaluate_myopic1`
  - `run_heuristic_benchmarks`

### B 部分：DQN 和 Actor-Critic

已完成：

- `Replaybuffer`
- `QNetwork`
- `Policy`
- `ValueNetwork`
- `DQN_agent`
- `AC_agent`

### C 部分：Policy Gradient、实验器、结果汇总

已完成：

- `PG_agent`
- `train_all_drl_agents`
- `run_single_setting`
- `summarize_results`
- `plot_training_curves`
- `save_results_json`
- `heuristic_guided_action`

## 当前实验结果

### Heuristic benchmark

当前 notebook 中已经跑出一组 heuristic 结果：

- `Basestock`: `avg_discounted_cost = 6.4256`
- `Cappedbasestock`: `avg_discounted_cost = 4.7275`
- `Constantorder`: `avg_discounted_cost = 5.5519`
- `Myopic1`: `avg_discounted_cost = 5.3555`

结论：

- 当前最优 heuristic 是 `Cappedbasestock`
- 当前最优参数为：
  - `S = 16`
  - `r = 5`

### 单组汇总结果（heuristic + DRL）

从当前导出的 notebook 结果中，单组实验表现为：

- `Basestock`: `6.4881`
- `Cappedbasestock`: `4.5923`
- `Constantorder`: `5.4462`
- `Myopic1`: `5.3282`
- `DQN`: `10.7178`
- `ActorCritic`: `16.2569`
- `PolicyGradient`: `19.9272`

当前排序：

1. `Cappedbasestock`
2. `Myopic1`
3. `Constantorder`
4. `Basestock`
5. `DQN`
6. `ActorCritic`
7. `PolicyGradient`

### 当前结果的含义

这说明：

- heuristic baseline 已经跑通，而且结果合理
- 三个 DRL 方法已经可以运行
- 但当前 DRL 明显弱于 heuristic baseline

目前更合理的判断是：

- 项目框架已经跑通
- benchmark 部分可信
- DRL 训练还不充分，尚未收敛到有竞争力的水平

## 如何运行

### 1. 先跑 sanity check

建议先运行：

```powershell
cd <your-project-root>

python sanity_check_part_a.py
python sanity_check_part_b.py
python sanity_check_part_c.py
```

如果本机没有直接挂好 `python`，请用你自己的解释器路径替代。

### 2. 再跑 notebook

打开 `project1_notebook.ipynb`，按顺序运行单元即可。

注意：

- 环境导入单元应该很快完成
- heuristic 搜索单元会比较慢，这是正常的

## 为什么 heuristic 部分会比较慢

当前 notebook 中 heuristic 搜索默认配置为：

- `S_values = range(0, 31)`
- `r_values = range(0, 16)`
- `horizon = 1000`
- `n_sim = 100`

其中最耗时的是 `Cappedbasestock`，因为它会枚举大量 `(S, r)` 组合。

### 调试配置

```python
benchmark_results = run_heuristic_benchmarks(
    env,
    S_values=range(0, 11),
    r_values=range(0, 6),
    horizon=200,
    n_sim=20,
    max_order=env.max_order,
)
```

### 正式配置

```python
benchmark_results = run_heuristic_benchmarks(
    env,
    S_values=range(0, 31),
    r_values=range(0, 16),
    horizon=1000,
    n_sim=100,
    max_order=env.max_order,
)
```

## 下一步任务

当前最重要的后续任务不是继续补框架，而是做正式实验和调参。

### 任务 1：提高 DRL 训练预算

建议先把：

- `episodes=50`

提高到：

- `episodes=200`

如果时间允许，再尝试：

- `episodes=300`

### 任务 2：提高评估稳定性

建议把：

- `eval_runs=10`

提高到：

- `eval_runs=20`

### 任务 3：优先优化 DQN

原因：

- 当前 DQN 是三个 DRL 里最好的
- 最有希望先接近 heuristic baseline

建议目标：

- 先把 DQN 从 `10.7` 降到 `7~8`
- 如果能接近 `5~6`，说明训练有明显改善

### 任务 4：尝试缩小动作空间

当前 `max_order=30`，动作空间偏大。可以尝试：

- `max_order=15`

看 DQN / AC / PG 是否更稳定。

### 任务 5：跑完整 6 组参数

最终需要系统比较：

- `(l, p) = (2,4)`
- `(2,9)`
- `(3,4)`
- `(3,9)`
- `(4,4)`
- `(4,9)`

### 任务 6：整理图表和报告

后续应基于 `REPORT_TEMPLATE.md` 逐步补充：

- methodology
- heuristic results
- DRL results
- comparison analysis
- labor of division

## 当前项目的总体判断

一句话总结：

当前项目已经完成了从环境、heuristic 到三类 DRL 的基础实现，benchmark 结果合理，DRL 已经跑通，但当前训练结果尚未达到与 heuristic baseline 竞争的水平。接下来的重点应转向正式实验、参数调优和结果分析。
