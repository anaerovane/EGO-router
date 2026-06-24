# Short Spec: EGO Budget-Aware Agent Orchestration

## 1. What this is
EGO（Entropy-Gated Orchestration）是一个 LLM agent 总控方法，用于在有限预算下动态决定：

- 是否继续
- 继续做什么
- 何时停止输出答案

支持的动作包括：

- `think`
- `tool:<name>`
- `delegate:<expert>`
- `stop`

---

## 2. Why we need it
当前很多 agent 系统的问题不是“模型不会回答”，而是：

- 什么时候该停没有原则
- 什么时候该调用工具没有统一标准
- 什么时候该委托 expert 没有统一决策逻辑
- token / latency / step 成本没有进入控制目标

EGO 要解决的是一个 **budget-aware orchestration** 问题，而不是单点回答问题。

---

## 3. Core idea
EGO 每轮根据当前状态估计：

- entropy
- margin
- disagreement
- verifier confidence
- remaining budget

然后分两步决策：

### Step A: 判断是否停止
如果：
- 当前不确定性已经足够低，或
- 所有继续动作的净收益都不高，

则执行 `stop`。

### Step B: 如果继续，选最值得的动作
对 think / tool / delegate 做统一打分：

- heuristic 版：`estimated_gain - cost`
- learned 版：`predicted_reward + exploration_bonus + λ * heuristic_gain - cost`

选择最高分动作执行。

---

## 4. Primary goals
1. 定义统一动作空间。
2. 定义 budget-aware stop rule。
3. 定义统一 routing / action scoring 机制。
4. 支持 heuristic 与 learned 两种 routing。
5. 输出可分析轨迹，支持 benchmark 与 ablation。

---

## 5. Scope
### In scope
- 单轮最终答案任务
- 有限 step / token / latency budget
- think / tool / delegate / stop 决策
- candidate posterior estimation
- synthetic stopping 实验
- controlled routing benchmark
- LangChain-like adapter

### Out of scope
- 生产级 agent 平台
- 长链工作流编排
- 多 agent 通信协议
- 大规模 RL 训练系统
- 真实世界 benchmark leaderboard

---

## 6. Required modules
### `ego_core`
- budget 定义
- uncertainty metrics
- stopping controller
- candidate posterior estimator

### `langchain_ego_adapter`
- 执行 think / tool / delegate
- 统一动作评分
- 输出轨迹与最终结果

### `learned_action_scorer`
- action feature builder
- LinUCB-style scorer
- online update

### `envs / solvers / policies`
- synthetic env
- oracle solver
- baselines
- theorem-aligned evaluation

---

## 7. Evaluation
### Stopping
比较：
- oracle threshold
- budget-aware threshold
- fixed threshold
- fixed depth
- immediate stop
- never stop early

指标：
- avg reward
- stop time
- final entropy

### Routing
比较：
- heuristic routing
- learned routing

指标：
- final reward
- first-action accuracy
- reward by task type
- first-action accuracy by task type

---

## 8. Current project status
当前项目已经有：

- formalization
- algorithm spec
- theorem scaffold
- synthetic stopping prototype
- LangChain-compatible adapter
- learned routing prototype
- mixed-task routing benchmark

所以当前定位应是：

> 面向 AAAI 方法论文的 research prototype。

不是产品项目，也不是纯 benchmark 项目。

---

## 9. Final positioning
一句话定位：

> EGO 是一个预算感知、基于不确定性的 agent orchestration 方法，用来统一解决 stop / think / tool / delegate 的控制问题。
