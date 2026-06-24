# AAAI Agent Method / EGO 项目说明

## 重要状态说明

关于当前数据状态、我生成过的伪数据文件、实际读取过的 benchmark 原始文件路径、以及后续如何构造真实训练数据，请先看：

- `docs/15_data_status_and_paths.md`

其中已经明确写明：
- 当前真实可用训练数据 = 0 条
- 哪些 `jsonl` 文件只是伪数据 / 预热格式文件
- 后续真实训练数据应该如何构造、字段格式是什么

## 这个项目是在做什么？

这个项目在做的不是一个具体业务应用，而是一个 **面向论文的方法型项目**：

> 研究一个 budget-aware 的 LLM agent orchestration 方法，
> 让控制器能够在 `思考（think）`、`调用工具（tool）`、`委托专家（delegate）`、`停止并作答（stop）` 之间做出合理选择。

项目当前的方法名是：

- **EGO = Entropy-Gated Orchestration**

它想解决的核心问题不是“LLM 会不会答题”，而是：

1. **什么时候应该继续思考，什么时候应该停？**
2. **什么时候应该调用工具？**
3. **什么时候应该把任务委托给某个 expert？**
4. **在有限 budget（步数 / token / latency）下，怎么做最优 orchestration？**

---

## 一句话理解

如果把一个 agent 系统看成“有很多可选动作的总控系统”，那这个项目做的就是：

> 给这个总控器设计一个有数学定义、可分析、可实验验证的决策策略。

它的目标是把 agent orchestration 从“靠 prompt 凭感觉调度”，推进到：

- 有明确状态定义
- 有 stop rule
- 有 action scoring
- 有 learned routing
- 有 theorem 和 benchmark 支撑

---

## 项目的核心研究对象

项目把 agent 每一步的选择形式化为：

- `THINK`
- `TOOL(m)`
- `DELEGATE(k)`
- `STOP`

也就是说，controller 每一轮都要判断：

- 再内部推理一次值不值？
- 调某个工具值不值？
- 问某个 expert 值不值？
- 还是说现在就应该直接停止输出答案？

这本质上是一个：

- **budgeted sequential decision problem**
- 带有 **stopping** 与 **routing** 的 meta-controller 问题

---

## EGO 方法的基本思路

当前项目中的 EGO 方法，大致由两层组成：

### 1. 停止层（Stopping）
控制器根据当前不确定性和剩余预算，判断是否应该停止。

项目里当前使用的不确定性摘要包括：

- **Entropy**：候选答案分布的熵
- **Margin**：第一候选和第二候选之间的差距
- **Disagreement**：不同 candidate / path / tool / expert 之间的一致程度
- **Verifier confidence**：verifier 对当前最好答案的置信度

如果：

- 当前不确定性已经足够低，或
- 继续动作的净收益已经不大，

那么 controller 就会选择 `STOP`。

### 2. 路由层（Routing）
如果还不该停，就要在所有 continuation actions 里选一个最值得做的动作。

项目里把它写成：

- 估计每个动作的 value-of-information（信息价值）
- 再减去动作成本
- 选择得分最高的动作

所以 EGO 不是简单“先想再搜再答”，而是一个**动态决策器**。

---

## 这个项目现在已经做到哪一步了？

项目现在已经有三条很清晰的工作线，而且都已经落了代码和文档。

### 第一条：方法 formalization
已经写了比较完整的论文式 formalization，定义了：

- 问题设定
- 状态表示
- 动作空间
- budget
- posterior approximation
- utility objective
- stopping rule
- action scoring

对应文档：

- `docs/01_formalization.md`
- `docs/03_algorithm_v1.md`

### 第二条：理论 / theorem scaffold
项目已经开始围绕方法本体写 theorem：

- Theorem A：budget-dependent threshold stopping
- Theorem B：uncertainty estimation error 下的性能损失界

而且已经为 Theorem A 做了一个 synthetic 环境和 oracle 对照。

对应文档：

- `docs/05_theorem_A_draft.md`
- `docs/06_theorem_A_refined.md`
- `docs/07_synthetic_env_for_theorem_A.md`
- `docs/08_theorem_B_draft.md`

### 第三条：可运行原型
项目不只是写文档，也已经有 prototype：

1. **Synthetic stopping prototype**
   - 用一个可控环境验证 budget-aware stopping 是否合理
   - 可以和 oracle threshold、fixed threshold、fixed depth 等 baseline 比较

2. **LangChain-compatible adapter**
   - 提供一个框架无关的 EGO 控制核心
   - 再套一层 LangChain 风格 adapter
   - 可以在 think / tool / delegate / stop 之间做动作评分和执行

3. **Learned routing prototype**
   - 加了一个 contextual bandit / LinUCB 风格的 learned action scorer
   - 用来研究 heuristic routing 和 learned routing 的差异

---

## 从代码结构看，这个项目主要分成什么模块？

### `src/envs/`
放 synthetic environment。

当前重点是：

- `synthetic_entropy_env.py`

这说明项目先从一个可控的 entropy 演化环境入手，验证 stop rule 和 threshold theorem。

### `src/solvers/`
放 oracle / DP 求解器。

当前有：

- `dp_oracle.py`

说明项目会先在 toy/synthetic setting 里求一个最优 stopping policy，当作理论和实验参照。

### `src/policies/`
放各种 stopping policy / baseline policy。

当前有：

- `baselines.py`
- `ego_threshold.py`

说明项目已经在比：

- oracle threshold
- fixed threshold
- fixed depth
- immediate stop
- never stop early
- EGO 风格 threshold policy

### `src/integrations/`
这是项目最关键的“方法落地层”。

当前有：

- `ego_core.py`
- `langchain_ego_adapter.py`
- `learned_action_scorer.py`

分别对应：

1. **EGO 核心停止控制器**
2. **LangChain 风格 agent orchestration 适配层**
3. **learned routing / LinUCB action scorer**

也就是说，这个项目已经从“理论上的控制器”走到“可以插进 agent stack 里的控制器”。

### `scripts/`
放实验和 demo。

包括：

- `run_synthetic_theorem_a.py`
- `langchain_ego_example.py`
- `langchain_ego_multitool_example.py`
- `langchain_ego_delegate_example.py`
- `langchain_ego_learned_scoring_demo.py`
- `mixed_task_routing_benchmark.py`

这说明项目已经同时在验证两件事：

1. **stopping 是否成立**
2. **routing 是否真的学到了 task-action 对应关系**

---

## 现在这个项目最像什么？

更准确地说，它现在像一个：

> **AAAI 方法论文的研究原型仓库**

而不是：

- 线上服务
- 产品功能
- 完整 agent 平台
- 面向生产的工具链

它的当前定位更像：

- 方法定义 + 理论草稿
- 受控实验环境
- 路由/停止策略原型
- 为论文实验做准备的 research codebase

---

## 目前已经能跑出什么结果？

### 1. 停止实验（Theorem A 对齐）
项目里已经可以跑一个 synthetic stopping 验证脚本：

- `python3 scripts/run_synthetic_theorem_a.py`

当前结果表明：

- oracle threshold 最优
- budget-aware threshold / EGO entropy gate 接近 oracle
- fixed-depth heuristic 更差
- 永不提前停止会明显更差

这说明：

- “按不确定性 + 剩余预算决定是否停止”这个方向是合理的

### 2. 混合任务 routing benchmark
项目里也已经有一个 mixed-task benchmark：

- `python3 scripts/mixed_task_routing_benchmark.py`

这个 benchmark 人为构造了几类任务：

- `math`
- `calc`
- `search`
- `code`
- `think`

并为每类任务指定最优首动作，例如：

- math → `delegate:math`
- calc → `tool:calculator`
- search → `tool:search`
- code → `delegate:code`
- think → `think`

这个 benchmark 的目的不是刷真实任务分数，而是验证：

> learned routing 能不能学会“什么任务应该先走什么动作”。

当前跑出来的结果大致是：

- heuristic avg reward ≈ `1.98`
- learned avg reward ≈ `2.29`
- reward gain ≈ `+0.31`
- first-action accuracy gain ≈ `+0.05`

这意味着：

- learned routing 已经开始比 heuristic routing 更有效
- 尤其在 `search` 类型任务上提升更明显

---

## 所以这个项目的真正贡献点是什么？

如果把这个项目抽象成论文贡献，它想讲的是三件事：

### 1. 一个新的 orchestration formalism
把 LLM agent 的控制过程形式化为：

- 不确定性感知
- budget-aware
- think/tool/delegate/stop 联合决策

### 2. 一个可解释的 stop + routing 方法
核心不是端到端黑盒学策略，而是：

- 用 uncertainty summary 描述状态
- 用 threshold 控制 stop
- 用 value-of-information / learned scorer 选择动作

### 3. 理论与实验对齐
项目不是只写工程 demo，而是尝试建立：

- theorem A ↔ stopping 现象
- theorem B ↔ uncertainty estimation robustness
- learned scorer ↔ routing improvement

这种“方法—理论—实验”三者对齐的结构。

---

## 当前项目的边界和不足

也要明确，这个项目现在还处在 research prototype 阶段。

### 还不是最终论文成品
原因包括：

1. theorem 还在 draft/refine 阶段
2. learned routing 还是一个简化版 LinUCB scaffold
3. mixed-task benchmark 还是 controlled / mock benchmark
4. verifier 仍然有较强人工设计成分
5. 还缺更正式的 train/eval protocol、oracle baseline、cost-aware utility 分析等

所以当前最准确的说法不是：

> “已经做完一个完整 benchmark 系统”

而是：

> “已经搭好了一个论文方法原型，并开始形成第一批可验证实验。”

---

## 如果要把它对外介绍，可以怎么说？

可以用下面这段话：

> 这是一个面向 AAAI 方法论文的 research prototype，研究 budget-aware LLM agent orchestration。项目提出了 EGO（Entropy-Gated Orchestration）框架，让 agent 在有限预算下动态决定是继续内部推理、调用工具、委托专家，还是停止输出答案。项目目前已经完成方法 formalization、初步 theorem scaffold、可运行的 stopping prototype、LangChain-compatible adapter，以及 learned routing 的混合任务 benchmark，用于验证 stop rule 与 routing policy 的有效性。

---

## 建议的项目定位

最合适的定位是：

- **主线：方法研究**
- **骨架：理论分析**
- **验证：synthetic + controlled benchmark + adapter demo**

不要把它说成“benchmark 项目”，因为 benchmark 在这里是实验载体；
真正的主角是：

- **EGO 这个 orchestration 方法本身**

---

## 可以从哪里开始看？

建议阅读顺序：

1. `todo.md`
   - 看整体目标和 roadmap
2. `docs/01_formalization.md`
   - 看方法定义
3. `docs/03_algorithm_v1.md`
   - 看算法冻结版本
4. `docs/09_prototype_status.md`
   - 看现在到底实现了什么
5. `docs/10_langchain_integration.md`
   - 看系统接入方式
6. `docs/11_learned_action_scoring.md`
   - 看 learned routing 方向
7. `docs/12_mixed_task_benchmark.md`
   - 看 routing benchmark 的定位

如果想直接跑：

```bash
python3 scripts/run_synthetic_theorem_a.py
python3 scripts/mixed_task_routing_benchmark.py
```

---

## 总结

这个项目本质上是在做：

> **一个带预算感知、带不确定性感知、可停止、可路由、可委托的 agent 控制方法。**

它目前已经从“想法 / 论文草图”进入到“有 formalization、有 prototype、有初步实验”的阶段。

最准确的理解不是“它在做一个 benchmark”，而是：

> **它在做一个 agent orchestration 方法，而 benchmark 和 demo 只是用来证明这个方法是有意义的。**

---

## 运行 mock / real 双模式实验

项目现在支持两种实验模式：

### 1. Mock mode
适用于：
- 不依赖外部 API 的本地验证
- 检查 routing / trajectory / logging 是否正常
- 跑受控 benchmark scaffold

运行：

```bash
python3 scripts/run_configurable_routing_experiment.py \
  --config configs/experiment_mock.json \
  --save-json outputs/mock_routing_results.json
```

说明：
- 使用 mock LLM、mock search、mock experts；
- calculator 仍然使用本地安全计算器；
- verifier 使用规则打分；
- 不需要任何 API key。

### 2. Real mode
适用于：
- 接真实 LLM API 跑 routing / delegation 实验
- 用同一个模型实例化主 agent 与 expert wrappers
- 用 LLM-as-judge 做初步质量评分

准备环境变量：

```bash
cp .env.example .env
```

填写 `.env` 中的：

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
OPENAI_JUDGE_MODEL=gpt-4.1-mini
```

运行：

```bash
python3 scripts/run_configurable_routing_experiment.py \
  --config configs/experiment_real.json \
  --save-json outputs/real_routing_results.json
```

说明：
- 主 LLM 使用 OpenAI-compatible chat completions API；
- `delegate:math` / `delegate:code` 由同一模型配合不同 system prompt 实现；
- `tool:calculator` 使用本地安全计算器；
- `tool:search` 默认搜索项目本地语料目录（配置里默认是 `docs/`）；
- verifier 默认是 LLM-as-judge，因此 real mode 需要额外 judge 调用成本。

### 配置文件
- `configs/experiment_mock.json`：离线 mock 配置
- `configs/experiment_real.json`：真实 API 配置
- `configs/query_sets/mock_benchmark.json`：mock query 集
- `configs/query_sets/real_project_queries.json`：面向本项目的 real query 集

### 当前限制
当前 real mode 是“真实模型 + 本地工具 + LLM judge”的研究原型，不等价于正式论文实验。它更适合：

- 检查 EGO 在真实模型上的动作轨迹
- 验证 think / tool / delegate 切换是否合理
- 形成下一步正式 evaluation 的工程基础

它还不等于：
- 完整真实 benchmark
- 稳健 paper-grade 实验协议
- 外部 Web search 工具评测

---

## 运行 benchmark v3 evaluator

项目现在已经支持对 `data/realistic_mixed_task_benchmark_v3.json` 做统一 benchmark 评测。当前支持的 `evaluation_type` 包括：

- `numeric_exact`
- `substring_or_evidence_match`
- `reference_points_judge`
- `rubric_judge`

其中：
- mock 模式下，开放题默认使用启发式 key-point / rubric 匹配；
- real 模式下，可以启用 LLM-as-judge 对开放题做更强评测。

### 1. 离线跑 benchmark v3（mock）

```bash
python3 scripts/run_configurable_routing_experiment.py \
  --config configs/experiment_benchmark_v3_mock.json \
  --quiet-trajectories \
  --save-json outputs/benchmark_v3_mock_results.json
```

输出将包含：
- `reward_by_type`
- `reward_by_difficulty`
- `reward_by_split`
- `reward_by_evaluation_type`
- `first_action_accuracy_by_type`

### 2. 真实模型跑 benchmark v3（real）
先准备 `.env`：

```bash
cp .env.example .env
```

填写：

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
OPENAI_JUDGE_MODEL=gpt-4.1-mini
```

再运行：

```bash
python3 scripts/run_configurable_routing_experiment.py \
  --config configs/experiment_benchmark_v3_real.json \
  --quiet-trajectories \
  --save-json outputs/benchmark_v3_real_results.json
```

### 3. Benchmark v3 的文件位置

```bash
data/realistic_mixed_task_benchmark_v3.json
```

### 4. Benchmark v3 的当前定位
这是一个：

> AI-assisted realistic mixed-task benchmark v3 (300 items, difficulty-aware)

它已经适合作为：
- evaluator 开发底座
- heuristic / learned routing 对比底座
- per-task-family / per-difficulty / per-split 分析底座

但仍然不等价于：
- fully human-gold benchmark
- 已完成的论文主实验

### 5. 运行日志（log）
现在 runner 支持实时日志和日志文件输出。运行时会记录：

- 当前进度 `task=i/N`
- benchmark item id
- task type / difficulty / split
- evaluation type
- first action
- reward
- stop reason
- 单题耗时与总耗时

示例：

```bash
python3 scripts/run_configurable_routing_experiment.py \
  --config configs/experiment_benchmark_v3_real.json \
  --quiet-trajectories \
  --log-file outputs/benchmark_v3_real.log \
  --save-json outputs/benchmark_v3_real_results.json
```

日志文件会写到你指定的位置，例如：

```bash
outputs/benchmark_v3_real.log
```
