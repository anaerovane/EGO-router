# Handoff Prompt for the Next Agent

你现在接手的项目是：

- 仓库：`aaai_agent_method`
- 目标：为 EGO / router / scorer 构造**真实可用的训练数据**

请注意：

## 第一原则

**不要重复生成伪数据。不要把 task 文本直接改写成 action_scores。不要把 benchmark 的 reference trajectory 直接当成 router 监督标签。**

当前仓库里已经有一些我之前生成的 `jsonl` 文件，但它们**不是**真实训练数据，不能作为主训练集使用。

请先阅读下面两个文件：

1. `README.md`
2. `docs/15_data_status_and_paths.md`

其中已经明确写明：

- 当前真实可用训练数据 = **0 条**
- 哪些 `jsonl` 文件只是伪数据 / 预热格式文件
- benchmark 的本地路径、原始文件路径、线上来源地址
- 后续真实训练数据应该如何构造

---

# 你的唯一任务

你的任务不是继续讲概念，也不是再造格式样例，而是：

> **从真实 benchmark / 真实 environment 中，做出至少 1 条真实 state-level、action-scored 训练数据。**

然后再扩展到更多条。

---

# 什么算“真实训练数据”

一条真实训练数据必须满足：

1. 来自一个真实 benchmark episode 的某个中间 decision point
2. 不是 task-level 样本，而是 **state-level 样本**
3. 包含当前真实状态：
   - query / latest user request
   - history summary
   - candidate answers
   - metrics（entropy / margin / disagreement / verifier_confidence / steps_remaining）
4. 包含当前真实可选动作集合
5. 包含每个动作的真实打分或可靠近似打分
6. 明确标记：
   - `is_real_collected_data: true`

如果不满足上面这些条件，就不要把它叫做真实训练数据。

---

# 当前项目里的 benchmark / 数据来源

## 本地路径

### tau2-bench
- `/private/tmp/tau2-bench_escalated`

### Gorilla / BFCL
- `/private/tmp/gorilla_escalated`

## 重点原始文件

### tau2
- `/private/tmp/tau2-bench_escalated/data/tau2/domains/airline/tasks.json`
- `/private/tmp/tau2-bench_escalated/data/tau2/domains/airline/split_tasks.json`
- `/private/tmp/tau2-bench_escalated/docs/evaluation.md`
- `/private/tmp/tau2-bench_escalated/src/tau2/data_model/tasks.py`

### BFCL
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/*.json`

但要注意：

- **这些原始文件本身不是训练数据**
- 它们只是任务定义 / benchmark 数据源
- 你真正要拿的是 **运行过程中的中间状态**

---

# 当前仓库里哪些东西不要继续当主数据用

这些文件只是历史遗留的伪数据 / 格式实验：

- `training/llamafactory_data/router_sft_seed.jsonl`
- `training/llamafactory_data/router_sft_tau2_airline_raw.jsonl`
- `training/llamafactory_data/router_sft_tau2_airline_raw_v2.jsonl`
- `training/llamafactory_data/router_sft_bfcl_raw.jsonl`
- `training/llamafactory_data/router_sft_mixed.jsonl`

可以保留作参考，但**不要**把它们继续包装成主训练数据。

---

# 正确的数据构造目标格式

请严格按下面这种结构落数据：

```json
{
  "sample_id": "tau2_airline_task_7_step_2",
  "is_real_collected_data": true,
  "benchmark": "tau2",
  "domain": "airline",
  "task_id": "7",
  "step_id": 2,
  "query": "...",
  "state": {
    "history_summary": "...",
    "candidate_answers": [
      {"source": "draft", "text": "..."},
      {"source": "tool:get_reservation_details", "text": "..."}
    ],
    "metrics": {
      "entropy": 0.58,
      "margin": 0.12,
      "disagreement": 0.21,
      "verifier_confidence": 0.41,
      "steps_remaining": 2
    }
  },
  "available_actions": [
    "stop",
    "think",
    "tool:get_reservation_details",
    "tool:update_reservation_flights",
    "tool:cancel_reservation"
  ],
  "action_scores": {
    "stop": -0.42,
    "think": 0.08,
    "tool:get_reservation_details": 0.64,
    "tool:update_reservation_flights": 0.21,
    "tool:cancel_reservation": -0.17
  },
  "best_action": "tool:get_reservation_details"
}
```

其中：

- `available_actions` 必须是当前真实可选动作
- `action_scores` 必须覆盖全部候选动作
- `best_action = argmax(action_scores)`

---

# 推荐的最短执行路线

请不要一上来就想做完整大系统。先做最小闭环：

## Phase 1：只做 1 条真实数据

目标：
- 从 `tau2` 跑一个 task
- 在某一个 step 截状态
- 导出 1 条真实数据

这是唯一优先级最高的目标。

## Phase 2：验证该样本是否合格

检查：
- 是不是来自真实运行
- 有没有真实 candidate pool
- 有没有真实 action set
- 有没有每动作分数
- `is_real_collected_data` 是否为 `true`

## Phase 3：再扩到 10 条 / 100 条

在第 1 条没做出来之前，不要做大规模生成。

---

# 推荐执行顺序（非常具体）

## Step A：确认环境真的能跑

优先尝试 `tau2`。

你需要先确认：
- tau2 环境是否能 import
- 依赖是否满足
- 是否能跑一个最小 task / simulation

如果跑不起来，明确记录卡在哪：
- Python 版本
- 缺失依赖
- API key
- CLI / import 问题

不要假装“能转 task 文件就算拿到数据”。

## Step B：找到一个真实 decision point

在运行中的 episode 里，找到某一步：
- 记录当前 user message
- 记录当前 tool outputs
- 记录当前 agent candidate answer(s)
- 记录当前 budget
- 记录当前 uncertainty

如果项目里还没有 candidate pool，你需要先明确构造一个最小 candidate pool，例如：
- 当前 draft answer
- think/refine answer
- tool-augmented answer

## Step C：枚举动作

对当前状态，列出当前真正能选的动作：
- stop
- think
- 当前 domain 下可用的工具动作
- 如果有 delegate，就加 delegate

动作集合必须和当前状态对齐。

## Step D：给动作打分

理想做法：

\[
score_t(a) = U(s_t \xrightarrow{a} s_{t+1}) - U(s_t) - cost(a)
\]

如果一开始还做不到完整 utility，也至少要有一个明确、可复现的近似：

- one-step reward improvement
- verifier improvement
- DB-state improvement
- communicate score improvement
- minus action cost

但必须是**由真实状态推进得到的分数**，不是 task 文本脑补。

## Step E：落盘

落成单条 json / jsonl 样本。

---

# 你应该避免的错误

1. **不要**再把 benchmark task description 直接改成 router 标签
2. **不要**把 `evaluation_criteria.actions` 直接等同于第一步最优动作
3. **不要**在没有真实 episode 的情况下声称“已经有训练数据”
4. **不要**先批量生成几千条伪数据
5. **不要**把格式样例当成训练集

---

# 你交付时必须写清楚的内容

当你完成下一步工作时，请在结果里明确写：

## 1. 数据条数
- 真实数据多少条
- 伪数据多少条

## 2. 数据位置
- 真实数据文件路径
- 采集脚本路径

## 3. 数据来源
- benchmark 名称
- task id
- step id

## 4. 打分方式
- action_scores 是怎么来的
- 是完整 utility 还是 one-step proxy

## 5. 剩余问题
- 还有哪些字段没做实
- 哪些地方还只是近似

---

# 最终要求

在你继续任何“训练”“微调”“混合数据集”“LLaMA-Factory 配置”之前，必须先做到：

> **至少 1 条真实 state-level、action-scored 样本**

如果 1 条都没有，就不要往下推进训练。

