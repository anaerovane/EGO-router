# Data Status and Paths

## 当前结论

截至 2026-05-28：

- 历史上生成的若干 `router_sft_*.jsonl` 文件，**仍然不是**真实主训练数据，只能算 warm-start / 伪数据 / 格式验证文件。
- 已经产出 state-level、action-scored 的 tau2 样本资产：
  - `outputs/tau2_real_sample.json`：1 条自动采样原型样本
  - `outputs/tau2_manual_sample_001.json` 至 `outputs/tau2_manual_sample_500.json`：共 500 条 manual tau2 decision-point samples
- 因此，当前应区分两件事：
  - **已产出真实来源 / 手工打分样本资产：501 条**
  - **已完成统一验收、统一转换、dataset 注册并可直接作为主训练集的正式数据：仍未冻结**
- 也就是说，项目已经不再是“真实样本 0 条”的状态，但也**还不能**把现有 manual 样本直接等同于最终主训练集。

---

## 一、我生成过的文件（不是可用真数据）

以下文件是本仓库内由我生成或改写的：

### 训练数据 / 伪数据文件

- `training/llamafactory_data/router_sft_seed.jsonl`
- `training/llamafactory_data/router_sft_tau2_airline_raw.jsonl`
- `training/llamafactory_data/router_sft_tau2_airline_raw_v2.jsonl`
- `training/llamafactory_data/router_sft_bfcl_raw.jsonl`
- `training/llamafactory_data/router_sft_mixed.jsonl`
- `training/llamafactory_data/router_sft_tau2_example.jsonl`
- `training/llamafactory_data/router_sft_tau2_airline_raw_v2.jsonl`
- `training/llamafactory_data/router_sft_tau2_example.jsonl`
- `training/llamafactory_data/router_sft_tau2_airline_raw_v2.jsonl`
- `training/llamafactory_data/router_sft_tau2_airline_raw.jsonl`
- `training/llamafactory_data/router_sft_bfcl_raw.jsonl`

> 说明：这些文件**不是**真实 benchmark episode 中间状态采样得到的训练数据，不能当作主训练集使用。

### 转换/生成脚本

- `scripts/export_router_sft_data.py`
- `scripts/convert_tau2_raw_to_router_sft.py`
- `scripts/convert_bfcl_raw_to_router_sft.py`
- `scripts/convert_benchmark_logs_to_router_sft.py`
- `scripts/build_router_mixed_dataset.sh`
- `scripts/train_router_with_llamafactory.sh`

### LLaMA-Factory 配置与模板

- `training/llamafactory_data/dataset_info.json`
- `training/llamafactory_configs/qwen25_router_lora_sft.yaml`
- `training/benchmark_adapter_templates/bfcl_router_adapter.json`
- `training/benchmark_adapter_templates/tau2_router_adapter.json`
- `training/benchmark_adapter_templates/openhands_router_adapter.json`
- `training/benchmark_adapter_templates/gaia_router_adapter.json`
- `training/README_router_finetune.md`

---

## 二、我本地克隆过并查看过的 benchmark 仓库路径

### tau2-bench

本地路径：

- `/private/tmp/tau2-bench_escalated`

### Gorilla / BFCL

本地路径：

- `/private/tmp/gorilla_escalated`

---

## 三、我实际读取过的 tau2 原始文件路径

- `/private/tmp/tau2-bench_escalated/README.md`
- `/private/tmp/tau2-bench_escalated/docs/getting-started.md`
- `/private/tmp/tau2-bench_escalated/docs/evaluation.md`
- `/private/tmp/tau2-bench_escalated/src/tau2/agent/README.md`
- `/private/tmp/tau2-bench_escalated/src/tau2/domains/README.md`
- `/private/tmp/tau2-bench_escalated/src/tau2/data_model/tasks.py`
- `/private/tmp/tau2-bench_escalated/data/tau2/domains/airline/tasks.json`
- `/private/tmp/tau2-bench_escalated/data/tau2/domains/airline/split_tasks.json`
- `/private/tmp/tau2-bench_escalated/data/tau2/domains/airline/db.json`

### tau2 目录结构（已确认存在）

- `/private/tmp/tau2-bench_escalated/data/tau2/domains/airline`
- `/private/tmp/tau2-bench_escalated/data/tau2/domains/retail`
- `/private/tmp/tau2-bench_escalated/data/tau2/domains/telecom`
- `/private/tmp/tau2-bench_escalated/data/tau2/domains/banking_knowledge`

---

## 四、我实际读取过的 BFCL 原始文件路径

- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/README.md`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/README.md`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_format_sensitivity.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_irrelevance.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_irrelevance.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_multiple.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_parallel.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_parallel_multiple.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_relevance.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_simple.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_memory.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multi_turn_base.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multi_turn_long_context.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multi_turn_miss_func.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multi_turn_miss_param.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multiple.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_parallel.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_parallel_multiple.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_java.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_javascript.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json`
- `/private/tmp/gorilla_escalated/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_web_search.json`

---

## 五、线上项目来源地址

### tau2-bench

- `https://github.com/sierra-research/tau2-bench`

### Gorilla / Berkeley Function Calling Leaderboard (BFCL)

- `https://github.com/ShishirPatil/gorilla`
- `https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard`

### OpenHands Benchmarks

- `https://github.com/OpenHands/benchmarks`

### GAIA Agent SDK / related project

- `https://github.com/gaia-agent/gaia-agent`

---

## 六、当前问题总结

1. 我前面生成的 `router_sft_*.jsonl` 不是 state-level 的真实训练数据。
2. 它们不是从真实 benchmark episode 中间状态采出来的。
3. 它们也不是对候选动作逐个 rollout 后得到的真实 action score。
4. 因此，**这些文件不能作为主训练数据使用**。

---

## 七、当前判断与后续正确方向

真正需要的主训练数据应当满足：

- 一条样本 = 一个真实 decision point
- 包含当前真实状态
- 包含当前候选动作集合
- 包含每个动作的真实打分（而不是任务级伪标签）

按这个标准看，当前状态应表述为：

- **已经有一批真实来源 / 手工打分的 state-level 样本资产**
  - `outputs/tau2_real_sample.json`
  - `outputs/tau2_manual_sample_001.json` 至 `outputs/tau2_manual_sample_500.json`
- 但这批样本**还没有**完成：
  - 统一验收
  - 统一转换为正式 router SFT 数据集
  - `dataset_info.json` 注册
  - train / dev / test split 协议冻结

因此，当前最准确的说法不是“真实样本仍为 0 条”，而是：

- **真实样本资产已经做出来了**
- **正式主训练集仍未收口完成**

---

## 八、后续如何构造真实训练数据

这一节描述**正确的数据构造方式**。目标不是再生成 task-level 伪标签，而是生成 **state-level, action-scored** 的真实训练样本。

### 8.1 总原则

一条真实训练样本必须对应：

- **某个 benchmark episode 的某个中间决策点**
- 当前 agent 已有的候选答案 / 历史 / budget / uncertainty 状态
- 当前时刻真实可选的动作集合
- 对每个动作的真实分数或相对优先级

也就是说：

- **一条样本 != 一道题**
- **一条样本 = 一次真实决策**

---

## 九、真实训练数据的构造流程

### Step 1：跑真实 benchmark episode

以 `tau2` 为例：

- 选定一个 task
- 让 agent 在环境中真实运行
- 在每一步（turn / action step）记录 agent 状态

这个状态不是 task 文本，而是例如：

- 当前用户 query / 最新用户消息
- 当前 message history
- 当前工具调用结果
- 当前 candidate answers
- 当前 metrics（entropy / margin / disagreement / verifier confidence）
- 当前剩余 budget

---

### Step 2：在某一步截取一个 decision point

假设现在来到第 `t` 步，记当前状态为 `s_t`。

这时需要记录：

- 当前 query
- 当前 history summary
- 当前 candidate pool
- 当前 uncertainty summary
- 当前可选动作 `A_t`

---

### Step 3：列出当前可选动作集合

候选动作必须是**当前这一步真的能做的动作**。

例如在 EGO router 里，动作集合可能是：

- `stop`
- `think`
- `tool:get_user_details`
- `tool:get_reservation_details`
- `tool:search_direct_flight`
- `tool:cancel_reservation`
- `tool:update_reservation_flights`
- `tool:update_reservation_baggages`
- `tool:update_reservation_passengers`
- `tool:book_reservation`
- `tool:transfer_to_human_agents`
- `tool:calculate`

注意：
- 候选动作不能是固定死的模板，应该由当前环境和当前 domain 决定
- 某些 benchmark / domain 下动作集合会不同

---

### Step 4：对每个动作计算真实分数

对当前每个动作 `a in A_t`，计算一个监督分数：

\[
score_t(a)
\]

最直接的构造方式是 **one-step value target**：

\[
score_t(a) = U(s_t \xrightarrow{a} s_{t+1}) - U(s_t) - cost(a)
\]

其中：

- `U(s_t)`：当前状态下的任务效用估计
- `U(s_t -> a -> s_{t+1})`：执行动作 `a` 后的新状态效用
- `cost(a)`：动作成本（step/token/latency/risk 等）

如果暂时算不出完整 utility，也可以先用简化版本：

\[
score_t(a) = \Delta reward + \lambda_1 \Delta verifier - \lambda_2 cost(a)
\]

---

### Step 5：保存为一条训练样本

一条训练样本就是：

- 当前状态 `s_t`
- 当前动作集合 `A_t`
- 每个动作分数 `score_t(a)`
- `best_action = argmax_a score_t(a)`

---

## 十、真实训练数据的目标格式

下面是**推荐的标准格式**。

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
      {
        "source": "draft",
        "text": "..."
      },
      {
        "source": "tool:get_reservation_details",
        "text": "..."
      }
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

---

## 十一、字段解释

### `sample_id`
样本唯一 ID，用于追踪。

### `is_real_collected_data`
是否来自真实 rollout。真实采样必须为 `true`。

### `benchmark`
样本来自哪个 benchmark，例如 `tau2`、`bfcl`、`gaia`。

### `domain`
具体域，例如 `airline`、`retail`、`telecom`。

### `task_id`
原始任务 ID。

### `step_id`
该任务 episode 内的第几个 decision point。

### `query`
当前任务或用户请求的原始文本。

### `state.history_summary`
对当前对话和工具结果的摘要。

### `state.candidate_answers`
当前时刻 agent 已经形成的候选答案池。

### `state.metrics`
当前时刻的不确定性与预算特征：
- `entropy`
- `margin`
- `disagreement`
- `verifier_confidence`
- `steps_remaining`

### `available_actions`
当前状态下真实可选动作列表。

### `action_scores`
每个动作的监督分数。**这是核心标签。**

### `best_action`
分数最高的动作，用于分类或排序学习。

---

## 十二、给 LLaMA / Qwen 做 SFT 时的格式

如果后续要给小模型做 SFT，可以把上面的 JSON 样本转成：

### 输入文本（user）

- query
- history_summary
- candidate_answers
- metrics
- available_actions

### 输出文本（assistant）

```json
{
  "should_stop": false,
  "best_action": "tool:get_reservation_details",
  "action_scores": {
    "stop": -0.42,
    "think": 0.08,
    "tool:get_reservation_details": 0.64,
    "tool:update_reservation_flights": 0.21,
    "tool:cancel_reservation": -0.17
  }
}
```

注意：
- `should_stop` 可以由 `best_action == "stop"` 推出
- 也可以单独作为一个标签

---

## 十三、最小可交付目标（已跨过）与当前阶段目标

在继续任何大规模训练之前，最小目标原本是先做出：

- **1 条真实样本**

这个门槛现在已经跨过：

- `outputs/tau2_real_sample.json` 提供了第一条自动采样原型样本；
- `outputs/tau2_manual_sample_001.json` 至 `outputs/tau2_manual_sample_500.json` 说明 manual tau2 样本收集已经推进到批量阶段。

因此，当前阶段的最小目标已经不再是“证明能不能做出 1 条”，而是：

- 对 manual tau2 样本做抽样审计；
- 区分可训练 / 需返工 / 仅参考样本；
- 把合格样本统一转换成正式 `router_sft_*.jsonl` 数据集；
- 在 `training/llamafactory_data/dataset_info.json` 中注册；
- 冻结 train / dev / test split 与是否混合 warm-start 数据的协议；
- 然后再训练第一版真正基于 manual tau2 数据的 scorer / router。

---

## 十四、当前状态与后续要求

当前状态：

- 文档格式已定义
- benchmark 路径已明确
- 真实数据构造流程已明确
- `outputs/tau2_real_sample.json` 已打通首条自动采样原型样本
- `outputs/tau2_manual_sample_001.json` 至 `outputs/tau2_manual_sample_500.json` 已形成一批 manual tau2 state-level 样本
- 但这批样本尚未完成统一验收、统一转换、dataset 注册和正式 split 协议
- 当前默认训练入口仍偏向 warm-start 数据，而不是已收口的 manual tau2 正式训练集

当前阶段的后续要求：

1. 不要把历史 `router_sft_*.jsonl` 重新包装成真实主训练集。
2. 不要因为已经有 500 条 manual 样本，就跳过验收、转换、注册、切分这些关键步骤。
3. 在把 manual 样本接入正式训练配置前，不要把它们表述成“已经可直接用于论文主结论的训练集”。
4. 后续任何新增数据，如果不是按这里的格式和流程构造，仍不应被视为真实训练数据。

---

## 十五、当前 TODO（与 learned scorer / finetuning 直接相关）

### 15.1 样本验收
- 抽样检查 `outputs/tau2_manual_sample_*.json`
- 检查是否存在 future leakage / hidden-task leakage / reference-action leakage
- 检查 `available_actions` 与 decision point 是否对齐
- 检查 `best_action = argmax(action_scores)` 是否一致

### 15.2 正式训练集落盘
- 将合格的 manual tau2 样本转换为正式 router SFT 数据文件
- 建议产出独立文件，例如：`training/llamafactory_data/router_sft_tau2_manual.jsonl`
- 将该数据集注册到 `training/llamafactory_data/dataset_info.json`

### 15.3 训练协议冻结
- 明确 manual tau2 数据的 `train / dev / test` 切分
- 决定是否与 `router_sft_seed.jsonl` / `router_sft_bfcl_raw.jsonl` 混合
- 如果混合，明确哪些只算 warm-start，哪些才算主要实验数据

### 15.4 第一版 finetuned scorer 验证
- 基于 manual tau2 正式数据集训练第一版 scorer / router
- 在 held-out benchmark 上比较 heuristic routing 与 finetuned scorer
- 记录 first-action accuracy、reward、reward_by_type、reward_by_difficulty

### 15.5 文档同步
- `training/README_router_finetune.md` 需要补 manual tau2 数据路线
- 训练配置需要补一份明确面向 manual tau2 数据的 config
- spec / README / data-status 三处表述需要保持一致
