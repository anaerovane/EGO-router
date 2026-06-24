# EGO Method Section Outline (Paper-Aligned Draft)

## 1. Method positioning
本文提出 **EGO (Entropy-Gated Orchestration)**，一种面向预算受限 LLM agents 的控制方法。与将 agent 行为写死为固定工作流不同，EGO 在每一轮显式决定：

- 是否继续收集信息；
- 若继续，应该执行内部推理、工具调用还是专家委托；
- 在预算约束下何时停止并输出当前答案。

EGO 的目标不是单纯最大化回答质量，而是最大化 **答案效用减去资源成本** 后的总体 utility。

---

## 2. Problem formulation
给定输入任务 `x`，控制器在离散轮次 `t = 1, 2, ...` 上做顺序决策。动作空间定义为：

\[
\mathcal{A} = \{\texttt{THINK}\} \cup \{\texttt{TOOL}(m): m \in \mathcal{M}\} \cup \{\texttt{DELEGATE}(k): k \in \mathcal{K}\} \cup \{\texttt{STOP}\}.
\]

其中：

- `THINK` 表示执行一次额外内部推理或 refinement；
- `TOOL(m)` 表示调用外部工具 `m`；
- `DELEGATE(k)` 表示查询专家 `k`；
- `STOP` 表示终止并返回当前最优答案。

每执行一次非停止动作，系统都会得到新的 observation，并更新候选答案池与控制状态。

---

## 3. State representation
EGO 不直接对完整历史建模，而是构造一个紧凑状态：

\[
s_t = (H_t, M_t, D_t, V_t, B_t^{\text{tok}}, B_t^{\text{lat}}, B_t^{\text{step}}).
\]

其中：

- `H_t`：当前候选答案 posterior 的 predictive entropy；
- `M_t`：top-1 与 top-2 候选的概率间隔；
- `D_t`：候选间 disagreement；
- `V_t`：verifier 对当前最优候选的置信度；
- `B_t^{tok}, B_t^{lat}, B_t^{step}`：剩余 token、latency、step 预算。

这一状态设计的核心目的，是让 stop 与 routing 都能围绕“当前不确定性 + 剩余资源”展开。

---

## 4. Posterior estimation from candidate answers
在第 `t` 轮，系统维护候选答案集合 `\mathcal{C}_t`。候选来源包括：

- 初始 draft；
- 内部 refinement 得到的新候选；
- 工具增强后的候选；
- 专家委托返回的候选。

每个候选 `y` 会收到一个综合得分 `S_t(y)`，得分可由 verifier、候选支持度、证据新鲜度等组成。基于这些得分，系统构造近似 posterior：

\[
p_t(y) = \frac{\exp(S_t(y)/\tau)}{\sum_{y' \in \mathcal{C}_t} \exp(S_t(y')/\tau)}.
\]

该 posterior 进一步诱导出 entropy、margin 与 disagreement 等控制变量。

---

## 5. Utility objective
EGO 的目标是最大化停止时刻的总体 utility：

\[
U_T = R(x, \hat y_T)
- \lambda_{\text{tok}} C_T^{\text{tok}}
- \lambda_{\text{lat}} C_T^{\text{lat}}
- \lambda_{\text{step}} C_T^{\text{step}}
- \lambda_{\text{risk}} C_T^{\text{risk}}.
\]

其中：

- `R(x, \hat y_T)` 是最终答案质量；
- 各类 `C_T` 对应累计 token、latency、step 与风险成本。

该目标强调：**额外思考、额外工具调用、额外专家委托只有在其收益超过成本时才值得发生。**

---

## 6. Budget-aware stopping rule
EGO 的停止机制由两层组成。

### 6.1 Uncertainty gate
定义一个依赖剩余预算的阈值函数，例如以 step budget 为主：

\[
h(B_t^{\text{step}}) = h_0 + \frac{\alpha_h}{B_t^{\text{step}} + 1}.
\]

当剩余预算变少时，系统会更容易接受较高的不确定性并停止；当剩余预算更多时，系统会更愿意继续直到不确定性进一步下降。

### 6.2 Value-based stopping
即便 uncertainty gate 仍然允许继续，如果所有 continuation actions 的净收益都不为正，系统也应停止：

\[
\max_{a \neq \texttt{STOP}} Q_t(a) \le 0.
\]

因此 EGO 的停止规则可以写为：

\[
\texttt{STOP if gate is closed or } \max_{a \neq \texttt{STOP}} Q_t(a) \le 0.
\]

---

## 7. Unified action scoring
对任意非停止动作 `a`，EGO 用统一的净收益形式打分：

\[
Q_t(a) = \widehat{\mathrm{VOI}}_t(a) - C_t(a).
\]

其中：

- `\widehat{VOI}_t(a)` 是该动作带来的预期信息价值；
- `C_t(a)` 是动作成本。

在 v1 中，`\widehat{VOI}` 用一个轻量代理来近似，它依赖：

- entropy
- disagreement
- verifier uncertainty
- action relevance
- prior relevance
- remaining budget

该统一打分让 think、tool、delegate 首次进入同一个比较空间。

---

## 8. Heuristic EGO-v1
在启发式版本中，EGO 使用手工构造的 gain proxy：

\[
\text{score}(a) = \widehat{\text{gain}}(a) - \text{cost}(a).
\]

其设计原则如下：

- 当 entropy 高时，继续动作更有价值；
- 当 disagreement 高时，工具或专家更有价值；
- 当 verifier confidence 低时，额外信息收集更有价值；
- 不同工具 / 专家可通过 relevance 和 prior relevance 区分适用性。

该版本的优点是：

- 可解释；
- 易于实现；
- 与 theorem-friendly stopping story 自然兼容。

---

## 9. Learned routing extension
为了让 routing 具备在线适应能力，EGO 进一步引入 learned action scorer。对动作 `a`，学习版打分形式为：

\[
\text{score}(a)
=
\hat r_\theta(s_t, a)
+ \mathrm{bonus}_t(s_t, a)
+ \lambda \cdot \widehat{\text{gain}}(a)
- C_t(a).
\]

其中：

- `\hat r_\theta(s_t, a)` 是 learned predictor；
- `bonus_t` 是 exploration bonus；
- `\widehat{gain}(a)` 是 heuristic gain proxy；
- `C_t(a)` 是动作成本。

当前实现采用 LinUCB 风格的 per-action 线性模型：

- 每个动作维护单独参数；
- 特征包含 uncertainty、budget、action cost、relevance 与 action-type indicators；
- 执行动作后根据 reward proxy 做在线更新。

这一设计使方法自然连接到 contextual bandit / regret 分析叙事。

---

## 10. Practical system instantiation
在实现层，EGO 被拆成三部分：

1. `EGOStoppingController`
   - 输入 uncertainty metrics 与 budget
   - 输出是否停止及停止原因

2. `CandidatePosteriorEstimator`
   - 从 candidate pool 估计 entropy / margin / disagreement / verifier confidence

3. `LangChainEGOAgent`
   - 包装 invoke 风格的 LLM、tool、expert
   - 在 think/tool/delegate/stop 之间统一调度
   - 输出最终答案与完整轨迹

因此，EGO 既有形式化定义，也能直接插入现有 agent stack 作为控制层。

---

## 11. Theoretical alignment
该方法部分与论文理论部分一一对应：

- **Theorem A**：说明 budget-aware threshold stopping 的合理性；
- **Theorem B**：说明 posterior / uncertainty 估计误差下的 stopping robustness；
- **Routing extension**：为后续 regret 或 suboptimality 分析预留 learned scorer 结构。

这种设计确保理论不是脱离实现的独立附录，而是直接解释控制器为什么要这样设计。

---

## 12. Experimental alignment
该方法对应两类关键实验：

### 12.1 Stopping experiments
在理论对齐的 synthetic entropy 环境中比较：

- oracle threshold
- budget-aware threshold
- fixed threshold
- fixed depth baselines

目标是验证：budget-aware stopping 比固定深度或过度计算更合理。

### 12.2 Routing experiments
在 mixed-task routing benchmark 中比较：

- heuristic routing
- learned routing

目标是验证：当任务分布混合时，learned routing 能更好地学习 task-action specialization。

---

## 13. Method takeaway
EGO 的核心贡献可以概括为：

1. 将 LLM agent 的 orchestration 形式化为一个 budget-aware sequential decision problem；
2. 用 uncertainty-aware threshold 实现 principled stopping；
3. 用统一 action scoring 实现 think / tool / delegate 的联合 routing；
4. 用 learned scorer 将方法扩展到在线适应与可学习 specialization。

因此，EGO 不是一个特定 benchmark 技巧，而是一个可理论化、可实现、可实验验证的 agent orchestration 方法。
