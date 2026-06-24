# Mixed-Task Routing Benchmark

## Goal
Create a controlled benchmark where different action types are truly useful on different task families, so we can compare:
- heuristic routing;
- learned / contextual-bandit routing;
- eventually oracle or tuned baselines.

This benchmark is meant to support the paper's routing claims, not just the stopping claims.

---

## File added
- `scripts/mixed_task_routing_benchmark.py`

---

## Benchmark design
The benchmark currently contains five task types:
- `math` -> best action should be `delegate:math`
- `calc` -> best action should be `tool:calculator`
- `search` -> best action should be `tool:search`
- `code` -> best action should be `delegate:code`
- `think` -> best action should be `think`

Each task type is repeated multiple times so that the learned scorer has a chance to adapt online.

---

## How evaluation works
For each task:
1. run the agent on the query;
2. record the final verifier reward;
3. record the first chosen action;
4. compare the first chosen action against the designated best action.

Reported metrics:
- average final reward;
- first-action accuracy;
- per-task-type reward;
- per-task-type first-action accuracy.

These metrics separate two questions:
- Did the routing choose the right first move?
- Even if not, did the final answer still improve?

---

## Initial results snapshot
In the current first run, learned routing outperformed heuristic routing on average reward and slightly on first-action accuracy.

Representative output:
- heuristic avg reward: about `1.98`
- learned avg reward: about `2.29`
- reward gain: about `+0.31`
- first-action accuracy gain: about `+0.05`

The main early gain comes from better handling of search-style tasks.

---

## Interpretation
This is already useful for the paper because it shows:
1. routing quality matters in a mixed task distribution;
2. learned routing can outperform heuristic routing even in a simple online setup;
3. the project now has a path from threshold theory to routing experiments.

However, this benchmark is still an early-stage scaffold, not yet a polished final result.

---

## Current limitations
1. The learned scorer is updated online with a simple proxy reward.
2. The benchmark uses mock LLM/tool/expert components rather than realistic task traces.
3. Some task types are still easier for delegate actions than they should be, which can blur action separation.
4. First-action accuracy is a strict metric; final reward can improve even when the first action is imperfect.

---

## Best next improvements
1. Add train/eval splits instead of only sequential online adaptation.
2. Improve the task generator so action advantages are more sharply separated.
3. Add oracle-routing or tuned-routing baselines.
4. Plot performance over time to show online adaptation curves.
5. Measure cost-aware utility, not just verifier reward.

These would turn the benchmark into a stronger experimental section for the paper.
