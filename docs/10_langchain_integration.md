# LangChain Integration Plan

## Goal
Make the current EGO prototype usable inside a LangChain-style stack without forcing the research code to depend on LangChain internals.

The integration is therefore structured as:
1. a **core EGO controller** that is framework-agnostic;
2. a **LangChain-compatible adapter** that works with `invoke()`-style LLMs, tools, and expert agents.

---

## Files added
- `src/integrations/ego_core.py`
- `src/integrations/langchain_ego_adapter.py`
- `scripts/langchain_ego_example.py`
- `scripts/langchain_ego_multitool_example.py`
- `scripts/langchain_ego_delegate_example.py`

---

## What is implemented

### 1. `EGOStoppingController`
This is the reusable control core.
It takes uncertainty statistics:
- entropy
- margin
- disagreement
- verifier confidence
- action cost

and returns a budget-aware stop/continue decision.

### 2. `CandidatePosteriorEstimator`
This converts multiple candidate answers into:
- entropy
- margin
- disagreement
- verifier confidence

So EGO can now be driven by sampled answer candidates, which is much closer to a real agent stack.

### 3. `LangChainEGOAgent`
This is the LangChain-compatible orchestration wrapper.
It expects:
- an LLM-like object exposing `invoke(prompt)` or being directly callable;
- tools exposing `name` and `invoke(input)` or being callable;
- experts exposing `name` and `invoke(input)` or being callable.

It now supports **action scoring** over:
- `think`
- `tool:<name>` for each registered tool
- `delegate:<name>` for each registered expert
- `stop`

This now matches the paper's target action space much more closely.

---

## How action scoring currently works
At each step:
1. generate / maintain candidate answers;
2. estimate posterior-derived uncertainty statistics;
3. compute a budget-aware stopping decision;
4. if continuing is still valuable, score all available actions;
5. execute the highest-scoring action.

Each action score is currently a lightweight proxy for value of information:

\[
\text{score}(a) = \widehat{\text{gain}}(a) - \text{cost}(a)
\]

where `\widehat{gain}(a)` depends on:
- entropy
- disagreement
- verifier uncertainty
- tool/expert relevance
- prior relevance of the tool/expert
- remaining budget

This is not yet the final theorem-grade routing model, but it is a practical research prototype aligned with the paper direction.

---

## Current status relative to the paper formulation

### Already mapped
- `STOP` -> budget-aware entropy gate + continuation score check
- `THINK` -> internal refinement action with explicit action score
- `TOOL(m)` -> per-tool action scores over registered tools
- `DELEGATE(k)` -> per-expert action scores over registered expert agents

So the current adapter now covers the full intended action set:
\[
\{\texttt{THINK},\; \texttt{TOOL}(m),\; \texttt{DELEGATE}(k),\; \texttt{STOP}\}
\]

That is a substantial milestone because the implementation now lines up directly with the paper's control formulation.

---

## Why this design is better than hard-coding LangChain imports
LangChain APIs move relatively quickly. For research code, a thin compatibility layer is safer than binding the whole project to one exact version.

This design gives you:
- immediate research usability;
- a clean paper story;
- easier future migration to actual LangChain agents, LangGraph, or other orchestration frameworks.

---

## How to use it

### Minimal pattern
```python
from src.integrations.langchain_ego_adapter import LangChainEGOAgent

agent = LangChainEGOAgent(
    llm=my_langchain_llm,
    tools=[my_langchain_tool_1, my_langchain_tool_2],
    experts=[my_expert_agent_1, my_expert_agent_2],
    max_steps=4,
)
result = agent.invoke("your query")
print(result.final_answer)
```

### Multi-tool example
Run:
```bash
python3 scripts/langchain_ego_multitool_example.py
```

### Delegate example
Run:
```bash
python3 scripts/langchain_ego_delegate_example.py
```

This example shows:
- tools and experts being scored in one common action space;
- EGO choosing the best action;
- the final answer after expert-assisted refinement.

---

## Recommended next extension
The next real LangChain step should be one of these two:

1. **LangGraph controller-node version**
   - make EGO the controller node in a graph-based agent workflow.

2. **Learned action scoring version**
   - replace the hand-crafted gain proxy with a learned value-of-information predictor.

For the paper, the second one is especially important if we want a stronger algorithmic contribution beyond threshold stopping.
