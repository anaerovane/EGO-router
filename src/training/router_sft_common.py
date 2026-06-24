from __future__ import annotations

import json
from typing import Dict, Iterable, List


def clip_score(x: float) -> float:
    return max(-1.0, min(1.0, round(float(x), 4)))


def action_costs() -> Dict[str, float]:
    return {
        "stop": 0.0,
        "think": 0.05,
        "tool:search": 0.16,
        "tool:calculator": 0.12,
        "delegate:math": 0.10,
        "delegate:code": 0.11,
    }


def normalize_split(raw: str | None) -> str:
    raw = (raw or "train").strip().lower()
    if raw in {"validation", "valid", "dev"}:
        return "dev"
    if raw not in {"train", "dev", "test"}:
        return "train"
    return raw


def format_recent_candidates(recent_candidates: Iterable[Dict[str, str]]) -> str:
    lines: List[str] = []
    for idx, item in enumerate(recent_candidates, start=1):
        source = str(item.get("source", "unknown"))
        text = str(item.get("text", "")).strip()
        lines.append(f"{idx}. source={source} | text={text}")
    return "\n".join(lines)


def serialize_router_prompt(
    *,
    query: str,
    state: Dict[str, float | int],
    best_candidate: str,
    second_candidate: str,
    recent_candidates: List[Dict[str, str]],
    available_actions: List[Dict[str, str | float]],
) -> str:
    action_lines = [
        f"{idx}. {a['name']} | cost={float(a.get('cost', 0.0)):.2f}"
        for idx, a in enumerate(available_actions, start=1)
    ]
    return (
        f"[QUERY]\n{query}\n\n"
        f"[STATE]\n"
        f"entropy: {float(state.get('entropy', 0.5)):.4f}\n"
        f"margin: {float(state.get('margin', 0.1)):.4f}\n"
        f"disagreement: {float(state.get('disagreement', 0.2)):.4f}\n"
        f"verifier_confidence: {float(state.get('verifier_confidence', 0.5)):.4f}\n"
        f"steps_remaining: {int(state.get('steps_remaining', 1))}\n"
        f"current_best_score: {float(state.get('current_best_score', 0.0)):.4f}\n\n"
        f"[BEST_CANDIDATE]\n{best_candidate}\n\n"
        f"[SECOND_CANDIDATE]\n{second_candidate}\n\n"
        f"[RECENT_CANDIDATES]\n{format_recent_candidates(recent_candidates)}\n\n"
        f"[AVAILABLE_ACTIONS]\n" + "\n".join(action_lines) + "\n\n"
        "[INSTRUCTION]\n"
        "You are a budget-aware agent controller. Return JSON only with fields: "
        "should_stop, best_action, action_scores. Higher score means more necessary."
    )


def build_llamafactory_example(
    *,
    query: str,
    state: Dict[str, float | int],
    best_candidate: str,
    second_candidate: str,
    recent_candidates: List[Dict[str, str]],
    available_actions: List[Dict[str, str | float]],
    should_stop: bool,
    best_action: str,
    action_scores: Dict[str, float],
    metadata: Dict,
) -> Dict:
    instruction = serialize_router_prompt(
        query=query,
        state=state,
        best_candidate=best_candidate,
        second_candidate=second_candidate,
        recent_candidates=recent_candidates,
        available_actions=available_actions,
    )
    output = json.dumps(
        {
            "should_stop": bool(should_stop),
            "best_action": best_action,
            "action_scores": {k: clip_score(v) for k, v in action_scores.items()},
        },
        ensure_ascii=False,
    )
    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "system": "You are EGO-Router, a compact controller that decides whether to stop or which action is most necessary.",
        "metadata": metadata,
    }
