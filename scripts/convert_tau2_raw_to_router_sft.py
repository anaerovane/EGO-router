from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.router_sft_common import build_llamafactory_example, clip_score, normalize_split

READ_ACTIONS = {
    "get_user_details",
    "get_reservation_details",
    "search_direct_flight",
    "calculate",
}
WRITE_ACTIONS = {
    "update_reservation_flights",
    "cancel_reservation",
    "book_reservation",
    "update_reservation_baggages",
    "update_reservation_passengers",
}
SPECIAL_ACTIONS = {
    "transfer_to_human_agents",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_query(task: Dict[str, Any]) -> str:
    instr = task.get("user_scenario", {}).get("instructions", {})
    reason = instr.get("reason_for_call") or ""
    known = instr.get("known_info") or ""
    unknown = instr.get("unknown_info") or ""
    extra = instr.get("task_instructions") or ""
    blocks = [b.strip() for b in [reason, known, unknown, extra] if b and b.strip()]
    return "\n\n".join(blocks)


def build_domain_action_space(tasks: List[Dict[str, Any]]) -> List[str]:
    names = set()
    for t in tasks:
        for a in t.get("evaluation_criteria", {}).get("actions", []):
            name = str(a.get("name", "")).strip()
            if name:
                names.add(name)
    ordered = sorted(names)
    return ["stop", "think"] + [f"tool:{x}" for x in ordered]


def action_cost(name: str) -> float:
    if name == "stop":
        return 0.0
    if name == "think":
        return 0.05
    raw = name.replace("tool:", "")
    if raw in READ_ACTIONS:
        return 0.10 if raw != "search_direct_flight" else 0.16
    if raw in WRITE_ACTIONS:
        return 0.14
    if raw in SPECIAL_ACTIONS:
        return 0.12
    return 0.12


def infer_stop_noop(task: Dict[str, Any]) -> bool:
    evalc = task.get("evaluation_criteria", {})
    if evalc.get("actions"):
        return False
    text = " ".join(
        [
            task.get("description", {}).get("purpose", ""),
            *evalc.get("nl_assertions", []),
            *evalc.get("communicate_info", []),
        ]
    ).lower()
    stop_patterns = [
        "should refuse",
        "does not offer",
        "should not allow",
        "should not approve",
        "does not cancel",
        "doesn't book any flight",
        "should not make any changes",
        "does not cancel insurance",
        "not possible",
        "criteria are not met",
    ]
    return any(p in text for p in stop_patterns)


def action_family(raw_name: str) -> str:
    if raw_name in READ_ACTIONS:
        return "read"
    if raw_name in WRITE_ACTIONS:
        return "write"
    if raw_name in SPECIAL_ACTIONS:
        return "special"
    return "other"


def infer_intent(query: str, purpose: str) -> Counter:
    text = f"{query} {purpose}".lower()
    tags = Counter()
    for kw, tag in [
        ("cancel", "cancel"),
        ("refund", "refund"),
        ("book", "book"),
        ("upgrade", "upgrade"),
        ("change", "change"),
        ("baggage", "baggage"),
        ("suitcase", "baggage"),
        ("passenger", "passenger"),
        ("insurance", "insurance"),
        ("compensation", "compensation"),
        ("delay", "delay"),
        ("nonstop", "search"),
        ("direct flight", "search"),
    ]:
        if kw in text:
            tags[tag] += 1
    return tags


def base_state(task: Dict[str, Any], should_stop: bool, num_ref_actions: int, intent: Counter) -> Dict[str, float | int]:
    if should_stop:
        return {
            "entropy": 0.12,
            "margin": 0.61,
            "disagreement": 0.08,
            "verifier_confidence": 0.86,
            "steps_remaining": 1,
            "current_best_score": 2.4,
        }
    ent = 0.62
    if num_ref_actions >= 4:
        ent += 0.08
    if intent["search"]:
        ent += 0.04
    if intent["change"] or intent["upgrade"]:
        ent += 0.03
    return {
        "entropy": min(ent, 0.82),
        "margin": 0.09 if num_ref_actions >= 2 else 0.14,
        "disagreement": 0.36 if num_ref_actions >= 3 else 0.22,
        "verifier_confidence": 0.34 if num_ref_actions >= 2 else 0.42,
        "steps_remaining": 2 if num_ref_actions <= 3 else 3,
        "current_best_score": 0.85,
    }


def score_actions(task: Dict[str, Any], action_space: List[str]) -> Tuple[bool, str, Dict[str, float]]:
    evalc = task.get("evaluation_criteria", {})
    ref_actions = [str(a.get("name", "")).strip() for a in evalc.get("actions", []) if str(a.get("name", "")).strip()]
    query = flatten_query(task)
    purpose = task.get("description", {}).get("purpose", "") or ""
    intent = infer_intent(query, purpose)
    should_stop = infer_stop_noop(task)

    scores: Dict[str, float] = {a: -0.55 for a in action_space}
    scores["stop"] = 0.72 if should_stop else -0.25
    scores["think"] = 0.12 if (not should_stop and not ref_actions) else (-0.10 if should_stop else -0.02)

    if should_stop:
        best_action = "stop"
        # if query is ambiguous, thinking is slightly less bad than taking a wrong tool action
        if intent["change"] or intent["cancel"]:
            scores["think"] = -0.08
        return should_stop, best_action, {k: clip_score(v) for k, v in scores.items()}

    # rank reference actions by first occurrence order
    first_pos: Dict[str, int] = {}
    for idx, raw in enumerate(ref_actions):
        first_pos.setdefault(raw, idx)
    counts = Counter(ref_actions)

    # reference actions get strong positive scores, earlier actions get more weight
    for raw, pos in first_pos.items():
        action = f"tool:{raw}"
        bonus = 0.92 - 0.12 * pos + 0.04 * (counts[raw] - 1)
        scores[action] = max(scores[action], bonus)

    # same-family / intent-matching actions get medium scores
    ref_families = {action_family(x) for x in ref_actions}
    for action in action_space:
        if action in {"stop", "think"}:
            continue
        raw = action.replace("tool:", "")
        fam = action_family(raw)
        if raw in first_pos:
            continue
        # read actions are often useful prerequisites when task has any references
        if fam == "read" and "read" in ref_families:
            scores[action] = max(scores[action], 0.18)
        if fam == "write" and "write" in ref_families:
            scores[action] = max(scores[action], 0.08)
        if fam == "special" and "special" in ref_families:
            scores[action] = max(scores[action], 0.10)

        # intent-specific nudges
        if raw == "search_direct_flight" and (intent["search"] or intent["book"] or intent["change"]):
            scores[action] = max(scores[action], 0.22)
        if raw == "cancel_reservation" and intent["cancel"]:
            scores[action] = max(scores[action], -0.05)
        if raw == "book_reservation" and intent["book"]:
            scores[action] = max(scores[action], -0.02)
        if raw == "update_reservation_flights" and (intent["change"] or intent["upgrade"]):
            scores[action] = max(scores[action], 0.02)
        if raw == "update_reservation_baggages" and intent["baggage"]:
            scores[action] = max(scores[action], 0.05)
        if raw == "update_reservation_passengers" and intent["passenger"]:
            scores[action] = max(scores[action], 0.03)

    # if there are only read actions in reference, stop should remain clearly negative
    if ref_actions and all(action_family(x) == "read" for x in ref_actions):
        scores["stop"] = -0.35
        scores["think"] = max(scores["think"], 0.06)

    # choose best action among scored actions excluding stop when not stop-task
    best_action = max((a for a in action_space if a != "stop"), key=lambda a: scores[a])
    return should_stop, best_action, {k: clip_score(v) for k, v in scores.items()}


def make_example(task: Dict[str, Any], split: str, domain: str, action_space: List[str]) -> Dict[str, Any]:
    query = flatten_query(task)
    should_stop, best_action, action_scores = score_actions(task, action_space)
    eval_criteria = task.get("evaluation_criteria", {})
    intent = infer_intent(query, task.get("description", {}).get("purpose", "") or "")
    state = base_state(task, should_stop, len(eval_criteria.get("actions", [])), intent)

    best_candidate = task.get("description", {}).get("purpose") or query[:300]
    second_candidate = " ".join(eval_criteria.get("nl_assertions", [])[:2]) or "A different resolution strategy may also be plausible."
    recent_candidates = [
        {"source": "purpose", "text": best_candidate},
        {"source": "assertion", "text": second_candidate},
        {"source": "known_info", "text": task.get("user_scenario", {}).get("instructions", {}).get("known_info", "") or query[:200]},
    ]
    available_actions = [{"name": a, "cost": action_cost(a)} for a in action_space]

    return build_llamafactory_example(
        query=query,
        state=state,
        best_candidate=best_candidate,
        second_candidate=second_candidate,
        recent_candidates=recent_candidates,
        available_actions=available_actions,
        should_stop=should_stop,
        best_action=best_action,
        action_scores=action_scores,
        metadata={
            "source_benchmark": f"tau2_raw_{domain}",
            "task_id": task.get("id"),
            "split": split,
            "step_index": 0,
            "teacher_best_action": best_action,
            "domain": domain,
            "num_reference_actions": len(eval_criteria.get("actions", [])),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw tau2 tasks into router SFT jsonl.")
    parser.add_argument("--domain-dir", required=True, help="e.g. /path/to/tau2-bench/data/tau2/domains/airline")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    domain_dir = Path(args.domain_dir)
    if not domain_dir.is_absolute():
        domain_dir = PROJECT_ROOT / domain_dir
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_json(domain_dir / "tasks.json")
    split_map = load_json(domain_dir / "split_tasks.json") if (domain_dir / "split_tasks.json").exists() else {"train": [], "test": []}
    split_lookup = {}
    for split_name, ids in split_map.items():
        canonical = normalize_split(split_name)
        for tid in ids:
            split_lookup[str(tid)] = canonical

    action_space = build_domain_action_space(tasks)
    rows = []
    for task in tasks:
        split = split_lookup.get(str(task.get("id")), "train")
        rows.append(make_example(task, split=split, domain=domain_dir.name, action_space=action_space))

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"output": str(output_path), "num_examples": len(rows), "domain": domain_dir.name, "action_space": action_space}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
