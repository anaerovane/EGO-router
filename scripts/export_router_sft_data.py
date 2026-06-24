from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_tasks(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def clip_score(x: float) -> float:
    return max(-1.0, min(1.0, round(x, 4)))


def normalize_split(raw: str | None) -> str:
    raw = (raw or "train").strip().lower()
    if raw not in {"train", "dev", "test"}:
        return "train"
    return raw


def action_costs() -> Dict[str, float]:
    return {
        "stop": 0.0,
        "think": 0.05,
        "tool:search": 0.16,
        "tool:calculator": 0.12,
        "delegate:math": 0.10,
        "delegate:code": 0.11,
    }


def default_action_scores(best_action: str, best_margin: float = 0.72) -> Dict[str, float]:
    base = {
        "stop": -0.35,
        "think": -0.05,
        "tool:search": -0.15,
        "tool:calculator": -0.25,
        "delegate:math": -0.20,
        "delegate:code": -0.20,
    }
    if best_action not in base:
        raise ValueError(f"Unknown best action: {best_action}")
    base[best_action] = best_margin
    if best_action != "stop":
        base["stop"] = -0.10
    return {k: clip_score(v) for k, v in base.items()}


def stop_action_scores() -> Dict[str, float]:
    scores = {
        "stop": 0.55,
        "think": -0.30,
        "tool:search": -0.45,
        "tool:calculator": -0.40,
        "delegate:math": -0.35,
        "delegate:code": -0.35,
    }
    return {k: clip_score(v) for k, v in scores.items()}


def query_hints(task: Dict) -> Tuple[str, str, str]:
    task_type = task.get("task_type", "general")
    query = task["query"]
    if task_type == "search":
        best = f"Current answer is weak and likely ungrounded: {query}"
        second = "Another candidate guesses a file name but provides no local evidence."
        recent = "Local evidence has not been retrieved yet."
        return best, second, recent
    if task_type == "calc":
        best = f"Current answer attempts the calculation but may contain arithmetic error: {query}"
        second = "A second candidate gives a different numeric result."
        recent = "No trusted calculator result has been incorporated yet."
        return best, second, recent
    if task_type == "code":
        best = f"Current answer mentions coding ideas but lacks module-level specifics: {query}"
        second = "A second candidate is generic and not grounded in implementation details."
        recent = "No code-specialized expert feedback has been incorporated yet."
        return best, second, recent
    if task_type == "think":
        best = f"Current answer is partially formed and may benefit from one more concise reasoning pass: {query}"
        second = "Another candidate is shorter but less coherent."
        recent = "No external evidence seems necessary right now."
        return best, second, recent
    # default -> math / conceptual
    best = f"Current answer is conceptually relevant but underspecified: {query}"
    second = "A second candidate is generic and misses the EGO-specific framing."
    recent = "No domain-specialized explanation has been incorporated yet."
    return best, second, recent


def serialize_user_prompt(
    query: str,
    best_candidate: str,
    second_candidate: str,
    recent_note: str,
    state: Dict[str, float | int],
    costs: Dict[str, float],
) -> str:
    recent_candidates = [
        f"1. source=draft | text={best_candidate}",
        f"2. source=alternative | text={second_candidate}",
        f"3. source=system_note | text={recent_note}",
    ]
    actions = [
        f"1. stop | cost={costs['stop']:.2f}",
        f"2. think | cost={costs['think']:.2f}",
        f"3. tool:search | cost={costs['tool:search']:.2f}",
        f"4. tool:calculator | cost={costs['tool:calculator']:.2f}",
        f"5. delegate:math | cost={costs['delegate:math']:.2f}",
        f"6. delegate:code | cost={costs['delegate:code']:.2f}",
    ]
    return (
        f"[QUERY]\n{query}\n\n"
        f"[STATE]\n"
        f"entropy: {state['entropy']:.4f}\n"
        f"margin: {state['margin']:.4f}\n"
        f"disagreement: {state['disagreement']:.4f}\n"
        f"verifier_confidence: {state['verifier_confidence']:.4f}\n"
        f"steps_remaining: {int(state['steps_remaining'])}\n"
        f"current_best_score: {state['current_best_score']:.4f}\n\n"
        f"[BEST_CANDIDATE]\n{best_candidate}\n\n"
        f"[SECOND_CANDIDATE]\n{second_candidate}\n\n"
        f"[RECENT_CANDIDATES]\n" + "\n".join(recent_candidates) + "\n\n"
        f"[AVAILABLE_ACTIONS]\n" + "\n".join(actions) + "\n\n"
        "[INSTRUCTION]\n"
        "You are a budget-aware agent controller. Return JSON only with fields: "
        "should_stop, best_action, action_scores. Higher score means more necessary."
    )



def make_example(task: Dict, variant: str, step_index: int) -> Dict:
    costs = action_costs()
    best_action = task.get("best_action") or "think"
    query = task["query"]
    best_candidate, second_candidate, recent_note = query_hints(task)

    if variant == "continue":
        state = {
            "entropy": 0.62,
            "margin": 0.08,
            "disagreement": 0.41,
            "verifier_confidence": 0.33,
            "steps_remaining": 2,
            "current_best_score": 0.9,
        }
        scores = default_action_scores(best_action, best_margin=0.72)
        should_stop = False
        selected = best_action
    elif variant == "late_continue":
        state = {
            "entropy": 0.44,
            "margin": 0.14,
            "disagreement": 0.28,
            "verifier_confidence": 0.46,
            "steps_remaining": 1,
            "current_best_score": 1.1,
        }
        scores = default_action_scores(best_action, best_margin=0.46)
        scores["stop"] = clip_score(0.05 if best_action != "stop" else 0.55)
        should_stop = False if best_action != "stop" else True
        selected = best_action if best_action != "stop" else "stop"
    elif variant == "stop":
        state = {
            "entropy": 0.11,
            "margin": 0.63,
            "disagreement": 0.06,
            "verifier_confidence": 0.88,
            "steps_remaining": 1,
            "current_best_score": 2.6,
        }
        scores = stop_action_scores()
        should_stop = True
        selected = "stop"
        recent_note = "Evidence across recent candidates is already consistent, so further actions look wasteful."
    else:
        raise ValueError(f"Unknown variant: {variant}")

    user_content = serialize_user_prompt(
        query=query,
        best_candidate=best_candidate,
        second_candidate=second_candidate,
        recent_note=recent_note,
        state=state,
        costs=costs,
    )
    assistant_content = json.dumps(
        {
            "should_stop": should_stop,
            "best_action": selected,
            "action_scores": scores,
        },
        ensure_ascii=False,
    )
    return {
        "instruction": user_content,
        "input": "",
        "output": assistant_content,
        "system": "You are EGO-Router, a compact controller that decides whether to stop or which action is most necessary.",
        "metadata": {
            "task_id": task.get("id"),
            "task_type": task.get("task_type"),
            "split": normalize_split(task.get("metadata", {}).get("split")),
            "variant": variant,
            "step_index": step_index,
            "teacher_best_action": task.get("best_action"),
        },
    }



def generate_examples(tasks: Iterable[Dict]) -> List[Dict]:
    examples: List[Dict] = []
    for task in tasks:
        examples.append(make_example(task, "continue", 1))
        examples.append(make_example(task, "late_continue", 2))
        examples.append(make_example(task, "stop", 3))
    return examples



def main() -> None:
    parser = argparse.ArgumentParser(description="Export seed router SFT data for LLaMA-Factory.")
    parser.add_argument(
        "--tasks",
        default="data/realistic_mixed_task_benchmark_v3.json",
        help="Input task JSON file relative to project root or absolute.",
    )
    parser.add_argument(
        "--output",
        default="training/llamafactory_data/router_sft_seed.jsonl",
        help="Output jsonl path relative to project root or absolute.",
    )
    args = parser.parse_args()

    task_path = Path(args.tasks)
    if not task_path.is_absolute():
        task_path = PROJECT_ROOT / task_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(task_path)
    examples = generate_examples(tasks)
    with output_path.open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_counts: Dict[str, int] = {}
    for row in examples:
        split = row["metadata"]["split"]
        split_counts[split] = split_counts.get(split, 0) + 1

    print(json.dumps({
        "output": str(output_path),
        "num_examples": len(examples),
        "split_counts": split_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
