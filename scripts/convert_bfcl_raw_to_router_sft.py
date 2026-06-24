from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.router_sft_common import build_llamafactory_example, clip_score


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return
    if raw.startswith("{") and "\n" in raw and path.stem.lower().endswith("format_sensitivity"):
        payload = json.loads(raw)
        for category, ids in payload.items():
            if isinstance(ids, list):
                for i, item_id in enumerate(ids):
                    yield {"id": f"format_sensitivity_{category}_{i}", "question": category, "function": [], "_format_sensitivity_id": item_id}
        return
    for line in raw.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def flatten_question(question: Any) -> str:
    if isinstance(question, str):
        return question
    if isinstance(question, list):
        texts: List[str] = []
        for turn in question:
            if isinstance(turn, list):
                for msg in turn:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        texts.append(str(msg.get("content", "")))
            elif isinstance(turn, dict) and turn.get("role") == "user":
                texts.append(str(turn.get("content", "")))
        return "\n\n".join(t for t in texts if t)
    return ""


def category_from_path(path: Path) -> str:
    name = path.stem.lower()
    if "web_search" in name:
        return "web_search"
    if "memory" in name:
        return "memory"
    if "irrelevance" in name:
        return "irrelevance"
    if "format_sensitivity" in name:
        return "format_sensitivity"
    if "miss_param" in name or "miss_func" in name:
        return "missing_info"
    if "multi_turn" in name:
        return "multi_turn"
    if "parallel" in name:
        return "parallel"
    if "multiple" in name:
        return "multiple"
    return "function_call"


def build_actions(category: str) -> List[Dict[str, Any]]:
    actions = [{"name": "stop", "cost": 0.0}, {"name": "think", "cost": 0.05}]
    if category == "web_search":
        actions.append({"name": "tool:web_search", "cost": 0.16})
    elif category == "memory":
        actions.append({"name": "tool:memory_write", "cost": 0.12})
        actions.append({"name": "tool:memory_read", "cost": 0.10})
    else:
        actions.append({"name": "tool:function_call", "cost": 0.12})
    return actions


def choose_label(category: str) -> tuple[bool, str]:
    if category == "irrelevance":
        return True, "stop"
    if category == "web_search":
        return False, "tool:web_search"
    if category == "memory":
        return False, "tool:memory_write"
    if category == "missing_info":
        return False, "think"
    return False, "tool:function_call"


def make_scores(actions: List[Dict[str, Any]], best_action: str, should_stop: bool) -> Dict[str, float]:
    scores = {a["name"]: -0.2 for a in actions}
    if should_stop:
        scores["stop"] = 0.7
        for k in list(scores):
            if k != "stop":
                scores[k] = -0.35
        return {k: clip_score(v) for k, v in scores.items()}
    scores[best_action] = 0.78
    scores["stop"] = -0.15
    if "think" in scores and best_action != "think":
        scores["think"] = 0.05 if best_action != "tool:function_call" else -0.05
    return {k: clip_score(v) for k, v in scores.items()}


def make_example(item: Dict[str, Any], category: str, source_file: str) -> Dict[str, Any]:
    query = flatten_question(item.get("question"))
    functions = item.get("function") or []
    func_names = [str(f.get("name", "")) for f in functions[:3] if isinstance(f, dict)]
    should_stop, best_action = choose_label(category)
    actions = build_actions(category)
    action_scores = make_scores(actions, best_action, should_stop)

    best_candidate = f"User asks: {query}"
    second_candidate = "Available function docs: " + (", ".join(func_names) if func_names else "none")
    recent_candidates = [
        {"source": "user", "text": query},
        {"source": "functions", "text": second_candidate},
        {"source": "category", "text": f"BFCL category: {category}"},
    ]
    state = {
        "entropy": 0.60 if not should_stop else 0.12,
        "margin": 0.09 if not should_stop else 0.66,
        "disagreement": 0.30 if len(functions) > 1 else 0.12,
        "verifier_confidence": 0.35 if not should_stop else 0.88,
        "steps_remaining": 2 if not should_stop else 1,
        "current_best_score": 0.8 if not should_stop else 2.5,
    }
    return build_llamafactory_example(
        query=query,
        state=state,
        best_candidate=best_candidate,
        second_candidate=second_candidate,
        recent_candidates=recent_candidates,
        available_actions=actions,
        should_stop=should_stop,
        best_action=best_action,
        action_scores=action_scores,
        metadata={
            "source_benchmark": "bfcl_raw",
            "task_id": item.get("id"),
            "split": "train",
            "step_index": 0,
            "teacher_best_action": best_action,
            "bfcl_category": category,
            "source_file": source_file,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw BFCL category files into router SFT jsonl.")
    parser.add_argument("--input-dir", required=True, help="Directory containing BFCL_v4_*.json files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-per-file", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    files = sorted(input_dir.glob('BFCL_v4_*.json'))
    for path in files:
        category = category_from_path(path)
        count = 0
        for item in iter_jsonl(path):
            rows.append(make_example(item, category=category, source_file=path.name))
            count += 1
            if args.limit_per_file and count >= args.limit_per_file:
                break

    with output_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(json.dumps({"output": str(output_path), "num_examples": len(rows), "num_files": len(files)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
