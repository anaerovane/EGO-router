from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.router_sft_common import action_costs, build_llamafactory_example, clip_score, normalize_split


DEFAULT_ACTION_SPACE = [
    "stop",
    "think",
    "tool:search",
    "tool:calculator",
    "delegate:math",
    "delegate:code",
]


class AdapterConfig:
    def __init__(self, raw: Dict[str, Any]):
        self.name = raw.get("name", "custom")
        self.field_aliases: Dict[str, List[str]] = raw.get("field_aliases", {})
        self.action_aliases: Dict[str, List[str]] = raw.get("action_aliases", {})
        self.default_costs = action_costs()
        self.default_costs.update(raw.get("action_costs", {}))
        self.default_action_space = raw.get("default_action_space", DEFAULT_ACTION_SPACE)
        self.default_should_stop = bool(raw.get("default_should_stop", False))
        self.default_best_action = raw.get("default_best_action", "think")
        self.assume_stop_if_missing = bool(raw.get("assume_stop_if_missing", False))

    def get_field(self, row: Dict[str, Any], key: str, default: Any = None) -> Any:
        for alias in self.field_aliases.get(key, [key]):
            value = self._get_nested(row, alias)
            if value is not None:
                return value
        return default

    def _get_nested(self, row: Dict[str, Any], alias: str) -> Any:
        current: Any = row
        for part in alias.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def normalize_action(self, raw_action: str | None) -> str:
        if raw_action is None:
            return self.default_best_action
        action = str(raw_action).strip().lower()
        for canonical, aliases in self.action_aliases.items():
            if action == canonical.lower():
                return canonical
            for alias in aliases:
                if action == alias.lower():
                    return canonical
        if action in {a.lower() for a in self.default_action_space}:
            for a in self.default_action_space:
                if a.lower() == action:
                    return a
        return self.default_best_action


def load_config(path: Path) -> AdapterConfig:
    return AdapterConfig(json.loads(path.read_text(encoding="utf-8")))


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            for item in payload["records"]:
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(payload.get("data"), list):
            for item in payload["data"]:
                if isinstance(item, dict):
                    yield item
            return
    raise ValueError(f"Unsupported input format for {path}")


def normalize_available_actions(raw_actions: Any, cfg: AdapterConfig) -> List[Dict[str, Any]]:
    if not raw_actions:
        raw_actions = [{"name": a} for a in cfg.default_action_space]
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for item in raw_actions:
        if isinstance(item, str):
            name = cfg.normalize_action(item)
            cost = cfg.default_costs.get(name, 0.0)
        elif isinstance(item, dict):
            name = cfg.normalize_action(item.get("name") or item.get("action") or item.get("tool_name"))
            cost = float(item.get("cost", cfg.default_costs.get(name, 0.0)))
        else:
            continue
        if name in seen:
            continue
        seen.add(name)
        normalized.append({"name": name, "cost": cost})
    if "stop" not in seen:
        normalized.insert(0, {"name": "stop", "cost": 0.0})
    return normalized


def normalize_recent_candidates(raw: Any, best: str, second: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if isinstance(raw, list):
        for obj in raw[:3]:
            if isinstance(obj, dict):
                items.append({
                    "source": str(obj.get("source", obj.get("role", "unknown"))),
                    "text": str(obj.get("text", obj.get("content", obj.get("answer", "")))),
                })
            elif isinstance(obj, str):
                items.append({"source": "trace", "text": obj})
    if not items:
        items = [
            {"source": "best", "text": best},
            {"source": "second", "text": second},
        ]
    while len(items) < 3:
        items.append({"source": "padding", "text": second if len(items) == 1 else best})
    return items[:3]


def derive_state(row: Dict[str, Any], cfg: AdapterConfig) -> Dict[str, Any]:
    state = cfg.get_field(row, "state", {}) or {}
    metrics = cfg.get_field(row, "metrics", {}) or {}
    return {
        "entropy": cfg.get_field(row, "entropy", state.get("entropy", metrics.get("entropy", 0.5))),
        "margin": cfg.get_field(row, "margin", state.get("margin", metrics.get("margin", 0.1))),
        "disagreement": cfg.get_field(row, "disagreement", state.get("disagreement", metrics.get("disagreement", 0.2))),
        "verifier_confidence": cfg.get_field(row, "verifier_confidence", state.get("verifier_confidence", metrics.get("verifier_confidence", 0.5))),
        "steps_remaining": cfg.get_field(row, "steps_remaining", state.get("steps_remaining", 1)),
        "current_best_score": cfg.get_field(row, "current_best_score", state.get("current_best_score", 0.0)),
    }


def derive_labels(row: Dict[str, Any], cfg: AdapterConfig, available_actions: List[Dict[str, Any]]) -> tuple[bool, str, Dict[str, float]]:
    raw_scores = cfg.get_field(row, "action_scores", None)
    best_action = cfg.normalize_action(cfg.get_field(row, "best_action", None))
    should_stop = cfg.get_field(row, "should_stop", None)

    score_map: Dict[str, float] = {a["name"]: -0.25 for a in available_actions}
    score_map.setdefault("stop", 0.0)

    if isinstance(raw_scores, dict):
        for k, v in raw_scores.items():
            try:
                score_map[cfg.normalize_action(k)] = clip_score(v)
            except Exception:
                continue
        if should_stop is None:
            should_stop = score_map.get("stop", 0.0) >= max(score_map.get(a["name"], -1.0) for a in available_actions if a["name"] != "stop")
        if best_action == cfg.default_best_action and raw_scores:
            best_action = max(score_map.items(), key=lambda kv: kv[1])[0]
    else:
        if should_stop is None:
            should_stop = cfg.assume_stop_if_missing or False
        if bool(should_stop):
            best_action = "stop"
            score_map["stop"] = 0.55
        else:
            score_map[best_action] = 0.65
            if best_action != "stop":
                score_map["stop"] = -0.10

    should_stop = bool(should_stop)
    if should_stop:
        best_action = "stop"
        score_map["stop"] = max(score_map.get("stop", 0.0), 0.25)
    return should_stop, best_action, {k: clip_score(v) for k, v in score_map.items()}


def convert_row(row: Dict[str, Any], cfg: AdapterConfig) -> Optional[Dict[str, Any]]:
    query = cfg.get_field(row, "query", None)
    if not query:
        return None
    best_candidate = str(cfg.get_field(row, "best_candidate", cfg.get_field(row, "current_answer", "No current answer available.")))
    second_candidate = str(cfg.get_field(row, "second_candidate", "No alternative candidate available."))
    recent_candidates = normalize_recent_candidates(cfg.get_field(row, "recent_candidates", None), best_candidate, second_candidate)
    available_actions = normalize_available_actions(cfg.get_field(row, "available_actions", None), cfg)
    state = derive_state(row, cfg)
    should_stop, best_action, action_scores = derive_labels(row, cfg, available_actions)

    metadata = {
        "source_benchmark": cfg.name,
        "task_id": cfg.get_field(row, "task_id", cfg.get_field(row, "id", None)),
        "split": normalize_split(cfg.get_field(row, "split", "train")),
        "step_index": cfg.get_field(row, "step_index", 0),
        "teacher_best_action": best_action,
        "original_best_action": cfg.get_field(row, "best_action", None),
    }
    return build_llamafactory_example(
        query=str(query),
        state=state,
        best_candidate=best_candidate,
        second_candidate=second_candidate,
        recent_candidates=recent_candidates,
        available_actions=available_actions,
        should_stop=should_stop,
        best_action=best_action,
        action_scores=action_scores,
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert benchmark trajectory logs to router SFT jsonl.")
    parser.add_argument("--input", required=True, help="Input benchmark export (.jsonl or .json).")
    parser.add_argument("--adapter", required=True, help="Adapter template JSON path.")
    parser.add_argument("--output", required=True, help="Output router SFT jsonl path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    adapter_path = Path(args.adapter)
    if not adapter_path.is_absolute():
        adapter_path = PROJECT_ROOT / adapter_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(adapter_path)
    num_in = 0
    num_out = 0
    split_counts: Dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as fout:
        for row in iter_rows(input_path):
            num_in += 1
            example = convert_row(row, cfg)
            if example is None:
                continue
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            num_out += 1
            split = example["metadata"]["split"]
            split_counts[split] = split_counts.get(split, 0) + 1
    print(json.dumps({
        "adapter": cfg.name,
        "input_rows": num_in,
        "output_rows": num_out,
        "output": str(output_path),
        "split_counts": split_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
