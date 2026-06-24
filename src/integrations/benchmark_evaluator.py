from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import re

from src.integrations.experiment_runtime import OpenAICompatibleChatModel, QueryTask


@dataclass
class EvaluationResult:
    score: float
    matched_reference: bool = False
    matched_points: int = 0
    total_points: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


class BenchmarkEvaluator:
    """Evaluate benchmark items using task metadata.

    Supports:
    - numeric_exact
    - substring_or_evidence_match
    - reference_points_judge
    - rubric_judge

    Open-ended evaluation can optionally use an LLM judge. When no judge model is
    available, the evaluator falls back to deterministic string / key-point heuristics.
    """

    def __init__(
        self,
        judge_model: Optional[OpenAICompatibleChatModel] = None,
        use_llm_for_open_ended: bool = False,
    ):
        self.judge_model = judge_model
        self.use_llm_for_open_ended = use_llm_for_open_ended and judge_model is not None

    def evaluate(self, task: QueryTask, answer: str) -> EvaluationResult:
        eval_type = task.evaluation_type or "rubric_judge"
        if eval_type == "numeric_exact":
            return self._evaluate_numeric(task, answer)
        if eval_type == "substring_or_evidence_match":
            return self._evaluate_substring_or_evidence(task, answer)
        if eval_type == "reference_points_judge":
            return self._evaluate_open_ended(task, answer, use_reference_points=True)
        if eval_type == "rubric_judge":
            return self._evaluate_open_ended(task, answer, use_reference_points=False)
        raise ValueError(f"Unsupported evaluation_type: {eval_type}")

    def _evaluate_numeric(self, task: QueryTask, answer: str) -> EvaluationResult:
        target_raw = task.reference_answer
        if target_raw is None:
            raise ValueError(f"numeric_exact task {task.id} is missing reference_answer")
        tolerance = float(task.metadata.get("numeric_tolerance", 1e-9))
        target = self._extract_first_number(str(target_raw))
        pred = self._extract_first_number(answer)
        if pred is None or target is None:
            return EvaluationResult(score=-1.0, details={"reason": "missing_numeric_parse"})

        abs_err = abs(pred - target)
        rel_err = abs_err / max(abs(target), 1.0)
        if abs_err <= tolerance or rel_err <= tolerance:
            return EvaluationResult(
                score=3.0,
                matched_reference=True,
                details={"pred": pred, "target": target, "abs_err": abs_err, "rel_err": rel_err},
            )

        if rel_err <= 0.01:
            score = 2.0
        elif rel_err <= 0.05:
            score = 1.0
        elif rel_err <= 0.20:
            score = 0.0
        else:
            score = -1.0
        return EvaluationResult(
            score=score,
            matched_reference=False,
            details={"pred": pred, "target": target, "abs_err": abs_err, "rel_err": rel_err},
        )

    def _evaluate_substring_or_evidence(self, task: QueryTask, answer: str) -> EvaluationResult:
        norm_answer = self._normalize_text(answer)
        ref = self._normalize_text(task.reference_answer or "")
        points = task.reference_points or []

        ref_hit = bool(ref and ref in norm_answer)
        matched_points = sum(1 for point in points if self._normalize_text(point) in norm_answer)
        total_points = len(points)
        point_frac = matched_points / total_points if total_points > 0 else 0.0

        if ref_hit or (total_points > 0 and matched_points == total_points):
            score = 3.0
        elif ref_hit or point_frac >= 0.5:
            score = 2.0
        elif matched_points > 0:
            score = 1.0
        else:
            score = -1.0

        return EvaluationResult(
            score=score,
            matched_reference=ref_hit,
            matched_points=matched_points,
            total_points=total_points,
            details={"reference_hit": ref_hit, "point_fraction": point_frac},
        )

    def _evaluate_open_ended(self, task: QueryTask, answer: str, use_reference_points: bool) -> EvaluationResult:
        points = list(task.reference_points or [])
        rubric = list(task.rubric or [])
        if self.use_llm_for_open_ended:
            return self._evaluate_with_llm(task, answer, points, rubric)
        return self._evaluate_with_heuristics(task, answer, points, rubric, use_reference_points)

    def _evaluate_with_heuristics(
        self,
        task: QueryTask,
        answer: str,
        points: Sequence[str],
        rubric: Sequence[str],
        use_reference_points: bool,
    ) -> EvaluationResult:
        norm_answer = self._normalize_text(answer)
        matched_points = sum(1 for point in points if self._soft_contains(norm_answer, point))
        matched_rubric = sum(1 for rule in rubric if self._soft_contains(norm_answer, rule))
        total_points = len(points) + len(rubric)
        matched_total = matched_points + matched_rubric
        frac = matched_total / total_points if total_points > 0 else 0.0

        if total_points == 0:
            score = -0.2 if norm_answer.strip() else -1.0
        elif frac >= 0.80:
            score = 3.0
        elif frac >= 0.55:
            score = 2.0
        elif frac >= 0.30:
            score = 1.0
        elif matched_total > 0:
            score = 0.0
        else:
            score = -1.0

        details = {
            "matched_rubric": matched_rubric,
            "total_rubric": len(rubric),
            "match_fraction": frac,
        }
        return EvaluationResult(
            score=score,
            matched_reference=False,
            matched_points=matched_points,
            total_points=len(points),
            details=details,
        )

    def _evaluate_with_llm(
        self,
        task: QueryTask,
        answer: str,
        points: Sequence[str],
        rubric: Sequence[str],
    ) -> EvaluationResult:
        prompt = self._build_llm_prompt(task, answer, points, rubric)
        raw = self.judge_model.complete(
            prompt,
            system_prompt="You are a strict benchmark evaluator. Return valid JSON only.",
        )
        score = self._parse_llm_score(raw)
        matched_points = self._extract_json_int(raw, "matched_points")
        total_points = self._extract_json_int(raw, "total_points")
        return EvaluationResult(
            score=score,
            matched_reference=False,
            matched_points=matched_points if matched_points is not None else 0,
            total_points=total_points if total_points is not None else len(points),
            details={"raw_judge_output": raw},
        )

    def _build_llm_prompt(
        self,
        task: QueryTask,
        answer: str,
        points: Sequence[str],
        rubric: Sequence[str],
    ) -> str:
        return (
            "Evaluate the candidate answer for a benchmark item.\n"
            "Return JSON with keys: score, matched_points, total_points, short_reason.\n"
            "score must be one of [-1.0, 0.0, 1.0, 2.0, 3.0].\n\n"
            f"Task type: {task.task_type}\n"
            f"Query: {task.query}\n"
            f"Expected best action: {task.best_action}\n"
            f"Reference answer: {task.reference_answer}\n"
            f"Reference points: {list(points)}\n"
            f"Rubric: {list(rubric)}\n"
            f"Candidate answer: {answer}\n"
        )

    def _parse_llm_score(self, raw: str) -> float:
        match = re.search(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)', raw)
        if match:
            score = float(match.group(1))
            return max(-1.0, min(3.0, score))
        number = self._extract_first_number(raw)
        if number is None:
            return -1.0
        return max(-1.0, min(3.0, number))

    def _extract_json_int(self, raw: str, key: str) -> Optional[int]:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*(\d+)', raw)
        return int(match.group(1)) if match else None

    def _soft_contains(self, normalized_answer: str, target: str) -> bool:
        norm_target = self._normalize_text(target)
        if not norm_target:
            return False
        if norm_target in normalized_answer:
            return True
        target_tokens = set(norm_target.split())
        answer_tokens = set(normalized_answer.split())
        if not target_tokens:
            return False
        overlap = len(target_tokens & answer_tokens) / len(target_tokens)
        return overlap >= 0.7

    def _normalize_text(self, text: str) -> str:
        lowered = (text or "").lower()
        lowered = re.sub(r"[^a-z0-9_./\-\s]+", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    def _extract_first_number(self, text: str) -> Optional[float]:
        match = re.search(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text.lower())
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None
