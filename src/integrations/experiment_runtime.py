from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import ast
import json
import math
import os
import re
import urllib.error
import urllib.request


def load_dotenv(dotenv_path: Path) -> Dict[str, str]:
    """Load a minimal .env file without external dependencies.

    Existing environment variables take precedence.
    """
    loaded: Dict[str, str] = {}
    if not dotenv_path.exists():
        return loaded

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


@dataclass
class QueryTask:
    query: str
    task_type: str
    best_action: Optional[str] = None
    id: Optional[str] = None
    reference_answer: Optional[str] = None
    reference_points: List[str] = field(default_factory=list)
    rubric: List[str] = field(default_factory=list)
    evaluation_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OpenAICompatibleChatModel:
    """Minimal OpenAI-compatible chat client using only the stdlib."""

    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: str = "https://api.openai.com/v1",
        temperature: float = 0.2,
        timeout_seconds: float = 60.0,
    ):
        if not api_key:
            raise ValueError("API key is required for OpenAICompatibleChatModel.")
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def invoke(self, prompt: str) -> str:
        return self.complete(prompt)

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.api_base}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Chat completion failed: HTTP {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Chat completion failed: {exc}") from exc

        parsed = json.loads(body)
        choices = parsed.get("choices", [])
        if not choices:
            raise RuntimeError(f"No choices in completion response: {parsed}")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join(part for part in text_parts if part)
        return str(content)


class PromptedExpert:
    """Expert wrapper that uses the same underlying model with a role prompt."""

    def __init__(self, model: OpenAICompatibleChatModel, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt

    def invoke(self, query: str) -> str:
        return self.model.complete(query, system_prompt=self.system_prompt)


class SafeCalculatorTool:
    name = "calculator"
    description = "Evaluate arithmetic expressions safely using the local Python runtime."

    _allowed_binary_ops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a ** b,
        ast.Mod: lambda a, b: a % b,
        ast.FloorDiv: lambda a, b: a // b,
    }
    _allowed_unary_ops = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }
    _allowed_names = {
        "pi": math.pi,
        "e": math.e,
    }
    _allowed_functions = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "abs": abs,
        "round": round,
    }

    def invoke(self, query: str) -> str:
        expression = self._extract_expression(query)
        if not expression:
            return f"calculator could not find a clean arithmetic expression in: {query}"
        try:
            value = self._eval_expr(expression)
            return f"calculator result for: {expression} = {value}"
        except Exception as exc:  # pragma: no cover - defensive path
            return f"calculator error for expression '{expression}': {exc}"

    def _extract_expression(self, query: str) -> str:
        fenced = re.findall(r"`([^`]+)`", query)
        if fenced:
            return fenced[0]
        allowed = set("0123456789+-*/().,% eEpiqrtsincaoblgx")
        cleaned = "".join(ch for ch in query if ch in allowed)
        return cleaned.strip()

    def _eval_expr(self, expression: str) -> float:
        node = ast.parse(expression, mode="eval")
        return self._eval_node(node.body)

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in self._allowed_names:
            return float(self._allowed_names[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in self._allowed_binary_ops:
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return float(self._allowed_binary_ops[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._allowed_unary_ops:
            value = self._eval_node(node.operand)
            return float(self._allowed_unary_ops[type(node.op)](value))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in self._allowed_functions:
                raise ValueError(f"function '{func_name}' is not allowed")
            args = [self._eval_node(arg) for arg in node.args]
            return float(self._allowed_functions[func_name](*args))
        raise ValueError(f"unsupported expression node: {ast.dump(node)}")


class LocalCorpusSearchTool:
    name = "search"
    description = "Search a local project corpus and return the most relevant snippets."

    def __init__(self, corpus_dir: Path, include_extensions: Optional[Sequence[str]] = None, max_results: int = 3):
        self.corpus_dir = corpus_dir
        self.include_extensions = tuple(include_extensions or (".md", ".py", ".txt"))
        self.max_results = max_results

    def invoke(self, query: str) -> str:
        if not self.corpus_dir.exists():
            return f"search corpus directory does not exist: {self.corpus_dir}"

        query_terms = self._terms(query)
        scored: List[tuple[int, Path, str]] = []
        for path in self._iter_files(self.corpus_dir):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            score = self._score_text(text, query_terms)
            if score <= 0:
                continue
            snippet = self._make_snippet(text, query_terms)
            scored.append((score, path, snippet))

        if not scored:
            return f"No local corpus results found for query: {query}"

        scored.sort(key=lambda item: item[0], reverse=True)
        lines = [f"Local corpus search results for: {query}"]
        for score, path, snippet in scored[: self.max_results]:
            rel_path = path.relative_to(self.corpus_dir)
            lines.append(f"- {rel_path} (score={score})")
            lines.append(snippet)
        return "\n".join(lines)

    def _iter_files(self, root: Path) -> Iterable[Path]:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.include_extensions:
                yield path

    def _terms(self, query: str) -> List[str]:
        return [term for term in re.findall(r"[a-zA-Z_]{3,}", query.lower()) if term not in {"what", "which", "that", "with", "from"}]

    def _score_text(self, text: str, query_terms: Sequence[str]) -> int:
        lowered = text.lower()
        return sum(lowered.count(term) for term in query_terms)

    def _make_snippet(self, text: str, query_terms: Sequence[str], window: int = 260) -> str:
        lowered = text.lower()
        start = 0
        for term in query_terms:
            idx = lowered.find(term)
            if idx >= 0:
                start = max(idx - window // 3, 0)
                break
        snippet = text[start : start + window].replace("\n", " ").strip()
        return snippet


class LLMJudgeVerifier:
    """LLM-as-judge verifier returning a scalar score in [-1, 3]."""

    def __init__(
        self,
        model: OpenAICompatibleChatModel,
        query_to_task_type: Optional[Dict[str, str]] = None,
    ):
        self.model = model
        self.query_to_task_type = query_to_task_type or {}

    def __call__(self, query: str, answer: str) -> float:
        task_type = self.query_to_task_type.get(query, "general")
        prompt = self._build_prompt(query, answer, task_type)
        raw = self.model.complete(prompt, system_prompt="You are a strict research evaluator. Return only a numeric score.")
        return self._parse_score(raw)

    def _build_prompt(self, query: str, answer: str, task_type: str) -> str:
        rubric = {
            "math": "Reward rigor, derivation quality, correctness, and explicit proof reasoning.",
            "calc": "Reward numerical correctness, explicit arithmetic verification, and precise final values.",
            "search": "Reward evidence-backed grounding, citation-like support from retrieved material, and factual alignment.",
            "code": "Reward implementation specificity, debugging usefulness, and code-level correctness.",
            "think": "Reward strong internal reasoning, clarity, and concise synthesis.",
            "general": "Reward answer usefulness, correctness, and support quality.",
        }.get(task_type, "Reward answer usefulness, correctness, and support quality.")
        return (
            "Score the candidate answer on a scale from -1.0 to 3.0.\n"
            "Return only one number, no explanation.\n\n"
            f"Task type: {task_type}\n"
            f"Rubric: {rubric}\n\n"
            f"Query: {query}\n\n"
            f"Candidate answer: {answer}\n"
        )

    def _parse_score(self, raw: str) -> float:
        match = re.search(r"-?\d+(?:\.\d+)?", raw)
        if not match:
            raise ValueError(f"Could not parse judge score from: {raw!r}")
        score = float(match.group(0))
        return max(-1.0, min(3.0, score))


class MockKeywordLLM:
    """Keyword-conditioned mock LLM used for offline benchmark mode."""

    def invoke(self, prompt: str) -> str:
        lower = prompt.lower()
        if "delegated expert: math" in lower:
            return "Math-expert-supported answer: rigorous derivation with validated final result."
        if "delegated expert: code" in lower:
            return "Code-expert-supported answer: implementation details and debugging steps are correct."
        if "tool used: calculator" in lower:
            return "Calculator-supported answer: arithmetic verification confirms the numerical result."
        if "tool used: search" in lower:
            return "Search-supported answer: retrieved evidence confirms the factual claim."
        if "refining an answer under budget-aware orchestration" in lower:
            return "Refined reasoning answer: internally improved but not externally validated."
        return "Initial draft answer: uncertain and only partially supported."


class MockSearchTool:
    name = "search"
    description = "Retrieve evidence for factual and latest-information queries."

    def invoke(self, query: str) -> str:
        return f"search evidence for: {query}"


class MockMathExpert:
    name = "math"
    description = "Expert for proofs, derivations, and theorem-style reasoning."

    def invoke(self, query: str) -> str:
        return f"math expert response for: {query}"


class MockCodeExpert:
    name = "code"
    description = "Expert for code, scripts, implementation, and debugging."

    def invoke(self, query: str) -> str:
        return f"code expert response for: {query}"


def mock_keyword_verifier(query: str, answer: str) -> float:
    q = query.lower()
    a = answer.lower()
    score = -0.4

    if "deriv" in q or "proof" in q or "theorem" in q:
        if "math-expert-supported" in a:
            score = 3.0
        elif "refined reasoning" in a:
            score = 1.0
        elif "calculator-supported" in a:
            score = 0.4
    elif "calculate" in q or "numerical" in q or "arithmetic" in q:
        if "calculator-supported" in a:
            score = 3.0
        elif "math-expert-supported" in a:
            score = 2.2
        elif "refined reasoning" in a:
            score = 0.8
    elif "evidence" in q or "factual" in q or "claim" in q or "file" in q:
        if "search-supported" in a:
            score = 3.0
        elif "refined reasoning" in a:
            score = 0.7
        elif "math-expert-supported" in a:
            score = 0.5
    elif "implement" in q or "debug" in q or "script" in q or "module" in q:
        if "code-expert-supported" in a:
            score = 3.0
        elif "refined reasoning" in a:
            score = 0.9
        elif "search-supported" in a:
            score = 0.3
    elif "internally" in q or "concise answer" in q or "summarize" in q:
        if "refined reasoning" in a:
            score = 2.6
        elif "math-expert-supported" in a or "code-expert-supported" in a:
            score = 1.2
        elif "search-supported" in a or "calculator-supported" in a:
            score = 0.9

    return score
