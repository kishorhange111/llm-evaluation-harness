"""Agent trajectory & tool-call accuracy/efficiency evaluator.

Scores multi-step tool-using agent runs along six dimensions:

* **Tool-name F1**: multiset precision/recall/F1 of invoked tools vs. reference.
* **Argument accuracy**: fraction of expected tool arguments matched accurately.
* **Sequence accuracy**: LCS-based ordering alignment against expected tool sequence.
* **Step efficiency**: penalises unnecessary retries, loops, and excess steps.
* **Error rate**: fraction of tool invocations that returned an error.
* **Goal completion**: optional final-answer check (normalized token F1 / exact).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from evalkit.metrics.text import token_f1
from evalkit.registry import register_metric, register_task
from evalkit.tasks.base import GenerationTask
from evalkit.types import Example, ExampleResult


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation in an agent trajectory."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0


@dataclass(frozen=True)
class Trajectory:
    """Observed sequence of tool calls and optional final answer."""

    steps: list[ToolCall]
    final_answer: str = ""
    total_tokens: int = 0


@dataclass(frozen=True)
class TrajectoryReference:
    """Ground-truth specification for a tool-using task."""

    expected_calls: list[ToolCall]
    expected_answer: str = ""
    max_steps: int | None = None
    ordered: bool = True


@dataclass(frozen=True)
class TrajectoryScore:
    """Detailed breakdown of an agent trajectory evaluation."""

    tool_name_f1: float
    argument_accuracy: float
    sequence_accuracy: float
    step_efficiency: float
    error_rate: float
    goal_completion: float
    composite: float
    details: dict[str, Any] = field(default_factory=dict)


@register_metric("trajectory_tool_f1", category="agent", tags=("agent", "tool-use"))
def tool_name_f1(predicted: list[ToolCall], expected: list[ToolCall]) -> float:
    """Multiset F1 over invoked tool names."""
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    cp = Counter(c.name for c in predicted)
    ce = Counter(c.name for c in expected)
    overlap = sum((cp & ce).values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(predicted)
    rec = overlap / len(expected)
    return 2 * prec * rec / (prec + rec)


def _arg_value_equal(actual: Any, expected: Any, tol: float = 1e-6) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and not isinstance(actual, bool):
        return abs(float(actual) - float(expected)) <= tol
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip().lower() == expected.strip().lower()
    return actual == expected


@register_metric("trajectory_arg_accuracy", category="agent", tags=("agent", "tool-use"))
def argument_accuracy(predicted: list[ToolCall], expected: list[ToolCall]) -> float:
    """Fraction of expected tool-call arguments accurately matched in predicted calls."""
    if not expected:
        return 1.0 if not predicted else 0.0
    used: set[int] = set()
    total_keys = 0
    matched_keys = 0

    for exp in expected:
        exp_args = exp.arguments
        # Find the best unused predicted call with the same tool name.
        best_idx: int | None = None
        best_matches = -1
        for idx, pred in enumerate(predicted):
            if idx in used or pred.name != exp.name:
                continue
            matches = sum(1 for k, v in exp_args.items() if k in pred.arguments and _arg_value_equal(pred.arguments[k], v))
            if matches > best_matches:
                best_matches = matches
                best_idx = idx
        if not exp_args:
            total_keys += 1
            if best_idx is not None:
                used.add(best_idx)
                matched_keys += 1
        else:
            total_keys += len(exp_args)
            if best_idx is not None:
                used.add(best_idx)
                matched_keys += max(0, best_matches)

    return matched_keys / total_keys if total_keys > 0 else 1.0


@register_metric("trajectory_sequence_accuracy", category="agent", tags=("agent", "tool-use"))
def sequence_accuracy(predicted: list[ToolCall], expected: list[ToolCall]) -> float:
    """Normalized Longest Common Subsequence (LCS) alignment over tool names."""
    p_names = [c.name for c in predicted]
    e_names = [c.name for c in expected]
    if not p_names and not e_names:
        return 1.0
    if not p_names or not e_names:
        return 0.0
    m, n = len(p_names), len(e_names)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p_names[i - 1] == e_names[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    return lcs / max(m, n)


@register_metric("trajectory_step_efficiency", category="agent", tags=("agent", "efficiency"))
def step_efficiency(
    predicted: list[ToolCall],
    optimal_steps: int,
    max_steps: int | None = None,
) -> float:
    """Efficiency score in [0, 1] penalizing redundant loops and excess steps."""
    n = len(predicted)
    if n == 0:
        return 1.0 if optimal_steps == 0 else 0.0
    if optimal_steps == 0:
        return 0.0

    # Penalize consecutive identical tool+argument calls (retry loops).
    redundant = 0
    for prev, curr in zip(predicted, predicted[1:]):
        if prev.name == curr.name and prev.arguments == curr.arguments:
            redundant += 1

    base = min(1.0, optimal_steps / n)
    loop_penalty = max(0.0, 1.0 - 0.25 * redundant)
    budget_penalty = 0.5 if (max_steps is not None and n > max_steps) else 1.0
    return base * loop_penalty * budget_penalty


def evaluate_trajectory(trajectory: Trajectory, reference: TrajectoryReference) -> TrajectoryScore:
    """Score an observed agent ``trajectory`` against ``reference``."""
    t_f1 = tool_name_f1(trajectory.steps, reference.expected_calls)
    a_acc = argument_accuracy(trajectory.steps, reference.expected_calls)
    seq_acc = sequence_accuracy(trajectory.steps, reference.expected_calls) if reference.ordered else t_f1
    eff = step_efficiency(trajectory.steps, len(reference.expected_calls), max_steps=reference.max_steps)
    err_rate = sum(1 for s in trajectory.steps if s.error) / len(trajectory.steps) if trajectory.steps else 0.0
    goal = token_f1(trajectory.final_answer, reference.expected_answer) if reference.expected_answer else 1.0

    # Weighted composite emphasizing tool/arg correctness and efficiency, penalizing tool errors.
    if reference.expected_answer:
        comp = 0.25 * t_f1 + 0.25 * a_acc + 0.15 * seq_acc + 0.15 * eff + 0.20 * goal
    else:
        comp = 0.30 * t_f1 + 0.30 * a_acc + 0.20 * seq_acc + 0.20 * eff
    comp *= 1.0 - 0.5 * err_rate

    return TrajectoryScore(
        tool_name_f1=t_f1,
        argument_accuracy=a_acc,
        sequence_accuracy=seq_acc,
        step_efficiency=eff,
        error_rate=err_rate,
        goal_completion=goal,
        composite=max(0.0, min(1.0, comp)),
        details={
            "num_steps": len(trajectory.steps),
            "expected_steps": len(reference.expected_calls),
            "num_errors": sum(1 for s in trajectory.steps if s.error),
            "total_latency_ms": sum(s.latency_ms for s in trajectory.steps),
            "total_tokens": trajectory.total_tokens,
        },
    )


def evaluate_trajectory_batch(pairs: list[tuple[Trajectory, TrajectoryReference]]) -> dict[str, Any]:
    """Evaluate multiple trajectories and return aggregate metrics."""
    scores = [evaluate_trajectory(t, r) for t, r in pairs]
    if not scores:
        nan = float("nan")
        return {
            "tool_name_f1": nan,
            "argument_accuracy": nan,
            "sequence_accuracy": nan,
            "step_efficiency": nan,
            "error_rate": nan,
            "goal_completion": nan,
            "composite": nan,
            "n": 0,
            "samples": [],
        }
    n = len(scores)
    return {
        "tool_name_f1": sum(s.tool_name_f1 for s in scores) / n,
        "argument_accuracy": sum(s.argument_accuracy for s in scores) / n,
        "sequence_accuracy": sum(s.sequence_accuracy for s in scores) / n,
        "step_efficiency": sum(s.step_efficiency for s in scores) / n,
        "error_rate": sum(s.error_rate for s in scores) / n,
        "goal_completion": sum(s.goal_completion for s in scores) / n,
        "composite": sum(s.composite for s in scores) / n,
        "n": n,
        "samples": scores,
    }


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def parse_tool_calls_from_text(text: str) -> tuple[list[ToolCall], str]:
    """Parse JSON tool calls (and optional final_answer) from a model's completion."""
    candidate = text.strip()
    m = _FENCE_RE.search(candidate)
    if m:
        candidate = m.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return [], text.strip()

    final_answer = ""
    raw_calls: list[Any] = []
    if isinstance(parsed, dict):
        if "steps" in parsed or "tool_calls" in parsed:
            raw_calls = list(parsed.get("steps", parsed.get("tool_calls", [])))
            final_answer = str(parsed.get("final_answer", ""))
        elif "name" in parsed:
            raw_calls = [parsed]
    elif isinstance(parsed, list):
        raw_calls = parsed

    calls: list[ToolCall] = []
    for item in raw_calls:
        if isinstance(item, dict) and "name" in item:
            calls.append(
                ToolCall(
                    name=str(item["name"]),
                    arguments=dict(item.get("arguments", item.get("args", {})) or {}),
                    result=item.get("result"),
                    error=item.get("error"),
                    latency_ms=float(item.get("latency_ms", 0.0)),
                )
            )
    return calls, final_answer


@register_task("agent_trajectory", category="agent", tags=("agent", "tool-use", "generation"))
class AgentTrajectoryTask(GenerationTask):
    """Evaluate agent tool-call planning and trajectory efficiency from JSONL fixtures."""

    name = "agent_trajectory"
    primary_metric = "trajectory_composite"
    eval_split = "test"
    fewshot_split = None

    def build_prompt(self, example: Example) -> str:
        tools = example.data.get("available_tools", [])
        tool_list = ", ".join(tools) if tools else "any registered tool"
        return (
            f"Available tools: {tool_list}\n"
            f"Task: {example.data['prompt']}\n"
            'Respond with JSON: {"tool_calls": [{"name": "...", "arguments": {...}}], "final_answer": "..."}'
        )

    def score(self, example: Example, completions: list[str]) -> ExampleResult:
        raw = completions[0] if completions else ""
        calls, final_answer = parse_tool_calls_from_text(raw)
        expected_calls = [ToolCall(name=c["name"], arguments=dict(c.get("arguments", {}))) for c in example.data.get("expected_calls", [])]
        ref = TrajectoryReference(
            expected_calls=expected_calls,
            expected_answer=str(example.data.get("expected_answer", "")),
            max_steps=example.data.get("max_steps"),
            ordered=bool(example.data.get("ordered", True)),
        )
        ts = evaluate_trajectory(Trajectory(steps=calls, final_answer=final_answer), ref)
        return ExampleResult(
            id=example.id,
            score=ts.composite,
            prediction=[{"name": c.name, "arguments": c.arguments} for c in calls],
            target=example.data.get("expected_calls", []),
            extra={
                "tool_name_f1": ts.tool_name_f1,
                "argument_accuracy": ts.argument_accuracy,
                "sequence_accuracy": ts.sequence_accuracy,
                "step_efficiency": ts.step_efficiency,
                "error_rate": ts.error_rate,
                "goal_completion": ts.goal_completion,
            },
        )

    def aggregate(self, results: list[ExampleResult]) -> dict[str, float]:
        if not results:
            nan = float("nan")
            return {
                "trajectory_composite": nan,
                "tool_name_f1": nan,
                "argument_accuracy": nan,
                "sequence_accuracy": nan,
                "step_efficiency": nan,
            }
        n = len(results)
        return {
            "trajectory_composite": sum(r.score for r in results) / n,
            "tool_name_f1": sum(r.extra["tool_name_f1"] for r in results) / n,
            "argument_accuracy": sum(r.extra["argument_accuracy"] for r in results) / n,
            "sequence_accuracy": sum(r.extra["sequence_accuracy"] for r in results) / n,
            "step_efficiency": sum(r.extra["step_efficiency"] for r in results) / n,
        }


__all__ = [
    "AgentTrajectoryTask",
    "ToolCall",
    "Trajectory",
    "TrajectoryReference",
    "TrajectoryScore",
    "argument_accuracy",
    "evaluate_trajectory",
    "evaluate_trajectory_batch",
    "parse_tool_calls_from_text",
    "sequence_accuracy",
    "step_efficiency",
    "tool_name_f1",
]
