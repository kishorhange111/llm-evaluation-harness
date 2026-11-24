"""Tests for agent trajectory & tool-call accuracy/efficiency evaluation."""

from __future__ import annotations

import json

import pytest

from evalkit.metrics.trajectory import (
    AgentTrajectoryTask,
    ToolCall,
    Trajectory,
    TrajectoryReference,
    argument_accuracy,
    evaluate_trajectory,
    evaluate_trajectory_batch,
    parse_tool_calls_from_text,
    sequence_accuracy,
    step_efficiency,
    tool_name_f1,
)
from evalkit.runner import run
from evalkit.tasks import build_task


def test_perfect_trajectory_scores_one():
    calls = [
        ToolCall("search_docs", {"query": "SLA policy", "top_k": 3}),
        ToolCall("create_ticket", {"priority": "P1", "team": "infra"}),
    ]
    ref = TrajectoryReference(expected_calls=calls, expected_answer="Created P1 ticket for infra.")
    score = evaluate_trajectory(Trajectory(steps=calls, final_answer="Created P1 ticket for infra."), ref)
    assert score.tool_name_f1 == pytest.approx(1.0)
    assert score.argument_accuracy == pytest.approx(1.0)
    assert score.sequence_accuracy == pytest.approx(1.0)
    assert score.step_efficiency == pytest.approx(1.0)
    assert score.error_rate == pytest.approx(0.0)
    assert score.composite == pytest.approx(1.0)


def test_out_of_order_and_redundant_loops_penalized():
    expected = [ToolCall("lookup_user", {"id": 42}), ToolCall("fetch_orders", {"user_id": 42})]
    reversed_with_loop = [
        ToolCall("fetch_orders", {"user_id": 42}),
        ToolCall("lookup_user", {"id": 42}),
        ToolCall("lookup_user", {"id": 42}),
    ]
    assert tool_name_f1(reversed_with_loop, expected) == pytest.approx(0.8)
    assert sequence_accuracy(reversed_with_loop, expected) < 0.5
    assert step_efficiency(reversed_with_loop, optimal_steps=2, max_steps=2) < 0.3


def test_argument_accuracy_and_error_penalty():
    expected = [ToolCall("sql_query", {"table": "orders", "limit": 10})]
    wrong_args = [ToolCall("sql_query", {"table": "users", "limit": 10}, error="permission denied")]
    assert argument_accuracy(wrong_args, expected) == pytest.approx(0.5)

    score = evaluate_trajectory(Trajectory(steps=wrong_args), TrajectoryReference(expected_calls=expected))
    assert score.error_rate == pytest.approx(1.0)
    assert score.composite < 0.5


def test_parse_tool_calls_from_fenced_and_raw_json():
    fenced = '```json\n{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Bengaluru"}}], "final_answer": "26C"}\n```'
    calls, ans = parse_tool_calls_from_text(fenced)
    assert len(calls) == 1 and calls[0].name == "get_weather" and calls[0].arguments == {"city": "Bengaluru"}
    assert ans == "26C"

    bad_calls, fallback = parse_tool_calls_from_text("not json at all")
    assert bad_calls == [] and fallback == "not json at all"


def test_batch_and_agent_trajectory_task_end_to_end(tmp_path):
    pair = (
        Trajectory(steps=[ToolCall("ping", {"host": "db-1"})], final_answer="ok"),
        TrajectoryReference(expected_calls=[ToolCall("ping", {"host": "db-1"})], expected_answer="ok"),
    )
    batch = evaluate_trajectory_batch([pair])
    assert batch["n"] == 1 and batch["composite"] == pytest.approx(1.0)

    data = tmp_path / "traj.jsonl"
    row = {
        "prompt": "Look up customer 99 and issue refund.",
        "available_tools": ["get_customer", "issue_refund"],
        "expected_calls": [
            {"name": "get_customer", "arguments": {"customer_id": 99}},
            {"name": "issue_refund", "arguments": {"customer_id": 99, "amount": 50}},
        ],
        "expected_answer": "Refunded 50 to customer 99.",
        "max_steps": 3,
    }
    data.write_text(json.dumps(row) + "\n")

    def oracle(_prompt: str) -> str:
        return json.dumps(
            {
                "tool_calls": row["expected_calls"],
                "final_answer": row["expected_answer"],
            }
        )

    task = build_task("agent_trajectory", data_files=str(data))
    assert isinstance(task, AgentTrajectoryTask)
    summary = run(
        {
            "run_name": "traj-smoke",
            "models": [{"backend": "dummy", "name": "agent-oracle", "responder": oracle}],
            "tasks": [{"name": "agent_trajectory", "data_files": str(data)}],
        },
        output_dir=tmp_path / "out",
    )
    assert summary["results"]["agent-oracle"]["agent_trajectory"]["trajectory_composite"] == pytest.approx(1.0)
