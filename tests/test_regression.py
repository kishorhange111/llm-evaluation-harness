"""Tests for eval run comparison and blocking regression detector."""

from __future__ import annotations

import json

import pytest

from evalkit.cli import main
from evalkit.regression import RegressionRule, compare_runs


def _make_summary(run_name: str, model: str, scores: dict[str, float]) -> dict:
    return {
        "run_name": run_name,
        "results": {model: {task: {"accuracy": val, "primary_metric": "accuracy", "n": 100} for task, val in scores.items()}},
    }


def test_compare_runs_passes_within_tolerance_and_fails_on_drop():
    base = _make_summary("base", "m1", {"gsm8k": 0.80, "mmlu": 0.70})
    good = _make_summary("cand-good", "m1", {"gsm8k": 0.79, "mmlu": 0.72})
    bad = _make_summary("cand-bad", "m1", {"gsm8k": 0.72, "mmlu": 0.70})

    rep_ok = compare_runs(base, good, default_max_drop=0.02)
    assert rep_ok.passed and not rep_ok.blocking_failures

    rep_bad = compare_runs(base, bad, default_max_drop=0.02)
    assert not rep_bad.passed
    assert len(rep_bad.blocking_failures) == 1
    assert rep_bad.blocking_failures[0].task == "gsm8k"
    assert rep_bad.blocking_failures[0].delta == pytest.approx(-0.08)


def test_compare_runs_supports_non_blocking_warn_and_min_score_floor():
    base = _make_summary("base", "m1", {"gsm8k": 0.60, "redteam": 0.90})
    cand = _make_summary("cand", "m1", {"gsm8k": 0.50, "redteam": 0.89})
    rules = [
        RegressionRule(task="gsm8k", max_drop=0.02, blocking=False),
        RegressionRule(task="redteam", max_drop=0.05, min_score=0.95, blocking=True),
    ]
    rep = compare_runs(base, cand, rules=rules)
    assert not rep.passed
    assert len(rep.warnings) == 1 and rep.warnings[0].task == "gsm8k"
    assert len(rep.blocking_failures) == 1 and rep.blocking_failures[0].task == "redteam"
    assert "min_score" in rep.blocking_failures[0].reason
    assert "## Eval Regression Gate: FAIL" in rep.to_markdown()


def test_compare_cli_with_run_dirs_and_paired_samples(tmp_path, capsys):
    base_dir = tmp_path / "base"
    cand_dir = tmp_path / "cand"
    (base_dir / "samples").mkdir(parents=True)
    (cand_dir / "samples").mkdir(parents=True)

    (base_dir / "summary.json").write_text(json.dumps(_make_summary("base", "m1", {"gsm8k": 0.90})))
    (cand_dir / "summary.json").write_text(json.dumps(_make_summary("cand", "m1", {"gsm8k": 0.60})))

    base_rows = [{"id": f"q{i}", "score": 1.0 if i < 9 else 0.0} for i in range(10)]
    cand_rows = [{"id": f"q{i}", "score": 1.0 if i < 6 else 0.0} for i in range(10)]
    (base_dir / "samples" / "m1__gsm8k.jsonl").write_text("\n".join(json.dumps(r) for r in base_rows) + "\n")
    (cand_dir / "samples" / "m1__gsm8k.jsonl").write_text("\n".join(json.dumps(r) for r in cand_rows) + "\n")

    md_out = tmp_path / "gate.md"
    code = main(["compare", str(base_dir), str(cand_dir), "--max-drop", "0.05", "--markdown-out", str(md_out), "--json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["deltas"][0]["p_value"] is not None
    assert md_out.exists() and "FAIL" in md_out.read_text()
