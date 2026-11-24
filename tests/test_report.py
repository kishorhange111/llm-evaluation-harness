"""Tests for rich HTML, Markdown, and JSON report generation."""

from __future__ import annotations

import json

import pytest

from evalkit.cli import main
from evalkit.report import render_html, render_json, render_markdown, write_reports


def _sample_summary_and_comparisons():
    summary = {
        "run_name": "eval-<release>-v2",
        "environment": {"python": "3.12", "evalkit": "0.1.0"},
        "results": {
            "baseline-7b": {
                "gsm8k": {"accuracy": 0.65, "ci95": [0.58, 0.72], "n": 100, "primary_metric": "accuracy"},
                "mmlu": {"accuracy": 0.60, "ci95": [0.54, 0.66], "n": 100, "primary_metric": "accuracy"},
            },
            "candidate-7b": {
                "gsm8k": {"accuracy": 0.78, "ci95": [0.72, 0.84], "n": 100, "primary_metric": "accuracy"},
                "mmlu": {"accuracy": 0.64, "ci95": [0.58, 0.70], "n": 100, "primary_metric": "accuracy"},
            },
        },
    }
    comparisons = {
        "baseline": "baseline-7b",
        "comparisons": {
            "candidate-7b": {
                "gsm8k": {"delta": 0.13, "ci95": [0.05, 0.21], "p_mcnemar": 0.002, "significant": True},
            }
        },
    }
    return summary, comparisons


def test_render_markdown_json_and_html():
    summary, comps = _sample_summary_and_comparisons()

    md = render_markdown(summary, comps)
    assert "# Evaluation report: eval-<release>-v2" in md
    assert "candidate-7b" in md and "Paired comparison" in md

    js = json.loads(render_json(summary, comps))
    assert js["schema_version"] == "1.0"
    assert js["leaderboard"][0]["model"] == "candidate-7b"
    assert js["leaderboard"][0]["macro_average"] == pytest.approx(0.71)

    doc = render_html(summary, comps)
    assert "<!DOCTYPE html>" in doc
    assert "<svg" in doc
    # Verify HTML escaping on run_name to prevent injection.
    assert "eval-&lt;release&gt;-v2" in doc
    assert "Paired Comparison vs. Baseline" in doc


def test_write_reports_and_cli_all_formats(tmp_path):
    summary, comps = _sample_summary_and_comparisons()
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "comparisons.json").write_text(json.dumps(comps), encoding="utf-8")

    written = write_reports(tmp_path)
    assert set(written) == {"markdown", "html", "json"}
    assert all(p.exists() for p in written.values())

    assert main(["report", str(tmp_path), "--format", "all"]) == 0
    assert (tmp_path / "report.html").read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["models"] == ["baseline-7b", "candidate-7b"]
