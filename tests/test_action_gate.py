"""Validate the GitHub Action definition, CI gate workflow, and the offline gate flow it drives."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from evalkit.cli import main

ROOT = Path(__file__).resolve().parent.parent


def _load(rel: str) -> dict:
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def test_action_yml_declares_inputs_outputs_and_composite_steps():
    action = _load("action.yml")
    assert action["runs"]["using"] == "composite"
    inputs = action["inputs"]
    assert inputs["config"]["required"] is True
    for key in ("baseline", "max-drop", "min-score", "output-dir", "python-version", "limit"):
        assert key in inputs
    assert set(action["outputs"]) == {"passed", "results-dir"}
    step_text = json.dumps(action["runs"]["steps"])
    for cmd in ("evalkit", "report", "compare", "--format all"):
        assert cmd in step_text
    # Every run step in a composite action must declare a shell.
    assert all("shell" in s for s in action["runs"]["steps"] if "run" in s)


def test_eval_gate_workflow_uses_local_action_and_ci_config():
    wf = _load(".github/workflows/eval-gate.yml")
    # PyYAML parses the bare key `on` as boolean True.
    triggers = wf.get("on", wf.get(True))
    assert "pull_request" in triggers
    steps = wf["jobs"]["eval-gate"]["steps"]
    action_step = next(s for s in steps if s.get("uses") == "./")
    assert action_step["with"]["config"] == "configs/ci_gate.yaml"
    assert wf["permissions"] == {"contents": "read"}


def test_ci_gate_config_runs_offline_and_gates_against_itself(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    base, cand = tmp_path / "base", tmp_path / "cand"
    assert main(["run", "configs/ci_gate.yaml", "--output-dir", str(base)]) == 0
    assert main(["run", "configs/ci_gate.yaml", "--output-dir", str(cand)]) == 0
    assert main(["report", str(cand), "--format", "all"]) == 0
    assert (cand / "report.html").exists()
    # Deterministic dummy model: identical runs must pass the gate.
    assert main(["compare", str(base), str(cand), "--max-drop", "0.02", "--markdown-out", str(cand / "gate.md")]) == 0
    assert "PASS" in (cand / "gate.md").read_text(encoding="utf-8")
