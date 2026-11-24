"""End-to-end integration tests: run -> report -> dataset manifest -> regression gate, all offline."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from evalkit.cli import main

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "benchmarks" / "baseline"


def _summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _write_summary(run_dir: Path, data: dict) -> None:
    (run_dir / "summary.json").write_text(json.dumps(data), encoding="utf-8")


def _scaled_copy(src: Path, dst: Path, score: float) -> Path:
    shutil.copytree(src, dst)
    data = copy.deepcopy(_summary(dst))
    for model in data["results"].values():
        for task in model.values():
            task[task["primary_metric"]] = score
    _write_summary(dst, data)
    return dst


def test_committed_baseline_is_well_formed():
    data = _summary(BASELINE)
    assert data["run_name"] == "ci-gate"
    task = data["results"]["ci-dummy"]["faq"]
    assert task["n"] == 2
    assert task["primary_metric"] in task


def test_pipeline_matches_committed_baseline(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "run"
    assert main(["run", "configs/ci_gate.yaml", "--output-dir", str(out)]) == 0
    assert main(["report", str(out), "--format", "all"]) == 0
    for name in ("report.md", "report.html", "report.json"):
        assert (out / name).exists()
    assert main(["compare", str(BASELINE), str(out), "--markdown-out", str(out / "gate.md")]) == 0
    assert "PASS" in (out / "gate.md").read_text(encoding="utf-8")


def test_gate_blocks_a_real_regression(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    good = _scaled_copy(BASELINE, tmp_path / "good", 0.9)
    bad = _scaled_copy(BASELINE, tmp_path / "bad", 0.5)
    assert main(["compare", str(good), str(bad), "--max-drop", "0.02"]) != 0
    assert main(["compare", str(good), str(good), "--max-drop", "0.02"]) == 0


def test_gate_tolerates_drop_within_threshold(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    base = _scaled_copy(BASELINE, tmp_path / "base", 0.80)
    cand = _scaled_copy(BASELINE, tmp_path / "cand", 0.79)
    assert main(["compare", str(base), str(cand), "--max-drop", "0.02"]) == 0
    assert main(["compare", str(base), str(cand), "--max-drop", "0.005"]) != 0


def test_min_score_floor_fails_even_without_drop(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    run = _scaled_copy(BASELINE, tmp_path / "low", 0.3)
    assert main(["compare", str(run), str(run), "--min-score", "0.5"]) != 0


def test_dataset_manifest_roundtrip_detects_tampering(tmp_path):
    ds = tmp_path / "qa.jsonl"
    shutil.copy(ROOT / "tests" / "fixtures" / "qa.jsonl", ds)
    manifest = tmp_path / "manifest.json"
    assert main(["dataset", "manifest", str(ds), "--out", str(manifest), "--version", "1.0.0"]) == 0
    assert main(["dataset", "verify", str(ds), str(manifest)]) == 0
    ds.write_text(ds.read_text(encoding="utf-8") + '{"id": "x", "question": "q", "answer": "a"}\n', encoding="utf-8")
    assert main(["dataset", "verify", str(ds), str(manifest)]) != 0
