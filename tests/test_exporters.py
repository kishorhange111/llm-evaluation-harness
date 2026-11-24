"""Tests for MLflow and OpenTelemetry evaluation metrics exporters."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from evalkit.exporters import MLflowExporter, OpenTelemetryExporter, flatten_summary_metrics
from evalkit.runner import run

FIX = Path(__file__).parent / "fixtures"


class _FakeMLflow:
    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.experiment: str | None = None
        self.run_name: str | None = None
        self.params: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.artifacts: list[str] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    @contextmanager
    def start_run(self, run_name: str):
        self.run_name = run_name
        yield self

    def log_params(self, params: dict[str, str]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(path)


class _FakeSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class _FakeTracer:
    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []

    @contextmanager
    def start_as_current_span(self, name: str):
        s = _FakeSpan(name)
        self.spans.append(s)
        yield s


def test_flatten_summary_metrics_and_mlflow_exporter(tmp_path):
    summary = {
        "run_name": "exp-01",
        "environment": {"python": "3.12"},
        "results": {
            "qwen/0.5b": {
                "gsm8k": {"accuracy": 0.72, "ci95": [0.65, 0.79], "n": 50, "primary_metric": "accuracy"},
            }
        },
    }
    flat = flatten_summary_metrics(summary)
    assert flat["qwen/0.5b.gsm8k.accuracy"] == pytest.approx(0.72)
    assert flat["qwen/0.5b.gsm8k.ci95_low"] == pytest.approx(0.65)

    art = tmp_path / "report.md"
    art.write_text("# report\n")
    fake_ml = _FakeMLflow()
    sink = tmp_path / "mlflow.jsonl"
    exp = MLflowExporter("llm-evals", tracking_uri="http://localhost:5000", mlflow_client=fake_ml, sink_path=sink)
    exp.export_summary(summary, artifact_paths=[art])

    assert fake_ml.experiment == "llm-evals"
    assert fake_ml.tracking_uri == "http://localhost:5000"
    assert fake_ml.metrics["qwen/0.5b.gsm8k.accuracy"] == pytest.approx(0.72)
    assert fake_ml.artifacts == [str(art)]
    assert sink.exists() and json.loads(sink.read_text().strip())["run_name"] == "exp-01"


def test_exporters_attach_to_runner_as_hooks(tmp_path):
    tracer = _FakeTracer()
    otlp_file = tmp_path / "otlp.json"
    otel_hook = OpenTelemetryExporter(service_name="evalkit-ci", tracer=tracer, otlp_json_path=otlp_file)
    ml_hook = MLflowExporter(sink_path=tmp_path / "ml.jsonl")

    config = {
        "run_name": "otel-hook-run",
        "models": [{"backend": "dummy", "name": "m1"}],
        "tasks": [{"name": "jsonl_qa", "data_files": str(FIX / "qa.jsonl"), "label": "qa"}],
    }
    run(config, output_dir=tmp_path / "out", hooks=[otel_hook, ml_hook])

    assert len(otel_hook.spans) == 1
    assert otel_hook.spans[0].attributes["eval.task"] == "qa"
    assert len(tracer.spans) == 1
    assert otlp_file.exists()
    payload = json.loads(otlp_file.read_text())
    assert payload["resource"]["attributes"]["service.name"] == "evalkit-ci"
    assert len(payload["metrics"]) >= 1
    assert len(ml_hook.records) == 1
