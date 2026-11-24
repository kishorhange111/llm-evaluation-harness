"""MLflow and OpenTelemetry exporters for evaluation metrics and traces.

Both exporters implement :class:`~evalkit.registry.BaseRunnerHook` so they can
attach directly to :func:`evalkit.runner.run`, and also support post-hoc export
of a finished ``summary.json`` / ``comparisons.json`` via ``export_summary``.

Optional third-party SDKs (``mlflow``, ``opentelemetry``) are imported lazily;
when absent (or in offline CI), both exporters can emit structured JSON/OTLP
records to memory or disk without external servers or API keys.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evalkit.registry import BaseRunnerHook


def _sanitize_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-/]+", "_", text).strip("_")


def flatten_summary_metrics(summary: dict[str, Any], comparisons: dict[str, Any] | None = None) -> dict[str, float]:
    """Flatten nested ``summary["results"]`` (and optional comparisons) into scalar metrics."""
    flat: dict[str, float] = {}
    for model, per_task in summary.get("results", {}).items():
        m_key = _sanitize_key(model)
        for task, metrics in per_task.items():
            t_key = _sanitize_key(task)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    flat[f"{m_key}.{t_key}.{_sanitize_key(k)}"] = float(v)
                elif k == "ci95" and isinstance(v, (list, tuple)) and len(v) == 2:
                    flat[f"{m_key}.{t_key}.ci95_low"] = float(v[0])
                    flat[f"{m_key}.{t_key}.ci95_high"] = float(v[1])
    if comparisons and comparisons.get("comparisons"):
        for model, per_task in comparisons["comparisons"].items():
            m_key = _sanitize_key(model)
            for task, c in per_task.items():
                t_key = _sanitize_key(task)
                flat[f"delta.{m_key}.{t_key}.delta"] = float(c["delta"])
                if "p_mcnemar" in c:
                    flat[f"delta.{m_key}.{t_key}.p_mcnemar"] = float(c["p_mcnemar"])
    return flat


class MLflowExporter(BaseRunnerHook):
    """Export eval params, metrics, and artifacts to MLflow (or a local JSONL sink)."""

    def __init__(
        self,
        experiment_name: str = "evalkit",
        tracking_uri: str | None = None,
        *,
        mlflow_client: Any = None,
        sink_path: str | Path | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._client = mlflow_client
        self.sink_path = Path(sink_path) if sink_path else None
        self.records: list[dict[str, Any]] = []

    def _resolve_mlflow(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import mlflow  # type: ignore[import-not-found]

            return mlflow
        except ImportError:
            return None

    def export_summary(
        self,
        summary: dict[str, Any],
        comparisons: dict[str, Any] | None = None,
        *,
        artifact_paths: list[str | Path] | None = None,
    ) -> dict[str, Any]:
        """Log run metadata, flattened scalar metrics, and optional artifacts."""
        run_name = str(summary.get("run_name", "evalkit-run"))
        params = {f"env.{k}": str(v) for k, v in summary.get("environment", {}).items()}
        params["num_models"] = str(len(summary.get("results", {})))
        metrics = flatten_summary_metrics(summary, comparisons)
        record = {
            "experiment": self.experiment_name,
            "run_name": run_name,
            "params": params,
            "metrics": metrics,
            "artifacts": [str(p) for p in (artifact_paths or [])],
        }
        self.records.append(record)

        if self.sink_path is not None:
            self.sink_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.sink_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        mlflow_mod = self._resolve_mlflow()
        if mlflow_mod is not None:
            if self.tracking_uri:
                mlflow_mod.set_tracking_uri(self.tracking_uri)
            mlflow_mod.set_experiment(self.experiment_name)
            with mlflow_mod.start_run(run_name=run_name):
                mlflow_mod.log_params(params)
                mlflow_mod.log_metrics(metrics)
                for p in artifact_paths or []:
                    if Path(p).exists():
                        mlflow_mod.log_artifact(str(p))
        return record

    def on_run_end(self, summary: dict[str, Any], comparisons: dict[str, Any]) -> None:
        self.export_summary(summary, comparisons)


@dataclass(frozen=True)
class EvalSpan:
    """OTLP-aligned span record representing one evaluated task."""

    trace_id: str
    span_id: str
    name: str
    start_time_ns: int
    end_time_ns: int
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalMetricPoint:
    """OTLP-aligned gauge data point for an evaluation metric."""

    name: str
    value: float
    timestamp_ns: int
    attributes: dict[str, Any] = field(default_factory=dict)


class OpenTelemetryExporter(BaseRunnerHook):
    """Emit OpenTelemetry spans and gauge metrics for evaluation runs.

    Works with an injected OTel ``tracer`` (or ``opentelemetry.trace`` when
    installed) and always records OTLP-JSON payloads in memory (and optionally
    to ``otlp_json_path``) for deterministic verification.
    """

    def __init__(
        self,
        service_name: str = "evalkit",
        *,
        tracer: Any = None,
        otlp_json_path: str | Path | None = None,
    ) -> None:
        self.service_name = service_name
        self._tracer = tracer
        self.otlp_json_path = Path(otlp_json_path) if otlp_json_path else None
        self.spans: list[EvalSpan] = []
        self.metric_points: list[EvalMetricPoint] = []
        self._trace_id: str = uuid.uuid4().hex
        self._run_name: str = ""
        self._task_starts: dict[tuple[str, str], int] = {}

    def on_run_start(self, run_name: str, config: dict[str, Any]) -> None:
        self._trace_id = uuid.uuid4().hex
        self._run_name = run_name
        self._task_starts.clear()

    def on_task_start(self, model_name: str, task_label: str, task_spec: dict[str, Any]) -> None:
        self._task_starts[(model_name, task_label)] = time.time_ns()

    def on_task_end(self, model_name: str, task_label: str, metrics: dict[str, Any]) -> None:
        end_ns = time.time_ns()
        start_ns = self._task_starts.pop((model_name, task_label), end_ns)
        primary = str(metrics.get("primary_metric", "score"))
        primary_val = float(metrics.get(primary, 0.0))

        attrs = {
            "eval.run_name": self._run_name,
            "eval.model": model_name,
            "eval.task": task_label,
            "eval.primary_metric": primary,
            "eval.score": primary_val,
            "eval.n": int(metrics.get("n", 0)),
        }
        span = EvalSpan(
            trace_id=self._trace_id,
            span_id=uuid.uuid4().hex[:16],
            name=f"evalkit.task.{task_label}",
            start_time_ns=start_ns,
            end_time_ns=end_ns,
            attributes=attrs,
        )
        self.spans.append(span)

        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self.metric_points.append(
                    EvalMetricPoint(
                        name=f"evalkit.{_sanitize_key(k)}",
                        value=float(v),
                        timestamp_ns=end_ns,
                        attributes={"model": model_name, "task": task_label, "run_name": self._run_name},
                    )
                )

        if self._tracer is not None:
            with self._tracer.start_as_current_span(span.name) as otel_span:
                for ak, av in attrs.items():
                    otel_span.set_attribute(ak, av)

    def to_otlp_dict(self) -> dict[str, Any]:
        """Serialize recorded spans and metrics into an OTLP-style JSON envelope."""
        return {
            "resource": {"attributes": {"service.name": self.service_name, "eval.run_name": self._run_name}},
            "spans": [asdict(s) for s in self.spans],
            "metrics": [asdict(m) for m in self.metric_points],
        }

    def on_run_end(self, summary: dict[str, Any], comparisons: dict[str, Any]) -> None:
        if self.otlp_json_path is not None:
            self.otlp_json_path.parent.mkdir(parents=True, exist_ok=True)
            self.otlp_json_path.write_text(json.dumps(self.to_otlp_dict(), indent=2) + "\n", encoding="utf-8")


__all__ = [
    "EvalMetricPoint",
    "EvalSpan",
    "MLflowExporter",
    "OpenTelemetryExporter",
    "flatten_summary_metrics",
]
