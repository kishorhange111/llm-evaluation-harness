"""Tests for the unified registry, plugin discovery, and runner lifecycle hooks."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalkit.models import MODEL_REGISTRY, DummyModel, build_model
from evalkit.registry import (
    METRIC_REGISTRY,
    TASK_REGISTRY,
    BaseRunnerHook,
    Registry,
    discover_plugins,
    register_metric,
    register_model,
    register_task,
)
from evalkit.runner import run
from evalkit.tasks import build_task
from evalkit.tasks.base import GenerationTask
from evalkit.types import Example, ExampleResult

FIX = Path(__file__).parent / "fixtures"


def test_registry_register_filter_override_and_unregister():
    reg: Registry[str] = Registry("widget")
    reg.register("alpha", "A", category="core", tags=("fast", "v1"), description="Alpha widget")
    reg.register("beta", "B", category="extra", tags=("slow",))

    assert "alpha" in reg and len(reg) == 2
    assert reg.get("alpha") == "A"
    assert reg.names() == ["alpha", "beta"]
    assert reg.names(category="core") == ["alpha"]
    assert reg.names(tag="slow") == ["beta"]
    assert reg.get_entry("alpha").description == "Alpha widget"

    with pytest.raises(ValueError, match="already registered"):
        reg.register("alpha", "A2")

    reg.register("alpha", "A2", override=True)
    assert reg.get("alpha") == "A2"

    removed = reg.unregister("beta")
    assert removed.target == "B"
    with pytest.raises(ValueError, match="unknown widget"):
        reg.get("beta")


def test_decorator_registration_for_task_model_and_metric():
    @register_task("custom_echo_test", category="custom", tags=("unit",))
    class EchoTask(GenerationTask):
        name = "custom_echo_test"

        def build_prompt(self, example: Example) -> str:
            return example.data["prompt"]

        def score(self, example: Example, completions: list[str]) -> ExampleResult:
            return ExampleResult(example.id, 1.0, completions[0], example.data["prompt"])

    @register_model("echo_backend_test")
    def _echo_factory(**kwargs):
        return DummyModel(name=kwargs.get("name", "echo"))

    @register_metric("identity_metric_test", tags=("unit",))
    def _identity(x: float) -> float:
        return x

    try:
        task = build_task("custom_echo_test")
        assert isinstance(task, EchoTask)
        model = build_model({"backend": "echo_backend_test", "name": "echo-1"})
        assert model.name == "echo-1"
        assert METRIC_REGISTRY.get("identity_metric_test")(0.75) == 0.75
    finally:
        TASK_REGISTRY.unregister("custom_echo_test")
        MODEL_REGISTRY.unregister("echo_backend_test")
        METRIC_REGISTRY.unregister("identity_metric_test")


def test_discover_plugins_loads_and_reports_failures(monkeypatch):
    called: list[str] = []

    class FakeEP:
        def __init__(self, name: str, fn):
            self.name = name
            self._fn = fn

        def load(self):
            return self._fn

    def broken():
        raise RuntimeError("boom")

    class FakeEntryPoints:
        def select(self, group: str):
            assert group == "evalkit.plugins"
            return [FakeEP("good_plugin", lambda: called.append("ok")), FakeEP("bad_plugin", broken)]

    monkeypatch.setattr("evalkit.registry.metadata.entry_points", lambda: FakeEntryPoints())
    result = discover_plugins()
    assert result.loaded == ("good_plugin",)
    assert "bad_plugin" in result.failed and "RuntimeError: boom" in result.failed["bad_plugin"]
    assert called == ["ok"]


def test_runner_invokes_lifecycle_hooks(tmp_path):
    events: list[tuple[str, str]] = []

    class RecordingHook(BaseRunnerHook):
        def on_run_start(self, run_name, config):
            events.append(("run_start", run_name))

        def on_task_start(self, model_name, task_label, task_spec):
            events.append(("task_start", f"{model_name}:{task_label}"))

        def on_task_end(self, model_name, task_label, metrics):
            events.append(("task_end", f"{model_name}:{task_label}:{metrics['n']}"))

        def on_run_end(self, summary, comparisons):
            events.append(("run_end", summary["run_name"]))

    config = {
        "run_name": "hook-test",
        "models": [{"backend": "dummy", "name": "m1"}],
        "tasks": [{"name": "jsonl_qa", "data_files": str(FIX / "qa.jsonl"), "label": "qa"}],
    }
    run(config, output_dir=tmp_path, hooks=[RecordingHook()])
    assert events[0] == ("run_start", "hook-test")
    assert events[1] == ("task_start", "m1:qa")
    assert events[2][0] == "task_end" and events[2][1].startswith("m1:qa:")
    assert events[3] == ("run_end", "hook-test")
