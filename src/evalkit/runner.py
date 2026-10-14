"""Run a set of tasks against a set of models and write reproducible results.

Output layout (one folder per run)::

    results/<run_name>/
      config.yaml                       # exact config used (copied)
      samples/<model>__<task>.jsonl     # every example: prompt id, prediction, target, score
      summary.json                      # metrics + 95% CIs + timings + environment
      comparisons.json                  # paired significance tests vs. the first (baseline) model
      report.md                         # human-readable tables

Models are evaluated one at a time and freed afterwards, so a sweep over
several 7B models fits on a single GPU.
"""

from __future__ import annotations

import gc
import json
import platform
import re
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from evalkit import __version__
from evalkit.cache import CachedModel
from evalkit.metrics.stats import bootstrap_ci, mcnemar_test, paired_bootstrap
from evalkit.models import Model, build_model
from evalkit.tasks import build_task
from evalkit.types import ExampleResult


def slug(text: str) -> str:
    """Make a filesystem-safe name: 'Qwen/Qwen2.5-1.5B' -> 'Qwen__Qwen2.5-1.5B'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "__", text)


def environment_info() -> dict[str, Any]:
    """Versions and hardware recorded next to results, so numbers can be reproduced later."""
    info: dict[str, Any] = {"evalkit": __version__, "python": platform.python_version(), "platform": platform.platform()}
    try:
        info["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        import torch
        import transformers

        info.update(torch=torch.__version__, transformers=transformers.__version__)
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def run_task(model: Model, task_spec: dict[str, Any]) -> tuple[str, list[ExampleResult], dict[str, Any]]:
    """Evaluate one task; returns (label, per-example results, metrics with CIs)."""
    spec = dict(task_spec)
    name = spec.pop("name")
    label = spec.pop("label", None) or name
    task = build_task(name, **spec)
    examples = task.examples()

    start = time.perf_counter()
    results = task.evaluate(model, examples)
    elapsed = time.perf_counter() - start

    metrics: dict[str, Any] = dict(task.aggregate(results))
    # Confidence interval on the headline metric, computed from per-example scores.
    ci = bootstrap_ci([r.score for r in results])
    metrics["ci95"] = [ci.low, ci.high]
    metrics["n"] = len(results)
    metrics["seconds"] = round(elapsed, 1)
    metrics["primary_metric"] = task.primary_metric
    return label, results, metrics


def compare_to_baseline(per_model: dict[str, dict[str, list[ExampleResult]]]) -> dict[str, Any]:
    """Paired tests of every model against the first one, per task, on shared example ids."""
    names = list(per_model)
    if len(names) < 2:
        return {}
    base = names[0]
    out: dict[str, Any] = {"baseline": base, "comparisons": {}}
    for other in names[1:]:
        for task, base_results in per_model[base].items():
            if task not in per_model[other]:
                continue
            a = {r.id: r.score for r in base_results}
            b = {r.id: r.score for r in per_model[other][task]}
            shared = sorted(a.keys() & b.keys())  # pair on the same questions only
            sa, sb = [a[i] for i in shared], [b[i] for i in shared]
            pb = paired_bootstrap(sa, sb)
            out["comparisons"].setdefault(other, {})[task] = {
                "delta": pb.delta,
                "ci95": [pb.low, pb.high],
                "p_bootstrap": pb.p_value,
                "p_mcnemar": mcnemar_test(sa, sb),
                "significant": pb.significant,
                "n": pb.n,
            }
    return out


def free_model(model: Model) -> None:
    """Release GPU memory before loading the next model.

    ``del model`` here would only drop this function's own reference; the
    caller's variable still keeps the weights alive. So we detach the heavy
    attributes from the wrapper object itself, then collect.
    """
    # CachedModel wraps another Model in ``.inner``; unwrap to reach the one holding the weights.
    while getattr(model, "inner", None) is not None:
        model = model.inner  # type: ignore[attr-defined]
    if getattr(model, "model", None) is not None:
        model.model = None  # type: ignore[attr-defined]
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run(config: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    """Execute a full evaluation config and write everything to ``output_dir``.

    Args:
        config: Parsed YAML with ``models`` and ``tasks`` lists (see ``configs/``).
        output_dir: Overrides ``config["output_dir"]``.

    Returns:
        The summary dict that is also written to ``summary.json``.
    """
    out = Path(output_dir or config.get("output_dir") or f"results/{config.get('run_name', 'run')}")
    (out / "samples").mkdir(parents=True, exist_ok=True)
    # Round-trip through JSON so non-serialisable values (e.g. a test's Python callable) become strings.
    (out / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(config, default=repr)), sort_keys=False))

    summary: dict[str, Any] = {"run_name": config.get("run_name", out.name), "environment": environment_info(), "results": {}}
    per_model: dict[str, dict[str, list[ExampleResult]]] = {}

    for model_spec in config["models"]:
        model_spec = dict(model_spec)
        display = model_spec.pop("label", None)
        model: Model = build_model(model_spec)
        if config.get("cache", True):
            model = CachedModel(model, out / "cache.sqlite")
        display = display or model.name
        per_model[display] = {}
        summary["results"][display] = {}

        for task_spec in config["tasks"]:
            label, results, metrics = run_task(model, task_spec)
            per_model[display][label] = results
            summary["results"][display][label] = metrics
            # Write per-example records immediately: if a later task crashes, this one is kept.
            with open(out / "samples" / f"{slug(display)}__{slug(label)}.jsonl", "w") as f:
                for r in results:
                    f.write(json.dumps(asdict(r), default=str) + "\n")
            print(f"[{display}] {label}: {metrics[metrics['primary_metric']]:.4f} (n={metrics['n']}, {metrics['seconds']}s)", flush=True)
            (out / "summary.json").write_text(json.dumps(summary, indent=2))
        free_model(model)

    comparisons = compare_to_baseline(per_model)
    (out / "comparisons.json").write_text(json.dumps(comparisons, indent=2))
    from evalkit.report import render_markdown

    (out / "report.md").write_text(render_markdown(summary, comparisons))
    return summary
