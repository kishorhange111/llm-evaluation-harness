"""Eval run comparison & regression detector with blocking CI thresholds.

Compares a candidate evaluation run against a baseline run (either in-memory
``summary.json`` dicts or full run directories containing ``samples/*.jsonl``),
checks per-task relative drops and absolute score floors, optionally gates on
paired statistical significance, and renders a Markdown summary for PR comments.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evalkit.metrics.stats import paired_bootstrap
from evalkit.runner import slug


@dataclass(frozen=True)
class RegressionRule:
    """Threshold policy for one task (or ``"*"`` wildcard)."""

    task: str = "*"
    metric: str | None = None
    max_drop: float = 0.02
    min_score: float | None = None
    require_significant: bool = False
    blocking: bool = True


@dataclass(frozen=True)
class TaskDelta:
    """Comparison outcome for one (model, task, metric) tuple."""

    model: str
    task: str
    metric: str
    baseline: float
    candidate: float
    delta: float
    p_value: float | None
    significant: bool | None
    status: str  # "pass" | "warn" | "fail"
    reason: str


@dataclass(frozen=True)
class RegressionReport:
    """Aggregate regression gate decision across all compared tasks."""

    passed: bool
    baseline_run: str
    candidate_run: str
    deltas: list[TaskDelta]
    blocking_failures: list[TaskDelta]
    warnings: list[TaskDelta]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "baseline_run": self.baseline_run,
            "candidate_run": self.candidate_run,
            "num_checked": len(self.deltas),
            "num_failures": len(self.blocking_failures),
            "num_warnings": len(self.warnings),
            "deltas": [asdict(d) for d in self.deltas],
        }

    def to_markdown(self) -> str:
        badge = "PASS" if self.passed else "FAIL"
        lines = [
            f"## Eval Regression Gate: {badge}",
            "",
            f"Baseline `{self.baseline_run}` vs. Candidate `{self.candidate_run}`",
            "",
            "| Model | Task | Metric | Baseline | Candidate | Delta | p-value | Status | Reason |",
            "|---|---|---|---:|---:|---:|---:|---|---|",
        ]
        for d in self.deltas:
            p_str = f"{d.p_value:.3g}" if d.p_value is not None else "-"
            lines.append(
                f"| {d.model} | {d.task} | {d.metric} | {d.baseline:.4f} | {d.candidate:.4f} | "
                f"{d.delta:+.4f} | {p_str} | {d.status.upper()} | {d.reason} |"
            )
        return "\n".join(lines) + "\n"


def _load_summary(source: dict[str, Any] | str | Path) -> tuple[dict[str, Any], Path | None]:
    if isinstance(source, dict):
        return source, None
    p = Path(source)
    if p.is_dir():
        return json.loads((p / "summary.json").read_text(encoding="utf-8")), p
    return json.loads(p.read_text(encoding="utf-8")), p.parent


def _load_sample_scores(run_dir: Path | None, model: str, task: str) -> dict[str, float]:
    if run_dir is None:
        return {}
    sample_file = run_dir / "samples" / f"{slug(model)}__{slug(task)}.jsonl"
    if not sample_file.exists():
        return {}
    scores: dict[str, float] = {}
    for line in sample_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            scores[str(row["id"])] = float(row["score"])
    return scores


def _match_rule(task: str, rules: list[RegressionRule], default_max_drop: float) -> RegressionRule:
    for r in rules:
        if r.task == task:
            return r
    for r in rules:
        if r.task == "*":
            return r
    return RegressionRule(task="*", max_drop=default_max_drop)


def compare_runs(
    baseline: dict[str, Any] | str | Path,
    candidate: dict[str, Any] | str | Path,
    *,
    rules: list[RegressionRule] | None = None,
    default_max_drop: float = 0.02,
    model_map: dict[str, str] | None = None,
) -> RegressionReport:
    """Compare ``candidate`` against ``baseline`` and flag blocking regressions."""
    base_summary, base_dir = _load_summary(baseline)
    cand_summary, cand_dir = _load_summary(candidate)
    active_rules = list(rules or [])

    base_results = base_summary.get("results", {})
    cand_results = cand_summary.get("results", {})

    # Map candidate models to baseline models (1:1 by name, or explicit map, or single-model pair).
    pairs: list[tuple[str, str]] = []
    if model_map:
        for cm, bm in model_map.items():
            if cm in cand_results and bm in base_results:
                pairs.append((bm, cm))
    else:
        shared_models = [m for m in cand_results if m in base_results]
        if shared_models:
            pairs = [(m, m) for m in shared_models]
        elif len(base_results) == 1 and len(cand_results) == 1:
            pairs = [(next(iter(base_results)), next(iter(cand_results)))]

    deltas: list[TaskDelta] = []
    failures: list[TaskDelta] = []
    warnings: list[TaskDelta] = []

    for base_model, cand_model in pairs:
        base_tasks = base_results[base_model]
        cand_tasks = cand_results[cand_model]
        for task_name, cand_metrics in cand_tasks.items():
            if task_name not in base_tasks:
                continue
            base_metrics = base_tasks[task_name]
            rule = _match_rule(task_name, active_rules, default_max_drop)
            metric_name = rule.metric or cand_metrics.get("primary_metric") or base_metrics.get("primary_metric")
            if not metric_name or metric_name not in base_metrics or metric_name not in cand_metrics:
                continue

            b_val = float(base_metrics[metric_name])
            c_val = float(cand_metrics[metric_name])
            delta = c_val - b_val

            p_val: float | None = None
            sig: bool | None = None
            b_samples = _load_sample_scores(base_dir, base_model, task_name)
            c_samples = _load_sample_scores(cand_dir, cand_model, task_name)
            shared_ids = sorted(b_samples.keys() & c_samples.keys())
            if len(shared_ids) >= 2:
                pb = paired_bootstrap([b_samples[i] for i in shared_ids], [c_samples[i] for i in shared_ids], n_resamples=2_000, seed=0)
                p_val = pb.p_value
                sig = pb.significant

            reasons: list[str] = []
            if rule.min_score is not None and c_val < rule.min_score:
                reasons.append(f"below min_score {rule.min_score:.4f}")
            drop = b_val - c_val
            if drop > rule.max_drop + 1e-12:
                if rule.require_significant and sig is False:
                    pass
                else:
                    reasons.append(f"dropped {drop:.4f} > max_drop {rule.max_drop:.4f}")

            if reasons:
                status = "fail" if rule.blocking else "warn"
                reason_str = "; ".join(reasons)
            else:
                status = "pass"
                reason_str = "ok"

            td = TaskDelta(
                model=cand_model,
                task=task_name,
                metric=metric_name,
                baseline=b_val,
                candidate=c_val,
                delta=delta,
                p_value=p_val,
                significant=sig,
                status=status,
                reason=reason_str,
            )
            deltas.append(td)
            if status == "fail":
                failures.append(td)
            elif status == "warn":
                warnings.append(td)

    return RegressionReport(
        passed=len(failures) == 0,
        baseline_run=str(base_summary.get("run_name", "baseline")),
        candidate_run=str(cand_summary.get("run_name", "candidate")),
        deltas=deltas,
        blocking_failures=failures,
        warnings=warnings,
    )


__all__ = ["RegressionReport", "RegressionRule", "TaskDelta", "compare_runs"]
