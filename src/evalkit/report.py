"""Render run results as Markdown tables (pasted into READMEs and PR descriptions)."""

from __future__ import annotations

from typing import Any


def _fmt(x: float) -> str:
    """0.6234 -> '62.3'."""
    return f"{100 * x:.1f}"


def render_markdown(summary: dict[str, Any], comparisons: dict[str, Any] | None = None) -> str:
    """Build a report with one row per model, one column per task, and significance tests.

    Each cell shows the task's primary metric with its 95 % bootstrap interval.
    """
    results = summary["results"]
    models = list(results)
    tasks: list[str] = []
    for m in models:
        tasks += [t for t in results[m] if t not in tasks]

    lines = [f"# Evaluation report: {summary.get('run_name', '')}", ""]
    env = summary.get("environment", {})
    if env:
        lines += ["Environment: " + ", ".join(f"{k}={v}" for k, v in env.items()), ""]

    header = "| Model | " + " | ".join(f"{t} ({results[models[0]].get(t, {}).get('primary_metric', '')})" for t in tasks) + " |"
    lines += [header, "|" + "---|" * (len(tasks) + 1)]
    for m in models:
        cells = []
        for t in tasks:
            r = results[m].get(t)
            if r is None:
                cells.append("–")
                continue
            lo, hi = r["ci95"]
            cells.append(f"**{_fmt(r[r['primary_metric']])}** [{_fmt(lo)}, {_fmt(hi)}]")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    if comparisons and comparisons.get("comparisons"):
        lines += ["", f"## Paired comparison vs. baseline `{comparisons['baseline']}`", ""]
        lines += ["| Model | Task | Δ (points) | 95% CI | p (McNemar) | Significant |", "|---|---|---|---|---|---|"]
        for m, per_task in comparisons["comparisons"].items():
            for t, c in per_task.items():
                lo, hi = c["ci95"]
                lines.append(
                    f"| {m} | {t} | {100 * c['delta']:+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}] | {c['p_mcnemar']:.3g} | {'yes' if c['significant'] else 'no'} |"
                )
    return "\n".join(lines) + "\n"
