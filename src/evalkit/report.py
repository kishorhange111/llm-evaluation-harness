"""Rich HTML, Markdown, and JSON report generators for evaluation runs."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _fmt(x: float) -> str:
    """0.6234 -> '62.3'."""
    return f"{100 * x:.1f}"


def _collect_tasks(results: dict[str, dict[str, Any]]) -> list[str]:
    tasks: list[str] = []
    for per_task in results.values():
        for t in per_task:
            if t not in tasks:
                tasks.append(t)
    return tasks


def _build_leaderboard(results: dict[str, dict[str, Any]], tasks: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, per_task in results.items():
        vals = [float(per_task[t][per_task[t]["primary_metric"]]) for t in tasks if t in per_task]
        macro = sum(vals) / len(vals) if vals else 0.0
        rows.append({"model": model, "macro_average": round(macro, 6), "tasks_evaluated": len(vals)})
    rows.sort(key=lambda r: r["macro_average"], reverse=True)
    return rows


def render_markdown(summary: dict[str, Any], comparisons: dict[str, Any] | None = None) -> str:
    """Build a report with one row per model, one column per task, and significance tests.

    Each cell shows the task's primary metric with its 95 % bootstrap interval.
    """
    results = summary["results"]
    models = list(results)
    tasks = _collect_tasks(results)

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


def render_json(summary: dict[str, Any], comparisons: dict[str, Any] | None = None, *, indent: int = 2) -> str:
    """Render a normalized machine-readable JSON report including a macro leaderboard."""
    results = summary.get("results", {})
    tasks = _collect_tasks(results)
    payload = {
        "schema_version": "1.0",
        "run_name": summary.get("run_name", ""),
        "environment": summary.get("environment", {}),
        "models": list(results),
        "tasks": tasks,
        "leaderboard": _build_leaderboard(results, tasks),
        "results": results,
        "comparisons": comparisons or {},
    }
    return json.dumps(payload, indent=indent) + "\n"


def _render_ci_bar_svg(score: float, lo: float, hi: float) -> str:
    """Compact inline SVG bar + 95% CI whisker for a score in [0, 1]."""
    w = 120
    x_val = max(0.0, min(1.0, score)) * w
    x_lo = max(0.0, min(1.0, lo)) * w
    x_hi = max(0.0, min(1.0, hi)) * w
    return (
        f'<svg width="{w}" height="14" viewBox="0 0 {w} 14" aria-label="score bar">'
        f'<rect x="0" y="3" width="{w}" height="8" rx="3" fill="#e5e7eb"/>'
        f'<rect x="0" y="3" width="{x_val:.1f}" height="8" rx="3" fill="#2563eb"/>'
        f'<line x1="{x_lo:.1f}" y1="7" x2="{x_hi:.1f}" y2="7" stroke="#0f172a" stroke-width="1.5"/>'
        f'<line x1="{x_lo:.1f}" y1="4" x2="{x_lo:.1f}" y2="10" stroke="#0f172a" stroke-width="1.5"/>'
        f'<line x1="{x_hi:.1f}" y1="4" x2="{x_hi:.1f}" y2="10" stroke="#0f172a" stroke-width="1.5"/>'
        "</svg>"
    )


def render_html(summary: dict[str, Any], comparisons: dict[str, Any] | None = None) -> str:
    """Render a standalone, self-contained HTML5 evaluation report with SVG CI bars."""
    results = summary.get("results", {})
    models = list(results)
    tasks = _collect_tasks(results)
    leaderboard = _build_leaderboard(results, tasks)
    run_name = html.escape(str(summary.get("run_name", "Evaluation Report")))
    env = summary.get("environment", {})
    env_badges = "".join(f'<span class="badge">{html.escape(str(k))}={html.escape(str(v))}</span>' for k, v in env.items())

    best_model = html.escape(leaderboard[0]["model"]) if leaderboard else "–"
    best_avg = f"{100 * leaderboard[0]['macro_average']:.1f}%" if leaderboard else "–"

    header_cols = "".join(f"<th>{html.escape(t)}</th>" for t in tasks)
    table_rows: list[str] = []
    for m in models:
        cells = [f"<td><strong>{html.escape(m)}</strong></td>"]
        for t in tasks:
            r = results[m].get(t)
            if r is None:
                cells.append("<td>–</td>")
                continue
            val = float(r[r["primary_metric"]])
            lo, hi = r.get("ci95", [val, val])
            svg = _render_ci_bar_svg(val, float(lo), float(hi))
            cells.append(f"<td><div><strong>{_fmt(val)}%</strong> [{_fmt(float(lo))}, {_fmt(float(hi))}]</div>{svg}</td>")
        table_rows.append("<tr>" + "".join(cells) + "</tr>")

    comp_html = ""
    if comparisons and comparisons.get("comparisons"):
        base_name = html.escape(str(comparisons.get("baseline", "")))
        comp_rows: list[str] = []
        for m, per_task in comparisons["comparisons"].items():
            for t, c in per_task.items():
                delta = float(c["delta"])
                lo, hi = c["ci95"]
                cls = "pos" if delta > 0 else "neg" if delta < 0 else "neu"
                sig_txt = "Yes" if c["significant"] else "No"
                comp_rows.append(
                    f"<tr><td>{html.escape(m)}</td><td>{html.escape(t)}</td>"
                    f'<td class="{cls}">{100 * delta:+.1f}</td>'
                    f"<td>[{100 * float(lo):+.1f}, {100 * float(hi):+.1f}]</td>"
                    f"<td>{float(c['p_mcnemar']):.3g}</td><td>{sig_txt}</td></tr>"
                )
        comp_html = (
            f"<h2>Paired Comparison vs. Baseline <code>{base_name}</code></h2>"
            "<table><thead><tr><th>Model</th><th>Task</th><th>Delta (pts)</th>"
            "<th>95% CI</th><th>p (McNemar)</th><th>Significant</th></tr></thead>"
            f"<tbody>{''.join(comp_rows)}</tbody></table>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EvalKit Report — {run_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 2rem auto; max-width: 1080px; color: #0f172a; line-height: 1.5; padding: 0 1rem; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .badges {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1.5rem; }}
  .badge {{ background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 999px; padding: 0.15rem 0.65rem; font-size: 0.8rem; color: #334155; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
  .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.9rem 1rem; }}
  .card .label {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: #64748b; }}
  .card .value {{ font-size: 1.35rem; font-weight: 700; margin-top: 0.2rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0 2rem; font-size: 0.92rem; }}
  th, td {{ border-bottom: 1px solid #e2e8f0; padding: 0.6rem 0.75rem; text-align: left; vertical-align: middle; }}
  th {{ background: #f8fafc; font-weight: 600; }}
  .pos {{ color: #15803d; font-weight: 600; }}
  .neg {{ color: #b91c1c; font-weight: 600; }}
  .neu {{ color: #475569; }}
</style>
</head>
<body>
<h1>Evaluation Report: {run_name}</h1>
<div class="badges">{env_badges}</div>
<div class="cards">
  <div class="card"><div class="label">Models</div><div class="value">{len(models)}</div></div>
  <div class="card"><div class="label">Tasks</div><div class="value">{len(tasks)}</div></div>
  <div class="card"><div class="label">Top Model</div><div class="value">{best_model}</div></div>
  <div class="card"><div class="label">Top Macro Avg</div><div class="value">{best_avg}</div></div>
</div>
<h2>Benchmark Results (95% Bootstrap CI)</h2>
<table>
  <thead><tr><th>Model</th>{header_cols}</tr></thead>
  <tbody>{"".join(table_rows)}</tbody>
</table>
{comp_html}
</body>
</html>
"""


def write_reports(
    run_dir: str | Path,
    *,
    formats: tuple[str, ...] | list[str] = ("markdown", "html", "json"),
) -> dict[str, Path]:
    """Render and write report artifacts for a completed run directory."""
    root = Path(run_dir)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    comp_path = root / "comparisons.json"
    comparisons = json.loads(comp_path.read_text(encoding="utf-8")) if comp_path.exists() else None

    written: dict[str, Path] = {}
    for fmt in formats:
        if fmt in ("markdown", "md"):
            out = root / "report.md"
            out.write_text(render_markdown(summary, comparisons), encoding="utf-8")
            written["markdown"] = out
        elif fmt == "html":
            out = root / "report.html"
            out.write_text(render_html(summary, comparisons), encoding="utf-8")
            written["html"] = out
        elif fmt == "json":
            out = root / "report.json"
            out.write_text(render_json(summary, comparisons), encoding="utf-8")
            written["json"] = out
        else:
            raise ValueError(f"unknown report format {fmt!r}; expected markdown, html, or json")
    return written


__all__ = ["render_html", "render_json", "render_markdown", "write_reports"]
