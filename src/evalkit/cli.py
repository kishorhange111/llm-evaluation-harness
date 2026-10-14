"""Command-line entry point: ``evalkit run configs/quick.yaml``.

Sub-commands:
    run      Evaluate the models and tasks listed in a YAML config.
    list     Show the registered tasks.
    report   Re-render report.md from an existing run folder (no model calls).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalkit", description="Evaluate LLMs on benchmarks with confidence intervals.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an evaluation config")
    p_run.add_argument("config", help="path to a YAML config")
    p_run.add_argument("--output-dir", help="override output_dir from the config")
    p_run.add_argument("--limit", type=int, help="cap examples per task (quick smoke test)")

    sub.add_parser("list", help="list available tasks")

    p_rep = sub.add_parser("report", help="re-render report.md for a finished run")
    p_rep.add_argument("run_dir")

    args = parser.parse_args(argv)

    if args.command == "list":
        from evalkit.tasks import TASKS

        for name, cls in sorted(TASKS.items()):
            print(f"{name:15s} {cls.__doc__.strip().splitlines()[0] if cls.__doc__ else ''}")
        return 0

    if args.command == "report":
        from evalkit.report import render_markdown

        run_dir = Path(args.run_dir)
        summary = json.loads((run_dir / "summary.json").read_text())
        comp_path = run_dir / "comparisons.json"
        comparisons = json.loads(comp_path.read_text()) if comp_path.exists() else None
        (run_dir / "report.md").write_text(render_markdown(summary, comparisons))
        print(run_dir / "report.md")
        return 0

    from evalkit.runner import run

    config = yaml.safe_load(Path(args.config).read_text())
    if args.limit is not None:
        # Apply the limit to every task: one flag turns any config into a smoke test.
        for task in config["tasks"]:
            task["limit"] = args.limit
    run(config, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
