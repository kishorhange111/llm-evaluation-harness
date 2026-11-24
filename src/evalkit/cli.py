"""Command-line entry point: ``evalkit run configs/quick.yaml``.

Sub-commands:
    run      Evaluate the models and tasks listed in a YAML config.
    list     Show the registered tasks.
    report   Re-render report.md from an existing run folder (no model calls).
    dataset  Build/verify golden dataset manifests and diff dataset versions.
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

    p_ds = sub.add_parser("dataset", help="golden dataset manifest, verification, and diff")
    ds_sub = p_ds.add_subparsers(dest="dataset_command", required=True)

    p_ds_man = ds_sub.add_parser("manifest", help="write a SHA-256 manifest for a JSONL dataset")
    p_ds_man.add_argument("dataset", help="path to JSONL dataset")
    p_ds_man.add_argument("--out", required=True, help="output manifest JSON path")
    p_ds_man.add_argument("--version", default="1.0.0", help="semantic version tag")
    p_ds_man.add_argument("--id-field", default="id", help="row identifier field name")

    p_ds_ver = ds_sub.add_parser("verify", help="verify a JSONL dataset against a manifest")
    p_ds_ver.add_argument("dataset", help="path to JSONL dataset")
    p_ds_ver.add_argument("manifest", help="path to manifest JSON")

    p_ds_diff = ds_sub.add_parser("diff", help="diff two JSONL dataset versions")
    p_ds_diff.add_argument("old_dataset", help="baseline JSONL dataset")
    p_ds_diff.add_argument("new_dataset", help="candidate JSONL dataset")
    p_ds_diff.add_argument("--id-field", default="id", help="row identifier field name")
    p_ds_diff.add_argument("--json", action="store_true", dest="as_json", help="output JSON instead of Markdown")

    args = parser.parse_args(argv)

    if args.command == "dataset":
        from dataclasses import asdict

        from evalkit.datasets import DatasetManifest, build_manifest, diff_datasets, verify_manifest

        if args.dataset_command == "manifest":
            m = build_manifest(args.dataset, version=args.version, id_field=args.id_field)
            out_path = m.save(args.out)
            print(f"{out_path} ({m.num_rows} rows, sha256={m.sha256[:12]})")
            return 0
        if args.dataset_command == "verify":
            m = DatasetManifest.load(args.manifest)
            res = verify_manifest(args.dataset, m)
            print(json.dumps(asdict(res), indent=2))
            return 0 if res.valid else 1
        diff = diff_datasets(args.old_dataset, args.new_dataset, id_field=args.id_field)
        print(json.dumps(asdict(diff), indent=2) if args.as_json else diff.to_markdown())
        return 0

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
