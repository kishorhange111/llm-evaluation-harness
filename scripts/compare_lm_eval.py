"""Compare an evalkit run with lm-evaluation-harness numbers for the same model.

python scripts/compare_lm_eval.py results/validate_vs_lm_eval/summary.json

The reference numbers were produced by ``lm_eval --model hf`` (0-shot, bf16) for
Qwen/Qwen2.5-0.5B in the mini-chatgpt project (results/benchmarks.json there).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# lm-eval metric name -> (evalkit task, evalkit metric)
LM_EVAL_QWEN25_05B = {
    "mmlu/acc": (0.4749, "mmlu", "accuracy"),
    "arc_challenge/acc": (0.2935, "arc_challenge", "accuracy"),
    "arc_challenge/acc_norm": (0.3234, "arc_challenge", "accuracy_norm"),
    "hellaswag/acc": (0.4053, "hellaswag", "accuracy"),
    "hellaswag/acc_norm": (0.5215, "hellaswag", "accuracy_norm"),
}


def compare(summary: dict, model: str = "Qwen2.5-0.5B") -> list[dict]:
    """One row per metric: lm-eval value, evalkit value and the difference in points."""
    results = summary["results"][model]
    rows = []
    for name, (ref, task, metric) in LM_EVAL_QWEN25_05B.items():
        value = results.get(task, {}).get(metric)
        # Metrics may be stored as plain numbers or as {"value": ..., "ci95": ...}.
        value = value["value"] if isinstance(value, dict) else value
        rows.append(
            {"metric": name, "lm_eval": ref, "evalkit": value, "diff_points": None if value is None else round(100 * (value - ref), 2)}
        )
    return rows


def main() -> None:
    summary = json.loads(Path(sys.argv[1]).read_text())
    rows = compare(summary)
    print(f"{'metric':<26}{'lm-eval':>9}{'evalkit':>9}{'diff (pts)':>12}")
    for r in rows:
        ev = "n/a" if r["evalkit"] is None else f"{r['evalkit']:.4f}"
        print(f"{r['metric']:<26}{r['lm_eval']:>9.4f}{ev:>9}{str(r['diff_points']):>12}")
    out = Path(sys.argv[1]).with_name("lm_eval_comparison.json")
    out.write_text(json.dumps(rows, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
