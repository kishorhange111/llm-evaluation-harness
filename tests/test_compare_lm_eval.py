"""The lm-eval comparison script reads a summary.json and reports point differences."""

import importlib.util
from pathlib import Path


def load_script():
    path = Path(__file__).parents[1] / "scripts" / "compare_lm_eval.py"
    spec = importlib.util.spec_from_file_location("compare_lm_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compare_reports_differences_and_missing_tasks():
    mod = load_script()
    summary = {"results": {"Qwen2.5-0.5B": {"mmlu": {"accuracy": 0.48}, "hellaswag": {"accuracy": 0.4053, "accuracy_norm": 0.52}}}}
    rows = {r["metric"]: r for r in mod.compare(summary)}
    assert rows["mmlu/acc"]["diff_points"] == 0.51
    assert rows["hellaswag/acc"]["diff_points"] == 0.0
    assert rows["arc_challenge/acc"]["evalkit"] is None  # task not run -> reported, not crashed
