"""How much can we trust an LLM judge? A position-bias experiment on MT-Bench prompts.

Steps:
  1. Three Qwen2.5-Instruct sizes (0.5B, 1.5B, 3B) answer the 80 first-turn
     MT-Bench questions.
  2. Qwen2.5-7B-Instruct judges every pair of models, in both presentation
     orders (A/B and B/A).
  3. We report win rates after debiasing (a verdict counts only if it survives
     the swap), and how often the judge flipped when only the order changed.

Usage (GPU):  python scripts/judge_experiment.py --out results/judge
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from evalkit.metrics.judge import judge_pairwise, win_rate
from evalkit.metrics.stats import bootstrap_ci
from evalkit.models.hf import HFModel
from evalkit.runner import free_model
from evalkit.types import GenerationRequest

CONTESTANTS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
JUDGE = "Qwen/Qwen2.5-7B-Instruct"


def load_questions(limit: int | None) -> list[str]:
    """First-turn questions from MT-Bench (writing, roleplay, reasoning, math, coding, STEM ...)."""
    from datasets import load_dataset

    rows = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    questions = [r["prompt"][0] for r in rows]
    return questions[:limit] if limit else questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/judge")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.limit)

    # 1. Collect answers, one model on the GPU at a time.
    answers: dict[str, list[str]] = {}
    for name in CONTESTANTS:
        model = HFModel(name, chat=True, batch_size=32)
        reqs = [GenerationRequest(model.apply_chat_template(q), max_new_tokens=512) for q in questions]
        answers[name] = [o[0].strip() for o in model.generate(reqs)]
        free_model(model)
    (out / "answers.json").write_text(json.dumps({"questions": questions, "answers": answers}, indent=2))

    # 2. Judge every pair in both orders.
    judge = HFModel(JUDGE, chat=True, batch_size=16)
    report = {"judge": JUDGE, "n_questions": len(questions), "pairs": {}}
    for a, b in itertools.combinations(CONTESTANTS, 2):
        outcomes = judge_pairwise(judge, questions, answers[a], answers[b])
        # Raw verdict with A first: shows what a naive single-order judge would report.
        first_pos = [o.verdict_ab == "A" for o in outcomes if o.verdict_ab in ("A", "B")]
        flips = [float(o.position_flip) for o in outcomes]
        per_q = [1.0 if o.winner == "B" else 0.5 if o.winner == "tie" else 0.0 for o in outcomes]
        ci = bootstrap_ci(per_q)
        report["pairs"][f"{b} vs {a}"] = {
            "win_rate_b_debiased": win_rate(outcomes, "B"),
            "win_rate_ci95": [ci.low, ci.high],
            "position_flip_rate": sum(flips) / len(flips),
            "naive_first_position_preference": sum(first_pos) / max(len(first_pos), 1),
            "unparseable_verdicts": sum(o.verdict_ab is None or o.verdict_ba is None for o in outcomes),
        }
        print(b, "vs", a, json.dumps(report["pairs"][f"{b} vs {a}"]), flush=True)
    (out / "judge_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
