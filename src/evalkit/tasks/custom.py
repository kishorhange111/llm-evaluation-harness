"""Bring-your-own-data task: evaluate on a JSONL file without writing any code.

Each line must contain ``question`` and ``answer`` (and optionally ``id``).
This is how teams evaluate on private, domain-specific data, which usually
predicts real-world quality better than public leaderboards.

Example line::

    {"id": "faq-7", "question": "What port does Postgres use by default?", "answer": "5432"}
"""

from __future__ import annotations

from evalkit.metrics.text import exact_match, token_f1
from evalkit.tasks.base import GenerationTask
from evalkit.types import Example, ExampleResult


class JsonlQA(GenerationTask):
    name = "jsonl_qa"
    eval_split = "test"
    fewshot_split = None
    primary_metric = "exact_match"
    max_new_tokens = 64
    stop = ("\n",)
    use_chat_template = True

    def example_id(self, index: int, row: dict) -> str:
        return str(row.get("id", f"{self.name}-{index}"))

    def build_prompt(self, example: Example) -> str:
        return f"Answer the question with a short phrase only.\nQuestion: {example.data['question']}\nAnswer:"

    def score(self, example: Example, completions: list[str]) -> ExampleResult:
        pred, gold = completions[0].strip(), str(example.data["answer"])
        return ExampleResult(id=example.id, score=exact_match(pred, gold), prediction=pred, target=gold, extra={"f1": token_f1(pred, gold)})

    def aggregate(self, results: list[ExampleResult]) -> dict[str, float]:
        n = max(len(results), 1)
        return {"exact_match": sum(r.score for r in results) / n, "f1": sum(r.extra["f1"] for r in results) / n}
