"""Multiple-choice benchmarks scored by log-likelihood: MMLU, ARC-Challenge, HellaSwag.

Each class only says how to render a question and where the answer is; the
scoring logic lives once in :class:`~evalkit.tasks.base.MultipleChoiceTask`.
"""

from __future__ import annotations

import re

from evalkit.tasks.base import MultipleChoiceTask
from evalkit.types import Example

LETTERS = "ABCDEFGHIJ"

# Prompt formats used for the prompt-sensitivity study. They all ask the same
# question; only the surface formatting differs. A robust model should score
# the same on all of them, and the spread shows how much a single reported
# number depends on formatting choices.
MMLU_TEMPLATES = {
    "standard": "{question}\n{options}\nAnswer:",
    "instruction": "The following is a multiple choice question. Reply with the letter of the correct option.\n\n{question}\n{options}\nAnswer:",
    "parenthesis": "Question: {question}\nChoices:\n{options_paren}\nThe correct choice is",
    "lowercase_label": "{question}\n{options}\nanswer:",
}


class MMLU(MultipleChoiceTask):
    """MMLU: 14,042 four-option questions across 57 subjects (law, medicine, CS ...)."""

    name = "mmlu"
    hf_path = "cais/mmlu"
    hf_name = "all"
    eval_split = "test"
    fewshot_split = "dev"

    def _render(self, row: dict, with_answer: bool) -> str:
        """Render one question (optionally with its answer, for few-shot examples)."""
        template = MMLU_TEMPLATES[self.options.get("prompt_variant", "standard")]
        options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
        options_paren = "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(row["choices"]))
        text = template.format(question=row["question"].strip(), options=options, options_paren=options_paren)
        if with_answer:
            text += f" {LETTERS[row['answer']]}"
        return text

    def build_context(self, example: Example) -> str:
        row = example.data
        subject = row.get("subject", "").replace("_", " ")
        header = f"The following are multiple choice questions (with answers) about {subject}.\n\n" if subject else ""
        shots = self._subject_shots(row.get("subject"))
        body = "\n\n".join([self._render(s, True) for s in shots] + [self._render(row, False)])
        return header + body

    def _subject_shots(self, subject: str | None) -> list[dict]:
        """First ``num_fewshot`` dev examples of the same subject (deterministic)."""
        if not self.num_fewshot:
            return []
        if self._fewshot_pool is None:
            self._fewshot_pool = self._load_rows(self.fewshot_split or "dev")
        same = [r for r in self._fewshot_pool if r.get("subject") == subject]
        return same[: self.num_fewshot]

    def choices(self, example: Example) -> list[str]:
        # Score just the letter (" A" ... " D"); the leading space matches how it would be tokenised.
        return [f" {LETTERS[i]}" for i in range(len(example.data["choices"]))]

    def gold_index(self, example: Example) -> int:
        return int(example.data["answer"])

    def evaluate(self, model, examples):
        """Score as usual, then tag each result with its subject for per-subject reporting."""
        results = super().evaluate(model, examples)
        for ex, res in zip(examples, results):
            res.extra["subject"] = ex.data.get("subject", "unknown")
        return results

    def aggregate(self, results):
        """Overall (micro) accuracy plus the macro average over subjects (each subject weighs equally)."""
        metrics = super().aggregate(results)
        by_subject: dict[str, list[float]] = {}
        for r in results:
            by_subject.setdefault(r.extra.get("subject", "unknown"), []).append(r.score)
        if by_subject:
            metrics["accuracy_macro"] = sum(sum(v) / len(v) for v in by_subject.values()) / len(by_subject)
        return metrics


class ARCChallenge(MultipleChoiceTask):
    """ARC-Challenge: 1,172 grade-school science questions that retrieval baselines get wrong."""

    name = "arc_challenge"
    hf_path = "allenai/ai2_arc"
    hf_name = "ARC-Challenge"
    eval_split = "test"
    fewshot_split = "train"

    def example_id(self, index: int, row: dict) -> str:
        return row.get("id", f"{self.name}-{index}")

    def build_context(self, example: Example) -> str:
        shots = [
            f"Question: {s['question']}\nAnswer: {s['choices']['text'][s['choices']['label'].index(s['answerKey'])]}"
            for s in self.fewshot_examples()
        ]
        return "\n\n".join(shots + [f"Question: {example.data['question']}\nAnswer:"])

    def choices(self, example: Example) -> list[str]:
        # Score the full option text: ARC options are short phrases, so this is the standard setup.
        return [f" {t}" for t in example.data["choices"]["text"]]

    def gold_index(self, example: Example) -> int:
        return example.data["choices"]["label"].index(example.data["answerKey"])


class HellaSwag(MultipleChoiceTask):
    """HellaSwag: pick the most plausible continuation of an everyday scenario (10,042 val items).

    Options are long sentences of different lengths, so ``accuracy_norm`` is the
    number people report.
    """

    name = "hellaswag"
    hf_path = "Rowan/hellaswag"
    eval_split = "validation"
    fewshot_split = "train"
    primary_metric = "accuracy_norm"

    @staticmethod
    def _clean(text: str) -> str:
        """Remove WikiHow artefacts like '[title]' and double spaces from the raw data."""
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        return text.replace("  ", " ").strip()

    def build_context(self, example: Example) -> str:
        d = example.data
        return self._clean(f"{d['activity_label']}: {d['ctx_a']} {d['ctx_b'].capitalize()}")

    def choices(self, example: Example) -> list[str]:
        return [" " + self._clean(e) for e in example.data["endings"]]

    def gold_index(self, example: Example) -> int:
        return int(example.data["label"])
