"""GSM8K: 1,319 grade-school math word problems that need multi-step reasoning.

The model is shown worked solutions (few-shot chain-of-thought) and must end
its answer with a number, which we extract and compare with the reference.
"""

from __future__ import annotations

import re

from evalkit.metrics.text import extract_final_number
from evalkit.tasks.base import GenerationTask
from evalkit.types import Example, ExampleResult


class GSM8K(GenerationTask):
    name = "gsm8k"
    hf_path = "openai/gsm8k"
    hf_name = "main"
    eval_split = "test"
    fewshot_split = "train"
    primary_metric = "accuracy"
    max_new_tokens = 384
    # Stop before the model starts inventing the next few-shot question.
    stop = ("\n\nQuestion:", "Question:", "</s>", "<|im_end|>")

    @staticmethod
    def reference_answer(answer_field: str) -> str:
        """GSM8K solutions end with '#### <number>'; that number is the label."""
        return extract_final_number(answer_field.split("####")[-1]) or ""

    def build_prompt(self, example: Example) -> str:
        parts = []
        for shot in self.fewshot_examples():
            # Reference solutions contain calculator annotations like <<3*4=12>>; drop them
            # so the few-shot examples look like natural reasoning.
            solution = re.sub(r"<<.*?>>", "", shot["answer"]).replace("####", "The answer is")
            parts.append(f"Question: {shot['question']}\nAnswer: {solution.strip()}")
        parts.append(f"Question: {example.data['question']}\nAnswer:")
        if self.options.get("chat", self.use_chat_template):
            # Instruct models answer better with an explicit instruction than with a bare prefix.
            parts.insert(0, "Solve the math problem step by step. End with 'The answer is <number>'.")
        return "\n\n".join(parts)

    def score(self, example: Example, completions: list[str]) -> ExampleResult:
        target = self.reference_answer(example.data["answer"])
        pred = extract_final_number(completions[0])
        return ExampleResult(
            id=example.id, score=float(pred == target), prediction=pred, target=target, extra={"completion": completions[0]}
        )
