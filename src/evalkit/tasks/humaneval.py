"""HumanEval: 164 hand-written Python problems scored by running unit tests.

The model sees a function signature and docstring and writes the body. We run
``prompt + completion + tests`` in a sandboxed subprocess. A problem counts as
solved only if every test passes, so correctness is checked functionally, not
by text similarity.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from evalkit.metrics.passk import pass_at_k
from evalkit.sandbox import run_python
from evalkit.tasks.base import GenerationTask
from evalkit.types import Example, ExampleResult


class HumanEval(GenerationTask):
    name = "humaneval"
    hf_path = "openai/openai_humaneval"
    eval_split = "test"
    fewshot_split = None
    primary_metric = "pass@1"
    max_new_tokens = 512
    # A new top-level statement means the function body is finished.
    stop = ("\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n```")

    def __init__(self, num_samples: int = 1, temperature: float = 0.0, k: tuple[int, ...] = (1,), **kwargs):
        super().__init__(**kwargs)
        # pass@k with k > 1 needs several *different* samples, i.e. temperature > 0.
        self.num_samples = num_samples
        self.temperature = temperature
        self.k = tuple(x for x in k if x <= num_samples)

    def example_id(self, index: int, row: dict) -> str:
        return row["task_id"]

    def build_prompt(self, example: Example) -> str:
        # Base models just continue the code. Instruct models (chat=True) get an
        # instruction, and their markdown fences are removed in score().
        if self.options.get("chat", False):
            return f"Complete the following Python function. Return only the code.\n\n```python\n{example.data['prompt']}```"
        return example.data["prompt"]

    def stop_sequences(self, chat: bool) -> tuple[str, ...]:
        # Chat models answer with a fenced block that itself starts with "def ...",
        # so the base-model stops would cut the answer off. We parse the fence instead.
        return () if chat else self.stop

    @staticmethod
    def extract_code(prompt: str, completion: str) -> str:
        """Build the full program from the problem prompt and the model's completion.

        * Base models continue the prompt, so we append the completion to it.
        * Instruct models usually re-write the whole function inside ```python
          fences. We take the first fenced block and still put the prompt in
          front of it: the prompt holds the imports (``from typing import List``),
          and the model's ``def`` simply overrides the stub.
        """
        # First fenced block: optional language tag, then everything up to the closing fence (or the end).
        match = re.search(r"```[\w+-]*\n(.*?)(?:```|$)", completion, flags=re.DOTALL)
        if match is None:
            return prompt + completion
        block = match.group(1)
        if "def " in block:
            # "pass" closes the docstring-only stub so the file stays valid Python.
            return prompt + "\n    pass\n\n" + block
        return prompt + block

    def score(self, example: Example, completions: list[str]) -> ExampleResult:
        d = example.data
        programs = [f"{self.extract_code(d['prompt'], c)}\n\n{d['test']}\n\ncheck({d['entry_point']})\n" for c in completions]
        # Each program is its own subprocess, so threads give real parallelism here.
        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(run_python, programs))
        passed = sum(o.passed for o in outcomes)
        per_k = {f"pass@{k}": pass_at_k(len(completions), passed, k) for k in self.k}
        return ExampleResult(
            id=example.id,
            score=per_k.get("pass@1", passed / len(completions)),
            prediction=completions[0],
            target=None,
            extra={**per_k, "num_passed": passed, "timeouts": sum(o.timed_out for o in outcomes)},
        )

    def aggregate(self, results: list[ExampleResult]) -> dict[str, float]:
        n = max(len(results), 1)
        return {f"pass@{k}": sum(r.extra[f"pass@{k}"] for r in results) / n for k in self.k}
