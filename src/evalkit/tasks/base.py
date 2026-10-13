"""Base classes for benchmarks.

There are two families of task, matching the two ways LLMs are evaluated:

* :class:`GenerationTask`: the model writes an answer (GSM8K, HumanEval, QA)
  and we parse and score it.
* :class:`MultipleChoiceTask`: the model never writes anything. We score
  ``log P(option | question)`` for every option and pick the most likely one
  (MMLU, ARC, HellaSwag). This is deterministic and immune to answer-parsing
  bugs, which is why leaderboards use it.

A subclass only describes *what* the benchmark is (how to load it, build a
prompt and score an answer). Batching, caching and statistics are handled once,
in the base classes and the runner.
"""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from typing import Any

from evalkit.models.base import Model
from evalkit.types import Example, ExampleResult, GenerationRequest, LoglikelihoodRequest


class Task(ABC):
    """Common behaviour: dataset loading, sub-sampling and few-shot selection.

    Args:
        limit: Evaluate only the first ``limit`` examples (after a seeded
            shuffle), which is handy for smoke tests and cheap iterations.
        num_fewshot: Number of solved examples shown before each question.
        seed: Controls the shuffle and few-shot sampling, for reproducibility.
        data_files: Path to a local JSONL file that replaces the Hugging Face
            dataset. Used by tests and for private or offline benchmarks.
    """

    #: Registry name, e.g. "gsm8k". Set by subclasses.
    name: str = "task"
    #: Hugging Face dataset coordinates: path, config name and the split we score.
    hf_path: str = ""
    hf_name: str | None = None
    eval_split: str = "test"
    fewshot_split: str | None = "train"
    #: Headline metric reported in summaries.
    primary_metric: str = "accuracy"

    def __init__(self, limit: int | None = None, num_fewshot: int = 0, seed: int = 1234, data_files: str | None = None, **kwargs: Any):
        self.limit = limit
        self.num_fewshot = num_fewshot
        self.seed = seed
        self.data_files = data_files
        self.options = kwargs  # task-specific knobs such as prompt_variant
        self._fewshot_pool: list[dict[str, Any]] | None = None

    # ----------------------------------------------------------------- loading
    def _load_rows(self, split: str) -> list[dict[str, Any]]:
        """Read raw rows from a local JSONL file or from the Hugging Face Hub."""
        if self.data_files:
            with open(self.data_files) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            # A local file may include a "split" column; otherwise all rows are the eval split.
            if rows and "split" in rows[0]:
                rows = [r for r in rows if r["split"] == split]
            elif split != self.eval_split:
                rows = []
            return rows
        from datasets import load_dataset  # lazy import: heavy, and not needed in unit tests

        return list(load_dataset(self.hf_path, self.hf_name, split=split))

    def examples(self) -> list[Example]:
        """Load, convert and (optionally) sub-sample the evaluation examples."""
        rows = self._load_rows(self.eval_split)
        examples = [Example(id=self.example_id(i, r), data=r) for i, r in enumerate(rows)]
        if self.limit is not None and self.limit < len(examples):
            # Shuffle before truncating so a "limit=200" run is a random sample, not
            # just the first 200 rows (which are often sorted by subject or difficulty).
            random.Random(self.seed).shuffle(examples)
            examples = sorted(examples[: self.limit], key=lambda e: e.id)
        return examples

    def example_id(self, index: int, row: dict[str, Any]) -> str:
        """Stable id for an example; subclasses override when the dataset has its own ids."""
        return f"{self.name}-{index}"

    def fewshot_examples(self, exclude_id: str | None = None) -> list[dict[str, Any]]:
        """Pick ``num_fewshot`` solved examples, the same ones for every question (seeded)."""
        if self.num_fewshot == 0 or self.fewshot_split is None:
            return []
        if self._fewshot_pool is None:
            pool = self._load_rows(self.fewshot_split)
            random.Random(self.seed).shuffle(pool)
            self._fewshot_pool = pool
        return self._fewshot_pool[: self.num_fewshot]

    # ------------------------------------------------------------------ scoring
    @abstractmethod
    def evaluate(self, model: Model, examples: list[Example]) -> list[ExampleResult]:
        """Run the model on ``examples`` and return one scored result each."""

    def aggregate(self, results: list[ExampleResult]) -> dict[str, float]:
        """Turn per-example scores into summary metrics (mean score by default)."""
        if not results:
            return {self.primary_metric: float("nan")}
        return {self.primary_metric: sum(r.score for r in results) / len(results)}


class GenerationTask(Task):
    """A task where the model writes free text that we then parse and score."""

    max_new_tokens: int = 256
    stop: tuple[str, ...] = ()
    temperature: float = 0.0
    num_samples: int = 1
    #: Wrap prompts with the model's chat template (for instruct models).
    use_chat_template: bool = False

    @abstractmethod
    def build_prompt(self, example: Example) -> str:
        """The text sent to the model for one example (before the chat template)."""

    def stop_sequences(self, chat: bool) -> tuple[str, ...]:
        """Stop strings for this run; tasks override this when chat output needs different ones."""
        return self.stop

    @abstractmethod
    def score(self, example: Example, completions: list[str]) -> ExampleResult:
        """Score the model's completion(s) for one example."""

    def evaluate(self, model: Model, examples: list[Example]) -> list[ExampleResult]:
        chat = self.options.get("chat", self.use_chat_template)
        requests = []
        for ex in examples:
            prompt = self.build_prompt(ex)
            if chat:
                prompt = model.apply_chat_template(prompt)
            requests.append(
                GenerationRequest(
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    stop=self.stop_sequences(chat),
                    temperature=self.temperature,
                    n=self.num_samples,
                )
            )
        completions = model.generate(requests)
        return [self.score(ex, outs) for ex, outs in zip(examples, completions)]


class MultipleChoiceTask(Task):
    """A task scored by comparing log-likelihoods of the answer options.

    Reported metrics:
        accuracy: argmax of the raw log-probability.
        accuracy_norm: argmax of log-probability divided by the option's token
            count. Raw log-probs favour short options (fewer tokens means fewer
            negative terms in the sum); normalising removes that bias, which
            matters a lot for HellaSwag-style tasks with long options.
    """

    primary_metric = "accuracy"

    @abstractmethod
    def build_context(self, example: Example) -> str:
        """Question text up to (not including) the answer, e.g. '...\\nAnswer:'."""

    @abstractmethod
    def choices(self, example: Example) -> list[str]:
        """The candidate continuations, e.g. [' A', ' B', ' C', ' D']."""

    @abstractmethod
    def gold_index(self, example: Example) -> int:
        """Index of the correct option in :meth:`choices`."""

    def evaluate(self, model: Model, examples: list[Example]) -> list[ExampleResult]:
        # Flatten every (example, option) pair into one list so the backend can batch
        # across examples, then regroup the answers afterwards.
        requests, spans = [], []
        for ex in examples:
            ctx, opts = self.build_context(ex), self.choices(ex)
            spans.append((len(requests), len(opts)))
            requests.extend(LoglikelihoodRequest(ctx, o) for o in opts)
        scored = model.loglikelihood(requests)

        results = []
        for ex, (start, count) in zip(examples, spans):
            lls = scored[start : start + count]
            raw = [r.logprob for r in lls]
            norm = [r.logprob / max(r.num_tokens, 1) for r in lls]
            pred = max(range(count), key=raw.__getitem__)
            pred_norm = max(range(count), key=norm.__getitem__)
            gold = self.gold_index(ex)
            results.append(
                ExampleResult(
                    id=ex.id,
                    score=float(pred == gold),
                    prediction=pred,
                    target=gold,
                    extra={"correct_norm": float(pred_norm == gold), "logprobs": raw},
                )
            )
        return results

    def aggregate(self, results: list[ExampleResult]) -> dict[str, float]:
        if not results:
            return {"accuracy": float("nan"), "accuracy_norm": float("nan")}
        n = len(results)
        return {
            "accuracy": sum(r.score for r in results) / n,
            "accuracy_norm": sum(r.extra["correct_norm"] for r in results) / n,
        }
