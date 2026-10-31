"""Task registry: maps the names used in configs to task classes."""

from __future__ import annotations

from typing import Any

from evalkit.tasks.base import GenerationTask, MultipleChoiceTask, Task
from evalkit.tasks.custom import JsonlQA
from evalkit.tasks.gsm8k import GSM8K
from evalkit.tasks.humaneval import HumanEval
from evalkit.tasks.multiple_choice import MMLU, ARCChallenge, HellaSwag

TASKS: dict[str, type[Task]] = {cls.name: cls for cls in (GSM8K, MMLU, ARCChallenge, HellaSwag, HumanEval, JsonlQA)}


def build_task(name: str, **kwargs: Any) -> Task:
    """Instantiate a registered task by name, e.g. ``build_task("gsm8k", limit=100)``."""
    try:
        return TASKS[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown task {name!r}; available: {', '.join(sorted(TASKS))}") from None


__all__ = ["TASKS", "Task", "GenerationTask", "MultipleChoiceTask", "build_task"]
