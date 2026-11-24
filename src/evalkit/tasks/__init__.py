"""Task registry: maps the names used in configs to task classes."""

from __future__ import annotations

from typing import Any

from evalkit.registry import TASK_REGISTRY, register_task
from evalkit.tasks.base import GenerationTask, MultipleChoiceTask, Task
from evalkit.tasks.custom import JsonlQA
from evalkit.tasks.gsm8k import GSM8K
from evalkit.tasks.humaneval import HumanEval
from evalkit.tasks.multiple_choice import MMLU, ARCChallenge, HellaSwag

_BUILTIN_TASKS: tuple[tuple[type[Task], str, tuple[str, ...]], ...] = (
    (GSM8K, "reasoning", ("math", "generation")),
    (MMLU, "knowledge", ("multiple-choice", "loglikelihood")),
    (ARCChallenge, "reasoning", ("science", "multiple-choice", "loglikelihood")),
    (HellaSwag, "commonsense", ("multiple-choice", "loglikelihood")),
    (HumanEval, "coding", ("code", "generation", "sandbox")),
    (JsonlQA, "custom", ("qa", "generation")),
)

for _cls, _category, _tags in _BUILTIN_TASKS:
    if _cls.name not in TASK_REGISTRY:
        TASK_REGISTRY.register(_cls.name, _cls, category=_category, tags=_tags)

TASKS: dict[str, type[Task]] = TASK_REGISTRY.as_dict()


def build_task(name: str, **kwargs: Any) -> Task:
    """Instantiate a registered task by name, e.g. ``build_task("gsm8k", limit=100)``."""
    task_cls = TASKS.get(name) if name in TASKS else TASK_REGISTRY.get(name)
    return task_cls(**kwargs)


__all__ = ["TASKS", "TASK_REGISTRY", "Task", "GenerationTask", "MultipleChoiceTask", "build_task", "register_task"]
