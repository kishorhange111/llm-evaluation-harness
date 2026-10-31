"""Plain data types shared by models, tasks and the runner.

Keeping these as small frozen dataclasses (instead of dicts) gives us type
checking, readable reprs in logs, and stable hashing for the response cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationRequest:
    """Ask the model to continue ``prompt`` with free-form text.

    Attributes:
        prompt: The full text (or rendered chat template) sent to the model.
        max_new_tokens: Hard cap on generated length, so a rambling model can't
            stall the run.
        stop: Strings that end generation early, e.g. ``"\\n\\nQuestion:"`` in a
            few-shot prompt so the model doesn't invent the next question.
        temperature: 0.0 means greedy decoding, which keeps results reproducible.
        n: Number of samples. ``n > 1`` with temperature > 0 is used for pass@k.
    """

    prompt: str
    max_new_tokens: int = 256
    stop: tuple[str, ...] = ()
    temperature: float = 0.0
    n: int = 1


@dataclass(frozen=True)
class LoglikelihoodRequest:
    """Ask the model how likely ``continuation`` is after ``context``.

    This is how multiple-choice benchmarks (MMLU, ARC) are usually scored: we do
    not let the model write an answer, we compare log P(option | question) for
    each option and pick the highest. It is deterministic and needs no answer
    parsing.
    """

    context: str
    continuation: str


@dataclass(frozen=True)
class LoglikelihoodResult:
    """Result of a :class:`LoglikelihoodRequest`.

    Attributes:
        logprob: Sum of token log-probabilities of the continuation.
        num_tokens: How many tokens the continuation took. Used for
            length-normalised scoring (``logprob / num_tokens``), which removes
            the bias towards short answer options.
        is_greedy: True if every continuation token was also the model's argmax
            prediction, i.e. greedy decoding would have produced it exactly.
    """

    logprob: float
    num_tokens: int
    is_greedy: bool


@dataclass
class Example:
    """One benchmark item after loading.

    Attributes:
        id: Stable identifier, used for caching and for paired statistics
            between two models (we must compare the *same* questions).
        data: The raw fields of the dataset row (question, choices, answer...).
    """

    id: str
    data: dict[str, Any]


@dataclass
class ExampleResult:
    """Everything we record for one example, so any score can be audited later."""

    id: str
    score: float
    prediction: Any
    target: Any
    extra: dict[str, Any] = field(default_factory=dict)
