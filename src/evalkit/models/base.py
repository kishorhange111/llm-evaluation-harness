"""The interface every model backend implements.

Tasks only ever talk to this interface, so the same benchmark code runs against
a local Hugging Face model, a vLLM server, or a fake model in unit tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from evalkit.types import GenerationRequest, LoglikelihoodRequest, LoglikelihoodResult


class Model(ABC):
    """Abstract model backend.

    Both methods take a *list* of requests rather than one at a time, so
    backends can batch them (one GPU forward pass for many prompts is far
    faster than many passes of one prompt each).
    """

    #: Human-readable identifier written into results files, e.g. "Qwen/Qwen2.5-1.5B-Instruct".
    name: str = "model"

    @abstractmethod
    def generate(self, requests: list[GenerationRequest]) -> list[list[str]]:
        """Generate text for each request.

        Returns:
            One list per request holding ``request.n`` completions. Stop strings
            must already be cut off, so tasks receive clean answers.
        """

    @abstractmethod
    def loglikelihood(self, requests: list[LoglikelihoodRequest]) -> list[LoglikelihoodResult]:
        """Score each (context, continuation) pair; see :class:`LoglikelihoodRequest`."""

    def apply_chat_template(self, user_message: str) -> str:
        """Wrap a single user turn in the model's chat format.

        The default returns the message unchanged, which is correct for base
        models. Instruct backends override this so prompts are formatted the
        way the model was trained (forgetting this can cost 10+ accuracy points).
        """
        return user_message


def truncate_at_stop(text: str, stop: tuple[str, ...]) -> str:
    """Cut ``text`` at the earliest occurrence of any stop string.

    Backends call this after decoding because not every engine supports stop
    strings natively, and the ones that do sometimes leave the stop text in.

    >>> truncate_at_stop("42\\n\\nQuestion: next", ("\\n\\nQuestion:",))
    '42'
    """
    cut = len(text)
    for s in stop:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]
