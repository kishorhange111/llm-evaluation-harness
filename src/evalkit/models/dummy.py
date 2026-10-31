"""A deterministic fake model for tests and dry runs.

It lets the whole pipeline (task loading -> prompting -> scoring -> stats ->
report) run in CI in milliseconds with no GPU and no downloads.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from evalkit.models.base import Model, truncate_at_stop
from evalkit.types import GenerationRequest, LoglikelihoodRequest, LoglikelihoodResult


def _stable_float(text: str) -> float:
    """Map a string to a repeatable number in [0, 1) (Python's ``hash`` is salted per process)."""
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class DummyModel(Model):
    """Answers with a user-supplied function, or a hash-based pseudo-random answer.

    Args:
        responder: Optional ``prompt -> completion`` function. Tests use this to
            build an "oracle" model that always answers correctly, or one that
            is right exactly half the time, and then check the reported metrics.
        name: Name written into result files.
    """

    def __init__(self, responder: Callable[[str], str] | None = None, name: str = "dummy"):
        self.responder = responder
        self.name = name
        # Counts calls so tests can verify that the cache really avoids re-querying.
        self.calls = 0

    def generate(self, requests: list[GenerationRequest]) -> list[list[str]]:
        outputs = []
        for req in requests:
            self.calls += 1
            samples = []
            for i in range(req.n):
                if self.responder is not None:
                    text = self.responder(req.prompt)
                else:
                    text = f"answer {int(_stable_float(req.prompt + str(i)) * 10)}"
                samples.append(truncate_at_stop(text, req.stop))
            outputs.append(samples)
        return outputs

    def loglikelihood(self, requests: list[LoglikelihoodRequest]) -> list[LoglikelihoodResult]:
        results = []
        for req in requests:
            self.calls += 1
            if self.responder is not None:
                # The responder says which continuation it "prefers" for this context;
                # give that one a high log-probability and everything else a low one.
                preferred = self.responder(req.context).strip()
                good = req.continuation.strip() == preferred
                lp = -0.1 if good else -5.0
            else:
                lp = -10 * _stable_float(req.context + "||" + req.continuation)
                good = False
            results.append(LoglikelihoodResult(logprob=lp, num_tokens=max(1, len(req.continuation.split())), is_greedy=good))
        return results
