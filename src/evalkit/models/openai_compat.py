"""Backend for any server that speaks the OpenAI ``/v1/completions`` API.

That includes vLLM (``vllm serve <model>``), Hugging Face TGI, llama.cpp's
server and hosted APIs. Using an inference server instead of in-process
``transformers`` is typically 5-20x faster for generation-heavy tasks because of
continuous batching and paged attention.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from evalkit.models.base import Model, truncate_at_stop
from evalkit.types import GenerationRequest, LoglikelihoodRequest, LoglikelihoodResult


class OpenAICompatibleModel(Model):
    """Talk to an OpenAI-compatible completion endpoint.

    Args:
        model: Model name the server expects.
        base_url: e.g. ``http://localhost:8000/v1``.
        api_key_env: Name of the environment variable holding the key (never hard-code keys).
        concurrency: Parallel HTTP requests; the server batches them internally.
        max_retries: Retries with exponential backoff for 429/5xx/network errors.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key_env: str = "OPENAI_API_KEY",
        concurrency: int = 32,
        max_retries: int = 5,
        timeout: float = 120.0,
    ):
        self.name = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "EMPTY")  # vLLM accepts any key
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.timeout = timeout

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to ``/completions`` with retries; raises after ``max_retries`` failures."""
        body = json.dumps({"model": self.name, **payload}).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(f"{self.base_url}/completions", data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except (urllib.error.URLError, TimeoutError) as err:
                # Client errors other than rate limiting are bugs in the request: fail fast.
                code = getattr(err, "code", None)
                if code is not None and code < 500 and code != 429:
                    raise
                if attempt == self.max_retries:
                    raise
                time.sleep(min(2**attempt, 30))  # 1, 2, 4, 8 ... seconds
        raise RuntimeError("unreachable")

    def _map(self, fn, items):
        """Run ``fn`` over ``items`` concurrently, keeping the input order."""
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(fn, items))

    def generate(self, requests: list[GenerationRequest]) -> list[list[str]]:
        def one(req: GenerationRequest) -> list[str]:
            data = self._post(
                {
                    "prompt": req.prompt,
                    "max_tokens": req.max_new_tokens,
                    "temperature": req.temperature,
                    "n": req.n,
                    "stop": list(req.stop) or None,
                }
            )
            return [truncate_at_stop(c["text"], req.stop) for c in data["choices"]]

        return self._map(one, requests)

    def loglikelihood(self, requests: list[LoglikelihoodRequest]) -> list[LoglikelihoodResult]:
        """Score continuations with the ``echo=True, logprobs=1, max_tokens=0`` trick.

        The server returns the log-probability of every *prompt* token; we sum the
        ones that belong to the continuation (located by character offset).
        """

        def one(req: LoglikelihoodRequest) -> LoglikelihoodResult:
            data = self._post({"prompt": req.context + req.continuation, "max_tokens": 0, "echo": True, "logprobs": 1, "temperature": 0})
            lp = data["choices"][0]["logprobs"]
            start = len(req.context)
            total, n, greedy = 0.0, 0, True
            for offset, token_lp, top in zip(lp["text_offset"], lp["token_logprobs"], lp["top_logprobs"]):
                if offset < start or token_lp is None:
                    continue
                total += token_lp
                n += 1
                # The token is "greedy" if it is the most likely one at its position.
                if top and max(top.values()) > token_lp + 1e-6:
                    greedy = False
            return LoglikelihoodResult(total, max(n, 1), greedy)

        return self._map(one, requests)
