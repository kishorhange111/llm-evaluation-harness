"""A persistent SQLite cache for model calls.

Evaluation runs get interrupted (Colab disconnects, OOM in the last task) and
are re-run with small changes (a new metric, a fixed parser). Caching every
model call means a re-run only pays for requests it has never seen. Re-scoring
10k MMLU questions with a fixed metric then takes seconds instead of GPU-hours.

The key is a SHA-256 hash of (model name, request type, every request field),
so changing the prompt, stop strings or temperature correctly misses the cache.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pickle
import sqlite3
from pathlib import Path

from evalkit.models.base import Model
from evalkit.types import GenerationRequest, LoglikelihoodRequest, LoglikelihoodResult


def request_key(model_name: str, request: GenerationRequest | LoglikelihoodRequest) -> str:
    """Deterministic cache key for one request (independent of Python's salted ``hash``)."""
    payload = json.dumps([model_name, type(request).__name__, dataclasses.asdict(request)], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class CachedModel(Model):
    """Wraps another :class:`Model`; looks up each request before calling the real model.

    Only cache misses are forwarded, still as one batched call, so batching
    efficiency is preserved.
    """

    def __init__(self, inner: Model, path: str | Path):
        self.inner = inner
        self.name = inner.name
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the OpenAI backend may score from worker threads.
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB)")
        self.hits = 0
        self.misses = 0

    def apply_chat_template(self, user_message: str) -> str:
        return self.inner.apply_chat_template(user_message)

    def _lookup(self, keys: list[str]) -> dict[str, object]:
        found = {}
        # SQLite caps bound parameters per query, so look keys up in chunks.
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            q = f"SELECT key, value FROM cache WHERE key IN ({','.join('?' * len(chunk))})"
            found.update({k: pickle.loads(v) for k, v in self.db.execute(q, chunk)})
        return found

    def _cached_call(self, requests, fn):
        """Shared logic: answer hits from the DB, send misses to ``fn`` in one batch, store them."""
        keys = [request_key(self.name, r) for r in requests]
        found = self._lookup(keys)
        missing = [i for i, k in enumerate(keys) if k not in found]
        self.hits += len(requests) - len(missing)
        self.misses += len(missing)
        if missing:
            fresh = fn([requests[i] for i in missing])
            rows = [(keys[i], pickle.dumps(v)) for i, v in zip(missing, fresh)]
            self.db.executemany("INSERT OR REPLACE INTO cache VALUES (?, ?)", rows)
            self.db.commit()  # commit per batch: a crash loses at most one batch of work
            found.update({keys[i]: v for i, v in zip(missing, fresh)})
        return [found[k] for k in keys]

    def generate(self, requests: list[GenerationRequest]) -> list[list[str]]:
        return self._cached_call(requests, self.inner.generate)

    def loglikelihood(self, requests: list[LoglikelihoodRequest]) -> list[LoglikelihoodResult]:
        return self._cached_call(requests, self.inner.loglikelihood)
