"""RAG evaluation metrics: faithfulness, answer relevance, context precision & recall.

Designed to run deterministically offline without external API keys (suitable for
CI regression gates), while matching the standard RAGAS/DeepEval metric definitions:

* **Faithfulness**: fraction of atomic claims in the answer supported by the
  retrieved context passages.
* **Answer relevance**: semantic/lexical alignment between the question and answer.
* **Context precision**: rank-aware Mean Average Precision (MAP@K) rewarding
  relevant passages placed earlier in the retrieved list.
* **Context recall**: fraction of ground-truth statements attributable to the
  retrieved contexts.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from evalkit.registry import register_metric, register_task
from evalkit.tasks.base import GenerationTask
from evalkit.types import Example, ExampleResult

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "this",
        "these",
        "those",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "does",
        "do",
        "did",
        "can",
        "could",
        "would",
        "should",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _content_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric content tokens excluding common stopwords."""
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


def _split_claims(text: str) -> list[str]:
    """Split text into non-empty sentence-level claims."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if len(_content_tokens(s)) > 0]


def _cosine_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Bag-of-words cosine similarity in [0, 1]."""
    if not tokens_a or not tokens_b:
        return 0.0
    ca, cb = Counter(tokens_a), Counter(tokens_b)
    dot = sum(ca[t] * cb[t] for t in ca.keys() & cb.keys())
    norm_a = math.sqrt(sum(v * v for v in ca.values()))
    norm_b = math.sqrt(sum(v * v for v in cb.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _claim_support(claim: str, context_tokens: set[str]) -> float:
    """Fraction of content tokens in ``claim`` present in ``context_tokens``."""
    tokens = _content_tokens(claim)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in context_tokens) / len(tokens)


@dataclass(frozen=True)
class RAGSample:
    """One RAG evaluation instance."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


@dataclass(frozen=True)
class RAGScore:
    """Per-sample RAG evaluation scores."""

    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    composite: float
    details: dict[str, Any] = field(default_factory=dict)


@register_metric("rag_faithfulness", category="rag", tags=("rag", "groundedness"))
def faithfulness(answer: str, contexts: list[str], support_threshold: float = 0.6) -> float:
    """Fraction of answer claims grounded in the retrieved contexts."""
    claims = _split_claims(answer)
    if not claims:
        return 0.0
    ctx_tokens: set[str] = set()
    for ctx in contexts:
        ctx_tokens.update(_content_tokens(ctx))
    if not ctx_tokens:
        return 0.0
    supported = sum(1 for c in claims if _claim_support(c, ctx_tokens) >= support_threshold)
    return supported / len(claims)


@register_metric("rag_answer_relevance", category="rag", tags=("rag", "relevance"))
def answer_relevance(question: str, answer: str) -> float:
    """Relevance of ``answer`` to ``question`` combining token recall and cosine similarity."""
    q_tokens = _content_tokens(question)
    a_tokens = _content_tokens(answer)
    if not q_tokens or not a_tokens:
        return 0.0
    q_set = set(q_tokens)
    a_set = set(a_tokens)
    coverage = len(q_set & a_set) / len(q_set)
    cosine = _cosine_similarity(q_tokens, a_tokens)
    # Blend question coverage and distributional similarity, bounded in [0, 1].
    return min(1.0, 0.6 * coverage + 0.4 * cosine)


@register_metric("rag_context_precision", category="rag", tags=("rag", "retrieval"))
def context_precision(
    contexts: list[str],
    ground_truth: str,
    question: str = "",
    relevance_threshold: float = 0.25,
) -> float:
    """Rank-aware precision (Average Precision@K) of retrieved contexts.

    A context is marked relevant when its token overlap with ``ground_truth``
    (or ``question`` if ``ground_truth`` is empty) meets ``relevance_threshold``.
    """
    if not contexts:
        return 0.0
    ref_tokens = set(_content_tokens(ground_truth or question))
    if not ref_tokens:
        return 0.0

    relevant_flags: list[bool] = []
    for ctx in contexts:
        ctx_tokens = set(_content_tokens(ctx))
        if not ctx_tokens:
            relevant_flags.append(False)
            continue
        overlap = len(ref_tokens & ctx_tokens) / len(ref_tokens)
        relevant_flags.append(overlap >= relevance_threshold)

    hits = 0
    precision_sum = 0.0
    for rank, is_rel in enumerate(relevant_flags, start=1):
        if is_rel:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / hits if hits > 0 else 0.0


@register_metric("rag_context_recall", category="rag", tags=("rag", "retrieval"))
def context_recall(contexts: list[str], ground_truth: str, support_threshold: float = 0.5) -> float:
    """Fraction of ground-truth claims attributable to the retrieved contexts."""
    claims = _split_claims(ground_truth)
    if not claims:
        return 0.0
    ctx_tokens: set[str] = set()
    for ctx in contexts:
        ctx_tokens.update(_content_tokens(ctx))
    if not ctx_tokens:
        return 0.0
    covered = sum(1 for c in claims if _claim_support(c, ctx_tokens) >= support_threshold)
    return covered / len(claims)


def evaluate_rag_sample(
    sample: RAGSample,
    *,
    weights: tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20),
) -> RAGScore:
    """Compute all four RAG metrics and their weighted composite score."""
    f_score = faithfulness(sample.answer, sample.contexts)
    ar_score = answer_relevance(sample.question, sample.answer)
    cp_score = context_precision(sample.contexts, sample.ground_truth, question=sample.question)
    cr_score = context_recall(sample.contexts, sample.ground_truth) if sample.ground_truth else 0.0

    w_f, w_ar, w_cp, w_cr = weights
    if sample.ground_truth:
        total_w = w_f + w_ar + w_cp + w_cr
        comp = (w_f * f_score + w_ar * ar_score + w_cp * cp_score + w_cr * cr_score) / total_w
    else:
        total_w = w_f + w_ar + w_cp
        comp = (w_f * f_score + w_ar * ar_score + w_cp * cp_score) / total_w

    claims = _split_claims(sample.answer)
    ctx_tokens: set[str] = set()
    for ctx in sample.contexts:
        ctx_tokens.update(_content_tokens(ctx))
    unsupported = [c for c in claims if _claim_support(c, ctx_tokens) < 0.6]

    return RAGScore(
        faithfulness=f_score,
        answer_relevance=ar_score,
        context_precision=cp_score,
        context_recall=cr_score,
        composite=comp,
        details={"num_claims": len(claims), "unsupported_claims": unsupported, "num_contexts": len(sample.contexts)},
    )


def evaluate_rag_batch(samples: list[RAGSample]) -> dict[str, Any]:
    """Evaluate a batch of RAG samples and return aggregate means plus per-sample scores."""
    scores = [evaluate_rag_sample(s) for s in samples]
    if not scores:
        nan = float("nan")
        return {
            "faithfulness": nan,
            "answer_relevance": nan,
            "context_precision": nan,
            "context_recall": nan,
            "composite": nan,
            "n": 0,
            "samples": [],
        }
    n = len(scores)
    return {
        "faithfulness": sum(s.faithfulness for s in scores) / n,
        "answer_relevance": sum(s.answer_relevance for s in scores) / n,
        "context_precision": sum(s.context_precision for s in scores) / n,
        "context_recall": sum(s.context_recall for s in scores) / n,
        "composite": sum(s.composite for s in scores) / n,
        "n": n,
        "samples": scores,
    }


@register_task("rag_eval", category="rag", tags=("rag", "generation", "groundedness"))
class RAGEvalTask(GenerationTask):
    """Evaluate grounded generation over retrieved contexts using RAG metrics.

    Expects local JSONL rows with ``question``, ``contexts`` (list of strings),
    and optional ``ground_truth`` / ``answer``.
    """

    name = "rag_eval"
    primary_metric = "rag_composite"
    eval_split = "test"
    fewshot_split = None

    def build_prompt(self, example: Example) -> str:
        contexts = example.data.get("contexts", [])
        ctx_block = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        return (
            "Answer the question using only the provided context.\n\n"
            f"Context:\n{ctx_block}\n\n"
            f"Question: {example.data['question']}\nAnswer:"
        )

    def score(self, example: Example, completions: list[str]) -> ExampleResult:
        answer = completions[0].strip() if completions else ""
        sample = RAGSample(
            question=example.data["question"],
            answer=answer,
            contexts=list(example.data.get("contexts", [])),
            ground_truth=str(example.data.get("ground_truth", example.data.get("answer", ""))),
        )
        rag = evaluate_rag_sample(sample)
        return ExampleResult(
            id=example.id,
            score=rag.composite,
            prediction=answer,
            target=sample.ground_truth,
            extra={
                "faithfulness": rag.faithfulness,
                "answer_relevance": rag.answer_relevance,
                "context_precision": rag.context_precision,
                "context_recall": rag.context_recall,
                "unsupported_claims": rag.details["unsupported_claims"],
            },
        )

    def aggregate(self, results: list[ExampleResult]) -> dict[str, float]:
        if not results:
            nan = float("nan")
            return {
                "rag_composite": nan,
                "faithfulness": nan,
                "answer_relevance": nan,
                "context_precision": nan,
                "context_recall": nan,
            }
        n = len(results)
        return {
            "rag_composite": sum(r.score for r in results) / n,
            "faithfulness": sum(r.extra["faithfulness"] for r in results) / n,
            "answer_relevance": sum(r.extra["answer_relevance"] for r in results) / n,
            "context_precision": sum(r.extra["context_precision"] for r in results) / n,
            "context_recall": sum(r.extra["context_recall"] for r in results) / n,
        }


__all__ = [
    "RAGEvalTask",
    "RAGSample",
    "RAGScore",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "evaluate_rag_batch",
    "evaluate_rag_sample",
    "faithfulness",
]
