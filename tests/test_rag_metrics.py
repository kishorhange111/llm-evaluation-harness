"""Tests for the RAG evaluation metrics suite and RAGEvalTask."""

from __future__ import annotations

import json

import pytest

from evalkit.metrics.rag import (
    RAGEvalTask,
    RAGSample,
    answer_relevance,
    context_precision,
    context_recall,
    evaluate_rag_batch,
    evaluate_rag_sample,
    faithfulness,
)
from evalkit.runner import run
from evalkit.tasks import build_task


def test_faithfulness_distinguishes_grounded_from_hallucinated():
    contexts = [
        "FastAPI is a modern Python web framework for building APIs with type hints.",
        "Uvicorn is an ASGI server commonly used to run FastAPI applications.",
    ]
    grounded = "FastAPI is a Python web framework that uses type hints. Uvicorn runs FastAPI applications as an ASGI server."
    hallucinated = "FastAPI was invented in 1995 for COBOL mainframes. It requires Java 8 to compile."

    assert faithfulness(grounded, contexts) == pytest.approx(1.0)
    assert faithfulness(hallucinated, contexts) == pytest.approx(0.0)
    assert faithfulness("", contexts) == 0.0
    assert faithfulness(grounded, []) == 0.0


def test_answer_relevance_scores_on_topic_above_off_topic():
    q = "How does LoRA reduce fine-tuning memory usage?"
    good = "LoRA reduces fine-tuning memory usage by freezing base weights and training low-rank adapter matrices."
    off_topic = "Bananas grow in tropical climates and are rich in potassium."

    assert answer_relevance(q, good) > 0.6
    assert answer_relevance(q, off_topic) == pytest.approx(0.0)


def test_context_precision_rewards_higher_ranking():
    gt = "Transformers use self-attention to model token dependencies."
    rel = "Self-attention lets Transformers model long-range token dependencies in parallel."
    irrel_1 = "PostgreSQL supports ACID transactions and JSONB columns."
    irrel_2 = "Docker containers package applications with their runtime dependencies."

    top_ranked = context_precision([rel, irrel_1, irrel_2], gt)
    bottom_ranked = context_precision([irrel_1, irrel_2, rel], gt)

    assert top_ranked == pytest.approx(1.0)
    assert bottom_ranked == pytest.approx(1 / 3)
    assert top_ranked > bottom_ranked
    assert context_precision([], gt) == 0.0


def test_context_recall_measures_ground_truth_coverage():
    gt = "Redis is an in-memory data store. It supports pub-sub messaging and TTL expiration."
    full_ctx = ["Redis is an in-memory data store supporting pub-sub messaging and TTL expiration."]
    partial_ctx = ["Redis is an in-memory data store."]

    assert context_recall(full_ctx, gt) == pytest.approx(1.0)
    assert context_recall(partial_ctx, gt) == pytest.approx(0.5)


def test_evaluate_rag_sample_and_batch():
    s1 = RAGSample(
        question="What is FAISS used for?",
        answer="FAISS is used for fast dense vector similarity search.",
        contexts=["FAISS is a library for fast dense vector similarity search and clustering."],
        ground_truth="FAISS is used for dense vector similarity search.",
    )
    s2 = RAGSample(
        question="What is FAISS used for?",
        answer="Mars has two small moons named Phobos and Deimos.",
        contexts=["FAISS is a library for fast dense vector similarity search."],
        ground_truth="FAISS is used for dense vector similarity search.",
    )
    r1 = evaluate_rag_sample(s1)
    r2 = evaluate_rag_sample(s2)
    assert r1.composite > r2.composite
    assert r2.details["unsupported_claims"]

    batch = evaluate_rag_batch([s1, s2])
    assert batch["n"] == 2
    assert batch["faithfulness"] == pytest.approx(0.5)


def test_rag_eval_task_runs_end_to_end(tmp_path):
    data_path = tmp_path / "rag.jsonl"
    rows = [
        {
            "question": "What does BM25 score?",
            "contexts": ["BM25 scores lexical relevance between a query and candidate documents."],
            "ground_truth": "BM25 scores lexical relevance between queries and documents.",
        },
        {
            "question": "Why use cross-encoders?",
            "contexts": ["Cross-encoders jointly attend to query and passage for high-precision reranking."],
            "ground_truth": "Cross-encoders perform high-precision reranking by jointly attending to query and passage.",
        },
    ]
    data_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def oracle(prompt: str) -> str:
        if "BM25" in prompt:
            return "BM25 scores lexical relevance between a query and candidate documents."
        return "Cross-encoders jointly attend to query and passage for high-precision reranking."

    task = build_task("rag_eval", data_files=str(data_path))
    assert isinstance(task, RAGEvalTask)
    summary = run(
        {
            "run_name": "rag-smoke",
            "models": [{"backend": "dummy", "name": "rag-oracle", "responder": oracle}],
            "tasks": [{"name": "rag_eval", "data_files": str(data_path)}],
        },
        output_dir=tmp_path / "out",
    )
    metrics = summary["results"]["rag-oracle"]["rag_eval"]
    assert metrics["faithfulness"] == pytest.approx(1.0)
    assert metrics["context_recall"] == pytest.approx(1.0)
    assert metrics["rag_composite"] > 0.8
