"""Scoring functions, statistics and LLM-as-judge utilities."""

from evalkit.metrics.passk import pass_at_k
from evalkit.metrics.rag import (
    RAGEvalTask,
    RAGSample,
    RAGScore,
    answer_relevance,
    context_precision,
    context_recall,
    evaluate_rag_batch,
    evaluate_rag_sample,
    faithfulness,
)
from evalkit.metrics.stats import Estimate, bootstrap_ci, mcnemar_test, paired_bootstrap
from evalkit.metrics.text import exact_match, extract_choice_letter, extract_final_number, normalize_text, token_f1

__all__ = [
    "Estimate",
    "RAGEvalTask",
    "RAGSample",
    "RAGScore",
    "answer_relevance",
    "bootstrap_ci",
    "context_precision",
    "context_recall",
    "evaluate_rag_batch",
    "evaluate_rag_sample",
    "exact_match",
    "extract_choice_letter",
    "extract_final_number",
    "faithfulness",
    "mcnemar_test",
    "normalize_text",
    "paired_bootstrap",
    "pass_at_k",
    "token_f1",
]
