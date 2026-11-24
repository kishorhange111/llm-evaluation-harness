"""Tests for the calibrated LLM-as-judge rubric framework and local judge."""

from __future__ import annotations

import pytest

from evalkit.metrics.rubric import (
    DEFAULT_QUALITY_RUBRIC,
    DeterministicLocalJudge,
    calibrate_judge,
    evaluate_batch_with_rubric,
    evaluate_with_rubric,
    parse_rubric_output,
)


def test_parse_rubric_output_json_and_text_fallback():
    json_out = '```json\n{"scores": {"correctness": 5, "completeness": 4, "clarity": 5}, "rationale": "Strong."}\n```'
    v1 = parse_rubric_output(json_out, DEFAULT_QUALITY_RUBRIC)
    assert v1.criterion_scores == {"correctness": 5.0, "completeness": 4.0, "clarity": 5.0}
    assert v1.weighted_score == pytest.approx(4.7)
    assert v1.normalized_score == pytest.approx(0.925)

    text_out = "correctness: 2\ncompleteness: 3\nclarity: 4"
    v2 = parse_rubric_output(text_out, DEFAULT_QUALITY_RUBRIC)
    assert v2.criterion_scores["correctness"] == 2.0
    assert v2.criterion_scores["clarity"] == 4.0


def test_deterministic_local_judge_ranks_strong_above_weak_answer():
    judge = DeterministicLocalJudge(DEFAULT_QUALITY_RUBRIC)
    q = "What is KV caching in Transformer inference?"
    ref = "KV caching stores past key and value tensors during autoregressive decoding to avoid recomputing attention."
    good = "KV caching stores previously computed key and value tensors during autoregressive decoding so attention is not recomputed."
    poor = "Bananas are yellow."

    v_good = evaluate_with_rubric(judge, DEFAULT_QUALITY_RUBRIC, q, good, reference=ref)
    v_poor = evaluate_with_rubric(judge, DEFAULT_QUALITY_RUBRIC, q, poor, reference=ref)
    assert v_good.weighted_score > v_poor.weighted_score
    assert v_good.normalized_score > 0.7
    assert v_poor.normalized_score < 0.4


def test_evaluate_batch_with_rubric():
    judge = DeterministicLocalJudge()
    items = [
        {"question": "Define precision", "answer": "Precision is TP / (TP + FP).", "reference": "Precision is TP / (TP + FP)."},
        {"question": "Define recall", "answer": "Recall is TP / (TP + FN).", "reference": "Recall is TP / (TP + FN)."},
    ]
    res = evaluate_batch_with_rubric(judge, DEFAULT_QUALITY_RUBRIC, items)
    assert res["n"] == 2
    assert res["mean_normalized"] > 0.8


def test_calibrate_judge_recovers_bias_and_computes_alignment():
    # Judge consistently over-scores by +1 point relative to human labels.
    human = [1.0, 2.0, 3.0, 4.0, 4.0]
    judge = [2.0, 3.0, 4.0, 5.0, 5.0]
    rep = calibrate_judge(judge, human)
    assert rep.spearman_rho == pytest.approx(1.0)
    assert rep.bias == pytest.approx(1.0)
    assert rep.mean_abs_error == pytest.approx(1.0)
    assert rep.cohen_kappa > 0.5
    # Calibrated scores remove the +1 bias.
    assert rep.calibrated_scores == pytest.approx(human, abs=1e-6)
    assert rep.calibrate(4.0) == pytest.approx(3.0)

    with pytest.raises(ValueError):
        calibrate_judge([1.0], [1.0, 2.0])
