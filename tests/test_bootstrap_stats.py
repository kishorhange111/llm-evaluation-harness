"""Tests for BCa bootstrap CIs, permutation tests, effect sizes, and multiple-testing control."""

from __future__ import annotations

import math

import pytest

from evalkit.stats.bootstrap import (
    bca_bootstrap_ci,
    benjamini_hochberg,
    cohens_d_paired,
    compare_paired_metrics,
    holm_bonferroni,
    paired_permutation_test,
)


def test_bca_bootstrap_ci_covers_mean_and_handles_edge_cases():
    scores = [0.0] * 20 + [1.0] * 80
    ci = bca_bootstrap_ci(scores, n_resamples=4_000, seed=7)
    assert ci.mean == pytest.approx(0.8)
    assert ci.low < ci.mean < ci.high
    assert ci.low > 0.65 and ci.high < 0.92

    empty = bca_bootstrap_ci([])
    assert math.isnan(empty.mean) and empty.n == 0

    const = bca_bootstrap_ci([0.5, 0.5, 0.5])
    assert const.low == const.high == 0.5


def test_paired_permutation_test_detects_real_lift_and_ignores_ties():
    a = [0.0] * 30 + [1.0] * 20
    b = [0.0] * 10 + [1.0] * 40
    p_lift = paired_permutation_test(a, b, n_permutations=4_000, seed=42)
    assert p_lift < 0.01

    p_same = paired_permutation_test(a, a, n_permutations=1_000, seed=42)
    assert p_same == 1.0


def test_cohens_d_paired_and_compare_paired_metrics():
    a = [0.4, 0.5, 0.45, 0.55, 0.5, 0.48, 0.52, 0.49, 0.51, 0.47]
    b = [0.7, 0.75, 0.68, 0.80, 0.72, 0.74, 0.76, 0.71, 0.73, 0.69]
    d = cohens_d_paired(a, b)
    assert d > 3.0

    summary = compare_paired_metrics(a, b, n_resamples=3_000, seed=0)
    assert summary.significant
    assert summary.delta == pytest.approx(sum(y - x for x, y in zip(a, b)) / len(a))
    assert summary.p_bootstrap < 0.01
    assert summary.p_permutation < 0.01
    assert summary.effect_size_d == pytest.approx(d)


def test_holm_bonferroni_and_benjamini_hochberg():
    raw = [0.004, 0.02, 0.04, 0.40]
    hb = holm_bonferroni(raw, alpha=0.05)
    assert hb.adjusted_p_values[0] == pytest.approx(0.016)
    assert hb.rejected[0] is True
    assert hb.rejected[-1] is False

    bh = benjamini_hochberg(raw, alpha=0.05)
    assert bh.rejected == [True, True, False, False]
    assert bh.adjusted_p_values[1] == pytest.approx(0.04)
