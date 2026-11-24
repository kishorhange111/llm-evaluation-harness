"""Statistical significance, BCa bootstrap CIs, effect sizes, and multiple-testing control.

Extends :mod:`evalkit.metrics.stats` with:

* :func:`bca_bootstrap_ci`: Bias-Corrected and Accelerated (BCa) bootstrap
  confidence intervals using jackknife acceleration for skewed score distributions.
* :func:`paired_permutation_test`: Monte-Carlo paired sign-flip permutation test.
* :func:`cohens_d_paired`: paired effect size (:math:`d_z`).
* :func:`holm_bonferroni` and :func:`benjamini_hochberg`: family-wise error rate
  (FWER) and false discovery rate (FDR) control across multi-benchmark suites.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from evalkit.metrics.stats import Estimate, PairedComparison, bootstrap_ci, mcnemar_test, paired_bootstrap

_NORM = NormalDist()


def bca_bootstrap_ci(
    scores: list[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Estimate:
    """Bias-Corrected and Accelerated (BCa) bootstrap confidence interval of the mean."""
    x = np.asarray(scores, dtype=float)
    n = int(x.size)
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    theta_hat = float(x.mean())
    if n == 1 or float(np.std(x)) == 0.0:
        return Estimate(theta_hat, theta_hat, theta_hat, n)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = x[idx].mean(axis=1)

    # Bias-correction factor z0.
    prop_less = float((boot_means < theta_hat).mean())
    eps = 1.0 / (2.0 * n_resamples)
    prop_less = min(max(prop_less, eps), 1.0 - eps)
    z0 = _NORM.inv_cdf(prop_less)

    # Acceleration factor a via leave-one-out jackknife.
    total = float(x.sum())
    jack = (total - x) / (n - 1)
    jack_mean = float(jack.mean())
    diff = jack_mean - jack
    denom = float(np.sum(diff**2)) ** 1.5
    acc = float(np.sum(diff**3) / (6.0 * denom)) if denom > 0.0 else 0.0

    alpha = (1.0 - confidence) / 2.0
    z_low = _NORM.inv_cdf(alpha)
    z_high = _NORM.inv_cdf(1.0 - alpha)

    def _adj_quantile(z_alpha: float) -> float:
        denom_z = 1.0 - acc * (z0 + z_alpha)
        if abs(denom_z) < 1e-12:
            return _NORM.cdf(z_alpha)
        return _NORM.cdf(z0 + (z0 + z_alpha) / denom_z)

    q_low = min(max(_adj_quantile(z_low), 0.0), 1.0)
    q_high = min(max(_adj_quantile(z_high), 0.0), 1.0)
    low, high = float(np.quantile(boot_means, q_low)), float(np.quantile(boot_means, q_high))
    return Estimate(theta_hat, min(low, high), max(low, high), n)


def paired_permutation_test(
    a: list[float],
    b: list[float],
    n_permutations: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided paired sign-flip permutation test for mean(b) - mean(a)."""
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if xa.shape != xb.shape:
        raise ValueError("paired permutation test requires equal-length arrays")
    if xa.size == 0:
        return 1.0
    diff = xb - xa
    obs = abs(float(diff.mean()))
    if obs == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, diff.size))
    perm_means = np.abs((signs * diff).mean(axis=1))
    # Add-one smoothing prevents zero p-values on finite Monte-Carlo draws.
    return float((int((perm_means >= obs - 1e-12).sum()) + 1) / (n_permutations + 1))


def cohens_d_paired(a: list[float], b: list[float]) -> float:
    """Paired effect size Cohen's d_z = mean(b - a) / sd(b - a)."""
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if xa.shape != xb.shape:
        raise ValueError("cohens_d_paired requires equal-length arrays")
    if xa.size < 2:
        return 0.0
    diff = xb - xa
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        mean_d = float(diff.mean())
        return 0.0 if mean_d == 0.0 else math.copysign(float("inf"), mean_d)
    return float(diff.mean() / sd)


@dataclass(frozen=True)
class MultipleTestResult:
    """Adjusted p-values and rejection flags from multiple-hypothesis correction."""

    method: str
    alpha: float
    raw_p_values: list[float]
    adjusted_p_values: list[float]
    rejected: list[bool]


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> MultipleTestResult:
    """Holm-Bonferroni step-down FWER correction."""
    m = len(p_values)
    if m == 0:
        return MultipleTestResult("holm_bonferroni", alpha, [], [], [])
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, p_values[idx] * (m - rank))
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    rejected = [p <= alpha for p in adjusted]
    return MultipleTestResult("holm_bonferroni", alpha, list(p_values), adjusted, rejected)


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> MultipleTestResult:
    """Benjamini-Hochberg step-up FDR correction."""
    m = len(p_values)
    if m == 0:
        return MultipleTestResult("benjamini_hochberg", alpha, [], [], [])
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = min(1.0, p_values[idx] * m / (rank + 1))
        running_min = min(running_min, val)
        adjusted[idx] = running_min
    rejected = [p <= alpha for p in adjusted]
    return MultipleTestResult("benjamini_hochberg", alpha, list(p_values), adjusted, rejected)


@dataclass(frozen=True)
class SignificanceSummary:
    """Comprehensive paired statistical comparison between baseline A and candidate B."""

    delta: float
    ci_low: float
    ci_high: float
    p_bootstrap: float
    p_permutation: float
    p_mcnemar: float
    effect_size_d: float
    significant: bool
    n: int


def compare_paired_metrics(
    a: list[float],
    b: list[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 5_000,
    seed: int = 0,
) -> SignificanceSummary:
    """Compute BCa CI on paired differences, bootstrap/permutation/McNemar p-values, and Cohen's d."""
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if xa.shape != xb.shape:
        raise ValueError("compare_paired_metrics requires equal-length arrays")
    diff = (xb - xa).tolist()
    ci = bca_bootstrap_ci(diff, confidence=confidence, n_resamples=n_resamples, seed=seed)
    pb = paired_bootstrap(a, b, n_resamples=n_resamples, seed=seed)
    p_perm = paired_permutation_test(a, b, n_permutations=n_resamples, seed=seed)
    p_mcn = mcnemar_test(a, b)
    d = cohens_d_paired(a, b)
    sig = ci.low > 0.0 or ci.high < 0.0
    return SignificanceSummary(
        delta=ci.mean,
        ci_low=ci.low,
        ci_high=ci.high,
        p_bootstrap=pb.p_value,
        p_permutation=p_perm,
        p_mcnemar=p_mcn,
        effect_size_d=d,
        significant=sig,
        n=ci.n,
    )


__all__ = [
    "Estimate",
    "MultipleTestResult",
    "PairedComparison",
    "SignificanceSummary",
    "bca_bootstrap_ci",
    "benjamini_hochberg",
    "bootstrap_ci",
    "cohens_d_paired",
    "compare_paired_metrics",
    "holm_bonferroni",
    "mcnemar_test",
    "paired_bootstrap",
    "paired_permutation_test",
]
