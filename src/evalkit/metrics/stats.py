"""Statistics that make benchmark numbers trustworthy.

A single accuracy number hides how noisy it is. On a 500-question benchmark a
model at 60 % has a 95 % confidence interval of roughly +/- 4.3 points, so "model
B beats model A by 2 points" is often noise. This module provides:

* :func:`bootstrap_ci`: a confidence interval for one model's mean score.
* :func:`paired_bootstrap`: is model B better than model A on the *same* items?
* :func:`mcnemar_test`: an exact significance test for paired binary outcomes.

All functions are pure and take a seed, so reports are reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """A mean with a confidence interval, e.g. 0.62 [0.58, 0.66]."""

    mean: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{100 * self.mean:.1f} [{100 * self.low:.1f}, {100 * self.high:.1f}] (n={self.n})"


def bootstrap_ci(scores: list[float], confidence: float = 0.95, n_resamples: int = 10_000, seed: int = 0) -> Estimate:
    """Percentile bootstrap CI of the mean.

    We resample the per-example scores with replacement many times and take the
    spread of the resampled means. Unlike the normal approximation, this needs
    no assumptions and also works for non-binary scores like F1.
    """
    x = np.asarray(scores, dtype=float)
    if x.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    # Matrix of resample indices: (n_resamples, n). Vectorised, so 10k resamples take milliseconds.
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    means = x[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    return Estimate(float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)), int(x.size))


@dataclass(frozen=True)
class PairedComparison:
    """Result of comparing model B against model A on identical examples."""

    delta: float  # mean(B) - mean(A)
    low: float
    high: float
    p_value: float  # two-sided: how often the resampled delta crosses zero
    n: int

    @property
    def significant(self) -> bool:
        """True if the 95 % interval excludes zero."""
        return self.low > 0 or self.high < 0


def paired_bootstrap(a: list[float], b: list[float], n_resamples: int = 10_000, seed: int = 0) -> PairedComparison:
    """Paired bootstrap of the score difference B - A.

    "Paired" matters: both models answered the same questions, so we resample
    *questions* and keep each question's two scores together. That cancels
    out per-question difficulty and gives much tighter intervals than comparing
    two independent CIs.
    """
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if xa.shape != xb.shape:
        raise ValueError("paired comparison needs the same examples for both models")
    diff = xb - xa
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_resamples, diff.size))
    deltas = diff[idx].mean(axis=1)
    # Two-sided p-value: probability mass on the "wrong" side of zero, doubled.
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return PairedComparison(
        float(diff.mean()), float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)), float(min(p, 1.0)), int(diff.size)
    )


def mcnemar_test(a: list[float], b: list[float]) -> float:
    """Exact McNemar test for two models' correct/incorrect answers on the same items.

    Only *discordant* items carry information: ``n01`` (A wrong, B right) and
    ``n10`` (A right, B wrong). Under "no difference" each discordant item is a
    fair coin flip, so we compute an exact two-sided binomial p-value.

    Returns:
        The p-value; below 0.05 means the accuracy difference is unlikely to be chance.
    """
    n01 = sum(1 for x, y in zip(a, b) if x < 0.5 <= y)
    n10 = sum(1 for x, y in zip(a, b) if y < 0.5 <= x)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    # P(X <= k) for X ~ Binomial(n, 0.5), doubled for a two-sided test.
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)
