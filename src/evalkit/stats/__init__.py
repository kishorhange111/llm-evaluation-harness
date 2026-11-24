"""Statistical inference utilities for reliable LLM benchmark comparisons."""

from evalkit.stats.bootstrap import (
    Estimate,
    MultipleTestResult,
    PairedComparison,
    SignificanceSummary,
    bca_bootstrap_ci,
    benjamini_hochberg,
    bootstrap_ci,
    cohens_d_paired,
    compare_paired_metrics,
    holm_bonferroni,
    mcnemar_test,
    paired_bootstrap,
    paired_permutation_test,
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
