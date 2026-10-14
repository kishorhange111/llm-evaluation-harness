"""The unbiased pass@k estimator from the Codex paper (Chen et al., 2021).

pass@k = "if we sample k programs, what is the chance at least one passes the
tests?". The naive approach (sample exactly k, check if any pass) has very high
variance. Instead we sample n >= k programs, count the c that pass, and compute
the probability that a random size-k subset contains at least one passing one:

    pass@k = 1 - C(n - c, k) / C(n, k)
"""

from __future__ import annotations

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem.

    Args:
        n: Samples generated.
        c: Samples that passed all tests.
        k: The k in pass@k (must be <= n).
    """
    if k > n:
        raise ValueError(f"k={k} must be <= n={n}")
    if n - c < k:
        # Fewer than k failures exist, so every k-subset contains a pass.
        return 1.0
    # Product form of 1 - C(n-c, k)/C(n, k); avoids huge factorials and overflow.
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
