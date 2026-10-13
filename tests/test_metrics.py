"""Unit tests for answer extraction, text metrics, pass@k and statistics."""

import math

import pytest

from evalkit.metrics import (
    bootstrap_ci,
    exact_match,
    extract_choice_letter,
    extract_final_number,
    mcnemar_test,
    paired_bootstrap,
    pass_at_k,
    token_f1,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("so the total is 12.\n#### 12", "12"),
        ("The answer is $1,234.", "1234"),
        ("\\boxed{42}", "42"),
        ("First 3 then 5, finally 8", "8"),  # falls back to the last number
        ("It costs 7.50 dollars", "7.5"),
        ("Result: 6.0", "6"),  # 6.0 and 6 must compare equal
        ("-3 degrees", "-3"),
        ("no digits here", None),
    ],
)
def test_extract_final_number(text, expected):
    assert extract_final_number(text) == expected


def test_explicit_answer_beats_trailing_numbers():
    # "the answer is" is more explicit than the last number in the text.
    assert extract_final_number("The answer is 10. Check: 2 * 5 = 10, step 3") == "10"


def test_exact_match_and_f1():
    assert exact_match("The Paris.", "paris") == 1.0
    assert exact_match("Lyon", "Paris") == 0.0
    assert token_f1("the big red apple", "red apple") == pytest.approx(0.8)
    assert token_f1("", "") == 1.0


def test_extract_choice_letter():
    assert extract_choice_letter("Answer: (C) because...") == "C"
    assert extract_choice_letter("I think B is right") == "B"
    assert extract_choice_letter("none") is None


def test_pass_at_k_known_values():
    assert pass_at_k(n=10, c=0, k=1) == 0.0
    assert pass_at_k(n=10, c=10, k=5) == 1.0
    assert pass_at_k(n=10, c=3, k=1) == pytest.approx(0.3)
    # 1 - C(7,2)/C(10,2) = 1 - 21/45
    assert pass_at_k(n=10, c=3, k=2) == pytest.approx(1 - 21 / 45)
    with pytest.raises(ValueError):
        pass_at_k(n=2, c=1, k=5)


def test_bootstrap_ci_contains_mean_and_matches_normal_approx():
    scores = [1.0] * 60 + [0.0] * 40
    est = bootstrap_ci(scores, seed=1)
    assert est.low < 0.6 < est.high
    # For a proportion the half-width should be close to 1.96 * sqrt(p(1-p)/n) ≈ 0.096.
    assert (est.high - est.low) / 2 == pytest.approx(1.96 * math.sqrt(0.24 / 100), abs=0.015)


def test_bootstrap_is_reproducible():
    s = [0, 1, 1, 0, 1]
    assert bootstrap_ci(s, seed=3) == bootstrap_ci(s, seed=3)


def test_paired_bootstrap_detects_clear_improvement():
    a = [0.0] * 50 + [1.0] * 50
    b = [1.0] * 100
    res = paired_bootstrap(a, b, seed=0)
    assert res.delta == pytest.approx(0.5)
    assert res.significant and res.p_value < 0.01


def test_paired_bootstrap_identical_models_not_significant():
    a = [1.0, 0.0] * 50
    res = paired_bootstrap(a, list(a))
    assert res.delta == 0 and not res.significant


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.0])


def test_mcnemar():
    # No discordant pairs: no evidence of a difference.
    assert mcnemar_test([1, 0, 1], [1, 0, 1]) == 1.0
    # 10 items where only B is right, 0 where only A is right: p = 2 * 0.5**10.
    a, b = [0] * 10, [1] * 10
    assert mcnemar_test(a, b) == pytest.approx(2 * 0.5**10)
