"""Tests for LLM-as-judge debiasing, the code sandbox and the response cache."""

from evalkit.cache import CachedModel
from evalkit.metrics.judge import combine_verdicts, judge_pairwise, parse_verdict, win_rate
from evalkit.models import DummyModel
from evalkit.sandbox import run_python
from evalkit.types import GenerationRequest, LoglikelihoodRequest

# ------------------------------------------------------------------- judge


def test_parse_verdict_uses_last_tag():
    assert parse_verdict("Format: [[A]] if A better... Final: [[B]]") == "B"
    assert parse_verdict("no verdict") is None


def test_combine_verdicts_requires_agreement_after_swap():
    # B wins in both orders (in the swapped prompt B is shown as "A").
    assert combine_verdicts("B", "A").winner == "B"
    # The judge always picks whichever answer is shown first: position bias, so a tie.
    out = combine_verdicts("A", "A")
    assert out.winner == "tie" and out.position_flip


def test_judge_pairwise_with_position_biased_judge():
    biased = DummyModel(responder=lambda prompt: "I prefer the first. [[A]]")
    outcomes = judge_pairwise(biased, ["q1", "q2"], ["a1", "a2"], ["b1", "b2"])
    assert all(o.winner == "tie" for o in outcomes)
    assert win_rate(outcomes, "B") == 0.5


def test_judge_pairwise_with_consistent_judge():
    # This judge always prefers the answer containing "good", wherever it is shown.
    def judge(prompt):
        a_part = prompt.split("[Assistant A]")[1].split("[Assistant B]")[0]
        return "[[A]]" if "good" in a_part else "[[B]]"

    outcomes = judge_pairwise(DummyModel(responder=judge), ["q"], ["bad"], ["good"])
    assert outcomes[0].winner == "B" and not outcomes[0].position_flip


# ----------------------------------------------------------------- sandbox


def test_sandbox_pass_fail_and_timeout():
    assert run_python("assert 1 + 1 == 2").passed
    failed = run_python("assert 1 + 1 == 3")
    assert not failed.passed and "AssertionError" in failed.stderr_tail
    looping = run_python("while True: pass", timeout=1)
    assert looping.timed_out and not looping.passed


def test_sandbox_does_not_leak_environment(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")
    assert run_python("import os; assert 'SECRET_TOKEN' not in os.environ").passed


# ------------------------------------------------------------------- cache


def test_cache_avoids_repeat_calls(tmp_path):
    inner = DummyModel(responder=lambda p: p.upper())
    model = CachedModel(inner, tmp_path / "c.sqlite")
    reqs = [GenerationRequest("a"), GenerationRequest("b")]
    assert model.generate(reqs) == [["A"], ["B"]]
    assert inner.calls == 2
    # Second call: all hits, and a new request is the only miss.
    assert model.generate(reqs + [GenerationRequest("c")]) == [["A"], ["B"], ["C"]]
    assert inner.calls == 3 and model.hits == 2


def test_cache_persists_across_instances_and_separates_settings(tmp_path):
    path = tmp_path / "c.sqlite"
    CachedModel(DummyModel(responder=lambda p: "x"), path).loglikelihood([LoglikelihoodRequest("ctx", " x")])
    fresh_inner = DummyModel(responder=lambda p: "x")
    again = CachedModel(fresh_inner, path)
    again.loglikelihood([LoglikelihoodRequest("ctx", " x")])
    assert fresh_inner.calls == 0  # served from disk
    # A different max_new_tokens is a different request and must miss.
    again.generate([GenerationRequest("p", max_new_tokens=5)])
    again.generate([GenerationRequest("p", max_new_tokens=6)])
    assert fresh_inner.calls == 2
