"""Tests for the pairwise A/B arena evaluator (Elo, Bradley-Terry, and bootstrap CIs)."""

from __future__ import annotations

import pytest

from evalkit.arena import (
    Battle,
    bootstrap_arena_ratings,
    compute_bradley_terry,
    compute_elo_ratings,
    run_arena,
    win_rate_matrix,
)
from evalkit.models import DummyModel


def _sample_battles() -> list[Battle]:
    # Strict dominance hierarchy: strong > mid > weak.
    battles: list[Battle] = []
    for i in range(12):
        battles.append(Battle("strong", "mid", "A", f"sm-{i}"))
        battles.append(Battle("mid", "weak", "A", f"mw-{i}"))
        battles.append(Battle("strong", "weak", "A", f"sw-{i}"))
    # Add a couple of ties so ratings stay finite and realistic.
    battles.append(Battle("strong", "mid", "tie", "t-1"))
    battles.append(Battle("mid", "weak", "tie", "t-2"))
    return battles


def test_elo_and_bradley_terry_recover_true_ranking():
    battles = _sample_battles()
    elo = compute_elo_ratings(battles, n_orders=30, seed=0)
    bt = compute_bradley_terry(battles)

    assert elo["strong"] > elo["mid"] > elo["weak"]
    assert bt["strong"] > bt["mid"] > bt["weak"]
    # Mean rating stays centered near 1500.
    assert sum(bt.values()) / len(bt) == pytest.approx(1500.0, abs=1e-6)


def test_bootstrap_arena_ratings_and_win_rate_matrix():
    battles = _sample_battles()
    standings = bootstrap_arena_ratings(battles, method="bradley_terry", n_resamples=100, seed=42)
    assert [s.model for s in standings] == ["strong", "mid", "weak"]
    assert standings[0].ci_low <= standings[0].rating <= standings[0].ci_high
    assert standings[0].win_rate > standings[1].win_rate > standings[2].win_rate

    mat = win_rate_matrix(battles)
    assert mat["strong"]["weak"] == pytest.approx(1.0)
    assert mat["weak"]["strong"] == pytest.approx(0.0)
    assert mat["mid"]["mid"] == pytest.approx(0.5)


def test_run_arena_end_to_end_with_position_debiased_judge():
    # Judge prefers answers containing "GOLD" > "SILVER" > "BRONZE".
    rank = {"GOLD": 3, "SILVER": 2, "BRONZE": 1}

    def judge_fn(prompt: str) -> str:
        part_a = prompt.split("[Assistant A]\n", 1)[1].split("\n\n[Assistant B]", 1)[0]
        part_b = prompt.split("[Assistant B]\n", 1)[1]
        sa = next((v for k, v in rank.items() if k in part_a), 0)
        sb = next((v for k, v in rank.items() if k in part_b), 0)
        if sa > sb:
            return "A is clearly stronger.\n[[A]]"
        if sb > sa:
            return "B is clearly stronger.\n[[B]]"
        return "Tie.\n[[C]]"

    models = [
        DummyModel(responder=lambda p: f"GOLD answer for {p}", name="m-gold"),
        DummyModel(responder=lambda p: f"SILVER answer for {p}", name="m-silver"),
        DummyModel(responder=lambda p: f"BRONZE answer for {p}", name="m-bronze"),
    ]
    judge = DummyModel(responder=judge_fn, name="deterministic-judge")
    res = run_arena(models, ["Explain attention", "Explain LoRA", "Explain RAG"], judge, n_resamples=50, seed=0)

    assert [s.model for s in res["standings"]] == ["m-gold", "m-silver", "m-bronze"]
    assert res["position_flip_rate"] == pytest.approx(0.0)
    assert res["num_battles"] == 9
