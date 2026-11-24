"""Pairwise A/B arena evaluator with order-invariant Elo & Bradley-Terry ratings.

Implements Chatbot Arena-style model ranking:

* **Order-invariant Elo**: averages online Elo updates across multiple seeded
  permutations of the battle log to eliminate match-order sensitivity.
* **Bradley-Terry MLE**: fits log-strength parameters via regularized
  Minorization-Maximization (MM), mapped onto the standard 1500/400 Elo scale.
* **Bootstrap rating CIs**: resamples battles to attach 95% confidence intervals
  to every model's leaderboard rating.
* **End-to-end arena runner**: generates answers for a pool of models, runs
  position-debiased pairwise judging (:func:`evalkit.metrics.judge.judge_pairwise`),
  and produces ranked standings plus a head-to-head win-rate matrix.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from evalkit.metrics.judge import judge_pairwise
from evalkit.models.base import Model
from evalkit.types import GenerationRequest


@dataclass(frozen=True)
class Battle:
    """One head-to-head pairwise match outcome."""

    model_a: str
    model_b: str
    winner: str  # "A", "B", or "tie"
    prompt_id: str = ""
    position_flip: bool = False


@dataclass(frozen=True)
class ArenaStanding:
    """Leaderboard row for one model."""

    model: str
    rating: float
    ci_low: float
    ci_high: float
    wins: int
    losses: int
    ties: int
    games: int
    win_rate: float


def _collect_models(battles: list[Battle]) -> list[str]:
    names: set[str] = set()
    for b in battles:
        names.add(b.model_a)
        names.add(b.model_b)
    return sorted(names)


def compute_elo_ratings(
    battles: list[Battle],
    *,
    initial: float = 1500.0,
    k_factor: float = 32.0,
    n_orders: int = 50,
    seed: int = 0,
) -> dict[str, float]:
    """Compute order-invariant Elo ratings by averaging over ``n_orders`` battle shuffles."""
    models = _collect_models(battles)
    if not models:
        return {}
    rng = np.random.default_rng(seed)
    totals = {m: 0.0 for m in models}
    n = len(battles)

    for _ in range(max(1, n_orders)):
        order = rng.permutation(n)
        ratings = {m: initial for m in models}
        for idx in order:
            b = battles[int(idx)]
            ra, rb = ratings[b.model_a], ratings[b.model_b]
            ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            sa = 1.0 if b.winner == "A" else 0.0 if b.winner == "B" else 0.5
            ratings[b.model_a] = ra + k_factor * (sa - ea)
            ratings[b.model_b] = rb + k_factor * ((1.0 - sa) - (1.0 - ea))
        for m, val in ratings.items():
            totals[m] += val

    return {m: totals[m] / max(1, n_orders) for m in models}


def compute_bradley_terry(
    battles: list[Battle],
    *,
    base_rating: float = 1500.0,
    scale: float = 400.0,
    prior: float = 1e-2,
    max_iter: int = 300,
    tol: float = 1e-8,
) -> dict[str, float]:
    """Fit Bradley-Terry ratings via regularized MM iteration, centered at ``base_rating``."""
    models = _collect_models(battles)
    m_count = len(models)
    if m_count == 0:
        return {}
    if m_count == 1:
        return {models[0]: base_rating}

    idx = {m: i for i, m in enumerate(models)}
    wins = np.full((m_count, m_count), prior, dtype=float)
    np.fill_diagonal(wins, 0.0)

    for b in battles:
        ia, ib = idx[b.model_a], idx[b.model_b]
        if ia == ib:
            continue
        if b.winner == "A":
            wins[ia, ib] += 1.0
        elif b.winner == "B":
            wins[ib, ia] += 1.0
        else:
            wins[ia, ib] += 0.5
            wins[ib, ia] += 0.5

    total_games = wins + wins.T
    win_sums = wins.sum(axis=1)
    p = np.ones(m_count, dtype=float)

    for _ in range(max_iter):
        p_old = p.copy()
        denom = np.zeros(m_count, dtype=float)
        for i in range(m_count):
            mask = total_games[i] > 0
            denom[i] = float(np.sum(total_games[i, mask] / (p[i] + p[mask])))
        p = np.where(denom > 0, win_sums / denom, p)
        # Normalize geometric mean to 1 for identifiability.
        p /= math.exp(float(np.mean(np.log(np.maximum(p, 1e-15)))))
        if float(np.max(np.abs(p - p_old))) < tol:
            break

    log10_p = np.log10(np.maximum(p, 1e-15))
    log10_p -= float(np.mean(log10_p))
    return {m: float(base_rating + scale * log10_p[idx[m]]) for m in models}


def win_rate_matrix(battles: list[Battle]) -> dict[str, dict[str, float]]:
    """Head-to-head win rate of row model vs. column model (ties count as 0.5)."""
    models = _collect_models(battles)
    scores: dict[tuple[str, str], list[float]] = {}
    for b in battles:
        sa = 1.0 if b.winner == "A" else 0.0 if b.winner == "B" else 0.5
        scores.setdefault((b.model_a, b.model_b), []).append(sa)
        scores.setdefault((b.model_b, b.model_a), []).append(1.0 - sa)

    matrix: dict[str, dict[str, float]] = {}
    for a in models:
        matrix[a] = {}
        for b in models:
            if a == b:
                matrix[a][b] = 0.5
            elif (a, b) in scores:
                vals = scores[(a, b)]
                matrix[a][b] = sum(vals) / len(vals)
            else:
                matrix[a][b] = float("nan")
    return matrix


def bootstrap_arena_ratings(
    battles: list[Battle],
    *,
    method: str = "bradley_terry",
    n_resamples: int = 200,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[ArenaStanding]:
    """Rank models with Elo or Bradley-Terry ratings plus bootstrap confidence intervals."""
    if method not in ("bradley_terry", "elo"):
        raise ValueError(f"unknown rating method {method!r}; expected 'bradley_terry' or 'elo'")
    models = _collect_models(battles)
    if not models:
        return []

    def _fit(sample: list[Battle], s: int) -> dict[str, float]:
        if method == "elo":
            return compute_elo_ratings(sample, n_orders=10, seed=s)
        return compute_bradley_terry(sample)

    point = _fit(battles, seed)
    rng = np.random.default_rng(seed)
    n = len(battles)
    boot: dict[str, list[float]] = {m: [] for m in models}
    for r in range(max(1, n_resamples)):
        idxs = rng.integers(0, n, size=n)
        sample = [battles[int(i)] for i in idxs]
        res = _fit(sample, seed + r + 1)
        for m in models:
            boot[m].append(res.get(m, point[m]))

    counts: dict[str, dict[str, int]] = {m: {"w": 0, "l": 0, "t": 0} for m in models}
    for b in battles:
        if b.winner == "A":
            counts[b.model_a]["w"] += 1
            counts[b.model_b]["l"] += 1
        elif b.winner == "B":
            counts[b.model_b]["w"] += 1
            counts[b.model_a]["l"] += 1
        else:
            counts[b.model_a]["t"] += 1
            counts[b.model_b]["t"] += 1

    alpha = (1.0 - confidence) / 2.0
    standings: list[ArenaStanding] = []
    for m in models:
        w, l_cnt, t = counts[m]["w"], counts[m]["l"], counts[m]["t"]
        games = w + l_cnt + t
        wr = (w + 0.5 * t) / games if games else 0.0
        lo = float(np.quantile(boot[m], alpha))
        hi = float(np.quantile(boot[m], 1.0 - alpha))
        standings.append(ArenaStanding(m, point[m], lo, hi, w, l_cnt, t, games, wr))

    standings.sort(key=lambda s: s.rating, reverse=True)
    return standings


def run_arena(
    models: list[Model],
    prompts: list[str],
    judge: Model,
    *,
    method: str = "bradley_terry",
    n_resamples: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Run an all-pairs A/B tournament over ``prompts`` with position-debiased judging."""
    if len(models) < 2:
        raise ValueError("arena requires at least 2 models")

    # Generate each model's answers once.
    answers: dict[str, list[str]] = {}
    reqs = [GenerationRequest(prompt=p, max_new_tokens=256) for p in prompts]
    for m in models:
        outs = m.generate(reqs)
        answers[m.name] = [o[0] if o else "" for o in outs]

    battles: list[Battle] = []
    for ma, mb in itertools.combinations(models, 2):
        outcomes = judge_pairwise(judge, prompts, answers[ma.name], answers[mb.name])
        for idx, o in enumerate(outcomes):
            battles.append(
                Battle(
                    model_a=ma.name,
                    model_b=mb.name,
                    winner=o.winner,
                    prompt_id=f"p-{idx}",
                    position_flip=o.position_flip,
                )
            )

    standings = bootstrap_arena_ratings(battles, method=method, n_resamples=n_resamples, seed=seed)
    flip_rate = sum(1 for b in battles if b.position_flip) / len(battles) if battles else 0.0
    return {
        "method": method,
        "standings": standings,
        "win_rate_matrix": win_rate_matrix(battles),
        "position_flip_rate": flip_rate,
        "num_battles": len(battles),
        "battles": battles,
    }


__all__ = [
    "ArenaStanding",
    "Battle",
    "bootstrap_arena_ratings",
    "compute_bradley_terry",
    "compute_elo_ratings",
    "run_arena",
    "win_rate_matrix",
]
