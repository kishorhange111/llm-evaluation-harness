# ADR 0002: Every score carries a confidence interval; comparisons are paired

- Status: accepted

## Context
Single-number leaderboards hide noise: HumanEval has 164 problems, so a 5-point gap is often not significant.

## Decision
Scores are reported with a 95 % bootstrap interval (BCa where sample size allows). Model comparisons use paired bootstrap and McNemar's test on per-example scores. Reports show intervals next to every number.

## Consequences
Runs cost slightly more CPU time; readers can tell signal from noise. Arena results use order-invariant Elo and Bradley-Terry ratings.
