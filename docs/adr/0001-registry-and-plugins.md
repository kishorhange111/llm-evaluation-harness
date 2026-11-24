# ADR 0001: Task/runner registry with plugin entry points

- Status: accepted

## Context
Benchmarks (MMLU, GSM8K, HumanEval, custom JSONL) and evaluators (RAG, trajectory, schema, red-team) each need their own prompt building and scoring, but share caching, statistics and reporting.

## Decision
A small registry (`evalkit.registry`) maps names to task and evaluator classes. Built-ins register on import and third-party packages can register through Python entry points. The runner only knows the registry interface.

## Consequences
New evaluators are added without touching the runner; the cost is a global registry,.
