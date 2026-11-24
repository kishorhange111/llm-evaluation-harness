# ADR 0003: Offline, deterministic regression gate in CI

- Status: accepted

## Context
A CI gate that needs GPUs or API keys is flaky and expensive, so it gets disabled.

## Decision
The gate runs `configs/ci_gate.yaml` against a deterministic dummy backend and committed fixtures, then compares with `benchmarks/baseline` using `evalkit compare --max-drop`. The same logic ships as the reusable composite action in `action.yml` so real model runs can use identical gating with their own baselines.

## Consequences
The gate verifies the harness itself (scoring, aggregation, reporting) on every PR; model-quality gates are opt-in per project via the action inputs. Baselines are refreshed deliberately by re-running the config and committing the result.
