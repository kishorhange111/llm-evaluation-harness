# GitHub Action: evalkit eval gate

The repository ships a reusable composite action (`action.yml`) that runs an
evalkit config, renders Markdown/HTML/JSON reports, and fails the job when a
candidate run regresses against a baseline.

## Usage

```yaml
jobs:
  eval-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: kishorhange111/llm-evaluation-harness@main
        with:
          config: configs/my_eval.yaml
          baseline: benchmarks/baseline      # run dir or summary.json
          max-drop: "0.02"                   # absolute drop allowed per task
          min-score: "0.60"                  # optional floor
```

## Inputs

| Input | Default | Description |
|---|---|---|
| `config` | required | evalkit YAML config to run |
| `baseline` | `""` | baseline run dir / `summary.json`; empty skips the gate |
| `max-drop` | `0.02` | max allowed absolute score drop per task |
| `min-score` | `""` | optional absolute score floor for every task |
| `output-dir` | `eval-results` | where results and reports are written |
| `python-version` | `3.12` | Python used to run evalkit |
| `install-command` | `pip install .` | how evalkit is installed |
| `limit` | `""` | per-task example cap for smoke gates |

## Outputs

- `passed`: `true`/`false` regression gate result.
- `results-dir`: directory holding `summary.json`, `report.md`, `report.html`, `report.json`, and `gate.md`.

The gate table is appended to the job summary and the full results directory is
uploaded as the `evalkit-results` artifact.

## Repository CI gate

`.github/workflows/eval-gate.yml` runs the action against `configs/ci_gate.yaml`
(deterministic dummy model, offline fixtures) and compares to `benchmarks/baseline`.
Refresh the baseline after an intentional change:

```bash
evalkit run configs/ci_gate.yaml --output-dir benchmarks/baseline
```

## Local equivalent

```bash
evalkit run configs/ci_gate.yaml --output-dir eval-results
evalkit report eval-results --format all
evalkit compare benchmarks/baseline eval-results --max-drop 0.02
```
