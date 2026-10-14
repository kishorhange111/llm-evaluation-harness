# evalkit: an LLM evaluation harness you can read in an afternoon

[![CI](https://github.com/kishorhange111/llm-evaluation-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/llm-evaluation-harness/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Benchmarks for large language models with the statistics most leaderboards leave out: every score has a 95 % confidence interval, every model comparison has a paired significance test, and the harness is checked against EleutherAI's lm-evaluation-harness on the same model.

**Author:** Kishor Hange

<!-- RESULTS:START -->
## Results

Runs on a Google Colab L4. Raw summaries are in [`results/colab/`](results/colab).

### Validation against EleutherAI's lm-evaluation-harness

Same model (Qwen2.5-0.5B base), 0-shot, bf16, full test sets. The lm-evaluation-harness numbers come from running `lm_eval --model hf` on this model in the [mini-chatgpt](https://github.com/kishorhange111/mini-chatgpt) project.

| Metric | lm-evaluation-harness | evalkit | Difference (points) |
|---|---:|---:|---:|
| MMLU, acc (14,042 questions) | 47.49 | 47.17 | −0.32 |
| ARC-Challenge, acc | 29.35 | 29.27 | −0.08 |
| ARC-Challenge, acc_norm | 32.34 | 31.91 | −0.43 |
| HellaSwag, acc | 40.53 | 40.47 | −0.06 |
| HellaSwag, acc_norm | 52.15 | 52.22 | +0.07 |

Every difference is under half a point and well inside each score's 95 % confidence interval (±0.8 points on MMLU, ±2.6 on ARC), so the scoring (tokenisation boundary, BOS handling, length normalisation) matches the reference implementation.

### Qwen2.5-0.5B-Instruct, full suite with 95 % bootstrap confidence intervals

| Benchmark | Score (%) | 95 % CI | n |
|---|---:|---:|---:|
| MMLU (5-shot), accuracy | 46.9 | 46.1 – 47.8 | 14,042 |
| ARC-Challenge (0-shot), accuracy | 30.5 | 28.0 – 33.1 | 1,172 |
| HellaSwag (0-shot), acc_norm | 52.6 | 51.6 – 53.6 | 10,042 |
| GSM8K (chat, 0-shot), exact match | 40.1 | 37.5 – 42.8 | 1,319 |
| HumanEval (chat), pass@1 | 26.8 | 20.1 – 33.5 | 164 |

The intervals show why small benchmarks need care: HumanEval has only 164 problems, so its interval spans 13 points, and two models 5 points apart on it are not reliably different. The planned scaling study (1.5B, 3B and 7B), prompt-sensitivity and LLM-judge experiments were not completed: the Colab runtime kept disconnecting before the larger models finished. The configs are in `configs/` and resume from the response cache.
<!-- RESULTS:END -->

## What it does

| Area | What is implemented | Why it matters |
|---|---|---|
| Multiple choice (MMLU, ARC, HellaSwag) | Log-likelihood scoring of each option, `acc` and length-normalised `acc_norm`, few-shot prompts per subject | The standard way leaderboards score base models; no generation, no parsing errors |
| Generation (GSM8K, HumanEval, custom JSONL) | Batched, left-padded generation, stop sequences, chat templates | Instruct models are used through chat templates, so they should be evaluated that way |
| Code execution | Each program runs in an isolated `python -I` subprocess with a timeout, memory cap and empty environment | Model-written code is untrusted |
| pass@k | The unbiased estimator from the Codex paper | Naive "k samples, any correct" is biased upward |
| Statistics | Bootstrap 95 % CIs; paired bootstrap and McNemar's test between models | A 1-point gap on 1,000 questions is often noise |
| LLM-as-judge | Pairwise judging in both orders; a verdict counts only if it survives the swap | Judges prefer whichever answer they read first |
| Backends | Hugging Face `transformers`, any OpenAI-compatible server (vLLM, TGI, Ollama), and a dummy model for tests | Same task code for local and served models |
| Response cache | SQLite cache keyed by model, request and settings | A crashed or preempted run resumes without recomputing |

## How a run works

```mermaid
flowchart LR
    C[YAML config] --> R[runner]
    R --> T[Task: build prompts]
    T --> M[Model backend<br/>HF / OpenAI-compatible]
    M --> K[(SQLite cache)]
    M --> S[Score per example]
    S --> A[Aggregate + bootstrap CI]
    A --> P[Paired tests vs baseline]
    P --> O[summary.json · report.md · samples/*.jsonl]
```

Every example's prompt, prediction, target and score is written to `samples/`, so any number in a report can be traced back to the exact model outputs behind it.

## Quickstart

```bash
pip install -e ".[hf]"                      # torch, transformers, datasets
evalkit list                                # available tasks
evalkit run configs/smoke.yaml --limit 10   # a two-minute end-to-end check
evalkit report results/smoke                # re-render the markdown report
```

A config names the models and tasks:

```yaml
run_name: qwen25_scaling
models:
  - {backend: hf, pretrained: Qwen/Qwen2.5-0.5B-Instruct, chat: true, batch_size: 64}
  - {backend: hf, pretrained: Qwen/Qwen2.5-7B-Instruct, chat: true, batch_size: 32}
tasks:
  - {name: mmlu, num_fewshot: 5}
  - {name: gsm8k, chat: true}
  - {name: humaneval, chat: true}
```

The first model is the baseline for the paired tests. To evaluate a model behind an OpenAI-compatible server, see `configs/example_openai_server.yaml`.

On Google Colab, `scripts/colab_job.sh` clones the repo, installs it and runs any list of configs; results are packed into `/content/results.tgz` even if a step fails.

## Correctness details that change scores

- **Tokenisation boundary.** `tok(context + continuation)` is not `tok(context) + tok(continuation)` for BPE tokenisers. Continuation tokens are taken from the joined string, and a trailing space moves from the context to the continuation (`"Answer:" + " B"`).
- **Only the needed logits.** Sequences are left-padded so every continuation ends in the last columns, and the model projects only those positions onto the 152k-word vocabulary. Projecting every position of a 5-shot MMLU batch would need tens of GB.
- **BOS tokens.** Llama- and Gemma-style models expect a beginning-of-sequence token; it is added exactly when the tokenizer itself would add one.
- **Primary metric everywhere.** HellaSwag is reported as `acc_norm`, so its confidence interval and significance tests are computed on `acc_norm` too.

## Project layout

```
src/evalkit/
├── models/        hf.py (local), openai_compat.py (served), dummy.py (tests)
├── tasks/         mmlu / arc / hellaswag, gsm8k, humaneval, custom JSONL
├── metrics/       text matching, bootstrap + McNemar, pass@k, LLM judge
├── sandbox.py     isolated code execution
├── cache.py       SQLite response cache
├── runner.py      config → results, CIs, paired tests
└── cli.py         evalkit run | list | report
configs/           smoke, scaling, prompt sensitivity, lm-eval validation
scripts/           Colab job, judge experiment, lm-eval comparison
tests/             CPU-only tests, including a tiny GPT-2 built locally
```

## Tests

```bash
pip install -e ".[dev]" torch transformers
pytest
```

The tests run on CPU without downloads. They cover batched log-likelihoods against an unpadded reference computation, batched generation against unbatched, token-budget batching, metric formulas checked by hand, the sandbox (timeouts, memory cap, thread safety, environment isolation), the judge's position-swap logic and the cache.

## Limitations

- Single-GPU only; data-parallel evaluation across GPUs is not implemented.
- Benchmarks are the public test sets, which may appear in models' training data.
- The code sandbox is defence in depth, not a security boundary; run untrusted evaluations inside a container without network.

## License

MIT
