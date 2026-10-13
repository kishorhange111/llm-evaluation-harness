"""evalkit: a small, readable, production-style evaluation harness for LLMs.

The package is organised around four ideas:

* **Model**: anything that can ``generate`` text or score the log-likelihood of
  a continuation (``evalkit.models``). Backends: Hugging Face ``transformers``,
  any OpenAI-compatible HTTP server (vLLM, TGI, llama.cpp), and a deterministic
  dummy model used in tests.
* **Task**: a benchmark. It knows how to load its examples, turn each one into a
  request for the model, and score the model's answer (``evalkit.tasks``).
* **Metric**: pure functions that turn per-example scores into numbers you can
  report, plus confidence intervals and significance tests (``evalkit.metrics``).
* **Runner**: glues the three together, caches model calls, and writes
  reproducible results to disk (``evalkit.runner``).
"""

__version__ = "0.1.0"
