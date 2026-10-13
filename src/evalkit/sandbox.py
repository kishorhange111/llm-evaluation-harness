"""Run untrusted, model-written Python code with time and memory limits.

Code benchmarks (HumanEval, MBPP) are scored by *executing* the model's code
against unit tests. Model output must be treated as untrusted, so each program:

* runs in a fresh ``python -I`` subprocess (isolated mode: ignores PYTHON* env
  vars and the user site-packages, and doesn't add the cwd to ``sys.path``),
* runs in an empty temporary directory that is deleted afterwards,
* is killed after ``timeout`` seconds (infinite loops are common),
* gets an address-space limit on Linux (runaway allocations can't take down the host),
* receives a minimal environment (no API keys or tokens inherited).

This is defence-in-depth for evaluation, **not** a security boundary. For
hostile inputs run the whole harness inside a container or VM without network.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of running a program: passed means exit code 0 within the time limit."""

    passed: bool
    timed_out: bool
    stderr_tail: str


def _limit_resources(memory_mb: int) -> None:  # pragma: no cover - runs in the child process
    """Called in the child just before exec; caps memory and disables core dumps."""
    import resource

    limit = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_python(program: str, timeout: float = 10.0, memory_mb: int = 1024) -> ExecutionResult:
    """Execute ``program`` and report whether it exited cleanly.

    Args:
        program: Full source, typically ``prompt + completion + tests + check(entry_point)``.
        timeout: Wall-clock limit in seconds.
        memory_mb: Address-space cap (Linux only; ignored elsewhere).
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "program.py")
        with open(path, "w") as f:
            f.write(program)
        preexec = (lambda: _limit_resources(memory_mb)) if sys.platform.startswith("linux") else None
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                preexec_fn=preexec,  # noqa: PLW1509 - no threads are running at this point
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(False, True, "timeout")
        return ExecutionResult(proc.returncode == 0, False, proc.stderr[-500:])
