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


# The child sets its own limits, then runs the program. Doing it in the child's
# own code (rather than with subprocess's preexec_fn) is safe when run_python is
# called from many threads at once: preexec_fn can deadlock in a threaded parent.
_BOOTSTRAP = """
import resource, runpy, sys
limit = int(sys.argv[2]) * 1024 * 1024
if limit > 0:
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))  # cap memory
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # no core dumps
sys.argv = [sys.argv[1]]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


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
        if sys.platform.startswith("linux"):
            cmd = [sys.executable, "-I", "-c", _BOOTSTRAP, path, str(memory_mb)]
        else:  # the resource module is POSIX-only; elsewhere run without the memory cap
            cmd = [sys.executable, "-I", path]
        try:
            proc = subprocess.run(
                cmd,
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(False, True, "timeout")
        return ExecutionResult(proc.returncode == 0, False, proc.stderr[-500:])
