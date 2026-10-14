#!/usr/bin/env bash
# Run the evaluation suite on a Google Colab GPU runtime.
#
#   colab new -s eval --gpu A100
#   colab upload -s eval scripts/colab_job.sh /content/colab_job.sh
#   echo 'import subprocess; subprocess.run(["bash", "/content/colab_job.sh", "smoke"])' | colab exec -s eval
#
# Arguments: one or more config names from configs/ (without .yaml), or "judge".
# Results land in /content/llm-evaluation-harness/results and are packed into
# /content/results.tgz on exit (the SQLite cache is left out because it is large).
# A failing job is logged and the next one still runs; results.tgz is written on exit no matter what.
set -uo pipefail

REPO=${REPO:-https://github.com/kishorhange111/llm-evaluation-harness}
BRANCH=${BRANCH:-main}
DIR=/content/llm-evaluation-harness
trap 'tar --exclude="*.sqlite" -czf /content/results.tgz -C "$DIR" results 2>/dev/null; echo "packed results ($(date -u +%T))"' EXIT
set -e

# Clone once, then update in place, so a re-run reuses the downloaded models and the cache.
if [ ! -d "$DIR" ]; then
  git clone --depth 1 -b "$BRANCH" "$REPO" "$DIR"
else
  git -C "$DIR" fetch --depth 1 origin "$BRANCH" && git -C "$DIR" reset --hard FETCH_HEAD
fi
cd "$DIR"
pip install -q -e ".[hf]"
nvidia-smi --query-gpu=name,memory.total --format=csv

mkdir -p results
set +e
for job in "$@"; do
  echo "=== $job ($(date -u +%T))"
  if [ "$job" = "judge" ]; then
    python scripts/judge_experiment.py --out results/judge 2>&1 | tee "results/judge.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "FAILED: $job" | tee -a results/failures.txt
  else
    evalkit run "configs/$job.yaml" 2>&1 | tee "results/$job.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "FAILED: $job" | tee -a results/failures.txt
    if [ "$job" = "validate_vs_lm_eval" ]; then
      python scripts/compare_lm_eval.py results/validate_vs_lm_eval/summary.json | tee -a "results/$job.log"
    fi
  fi
done

echo "DONE $(date -u +%T)"; cat results/failures.txt 2>/dev/null
