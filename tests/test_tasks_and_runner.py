"""End-to-end tests of tasks and the runner using offline fixtures and fake models."""

import json
import re
from pathlib import Path

import pytest

from evalkit.cli import main
from evalkit.models import DummyModel
from evalkit.runner import run
from evalkit.tasks import build_task

FIX = Path(__file__).parent / "fixtures"


def gsm8k_oracle(prompt: str) -> str:
    """Solve the fixture questions by looking up the last question in the prompt."""
    question = prompt.rsplit("Question:", 1)[-1]
    answers = {"cookies": "10 - 4 = 6. The answer is 6.", "train": "60 * 2 = 120\n#### 120", "1,000": "The answer is 1234"}
    for key, ans in answers.items():
        if key in question:
            return ans + "\n\nQuestion: invented follow-up"
    return "unknown"


def test_gsm8k_prompt_and_scoring():
    task = build_task("gsm8k", data_files=str(FIX / "gsm8k.jsonl"), num_fewshot=2)
    examples = task.examples()
    assert len(examples) == 3
    prompt = task.build_prompt(examples[0])
    # Few-shot answers are cleaned of calculator annotations and end with "The answer is".
    assert "<<" not in prompt and prompt.count("Question:") == 3 and "The answer is 5" in prompt
    results = task.evaluate(DummyModel(responder=gsm8k_oracle), examples)
    assert [r.score for r in results] == [1.0, 1.0, 1.0]
    # The stop string removed the invented follow-up question.
    assert all("invented" not in r.extra["completion"] for r in results)


def test_mmlu_loglikelihood_scoring_and_macro_average():
    task = build_task("mmlu", data_files=str(FIX / "mmlu.jsonl"), num_fewshot=1)
    examples = task.examples()
    gold = {"largest": " C", "Red planet": " A", "hash": " C", "HTTPS": " A"}  # the last one is deliberately wrong

    def picker(context: str) -> str:
        last_q = context.rsplit("\n\n", 1)[-1]
        return next(v for k, v in gold.items() if k in last_q)

    ctx = task.build_context(examples[0])
    assert "about astronomy" in ctx and "Closest star" in ctx  # same-subject few-shot
    results = task.evaluate(DummyModel(responder=picker), examples)
    metrics = task.aggregate(results)
    assert metrics["accuracy"] == pytest.approx(0.75)
    # astronomy 2/2, computer_security 1/2 -> macro (1.0 + 0.5) / 2
    assert metrics["accuracy_macro"] == pytest.approx(0.75)


@pytest.mark.parametrize("variant", ["standard", "instruction", "parenthesis", "lowercase_label"])
def test_mmlu_prompt_variants_render(variant):
    task = build_task("mmlu", data_files=str(FIX / "mmlu.jsonl"), prompt_variant=variant)
    ctx = task.build_context(task.examples()[0])
    assert "Jupiter" in ctx and ctx.rstrip().endswith(("Answer:", "answer:", "is"))


@pytest.mark.parametrize("chat", [False, True])
def test_humaneval_executes_code_and_handles_fenced_answers(chat):
    task = build_task("humaneval", data_files=str(FIX / "humaneval.jsonl"), chat=chat)
    examples = task.examples()
    if chat:
        # Instruct-style answers: prose plus a fenced, complete function (the second one is wrong).
        solutions = {
            "add": "Sure!\n```python\ndef add(a, b):\n    # sum\n    return a + b\n```\nDone.",
            "is_even": "```python\ndef is_even(n):\n    return n % 2 == 1\n```",
        }
    else:
        solutions = {"add": "    return a + b\n\ndef unused():\n    pass", "is_even": "    return n % 2 == 1\n"}

    def coder(prompt: str) -> str:
        name = re.search(r"def (\w+)", prompt).group(1)
        return solutions[name]

    results = task.evaluate(DummyModel(responder=coder), examples)
    assert [r.score for r in results] == [1.0, 0.0]
    assert task.aggregate(results) == {"pass@1": 0.5}


def test_limit_is_a_seeded_random_subset():
    a = build_task("gsm8k", data_files=str(FIX / "gsm8k.jsonl"), limit=2, seed=1).examples()
    b = build_task("gsm8k", data_files=str(FIX / "gsm8k.jsonl"), limit=2, seed=1).examples()
    assert [e.id for e in a] == [e.id for e in b] and len(a) == 2


def test_unknown_task_and_backend_raise():
    from evalkit.models import build_model

    with pytest.raises(ValueError, match="unknown task"):
        build_task("nope")
    with pytest.raises(ValueError, match="unknown backend"):
        build_model({"backend": "nope"})


def test_runner_end_to_end_writes_results_and_comparisons(tmp_path):
    config = {
        "run_name": "test",
        "models": [{"backend": "dummy", "name": "random"}, {"backend": "dummy", "name": "oracle", "responder": gsm8k_oracle}],
        "tasks": [
            {"name": "gsm8k", "data_files": str(FIX / "gsm8k.jsonl")},
            {"name": "jsonl_qa", "data_files": str(FIX / "qa.jsonl"), "label": "faq"},
        ],
    }
    summary = run(config, output_dir=tmp_path)
    assert summary["results"]["oracle"]["gsm8k"]["accuracy"] == 1.0
    assert summary["results"]["random"]["gsm8k"]["accuracy"] == 0.0
    assert (tmp_path / "samples" / "oracle__gsm8k.jsonl").read_text().count("\n") == 3
    comps = json.loads((tmp_path / "comparisons.json").read_text())
    assert comps["baseline"] == "random"
    assert comps["comparisons"]["oracle"]["gsm8k"]["delta"] == pytest.approx(1.0)
    report = (tmp_path / "report.md").read_text()
    assert "| oracle |" in report and "Paired comparison" in report


def test_cli_list_and_report(tmp_path, capsys):
    assert main(["list"]) == 0
    assert "gsm8k" in capsys.readouterr().out
    run({"models": [{"backend": "dummy"}], "tasks": [{"name": "jsonl_qa", "data_files": str(FIX / "qa.jsonl")}]}, output_dir=tmp_path)
    (tmp_path / "report.md").unlink()
    assert main(["report", str(tmp_path)]) == 0
    assert (tmp_path / "report.md").exists()
