"""LLM-as-a-judge: pairwise comparison of two models' answers.

Open-ended answers ("explain X", "write an email") have no single correct
string, so we ask a strong *judge* model which of two answers is better, as in
MT-Bench and Chatbot Arena.

Judges have known biases; this module controls the biggest one:

* **Position bias**: judges tend to prefer whichever answer is shown first.
  We ask twice with the order swapped. A verdict only counts as a win if it
  survives the swap; otherwise it is a tie. We also report how often the judge
  flipped, which measures how much its verdicts can be trusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from evalkit.models.base import Model
from evalkit.types import GenerationRequest

JUDGE_TEMPLATE = """You are an impartial judge. Compare two AI assistant answers to the user question below.
Judge helpfulness, correctness, depth and clarity. Ignore answer length and the order in which answers appear.
First write a short explanation, then output your verdict on the last line in exactly one of these formats:
[[A]] if assistant A is better, [[B]] if assistant B is better, [[C]] for a tie.

[User Question]
{question}

[Assistant A]
{answer_a}

[Assistant B]
{answer_b}
"""


def parse_verdict(text: str) -> str | None:
    """Return 'A', 'B' or 'C' from the judge's output, using the *last* verdict tag.

    The last tag is used because judges sometimes quote the format instructions
    ("...output [[A]] if...") before giving their real answer.
    """
    tags = re.findall(r"\[\[([ABC])\]\]", text)
    return tags[-1] if tags else None


@dataclass(frozen=True)
class PairwiseOutcome:
    """Debiased result for one question: 'A', 'B' or 'tie', plus the raw verdicts."""

    winner: str
    verdict_ab: str | None  # verdict with A shown first
    verdict_ba: str | None  # verdict with B shown first, mapped back to A/B labels
    position_flip: bool  # True if swapping the order changed the decision


def combine_verdicts(v_ab: str | None, v_ba_raw: str | None) -> PairwiseOutcome:
    """Merge the two orderings into one debiased outcome.

    ``v_ba_raw`` comes from the swapped prompt, where "A" refers to the original
    B. We map it back first, then accept a winner only if both orderings agree.
    """
    swap = {"A": "B", "B": "A", "C": "C", None: None}
    v_ba = swap[v_ba_raw]
    if v_ab is not None and v_ab == v_ba and v_ab != "C":
        winner = v_ab
    else:
        winner = "tie"
    flip = v_ab is not None and v_ba is not None and v_ab != v_ba
    return PairwiseOutcome(winner, v_ab, v_ba, flip)


def judge_pairwise(judge: Model, questions: list[str], answers_a: list[str], answers_b: list[str]) -> list[PairwiseOutcome]:
    """Judge every (question, answer A, answer B) triple in both orders."""
    prompts = []
    for q, a, b in zip(questions, answers_a, answers_b):
        prompts.append(judge.apply_chat_template(JUDGE_TEMPLATE.format(question=q, answer_a=a, answer_b=b)))
        prompts.append(judge.apply_chat_template(JUDGE_TEMPLATE.format(question=q, answer_a=b, answer_b=a)))
    outs = judge.generate([GenerationRequest(prompt=p, max_new_tokens=512) for p in prompts])
    results = []
    for i in range(0, len(outs), 2):
        results.append(combine_verdicts(parse_verdict(outs[i][0]), parse_verdict(outs[i + 1][0])))
    return results


def win_rate(outcomes: list[PairwiseOutcome], side: str = "B") -> float:
    """Win rate of ``side`` where a tie counts as half a win (Chatbot Arena convention)."""
    if not outcomes:
        return float("nan")
    return sum(1.0 if o.winner == side else 0.5 if o.winner == "tie" else 0.0 for o in outcomes) / len(outcomes)
