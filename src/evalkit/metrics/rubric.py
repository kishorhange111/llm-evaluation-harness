"""Calibrated LLM-as-judge rubric framework with a deterministic local judge.

Supports multi-criterion weighted rubrics (1-5 or custom scales), structured
JSON/text verdict parsing, a zero-API-key :class:`DeterministicLocalJudge` for
CI and offline testing, and human-alignment calibration metrics (Spearman rank
correlation, MAE, bias, quadratic-weighted Cohen's kappa, and affine debiasing).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evalkit.metrics.text import token_f1
from evalkit.models.base import Model
from evalkit.types import GenerationRequest, LoglikelihoodRequest, LoglikelihoodResult


@dataclass(frozen=True)
class RubricCriterion:
    """Single scoring dimension within a rubric."""

    name: str
    description: str
    weight: float = 1.0
    scale_min: int = 1
    scale_max: int = 5
    anchors: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Rubric:
    """Collection of weighted criteria rendered into an LLM-as-judge prompt."""

    name: str
    criteria: tuple[RubricCriterion, ...]
    instructions: str = (
        "Score the assistant answer on each criterion independently. "
        'Return JSON: {"scores": {"<criterion>": <int>, ...}, "rationale": "<brief explanation>"}.'
    )

    def render_prompt(self, question: str, answer: str, reference: str = "") -> str:
        lines = [f"[Rubric: {self.name}]", self.instructions, "", "[Criteria]"]
        for c in self.criteria:
            lines.append(f"- {c.name} ({c.scale_min}-{c.scale_max}, weight={c.weight}): {c.description}")
            for level, desc in sorted(c.anchors.items()):
                lines.append(f"    {level}: {desc}")
        lines += ["", f"[Question]\n{question}", "", f"[Answer]\n{answer}"]
        if reference:
            lines += ["", f"[Reference]\n{reference}"]
        return "\n".join(lines)


DEFAULT_QUALITY_RUBRIC = Rubric(
    name="general_quality",
    criteria=(
        RubricCriterion("correctness", "Factual accuracy and alignment with the reference.", weight=0.45),
        RubricCriterion("completeness", "Covers all key parts of the question.", weight=0.30),
        RubricCriterion("clarity", "Clear, well-structured, concise presentation.", weight=0.25),
    ),
)


@dataclass(frozen=True)
class RubricVerdict:
    """Parsed multi-criterion rubric evaluation."""

    criterion_scores: dict[str, float]
    weighted_score: float
    normalized_score: float
    rationale: str
    raw_output: str


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def parse_rubric_output(text: str, rubric: Rubric) -> RubricVerdict:
    """Parse JSON or ``criterion: score`` lines into a :class:`RubricVerdict`."""
    candidate = text.strip()
    m = _FENCE_RE.search(candidate)
    if m:
        candidate = m.group(1).strip()

    parsed_scores: dict[str, float] = {}
    rationale = ""
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            raw_map = payload.get("scores", payload)
            if isinstance(raw_map, dict):
                for c in rubric.criteria:
                    if c.name in raw_map and isinstance(raw_map[c.name], (int, float)):
                        val = float(raw_map[c.name])
                        parsed_scores[c.name] = max(float(c.scale_min), min(float(c.scale_max), val))
            rationale = str(payload.get("rationale", ""))
    except json.JSONDecodeError:
        for c in rubric.criteria:
            pat = re.compile(rf"{re.escape(c.name)}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
            hit = pat.search(text)
            if hit:
                val = float(hit.group(1))
                parsed_scores[c.name] = max(float(c.scale_min), min(float(c.scale_max), val))
        rationale = text.strip()[:200]

    # Default missing criteria to scale_min to stay conservative.
    for c in rubric.criteria:
        parsed_scores.setdefault(c.name, float(c.scale_min))

    total_w = sum(c.weight for c in rubric.criteria) or 1.0
    weighted = sum(parsed_scores[c.name] * c.weight for c in rubric.criteria) / total_w
    norm = sum(((parsed_scores[c.name] - c.scale_min) / max(1e-9, c.scale_max - c.scale_min)) * c.weight for c in rubric.criteria) / total_w

    return RubricVerdict(
        criterion_scores=parsed_scores,
        weighted_score=weighted,
        normalized_score=max(0.0, min(1.0, norm)),
        rationale=rationale,
        raw_output=text,
    )


class DeterministicLocalJudge(Model):
    """Offline, reproducible rubric judge for CI and local evaluation.

    Inspects the rendered rubric prompt (``[Question]``, ``[Answer]``, optional
    ``[Reference]``) and assigns deterministic scores per criterion based on
    reference overlap, question coverage, and response conciseness/structure.
    """

    def __init__(self, rubric: Rubric = DEFAULT_QUALITY_RUBRIC, name: str = "local-rubric-judge") -> None:
        self.rubric = rubric
        self.name = name

    @staticmethod
    def _extract_section(prompt: str, header: str) -> str:
        pat = re.compile(rf"\[{re.escape(header)}\]\n([\s\S]*?)(?=\n\[[A-Za-z ]+\]\n|\Z)")
        m = pat.search(prompt)
        return m.group(1).strip() if m else ""

    def _score_prompt(self, prompt: str) -> str:
        question = self._extract_section(prompt, "Question")
        answer = self._extract_section(prompt, "Answer")
        reference = self._extract_section(prompt, "Reference")

        scores: dict[str, int] = {}
        ref_f1 = token_f1(answer, reference) if reference else token_f1(answer, question)
        q_f1 = token_f1(answer, question)
        words = answer.split()

        for c in self.rubric.criteria:
            span = c.scale_max - c.scale_min
            lname = c.name.lower()
            if not answer.strip():
                frac = 0.0
            elif "correct" in lname or "accuracy" in lname or "faith" in lname:
                frac = ref_f1
            elif "complete" in lname or "coverage" in lname or "depth" in lname:
                length_factor = min(1.0, len(words) / max(5, len((reference or question).split())))
                frac = 0.6 * ref_f1 + 0.4 * length_factor
            elif "clar" in lname or "concis" in lname or "style" in lname:
                # Reward readable non-empty answers without excessive repetition.
                unique_ratio = len(set(w.lower() for w in words)) / max(1, len(words))
                frac = 0.5 * unique_ratio + 0.5 * min(1.0, max(q_f1, ref_f1) + 0.25)
            else:
                frac = 0.5 * ref_f1 + 0.5 * q_f1
            val = c.scale_min + round(max(0.0, min(1.0, frac)) * span)
            scores[c.name] = int(max(c.scale_min, min(c.scale_max, val)))

        return json.dumps({"scores": scores, "rationale": "Deterministic local rubric scoring."})

    def generate(self, requests: list[GenerationRequest]) -> list[list[str]]:
        return [[self._score_prompt(req.prompt) for _ in range(req.n)] for req in requests]

    def loglikelihood(self, requests: list[LoglikelihoodRequest]) -> list[LoglikelihoodResult]:
        return [LoglikelihoodResult(logprob=-1.0, num_tokens=1, is_greedy=False) for _ in requests]


def evaluate_with_rubric(
    judge: Model,
    rubric: Rubric,
    question: str,
    answer: str,
    reference: str = "",
) -> RubricVerdict:
    """Score a single ``(question, answer)`` pair using ``judge`` and ``rubric``."""
    prompt = judge.apply_chat_template(rubric.render_prompt(question, answer, reference=reference))
    raw = judge.generate([GenerationRequest(prompt=prompt, max_new_tokens=256)])[0][0]
    return parse_rubric_output(raw, rubric)


def evaluate_batch_with_rubric(
    judge: Model,
    rubric: Rubric,
    items: list[dict[str, str]],
) -> dict[str, Any]:
    """Evaluate a list of ``{"question", "answer", "reference"?}`` dicts in one batched call."""
    if not items:
        return {"mean_weighted": float("nan"), "mean_normalized": float("nan"), "n": 0, "verdicts": []}
    prompts = [
        judge.apply_chat_template(rubric.render_prompt(it["question"], it["answer"], reference=it.get("reference", ""))) for it in items
    ]
    outs = judge.generate([GenerationRequest(prompt=p, max_new_tokens=256) for p in prompts])
    verdicts = [parse_rubric_output(out[0], rubric) for out in outs]
    n = len(verdicts)
    return {
        "mean_weighted": sum(v.weighted_score for v in verdicts) / n,
        "mean_normalized": sum(v.normalized_score for v in verdicts) / n,
        "n": n,
        "verdicts": verdicts,
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-tie ranking for Spearman correlation without SciPy."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rankdata(x), _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum())))
    if denom == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


def _quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, scale_min: int = 1, scale_max: int = 5) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal judge scales."""
    k = scale_max - scale_min + 1
    if k <= 1 or y_true.size == 0:
        return 0.0
    t = np.clip(np.rint(y_true).astype(int) - scale_min, 0, k - 1)
    p = np.clip(np.rint(y_pred).astype(int) - scale_min, 0, k - 1)
    obs = np.zeros((k, k), dtype=float)
    for ti, pi in zip(t, p):
        obs[ti, pi] += 1.0
    n = float(y_true.size)
    hist_t = obs.sum(axis=1)
    hist_p = obs.sum(axis=0)
    exp = np.outer(hist_t, hist_p) / n
    idx = np.arange(k, dtype=float)
    weights = ((idx[:, None] - idx[None, :]) ** 2) / ((k - 1) ** 2)
    denom = float((weights * exp).sum())
    if denom == 0.0:
        return 1.0 if np.array_equal(t, p) else 0.0
    return float(1.0 - (weights * obs).sum() / denom)


@dataclass(frozen=True)
class CalibrationReport:
    """Alignment diagnostics and affine calibration parameters vs. human labels."""

    spearman_rho: float
    cohen_kappa: float
    mean_abs_error: float
    bias: float
    n: int
    affine_slope: float
    affine_intercept: float
    calibrated_scores: list[float]

    def calibrate(self, score: float, scale_min: float = 1.0, scale_max: float = 5.0) -> float:
        """Apply learned affine calibration to a raw judge score."""
        return max(scale_min, min(scale_max, self.affine_slope * score + self.affine_intercept))


def calibrate_judge(
    judge_scores: list[float],
    human_scores: list[float],
    *,
    scale_min: int = 1,
    scale_max: int = 5,
) -> CalibrationReport:
    """Measure judge-vs-human alignment and fit an affine debiasing calibration."""
    xj = np.asarray(judge_scores, dtype=float)
    xh = np.asarray(human_scores, dtype=float)
    if xj.shape != xh.shape or xj.size == 0:
        raise ValueError("judge_scores and human_scores must be non-empty and equal length")

    rho = _spearman_rho(xj, xh)
    kappa = _quadratic_weighted_kappa(xh, xj, scale_min=scale_min, scale_max=scale_max)
    mae = float(np.abs(xj - xh).mean())
    bias = float((xj - xh).mean())

    var_j = float(np.var(xj))
    if var_j > 1e-12:
        slope = float(np.cov(xj, xh, ddof=0)[0, 1] / var_j)
        intercept = float(xh.mean() - slope * xj.mean())
    else:
        slope = 1.0
        intercept = float(xh.mean() - xj.mean())

    calibrated = [float(max(scale_min, min(scale_max, slope * s + intercept))) for s in xj]
    return CalibrationReport(
        spearman_rho=rho,
        cohen_kappa=kappa,
        mean_abs_error=mae,
        bias=bias,
        n=int(xj.size),
        affine_slope=slope,
        affine_intercept=intercept,
        calibrated_scores=calibrated,
    )


__all__ = [
    "CalibrationReport",
    "DEFAULT_QUALITY_RUBRIC",
    "DeterministicLocalJudge",
    "Rubric",
    "RubricCriterion",
    "RubricVerdict",
    "calibrate_judge",
    "evaluate_batch_with_rubric",
    "evaluate_with_rubric",
    "parse_rubric_output",
]
