"""String-level scoring helpers: normalisation, exact match, F1, answer extraction.

Answer extraction is where many published numbers quietly differ. A model that
writes "The answer is $1,234." is correct for a GSM8K target of "1234", and a
strict string compare would mark it wrong. Every rule here is unit-tested.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation and articles, squeeze spaces."""
    text = text.lower().translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, target: str) -> float:
    """1.0 if the normalised strings are identical, else 0.0."""
    return float(normalize_text(prediction) == normalize_text(target))


def token_f1(prediction: str, target: str) -> float:
    """Bag-of-words F1 between prediction and target (partial credit for QA)."""
    pred, gold = normalize_text(prediction).split(), normalize_text(target).split()
    if not pred or not gold:
        return float(pred == gold)
    common = Counter(pred) & Counter(gold)  # multiset intersection
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


# A number with optional sign, thousands separators and decimals: -1,234.50
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _clean_number(raw: str) -> str:
    """'1,234.50' -> '1234.5', '07' -> '7', so equal values compare equal as strings."""
    raw = raw.replace(",", "").rstrip(".")
    try:
        value = float(raw)
    except ValueError:
        return raw
    return str(int(value)) if value.is_integer() else repr(value)


def extract_final_number(text: str) -> str | None:
    """Pull the final numeric answer out of a chain-of-thought response.

    Priority order (most to least explicit):
      1. ``#### 42`` (GSM8K's own answer format)
      2. ``\\boxed{42}`` (common in math-tuned models)
      3. "the answer is 42"
      4. otherwise the *last* number in the text, because reasoning chains state
         intermediate numbers first and the result last.
    """
    for pattern in (
        r"####\s*(" + _NUMBER.pattern + ")",
        r"\\boxed\{\s*\$?(" + _NUMBER.pattern + r")",
        r"answer is[:\s]*\$?(" + _NUMBER.pattern + ")",
    ):
        m = re.findall(pattern, text, flags=re.IGNORECASE)
        if m:
            return _clean_number(m[-1])
    numbers = _NUMBER.findall(text)
    return _clean_number(numbers[-1]) if numbers else None


def extract_choice_letter(text: str, letters: str = "ABCD") -> str | None:
    """Find the option letter a model picked in free text, e.g. "Answer: (C)" -> "C"."""
    m = re.search(rf"answer\s*(?:is)?\s*[:\-]?\s*\(?([{letters}])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(rf"\b([{letters}])\b", text)
    return m.group(1) if m else None
