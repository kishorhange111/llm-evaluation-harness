"""Structured JSON schema adherence & constraint validator.

Provides a zero-dependency JSON Schema validator (supporting object/array/scalar
types, required fields, nested properties, ``additionalProperties``, numeric and
length bounds, regex patterns, ``enum``/``const``, ``anyOf``/``oneOf``) plus
custom domain constraint checks and a registered ``structured_json`` task.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from evalkit.registry import register_metric, register_task
from evalkit.tasks.base import GenerationTask
from evalkit.types import Example, ExampleResult

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json(text: str) -> tuple[Any, str | None]:
    """Extract and parse JSON from raw text, fenced blocks, or balanced braces.

    Returns ``(parsed_object, None)`` on success or ``(None, error_message)``.
    """
    stripped = text.strip()
    if not stripped:
        return None, "empty output"

    candidates = [stripped]
    for match in _FENCE_RE.finditer(stripped):
        candidates.append(match.group(1).strip())

    # Fallback: slice from first '{' or '[' to matching last '}' or ']'.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = stripped.find(open_ch)
        end = stripped.rfind(close_ch)
        if start != -1 and end > start:
            candidates.append(stripped[start : end + 1])

    last_err = "unable to parse JSON"
    for cand in candidates:
        try:
            return json.loads(cand), None
        except json.JSONDecodeError as exc:
            last_err = f"JSONDecodeError: {exc.msg} (pos {exc.pos})"
    return None, last_err


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "null":
        return value is None
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


def validate_json_schema(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate ``instance`` against a JSON Schema dict, returning violation strings."""
    errors: list[str] = []

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}, got {instance!r}")

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: value {instance!r} not in enum {schema['enum']!r}")

    if "type" in schema:
        expected = schema["type"]
        allowed = [expected] if isinstance(expected, str) else list(expected)
        if not any(_matches_type(instance, t) for t in allowed):
            errors.append(f"{path}: expected type {expected!r}, got {type(instance).__name__}")
            return errors

    if "anyOf" in schema and not any(len(validate_json_schema(instance, sub, path)) == 0 for sub in schema["anyOf"]):
        errors.append(f"{path}: value did not satisfy anyOf schemas")

    if "oneOf" in schema:
        matched = sum(1 for sub in schema["oneOf"] if len(validate_json_schema(instance, sub, path)) == 0)
        if matched != 1:
            errors.append(f"{path}: value matched {matched} schemas in oneOf (expected exactly 1)")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: string length {len(instance)} < minLength {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: string length {len(instance)} > maxLength {schema['maxLength']}")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            errors.append(f"{path}: string {instance!r} does not match pattern {schema['pattern']!r}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {instance} <= exclusiveMinimum {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: {instance} >= exclusiveMaximum {schema['exclusiveMaximum']}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: array length {len(instance)} < minItems {schema['minItems']}")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: array length {len(instance)} > maxItems {schema['maxItems']}")
        if schema.get("uniqueItems"):
            seen: list[str] = []
            for item in instance:
                key = json.dumps(item, sort_keys=True, default=str)
                if key in seen:
                    errors.append(f"{path}: duplicate item violates uniqueItems")
                    break
                seen.append(key)
        if "items" in schema and isinstance(schema["items"], dict):
            for idx, item in enumerate(instance):
                errors.extend(validate_json_schema(item, schema["items"], f"{path}[{idx}]"))

    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in instance:
                errors.append(f"{path}: missing required property {req!r}")
        for key, val in instance.items():
            if key in props:
                errors.extend(validate_json_schema(val, props[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected additional property {key!r}")
            elif isinstance(schema.get("additionalProperties"), dict):
                errors.extend(validate_json_schema(val, schema["additionalProperties"], f"{path}.{key}"))

    return errors


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a model's structured JSON output."""

    valid: bool
    parsed: Any
    parse_error: str | None
    violations: list[str] = field(default_factory=list)
    constraint_score: float = 0.0
    field_coverage: float = 0.0


@register_metric("json_schema_adherence", category="structured", tags=("json", "schema", "constraints"))
def evaluate_structured_output(
    raw_text: str,
    schema: dict[str, Any],
    *,
    custom_constraints: dict[str, Callable[[Any], bool]] | None = None,
) -> ValidationResult:
    """Parse ``raw_text`` as JSON, validate against ``schema``, and run custom constraints."""
    parsed, parse_err = extract_json(raw_text)
    if parse_err is not None:
        return ValidationResult(
            valid=False,
            parsed=None,
            parse_error=parse_err,
            violations=[parse_err],
            constraint_score=0.0,
            field_coverage=0.0,
        )

    violations = validate_json_schema(parsed, schema)
    custom = custom_constraints or {}
    failed_custom = 0
    for cname, fn in custom.items():
        try:
            ok = bool(fn(parsed))
        except Exception as exc:  # noqa: BLE001
            ok = False
            violations.append(f"constraint {cname!r} raised {type(exc).__name__}: {exc}")
        if not ok:
            failed_custom += 1
            if not any(v.startswith(f"constraint {cname!r}") for v in violations):
                violations.append(f"constraint {cname!r} failed")

    # Field coverage: fraction of required top-level properties present.
    required = schema.get("required", [])
    if required and isinstance(parsed, dict):
        field_coverage = sum(1 for k in required if k in parsed) / len(required)
    elif isinstance(parsed, dict):
        field_coverage = 1.0
    else:
        field_coverage = 1.0 if not violations else 0.0

    # Constraint score: 1.0 when valid; smoothly decays with violation count.
    total_checks = max(1, len(required) + len(schema.get("properties", {})) + len(custom))
    constraint_score = max(0.0, 1.0 - len(violations) / total_checks) if violations else 1.0

    return ValidationResult(
        valid=len(violations) == 0,
        parsed=parsed,
        parse_error=None,
        violations=violations,
        constraint_score=constraint_score,
        field_coverage=field_coverage,
    )


@register_task("structured_json", category="structured", tags=("json", "schema", "generation"))
class StructuredJsonTask(GenerationTask):
    """Evaluate structured JSON generation against per-example JSON Schemas.

    Expects local JSONL rows with ``prompt`` and ``schema`` (dict).
    """

    name = "structured_json"
    primary_metric = "schema_pass_rate"
    eval_split = "test"
    fewshot_split = None

    def build_prompt(self, example: Example) -> str:
        schema_str = json.dumps(example.data["schema"], indent=2)
        return f"{example.data['prompt']}\n\nReturn valid JSON strictly matching this schema:\n{schema_str}\n"

    def score(self, example: Example, completions: list[str]) -> ExampleResult:
        raw = completions[0] if completions else ""
        res = evaluate_structured_output(raw, example.data["schema"])
        return ExampleResult(
            id=example.id,
            score=1.0 if res.valid else 0.0,
            prediction=res.parsed,
            target=example.data["schema"],
            extra={
                "valid": res.valid,
                "parse_error": res.parse_error,
                "violations": res.violations,
                "constraint_score": res.constraint_score,
                "field_coverage": res.field_coverage,
            },
        )

    def aggregate(self, results: list[ExampleResult]) -> dict[str, float]:
        if not results:
            nan = float("nan")
            return {"schema_pass_rate": nan, "constraint_score": nan, "field_coverage": nan}
        n = len(results)
        return {
            "schema_pass_rate": sum(r.score for r in results) / n,
            "constraint_score": sum(r.extra["constraint_score"] for r in results) / n,
            "field_coverage": sum(r.extra["field_coverage"] for r in results) / n,
        }


__all__ = [
    "StructuredJsonTask",
    "ValidationResult",
    "evaluate_structured_output",
    "extract_json",
    "validate_json_schema",
]
