"""Tests for structured JSON schema adherence and constraint validation."""

from __future__ import annotations

import json

import pytest

from evalkit.metrics.schema import (
    StructuredJsonTask,
    evaluate_structured_output,
    extract_json,
    validate_json_schema,
)
from evalkit.runner import run
from evalkit.tasks import build_task

INVOICE_SCHEMA = {
    "type": "object",
    "required": ["invoice_id", "currency", "items", "total"],
    "additionalProperties": False,
    "properties": {
        "invoice_id": {"type": "string", "pattern": r"^INV-\d{4}$"},
        "currency": {"type": "string", "enum": ["USD", "EUR", "INR"]},
        "total": {"type": "number", "minimum": 0},
        "items": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {
                "type": "object",
                "required": ["sku", "price"],
                "properties": {
                    "sku": {"type": "string", "minLength": 2},
                    "price": {"type": "number", "exclusiveMinimum": 0},
                },
            },
        },
    },
}


def test_extract_json_handles_raw_fenced_and_prose_wrapped():
    obj, err = extract_json('Here is the payload:\n```json\n{"a": 1}\n```\nDone.')
    assert err is None and obj == {"a": 1}

    obj2, err2 = extract_json('Output: {"b": [1, 2]} thanks')
    assert err2 is None and obj2 == {"b": [1, 2]}

    obj3, err3 = extract_json("not json")
    assert obj3 is None and err3 is not None


def test_validate_json_schema_passes_valid_and_catches_violations():
    valid = {
        "invoice_id": "INV-1042",
        "currency": "USD",
        "total": 30.0,
        "items": [{"sku": "A1", "price": 10.0}, {"sku": "B2", "price": 20.0}],
    }
    assert validate_json_schema(valid, INVOICE_SCHEMA) == []

    invalid = {
        "invoice_id": "BAD",
        "currency": "GBP",
        "total": -5,
        "extra_key": True,
        "items": [{"sku": "X", "price": 0}, {"sku": "X", "price": 0}],
    }
    errs = validate_json_schema(invalid, INVOICE_SCHEMA)
    assert any("pattern" in e for e in errs)
    assert any("enum" in e for e in errs)
    assert any("minimum" in e for e in errs)
    assert any("additional property" in e for e in errs)
    assert any("uniqueItems" in e for e in errs)
    assert any("exclusiveMinimum" in e for e in errs)


def test_anyof_oneof_and_custom_constraints():
    schema = {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "string", "minLength": 2}]}
    assert validate_json_schema(5, schema) == []
    assert validate_json_schema("ok", schema) == []
    assert validate_json_schema(-1, schema)

    payload = json.dumps(
        {
            "invoice_id": "INV-0001",
            "currency": "INR",
            "total": 99.0,
            "items": [{"sku": "A1", "price": 10.0}],
        }
    )
    res = evaluate_structured_output(
        payload,
        INVOICE_SCHEMA,
        custom_constraints={"total_matches_items": lambda d: d["total"] == sum(i["price"] for i in d["items"])},
    )
    assert not res.valid
    assert any("total_matches_items" in v for v in res.violations)
    assert res.field_coverage == pytest.approx(1.0)


def test_structured_json_task_runs_in_runner(tmp_path):
    data = tmp_path / "schema.jsonl"
    row = {"prompt": "Extract invoice fields.", "schema": INVOICE_SCHEMA}
    data.write_text(json.dumps(row) + "\n")

    def oracle(_prompt: str) -> str:
        return json.dumps(
            {
                "invoice_id": "INV-2026",
                "currency": "USD",
                "total": 25.5,
                "items": [{"sku": "AB", "price": 25.5}],
            }
        )

    task = build_task("structured_json", data_files=str(data))
    assert isinstance(task, StructuredJsonTask)
    summary = run(
        {
            "run_name": "schema-smoke",
            "models": [{"backend": "dummy", "name": "json-oracle", "responder": oracle}],
            "tasks": [{"name": "structured_json", "data_files": str(data)}],
        },
        output_dir=tmp_path / "out",
    )
    assert summary["results"]["json-oracle"]["structured_json"]["schema_pass_rate"] == pytest.approx(1.0)
