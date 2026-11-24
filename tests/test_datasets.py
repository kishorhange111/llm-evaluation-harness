"""Tests for golden dataset versioning, manifest hashing, verification, and diff CLI."""

from __future__ import annotations

import json

import pytest

from evalkit.cli import main
from evalkit.datasets import DatasetManifest, build_manifest, diff_datasets, verify_manifest


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_manifest_canonical_hash_and_tamper_verification(tmp_path):
    v1 = tmp_path / "golden_v1.jsonl"
    _write_jsonl(v1, [{"id": "q1", "question": "2+2?", "answer": "4"}, {"id": "q2", "question": "3+3?", "answer": "6"}])
    manifest = build_manifest(v1, name="math_golden", version="1.0.0")
    manifest_path = manifest.save(tmp_path / "manifest.json")

    loaded = DatasetManifest.load(manifest_path)
    assert loaded.sha256 == manifest.sha256
    assert loaded.num_rows == 2

    # Key-order changes still produce the exact same canonical hash.
    v1_reordered = tmp_path / "golden_reordered.jsonl"
    _write_jsonl(v1_reordered, [{"answer": "4", "question": "2+2?", "id": "q1"}, {"answer": "6", "id": "q2", "question": "3+3?"}])
    assert verify_manifest(v1_reordered, loaded).valid

    # Modifying a row invalidates verification and reports the modified id.
    v1_tampered = tmp_path / "golden_tampered.jsonl"
    _write_jsonl(v1_tampered, [{"id": "q1", "question": "2+2?", "answer": "5"}, {"id": "q2", "question": "3+3?", "answer": "6"}])
    ver = verify_manifest(v1_tampered, loaded)
    assert not ver.valid
    assert ver.modified_ids == ["q1"]


def test_diff_datasets_detects_added_removed_modified_and_schema(tmp_path):
    old_p = tmp_path / "old.jsonl"
    new_p = tmp_path / "new.jsonl"
    _write_jsonl(old_p, [{"id": "1", "q": "a"}, {"id": "2", "q": "b"}, {"id": "3", "q": "c"}])
    _write_jsonl(new_p, [{"id": "1", "q": "a", "tag": "math"}, {"id": "2", "q": "b_edited"}, {"id": "4", "q": "d"}])

    diff = diff_datasets(old_p, new_p)
    assert not diff.identical
    assert diff.added_ids == ["4"]
    assert diff.removed_ids == ["3"]
    assert diff.modified_ids == ["1", "2"]
    assert diff.unchanged_count == 0
    assert diff.schema_added == ["tag"]
    assert "## Dataset Diff" in diff.to_markdown()


def test_cli_dataset_manifest_verify_and_diff(tmp_path, capsys):
    ds1 = tmp_path / "ds1.jsonl"
    ds2 = tmp_path / "ds2.jsonl"
    man = tmp_path / "ds1.manifest.json"
    _write_jsonl(ds1, [{"id": "a", "x": 1}, {"id": "b", "x": 2}])
    _write_jsonl(ds2, [{"id": "a", "x": 1}, {"id": "b", "x": 3}])

    assert main(["dataset", "manifest", str(ds1), "--out", str(man), "--version", "1.2.0"]) == 0
    assert man.exists()

    assert main(["dataset", "verify", str(ds1), str(man)]) == 0
    assert main(["dataset", "verify", str(ds2), str(man)]) == 1

    capsys.readouterr()
    assert main(["dataset", "diff", str(ds1), str(ds2), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["modified_ids"] == ["b"]

    with pytest.raises(ValueError, match="duplicate row id"):
        dup = tmp_path / "dup.jsonl"
        _write_jsonl(dup, [{"id": "same"}, {"id": "same"}])
        build_manifest(dup)
