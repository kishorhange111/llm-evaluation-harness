"""Golden dataset versioning, canonical manifest hashing, verification, and diffing.

Ensures evaluation benchmarks are immutable and auditable across runs:

* :func:`build_manifest`: hashes every JSONL row canonically (whitespace- and
  key-order-invariant) and computes a deterministic dataset SHA-256 digest.
* :func:`verify_manifest`: verifies a JSONL file against a stored manifest and
  pinpoints added, removed, or tampered rows.
* :func:`diff_datasets`: compares two dataset versions at both row and schema
  level, with a human-readable Markdown summary for PR reviews.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _canonical_row_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        obj = json.loads(stripped)
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: expected JSON object row")
        rows.append(obj)
    return rows


def _extract_row_id(row: dict[str, Any], index: int, id_field: str) -> str:
    val = row.get(id_field)
    return str(val) if val is not None else f"row-{index}"


@dataclass(frozen=True)
class DatasetManifest:
    """Immutable content-addressed manifest for a golden evaluation dataset."""

    name: str
    version: str
    sha256: str
    num_rows: int
    id_field: str
    schema_fields: tuple[str, ...]
    row_hashes: dict[str, str]
    created_at: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "sha256": self.sha256,
            "num_rows": self.num_rows,
            "id_field": self.id_field,
            "schema_fields": list(self.schema_fields),
            "row_hashes": self.row_hashes,
            "created_at": self.created_at,
            "source_path": self.source_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            sha256=str(data["sha256"]),
            num_rows=int(data["num_rows"]),
            id_field=str(data.get("id_field", "id")),
            schema_fields=tuple(data.get("schema_fields", ())),
            row_hashes=dict(data.get("row_hashes", {})),
            created_at=str(data.get("created_at", "")),
            source_path=str(data.get("source_path", "")),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> DatasetManifest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _dataset_digest(row_hashes: dict[str, str]) -> str:
    h = hashlib.sha256()
    for row_id in sorted(row_hashes):
        h.update(f"{row_id}:{row_hashes[row_id]}\n".encode())
    return h.hexdigest()


def build_manifest(
    path: str | Path,
    *,
    name: str | None = None,
    version: str = "1.0.0",
    id_field: str = "id",
    metadata: dict[str, Any] | None = None,
) -> DatasetManifest:
    """Create a canonical :class:`DatasetManifest` for a JSONL dataset file."""
    p = Path(path)
    rows = _load_jsonl_rows(p)
    row_hashes: dict[str, str] = {}
    fields: set[str] = set()
    for i, row in enumerate(rows):
        rid = _extract_row_id(row, i, id_field)
        if rid in row_hashes:
            raise ValueError(f"duplicate row id {rid!r} in {p}")
        row_hashes[rid] = _canonical_row_hash(row)
        fields.update(row.keys())

    return DatasetManifest(
        name=name or p.stem,
        version=version,
        sha256=_dataset_digest(row_hashes),
        num_rows=len(rows),
        id_field=id_field,
        schema_fields=tuple(sorted(fields)),
        row_hashes=row_hashes,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_path=str(p),
        metadata=dict(metadata or {}),
    )


@dataclass(frozen=True)
class ManifestVerification:
    """Result of verifying a dataset file against a stored manifest."""

    valid: bool
    expected_sha256: str
    actual_sha256: str
    added_ids: list[str]
    removed_ids: list[str]
    modified_ids: list[str]


def verify_manifest(path: str | Path, manifest: DatasetManifest) -> ManifestVerification:
    """Verify ``path`` matches ``manifest`` and report any row-level drift."""
    current = build_manifest(path, name=manifest.name, version=manifest.version, id_field=manifest.id_field)
    old_ids = set(manifest.row_hashes)
    new_ids = set(current.row_hashes)
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    modified = sorted(rid for rid in (old_ids & new_ids) if manifest.row_hashes[rid] != current.row_hashes[rid])
    valid = current.sha256 == manifest.sha256 and not added and not removed and not modified
    return ManifestVerification(
        valid=valid,
        expected_sha256=manifest.sha256,
        actual_sha256=current.sha256,
        added_ids=added,
        removed_ids=removed,
        modified_ids=modified,
    )


@dataclass(frozen=True)
class DatasetDiff:
    """Row-level and schema-level diff between two JSONL dataset versions."""

    old_sha256: str
    new_sha256: str
    old_count: int
    new_count: int
    added_ids: list[str]
    removed_ids: list[str]
    modified_ids: list[str]
    unchanged_count: int
    schema_added: list[str]
    schema_removed: list[str]

    @property
    def identical(self) -> bool:
        return self.old_sha256 == self.new_sha256

    def to_markdown(self) -> str:
        lines = [
            "## Dataset Diff",
            "",
            f"- **Old SHA-256:** `{self.old_sha256[:12]}` ({self.old_count} rows)",
            f"- **New SHA-256:** `{self.new_sha256[:12]}` ({self.new_count} rows)",
            f"- **Added:** {len(self.added_ids)} | **Removed:** {len(self.removed_ids)} | "
            f"**Modified:** {len(self.modified_ids)} | **Unchanged:** {self.unchanged_count}",
        ]
        if self.schema_added or self.schema_removed:
            lines.append(f"- **Schema added:** {self.schema_added} | **Schema removed:** {self.schema_removed}")
        return "\n".join(lines) + "\n"


def diff_datasets(old_path: str | Path, new_path: str | Path, *, id_field: str = "id") -> DatasetDiff:
    """Compute a structured diff between two JSONL dataset files."""
    m_old = build_manifest(old_path, id_field=id_field)
    m_new = build_manifest(new_path, id_field=id_field)
    old_ids, new_ids = set(m_old.row_hashes), set(m_new.row_hashes)
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    shared = old_ids & new_ids
    modified = sorted(rid for rid in shared if m_old.row_hashes[rid] != m_new.row_hashes[rid])
    unchanged = len(shared) - len(modified)
    old_fields, new_fields = set(m_old.schema_fields), set(m_new.schema_fields)
    return DatasetDiff(
        old_sha256=m_old.sha256,
        new_sha256=m_new.sha256,
        old_count=m_old.num_rows,
        new_count=m_new.num_rows,
        added_ids=added,
        removed_ids=removed,
        modified_ids=modified,
        unchanged_count=unchanged,
        schema_added=sorted(new_fields - old_fields),
        schema_removed=sorted(old_fields - new_fields),
    )


__all__ = [
    "DatasetDiff",
    "DatasetManifest",
    "ManifestVerification",
    "build_manifest",
    "diff_datasets",
    "verify_manifest",
]
