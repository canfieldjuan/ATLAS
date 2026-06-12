"""Tenant calibration-library persistence for Content Ops review.

The extracted calibration library (slice 5a) owns the deterministic anchor types
and selection. This module is the host reader around the
``content_ops_calibration_library`` table: it returns a tenant's curated
calibration examples for the marketer verify flow, so the connector need not
resend the whole library on every call.

Read-only by design for this slice: the write surface (create/update/archive)
lands with the admin slice. Anchors are evidence, not a gate, so the review
treats a read failure as "no server-side anchors available" -- this reader still
fails closed on an invalid tenant scope (returns nothing) and wraps database
errors, and the review's merge helper degrades to request-supplied anchors.

No FastAPI, MCP, or LLM code lives here.
"""

from __future__ import annotations

import json
import logging
import uuid as _uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional

from extracted_content_pipeline.calibration_library import CalibrationExample, CalibrationLabel
from extracted_content_pipeline.campaign_ports import TenantScope

logger = logging.getLogger("atlas.content_ops_calibration_library")

_LABEL_VALUES = frozenset(label.value for label in CalibrationLabel)


class ContentOpsCalibrationLibraryReadError(RuntimeError):
    """Raised when tenant calibration data cannot be read safely."""


@dataclass(frozen=True)
class ContentOpsCalibrationLibraryRecord:
    """Display-safe saved calibration-library row."""

    id: _uuid.UUID
    account_id: _uuid.UUID
    example_id: str
    label: str
    excerpt: str
    reasoning: str
    source: str
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    archived_at: Optional[datetime]

    def as_calibration_example(self) -> CalibrationExample:
        return CalibrationExample(
            example_id=self.example_id,
            excerpt=self.excerpt,
            label=CalibrationLabel(self.label),
            reasoning=self.reasoning,
            source=self.source or "curated",
        )


@dataclass(frozen=True)
class _ValidatedExample:
    example_id: str
    label: str
    excerpt: str
    reasoning: str
    source: str
    metadata: dict[str, Any]


class ContentOpsCalibrationLibraryRepository:
    """Postgres-backed implementation of the calibration-library reader."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def list_calibration_examples(
        self,
        *,
        scope: TenantScope,
    ) -> tuple[CalibrationExample, ...]:
        account_id = _scope_account_uuid(scope)
        if account_id is None:
            # Calibration anchors are evidence, not a gate: an unusable scope
            # means "no server-side anchors", not a hard error.
            logger.warning("calibration library read: invalid tenant scope")
            return ()

        try:
            rows = await list_calibration_example_rows(self._pool, account_id=account_id)
        except Exception as exc:
            logger.exception("calibration library read failed")
            raise ContentOpsCalibrationLibraryReadError(
                "calibration library read failed"
            ) from exc

        examples: list[CalibrationExample] = []
        seen_ids: set[str] = set()
        for row in rows:
            example = _example_from_row(row)
            if example is None or example.example_id in seen_ids:
                continue
            seen_ids.add(example.example_id)
            examples.append(example)
        return tuple(examples)


async def list_calibration_example_rows(
    pool: Any,
    *,
    account_id: _uuid.UUID,
) -> list[Mapping[str, Any]]:
    """Return active calibration-library rows for one tenant, newest first."""

    return await pool.fetch(
        """
        SELECT id, account_id, example_id, label, excerpt, reasoning,
               source, metadata, created_at, updated_at, archived_at
          FROM content_ops_calibration_library
         WHERE account_id = $1
           AND archived_at IS NULL
         ORDER BY updated_at DESC
        """,
        account_id,
    )


async def create_calibration_example(
    pool: Any,
    *,
    account_id: _uuid.UUID,
    payload: Mapping[str, Any],
) -> ContentOpsCalibrationLibraryRecord:
    """Create one active calibration-library row for an account."""

    example = _validated_example(payload)
    row = await pool.fetchrow(
        """
        INSERT INTO content_ops_calibration_library (
            account_id, example_id, label, excerpt, reasoning, source, metadata
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
        RETURNING id, account_id, example_id, label, excerpt, reasoning,
                  source, metadata, created_at, updated_at, archived_at
        """,
        account_id,
        example.example_id,
        example.label,
        example.excerpt,
        example.reasoning,
        example.source,
        _json_dump(example.metadata),
    )
    return _display_record(row)


async def update_calibration_example(
    pool: Any,
    *,
    account_id: _uuid.UUID,
    example_row_id: _uuid.UUID,
    payload: Mapping[str, Any],
) -> ContentOpsCalibrationLibraryRecord | None:
    """Replace one active tenant calibration row. Missing/cross-tenant -> None."""

    example = _validated_example(payload)
    row = await pool.fetchrow(
        """
        UPDATE content_ops_calibration_library
           SET example_id = $3,
               label = $4,
               excerpt = $5,
               reasoning = $6,
               source = $7,
               metadata = $8::jsonb,
               updated_at = NOW()
         WHERE id = $1
           AND account_id = $2
           AND archived_at IS NULL
        RETURNING id, account_id, example_id, label, excerpt, reasoning,
                  source, metadata, created_at, updated_at, archived_at
        """,
        example_row_id,
        account_id,
        example.example_id,
        example.label,
        example.excerpt,
        example.reasoning,
        example.source,
        _json_dump(example.metadata),
    )
    return _display_record(row) if row is not None else None


async def list_calibration_example_records(
    pool: Any,
    *,
    account_id: _uuid.UUID,
) -> list[ContentOpsCalibrationLibraryRecord]:
    """Return active calibration-library rows for one tenant as display records."""

    rows = await list_calibration_example_rows(pool, account_id=account_id)
    return [_display_record(row) for row in rows]


async def archive_calibration_example(
    pool: Any,
    *,
    account_id: _uuid.UUID,
    example_row_id: _uuid.UUID,
) -> bool:
    """Soft-delete one tenant calibration-library row."""

    row = await pool.fetchrow(
        """
        UPDATE content_ops_calibration_library
           SET archived_at = NOW(),
               updated_at = NOW()
         WHERE id = $1
           AND account_id = $2
           AND archived_at IS NULL
        RETURNING id
        """,
        example_row_id,
        account_id,
    )
    return row is not None


def _validated_example(payload: Mapping[str, Any]) -> _ValidatedExample:
    if not isinstance(payload, Mapping):
        raise ValueError("Calibration example payload must be an object")

    example_id = _clean(payload.get("example_id"))
    if not example_id:
        raise ValueError("Calibration example id is required")

    label = _clean(payload.get("label"))
    if label not in _LABEL_VALUES:
        raise ValueError(f"Invalid calibration label: {payload.get('label')!r}")

    excerpt = _clean(payload.get("excerpt"))
    if not excerpt:
        raise ValueError("Calibration excerpt is required")

    reasoning = _clean(payload.get("reasoning"))
    if not reasoning:
        raise ValueError("Calibration reasoning is required")

    return _ValidatedExample(
        example_id=example_id,
        label=label,
        excerpt=excerpt,
        reasoning=reasoning,
        source=_clean(payload.get("source")) or "curated",
        metadata=_metadata(payload.get("metadata")),
    )


def _display_record(row: Mapping[str, Any]) -> ContentOpsCalibrationLibraryRecord:
    return ContentOpsCalibrationLibraryRecord(
        id=row["id"],
        account_id=row["account_id"],
        example_id=_clean(row["example_id"]),
        label=_clean(row["label"]),
        excerpt=_clean(row["excerpt"]),
        reasoning=_clean(row["reasoning"]),
        source=_clean(row["source"]) or "curated",
        metadata=_metadata(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


def _example_from_row(row: Mapping[str, Any]) -> CalibrationExample | None:
    """Build a teachable ``CalibrationExample`` from a DB row, or ``None``.

    The table enforces non-empty excerpt/reasoning and a known label, but the
    reader stays defensive: a row that somehow lost its excerpt, reasoning, or a
    recognized label is skipped rather than surfaced as a broken anchor.
    """

    example_id = _clean(row.get("example_id"))
    excerpt = _clean(row.get("excerpt"))
    reasoning = _clean(row.get("reasoning"))
    label = _clean(row.get("label"))
    if not example_id or not excerpt or not reasoning or label not in _LABEL_VALUES:
        return None
    return CalibrationExample(
        example_id=example_id,
        excerpt=excerpt,
        label=CalibrationLabel(label),
        reasoning=reasoning,
        source=_clean(row.get("source")) or "curated",
    )


def _scope_account_uuid(scope: TenantScope | None) -> _uuid.UUID | None:
    value = getattr(scope, "account_id", None)
    if not isinstance(value, str):
        return None
    try:
        return _uuid.UUID(value.strip())
    except ValueError:
        return None


def _clean(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(decoded) if isinstance(decoded, Mapping) else {}
    return {}


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = [
    "ContentOpsCalibrationLibraryReadError",
    "ContentOpsCalibrationLibraryRecord",
    "ContentOpsCalibrationLibraryRepository",
    "archive_calibration_example",
    "create_calibration_example",
    "list_calibration_example_records",
    "list_calibration_example_rows",
    "update_calibration_example",
]
