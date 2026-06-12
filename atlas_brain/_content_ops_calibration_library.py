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

import logging
import uuid as _uuid
from typing import Any, Mapping

from extracted_content_pipeline.calibration_library import CalibrationExample, CalibrationLabel
from extracted_content_pipeline.campaign_ports import TenantScope

logger = logging.getLogger("atlas.content_ops_calibration_library")

_LABEL_VALUES = frozenset(label.value for label in CalibrationLabel)


class ContentOpsCalibrationLibraryReadError(RuntimeError):
    """Raised when tenant calibration data cannot be read safely."""


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


__all__ = [
    "ContentOpsCalibrationLibraryReadError",
    "ContentOpsCalibrationLibraryRepository",
    "list_calibration_example_rows",
]
