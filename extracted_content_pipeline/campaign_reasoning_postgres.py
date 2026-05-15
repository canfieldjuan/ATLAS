"""Postgres adapter for the Content Ops campaign reasoning context provider.

DB-backed counterpart to ``FileCampaignReasoningContextProvider``
(``extracted_content_pipeline/campaign_reasoning_data.py``). Both
implement ``CampaignReasoningProviderPort`` so the host can swap
providers without changing the route mount (PR #402 / PR #462) or
the bundle's ``with_reasoning_context()`` derivation.

Read path mirrors the file-backed provider's ``_candidate_keys``
selector set: the per-call lookup gathers ``target_id`` plus the
opportunity's ``id`` / ``company_name`` / ``contact_email`` /
``vendor_name`` (and lowercase variants) into a TEXT[] argument
matched via the GIN-indexed ``selectors && $2::text[]`` predicate.
``ORDER BY updated_at DESC LIMIT 1`` gives the "newest match wins"
tie-breaker that matches ``setdefault`` first-key-wins in the
file index.

Write path (``save_context``) is optional -- hosts that already
populate the table from their own ETL don't need it. The bundled
implementation lets host scripts persist a normalized
``CampaignReasoningContext`` payload directly, mirroring
``PostgresBlogBlueprintRepository.save_blueprints``.

target_mode is persisted and filtered on reads. Blank target-mode
rows are treated as global fallback contexts so legacy seed data can
still serve multiple outputs, but a row saved for one nonblank mode
does not satisfy another mode when selectors overlap.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from typing import Any

from .campaign_ports import CampaignReasoningContext, JsonDict, TenantScope
from .services.campaign_reasoning_context import (
    campaign_reasoning_context_metadata,
    normalize_campaign_reasoning_context,
)
from .storage._jsonb_helpers import (
    decode_jsonb_field,
    json_dump_jsonb,
    row_to_dict,
)


def _candidate_selectors(
    *,
    target_id: str,
    opportunity: Mapping[str, Any],
) -> tuple[str, ...]:
    """Build the lookup TEXT[] for a per-target read.

    Mirrors ``_candidate_keys`` in
    ``extracted_content_pipeline/campaign_reasoning_data.py`` so a
    DB row indexed under any of the file-loader's selector forms is
    findable by the same opportunity payload. Order is preserved
    for callers that may inspect the args; uniqueness is enforced
    with both case-as-given and lowercase variants.
    """

    raw_values = [
        target_id,
        opportunity.get("target_id"),
        opportunity.get("id"),
        opportunity.get("company_name"),
        opportunity.get("company"),
        opportunity.get("contact_email"),
        opportunity.get("email"),
        opportunity.get("vendor_name"),
        opportunity.get("vendor"),
    ]
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in raw_values:
        text = str(value or "").strip()
        if not text:
            continue
        for key in (text, text.lower()):
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(key)
    return tuple(cleaned)


def _dedupe_selectors(values: Sequence[Any]) -> tuple[str, ...]:
    """Dedupe a save-side selector list, persisting both
    case-as-given and lowercase forms so the read path's
    lowercase normalization keeps matching without a runtime
    LOWER() over the column.
    """

    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        for variant in (text, text.lower()):
            if variant in seen:
                continue
            seen.add(variant)
            cleaned.append(variant)
    return tuple(cleaned)


def _selector_key(values: Sequence[str]) -> str:
    """Return the order-independent persistence key for selectors."""

    canonical = "\x1f".join(sorted(values))
    return hashlib.md5(
        canonical.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()


def _identifier(value: str) -> str:
    parts = str(value or "").strip().split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid SQL identifier: {value!r}")
    for part in parts:
        if not all(char.isalnum() or char == "_" for char in part):
            raise ValueError(f"invalid SQL identifier: {value!r}")
    return ".".join(f'"{part}"' for part in parts)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, default=str, separators=(",", ":"))
    return "" if value is None else value


def _serializable_context_row(row: Mapping[str, Any]) -> JsonDict:
    payload = decode_jsonb_field(row.get("payload"), default=None)
    output = {
        key: _json_ready(value)
        for key, value in row.items()
        if key != "payload"
    }
    output["payload"] = _json_ready({} if payload is None else payload)
    return output


@dataclass(frozen=True)
class CampaignReasoningContextListResult:
    rows: tuple[JsonDict, ...]
    limit: int
    filters: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.rows),
            "limit": self.limit,
            "filters": dict(self.filters),
            "rows": [dict(row) for row in self.rows],
        }

    def as_csv(self) -> str:
        columns = (
            "id",
            "account_id",
            "target_mode",
            "selectors",
            "selector_key",
            "updated_at",
            "payload",
        )
        handle = StringIO()
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in self.rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})
        return handle.getvalue()


@dataclass(frozen=True)
class PostgresCampaignReasoningContextRepository:
    """Async Postgres adapter for per-tenant campaign reasoning contexts."""

    pool: Any
    table: str = "campaign_reasoning_contexts"

    async def read_campaign_reasoning_context(
        self,
        *,
        scope: TenantScope,
        target_id: str,
        target_mode: str,
        opportunity: Mapping[str, Any],
    ) -> CampaignReasoningContext | None:
        """Return the newest matching context for ``target_id``,
        or ``None`` when no row matches.

        ``target_mode`` filters nonblank rows so mode-specific
        contexts do not leak across assets. Blank target-mode rows
        remain global fallbacks. The returned context is
        ``normalize_campaign_reasoning_context``-normalized and
        ``has_content()``-checked; rows that decode to empty contexts
        resolve to ``None`` so callers fall back to zero-context
        defaults rather than passing an empty bundle through to the
        prompt.
        """

        mode = str(target_mode or "").strip().lower()
        selectors = _candidate_selectors(
            target_id=target_id,
            opportunity=opportunity,
        )
        if not selectors:
            return None
        table = _identifier(self.table)

        # Preserve the file-backed provider's candidate-key
        # priority: ``_candidate_selectors`` emits target_id first,
        # then opportunity.target_id / id / company_name / company /
        # contact_email / email / vendor_name / vendor (with
        # lowercase variants). A pure ``ORDER BY updated_at DESC``
        # makes a broader newer row beat an exact ``target_id``
        # row whose selectors also overlap -- divergent from
        # ``FileCampaignReasoningContextProvider`` which probes
        # candidates in order via ``setdefault``. The subquery
        # computes the row's "priority" as the MIN candidate
        # position whose value overlaps the row's selectors;
        # lower priority wins, with ``updated_at`` as the
        # within-priority tie-break.
        row = await self.pool.fetchrow(
            f"""
            SELECT payload
            FROM {table}
            WHERE account_id = $1
              AND selectors && $2::text[]
              AND ($3 = '' OR target_mode = $3 OR target_mode = '')
            ORDER BY (
                SELECT MIN(c.idx)
                FROM unnest($2::text[]) WITH ORDINALITY AS c(val, idx)
                WHERE c.val = ANY(selectors)
            ) ASC NULLS LAST,
            CASE WHEN $3 <> '' AND target_mode = $3 THEN 0 ELSE 1 END ASC,
            updated_at DESC
            LIMIT 1
            """,
            scope.account_id or "",
            list(selectors),
            mode,
        )
        if row is None:
            return None
        payload = decode_jsonb_field(row_to_dict(row).get("payload"), default={})
        if not isinstance(payload, Mapping) or not payload:
            return None
        context = normalize_campaign_reasoning_context(payload)
        return context if context.has_content() else None

    async def save_context(
        self,
        *,
        scope: TenantScope,
        selectors: Sequence[str],
        context: CampaignReasoningContext | Mapping[str, Any],
        target_mode: str = "",
    ) -> str:
        """Persist a normalized context row and return its id.

        Outside the upstream Protocol -- hosts with their own
        reasoning ETL don't need this. Callers can pass either a
        normalized ``CampaignReasoningContext`` or a raw mapping
        (the same shape ``FileCampaignReasoningContextProvider``
        accepts); both round-trip through
        ``campaign_reasoning_context_metadata`` so the persisted
        JSONB matches the file-backed loader's expected layout.

        Selectors are deduped (case-as-given + lowercase) so the
        GIN-indexed read predicate matches without a runtime
        LOWER(). An empty cleaned-selectors list raises
        ``ValueError`` -- a row with no selectors is unreachable
        and almost certainly an upstream ETL bug.
        """

        if isinstance(context, CampaignReasoningContext):
            normalized = context
        else:
            normalized = normalize_campaign_reasoning_context(context)

        cleaned = _dedupe_selectors(selectors)
        if not cleaned:
            raise ValueError(
                "save_context requires at least one non-empty selector"
            )

        payload: JsonDict = campaign_reasoning_context_metadata(normalized)
        selector_key = _selector_key(cleaned)
        table = _identifier(self.table)

        context_id = await self.pool.fetchval(
            f"""
            INSERT INTO {table} (
                account_id, target_mode, selectors, selector_key,
                payload, updated_at
            )
            VALUES ($1, $2, $3::text[], $4, $5::jsonb, NOW())
            ON CONFLICT (account_id, target_mode, selector_key)
            DO UPDATE SET
                selectors = EXCLUDED.selectors,
                payload = EXCLUDED.payload,
                updated_at = NOW()
            RETURNING id
            """,
            scope.account_id or "",
            str(target_mode or "").strip().lower(),
            list(cleaned),
            selector_key,
            json_dump_jsonb(payload),
        )
        return str(context_id)

    async def delete_stale_contexts(
        self,
        *,
        older_than_days: int,
        scope: TenantScope | None = None,
        target_mode: str | None = None,
        dry_run: bool = False,
    ) -> int:
        """Delete or count stale reasoning contexts.

        This is an operational cleanup hook for host installs that run
        periodic reasoning ETL. ``target_mode=None`` means all modes;
        a string value filters to that normalized mode, including the
        blank global-fallback mode when ``target_mode=""``.
        """

        if older_than_days < 1:
            raise ValueError("older_than_days must be at least 1")

        account_id = None if scope is None else str(scope.account_id or "")
        mode = None if target_mode is None else str(target_mode or "").strip().lower()
        table = _identifier(self.table)
        if dry_run:
            stale_count = await self.pool.fetchval(
                f"""
                WITH stale AS (
                    SELECT id
                    FROM {table}
                    WHERE updated_at < NOW() - ($1::int * INTERVAL '1 day')
                      AND ($2::text IS NULL OR account_id = $2)
                      AND ($3::text IS NULL OR target_mode = $3)
                )
                SELECT COUNT(*) FROM stale
                """,
                int(older_than_days),
                account_id,
                mode,
            )
            return int(stale_count or 0)

        stale_count = await self.pool.fetchval(
            f"""
            WITH stale AS (
                SELECT id
                FROM {table}
                WHERE updated_at < NOW() - ($1::int * INTERVAL '1 day')
                  AND ($2::text IS NULL OR account_id = $2)
                  AND ($3::text IS NULL OR target_mode = $3)
            ),
            deleted AS (
                DELETE FROM {table}
                WHERE id IN (SELECT id FROM stale)
                RETURNING 1
            )
            SELECT COUNT(*) FROM deleted
            """,
            int(older_than_days),
            account_id,
            mode,
        )
        return int(stale_count or 0)

    async def list_contexts(
        self,
        *,
        scope: TenantScope | None = None,
        target_mode: str | None = None,
        selectors: Sequence[str] = (),
        limit: int = 20,
    ) -> CampaignReasoningContextListResult:
        """Return reasoning-context rows for operator inventory/export."""

        table = _identifier(self.table)
        normalized_limit = int(limit)
        if normalized_limit < 0:
            raise ValueError("limit must be non-negative")

        account_id = None if scope is None else str(scope.account_id or "")
        mode = None if target_mode is None else str(target_mode or "").strip().lower()
        cleaned_selectors = _dedupe_selectors(selectors)
        rows = await self.pool.fetch(
            f"""
            SELECT
                id, account_id, target_mode, selectors, selector_key,
                COALESCE(payload, '{{}}'::jsonb) AS payload,
                updated_at
            FROM {table}
            WHERE ($1::text IS NULL OR account_id = $1)
              AND ($2::text IS NULL OR target_mode = $2)
              AND ($3::text[] IS NULL OR selectors && $3::text[])
            ORDER BY updated_at DESC
            LIMIT $4
            """,
            account_id,
            mode,
            list(cleaned_selectors) if cleaned_selectors else None,
            normalized_limit,
        )
        return CampaignReasoningContextListResult(
            rows=tuple(_serializable_context_row(row_to_dict(row)) for row in rows),
            limit=normalized_limit,
            filters={
                "account_id": account_id,
                "target_mode": mode,
                "selectors": cleaned_selectors,
            },
        )


__all__ = [
    "CampaignReasoningContextListResult",
    "PostgresCampaignReasoningContextRepository",
]
