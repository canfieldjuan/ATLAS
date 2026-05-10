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

target_mode is persisted but NOT filtered in the default read
path -- this preserves parity with the file-backed provider's
``del target_mode`` behavior so a single context row can serve
all five LLM-using outputs. Per-mode filtering is a future
slice; the column is already there.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

        ``target_mode`` is accepted for protocol parity but not
        filtered in this read -- mirrors the file-backed provider's
        ``del target_mode`` behavior. The returned context is
        ``normalize_campaign_reasoning_context``-normalized and
        ``has_content()``-checked; rows that decode to empty
        contexts resolve to ``None`` so callers fall back to
        zero-context defaults rather than passing an empty bundle
        through to the prompt.
        """

        del target_mode
        selectors = _candidate_selectors(
            target_id=target_id,
            opportunity=opportunity,
        )
        if not selectors:
            return None

        row = await self.pool.fetchrow(
            f"""
            SELECT payload
            FROM {self.table}
            WHERE account_id = $1
              AND selectors && $2::text[]
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            scope.account_id or "",
            list(selectors),
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

        context_id = await self.pool.fetchval(
            f"""
            INSERT INTO {self.table} (
                account_id, target_mode, selectors, payload, updated_at
            )
            VALUES ($1, $2, $3::text[], $4::jsonb, NOW())
            RETURNING id
            """,
            scope.account_id or "",
            target_mode or "",
            list(cleaned),
            json_dump_jsonb(payload),
        )
        return str(context_id)


__all__ = [
    "PostgresCampaignReasoningContextRepository",
]
