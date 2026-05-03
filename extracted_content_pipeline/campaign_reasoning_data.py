"""File-backed reasoning context provider for campaign generation examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .campaign_ports import CampaignReasoningContext, TenantScope
from .services.campaign_reasoning_context import normalize_campaign_reasoning_context


_ROW_KEYS = ("contexts", "rows", "data", "reasoning_contexts")
_MATCH_KEYS = {
    "target_id",
    "id",
    "company",
    "company_name",
    "account",
    "account_name",
    "email",
    "contact_email",
    "vendor",
    "vendor_name",
}
_CONTEXT_SELECTOR_KEYS = _MATCH_KEYS | {"target_mode"}
_CONTEXT_FIELD_KEYS = {"context", "reasoning_context", "campaign_reasoning_context"}


@dataclass(frozen=True)
class FileCampaignReasoningContextProvider:
    """CampaignReasoningContextProvider backed by loaded JSON rows.

    This is a reference adapter for hosts that already produce reasoning
    context outside AI Content Ops. It indexes rows by target/company/email
    selectors and returns normalized prompt context without importing a
    reasoning producer.
    """

    contexts: Mapping[str, CampaignReasoningContext]
    source: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "FileCampaignReasoningContextProvider":
        source = Path(path)
        return cls.from_payload(
            json.loads(source.read_text(encoding="utf-8")),
            source=str(source),
        )

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        source: str | None = None,
    ) -> "FileCampaignReasoningContextProvider":
        return cls(contexts=_index_contexts(_context_rows(payload)), source=source)

    async def read_campaign_reasoning_context(
        self,
        *,
        scope: TenantScope,
        target_id: str,
        target_mode: str,
        opportunity: Mapping[str, Any],
    ) -> CampaignReasoningContext | None:
        del scope
        del target_mode
        for key in _candidate_keys(target_id=target_id, opportunity=opportunity):
            context = self.contexts.get(key)
            if context is not None:
                return context
        return None


def load_campaign_reasoning_context_provider(
    path: str | Path,
) -> FileCampaignReasoningContextProvider:
    """Load a file-backed reasoning context provider from JSON."""

    return FileCampaignReasoningContextProvider.from_file(path)


def _context_rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [row for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        raise ValueError("reasoning context JSON must be an object or array")

    for key in _ROW_KEYS:
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [row for row in value if isinstance(row, Mapping)]
        if isinstance(value, Mapping):
            return [
                {"target_id": str(row_key), "context": row_value}
                for row_key, row_value in value.items()
                if isinstance(row_value, Mapping)
            ]

    if _looks_like_mapping_index(payload):
        return [
            {"target_id": str(row_key), "context": row_value}
            for row_key, row_value in payload.items()
            if isinstance(row_value, Mapping)
        ]
    return [payload]


def _looks_like_mapping_index(payload: Mapping[str, Any]) -> bool:
    if not payload:
        return False
    if any(key in payload for key in _CONTEXT_SELECTOR_KEYS | _CONTEXT_FIELD_KEYS):
        return False
    return all(isinstance(value, Mapping) for value in payload.values())


def _index_contexts(rows: Sequence[Mapping[str, Any]]) -> dict[str, CampaignReasoningContext]:
    indexed: dict[str, CampaignReasoningContext] = {}
    for row in rows:
        selectors = _row_selectors(row)
        context = normalize_campaign_reasoning_context(_row_context(row))
        if not selectors or not context.has_content():
            continue
        for selector in selectors:
            indexed.setdefault(selector, context)
    return indexed


def _row_context(row: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = row.get("context")
    if isinstance(nested, Mapping):
        return nested
    return {
        str(key): value
        for key, value in row.items()
        if key not in _CONTEXT_SELECTOR_KEYS and value not in (None, "", [], {})
    }


def _row_selectors(row: Mapping[str, Any]) -> tuple[str, ...]:
    values = [row.get(key) for key in _MATCH_KEYS]
    return _clean_keys(values)


def _candidate_keys(
    *,
    target_id: str,
    opportunity: Mapping[str, Any],
) -> tuple[str, ...]:
    values = [
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
    return _clean_keys(values)


def _clean_keys(values: Sequence[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        for key in (text, text.lower()):
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(key)
    return tuple(cleaned)


__all__ = [
    "FileCampaignReasoningContextProvider",
    "load_campaign_reasoning_context_provider",
]
