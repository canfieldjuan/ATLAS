"""Read-only landing page draft export helpers for AI Content Ops hosts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from io import StringIO
import json
from typing import Any
from urllib.parse import urlsplit

from .campaign_ports import TenantScope
from .landing_page_ports import LandingPageDraft, LandingPageRepository
from .landing_page_readiness import (
    landing_page_geo_readiness,
    landing_page_seo_aeo_readiness,
)
from .landing_page_structured_data import build_landing_page_structured_data


JsonDict = dict[str, Any]

_PUBLIC_CTA_PLACEHOLDER_URLS = frozenset({
    "#",
    "/#",
    "/demo",
    "javascript:void(0)",
    "javascript:;",
})


_EXPORT_COLUMNS = (
    "campaign_name",
    "persona",
    "value_prop",
    "title",
    "slug",
    "section_count",
    "reference_count",
    "generation_input_tokens",
    "generation_output_tokens",
    "generation_total_tokens",
    "generation_parse_attempts",
    "generation_quality_repair_attempts",
    "generation_quality_repair_history",
    "reasoning_context_used",
    "reasoning_wedge",
    "reasoning_confidence",
    "passed_output_checks",
    "output_checks",
    "seo_aeo_readiness",
    "geo_readiness",
    "structured_data",
    "hero",
    "sections",
    "cta",
    "meta",
    "reference_ids",
    "metadata",
    "id",
    "status",
)


@dataclass(frozen=True)
class LandingPageDraftExportResult:
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
        handle = StringIO()
        writer = csv.DictWriter(handle, fieldnames=list(_EXPORT_COLUMNS))
        writer.writeheader()
        for row in self.rows:
            writer.writerow({
                column: _csv_value(row.get(column))
                for column in _EXPORT_COLUMNS
            })
        return handle.getvalue()


async def export_landing_page_drafts(
    repository: LandingPageRepository,
    *,
    scope: TenantScope | Mapping[str, Any] | None = None,
    status: str | None = "draft",
    campaign_name: str | None = None,
    slug: str | None = None,
    ids: Sequence[str] | None = None,
    limit: int = 20,
) -> LandingPageDraftExportResult:
    """Return generated landing page drafts for host review/export workflows."""

    tenant = _tenant_scope(scope)
    normalized_limit = _normalize_limit(limit)
    normalized_ids = _normalize_ids(ids)
    filters: dict[str, Any] = {}
    if status:
        filters["status"] = status
    if tenant.account_id:
        filters["account_id"] = tenant.account_id
    if campaign_name:
        filters["campaign_name"] = campaign_name
    if slug:
        filters["slug"] = slug
    if normalized_ids:
        filters["ids"] = list(normalized_ids)
    drafts = await repository.list_drafts(
        scope=tenant,
        status=status,
        campaign_name=campaign_name,
        slug=slug,
        ids=normalized_ids or None,
        limit=normalized_limit,
    )
    return LandingPageDraftExportResult(
        rows=tuple(_draft_row(draft) for draft in drafts),
        limit=normalized_limit,
        filters=filters,
    )


def _draft_row(draft: LandingPageDraft) -> JsonDict:
    row = draft.as_dict()
    row["section_count"] = len(draft.sections)
    row["reference_count"] = len(draft.reference_ids)
    row.update(_metadata_summary(draft.metadata))
    readiness = landing_page_seo_aeo_readiness(draft)
    row["output_checks"] = readiness["checks"]
    row["passed_output_checks"] = readiness["passed"]
    row["seo_aeo_readiness"] = readiness
    row["geo_readiness"] = landing_page_geo_readiness(draft)
    row["structured_data"] = build_landing_page_structured_data(draft)
    return row


def landing_page_draft_export_row(draft: LandingPageDraft) -> JsonDict:
    """Return the host-facing export row for a single landing-page draft."""

    return _draft_row(draft)


def public_landing_page_draft_row(draft: LandingPageDraft) -> JsonDict:
    """Return the unauthenticated public renderer payload for one draft."""

    return {
        "id": draft.id,
        "slug": draft.slug,
        "title": draft.title,
        "persona": draft.persona,
        "value_prop": draft.value_prop,
        "hero": draft.hero,
        "sections": list(draft.sections),
        "cta": draft.cta,
        "meta": draft.meta,
        "robots": public_landing_page_robots(draft),
        "structured_data": build_landing_page_structured_data(draft),
    }


def public_landing_page_robots(draft: LandingPageDraft) -> str:
    """Return the public robots policy for a generated landing-page draft."""

    if draft.status != "approved":
        return "noindex,follow"
    if landing_page_seo_aeo_readiness(draft).get("status") != "ready":
        return "noindex,follow"
    if landing_page_geo_readiness(draft).get("status") != "ready":
        return "noindex,follow"
    if not _public_cta_url_indexable(draft.cta):
        return "noindex,follow"
    return "index,follow"


def _public_cta_url_indexable(value: Any) -> bool:
    cta = _metadata_mapping(value)
    url = str(cta.get("url") or "").strip()
    normalized = url.lower()
    if not normalized or normalized in _PUBLIC_CTA_PLACEHOLDER_URLS:
        return False
    if normalized.startswith("javascript:"):
        return False
    if _public_cta_local_path(normalized) == "/demo":
        return False
    if normalized.startswith(("https://", "http://", "mailto:", "tel:", "/")):
        return True
    return False


def _public_cta_local_path(url: str) -> str:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return ""
    if parsed.scheme or parsed.netloc:
        return ""
    return parsed.path.rstrip("/") or parsed.path


def _metadata_summary(value: Any) -> JsonDict:
    metadata = _metadata_mapping(value)
    usage = _metadata_mapping(metadata.get("generation_usage"))
    reasoning = _metadata_mapping(metadata.get("reasoning_context"))
    return {
        "generation_input_tokens": usage.get("input_tokens"),
        "generation_output_tokens": usage.get("output_tokens"),
        "generation_total_tokens": usage.get("total_tokens"),
        "generation_parse_attempts": metadata.get("generation_parse_attempts"),
        "generation_quality_repair_attempts": metadata.get(
            "generation_quality_repair_attempts"
        ),
        "generation_quality_repair_history": metadata.get(
            "generation_quality_repair_history"
        ),
        "reasoning_context_used": bool(reasoning),
        "reasoning_wedge": reasoning.get("wedge"),
        "reasoning_confidence": reasoning.get("confidence"),
    }


def _metadata_mapping(value: Any) -> JsonDict:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return {str(key): item for key, item in parsed.items()}
    return {}


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, default=str, separators=(",", ":"))
    return "" if value is None else value


def _normalize_limit(value: Any) -> int:
    limit = int(value)
    if limit < 0:
        raise ValueError("limit must be non-negative")
    return limit


def _normalize_ids(value: Sequence[str] | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _tenant_scope(value: TenantScope | Mapping[str, Any] | None) -> TenantScope:
    if isinstance(value, TenantScope):
        return value
    if isinstance(value, Mapping):
        return TenantScope(
            account_id=str(value.get("account_id") or "") or None,
            user_id=str(value.get("user_id") or "") or None,
        )
    return TenantScope()


__all__ = [
    "LandingPageDraftExportResult",
    "export_landing_page_drafts",
    "landing_page_draft_export_row",
    "public_landing_page_draft_row",
    "public_landing_page_robots",
]
