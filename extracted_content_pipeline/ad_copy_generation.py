"""Deterministic ad-copy drafts from Content Ops source material."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .ad_copy_ports import AdCopyDraft, AdCopyRepository
from .campaign_customer_data import CampaignOpportunityWarning
from .campaign_ports import TenantScope
from .campaign_source_adapters import source_row_to_campaign_opportunity

_ROW_LIST_KEYS = ("sources", "opportunities", "reviews", "documents", "rows", "data")


@dataclass(frozen=True)
class AdCopyGenerationConfig:
    """Config for deterministic source-material ad copy."""

    limit: int = 3
    max_text_chars: int = 600
    max_headline_chars: int = 90
    max_primary_text_chars: int = 240


@dataclass(frozen=True)
class AdCopyGenerationResult:
    """Generated ad copy plus non-fatal source-material warnings."""

    ads: tuple[dict[str, Any], ...]
    warnings: tuple[CampaignOpportunityWarning, ...] = ()
    target_mode: str = "vendor_retention"
    saved_ids: tuple[str, ...] = ()

    @property
    def generated(self) -> int:
        return len(self.ads)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated": self.generated,
            "target_mode": self.target_mode,
            "ads": [dict(ad) for ad in self.ads],
            "warnings": [warning.as_dict() for warning in self.warnings],
            "saved_ids": list(self.saved_ids),
        }


class AdCopyGenerationService:
    """Build short, evidence-backed ad copy from source material."""

    def __init__(
        self,
        config: AdCopyGenerationConfig | None = None,
        *,
        ad_copy_drafts: AdCopyRepository | None = None,
    ) -> None:
        self.config = config or AdCopyGenerationConfig()
        self._ad_copy_drafts = ad_copy_drafts

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        source_material: Any,
        limit: int | None = None,
        max_text_chars: int | None = None,
        **kwargs: Any,
    ) -> AdCopyGenerationResult:
        del kwargs
        resolved_limit = int(limit) if limit is not None else self.config.limit
        if resolved_limit < 1:
            raise ValueError("limit must be at least 1")
        resolved_max_text_chars = (
            int(max_text_chars)
            if max_text_chars is not None
            else self.config.max_text_chars
        )
        if resolved_max_text_chars < 1:
            raise ValueError("max_text_chars must be at least 1")
        result = _generate_ad_copy(
            _rows_from_source_material(source_material),
            target_mode=target_mode,
            limit=resolved_limit,
            max_text_chars=resolved_max_text_chars,
            max_headline_chars=self.config.max_headline_chars,
            max_primary_text_chars=self.config.max_primary_text_chars,
        )
        if self._ad_copy_drafts is None or not result.ads:
            return result
        saved_ids = tuple(
            str(item)
            for item in await self._ad_copy_drafts.save_drafts(
                _drafts_from_ads(result.ads, target_mode=target_mode),
                scope=scope,
            )
        )
        return replace(result, saved_ids=saved_ids)


def _generate_ad_copy(
    rows: Sequence[Any],
    *,
    target_mode: str,
    limit: int,
    max_text_chars: int,
    max_headline_chars: int,
    max_primary_text_chars: int,
) -> AdCopyGenerationResult:
    ads: list[dict[str, Any]] = []
    warnings: list[CampaignOpportunityWarning] = []
    for index, row in enumerate(rows, start=1):
        if len(ads) >= limit:
            break
        if not isinstance(row, Mapping):
            warnings.append(CampaignOpportunityWarning(
                code="row_not_object",
                row_index=index,
                message="Skipped source row because it is not an object.",
            ))
            continue
        opportunity, row_warnings = source_row_to_campaign_opportunity(
            row,
            row_index=index,
            max_text_chars=max_text_chars,
        )
        warnings.extend(row_warnings)
        ad = _ad_from_opportunity(
            opportunity,
            index=len(ads) + 1,
            max_headline_chars=max_headline_chars,
            max_primary_text_chars=max_primary_text_chars,
        )
        if ad is None:
            warnings.append(CampaignOpportunityWarning(
                code="missing_ad_copy_evidence",
                row_index=index,
                message="Skipped source row because it did not contain usable evidence.",
            ))
            continue
        ads.append(ad)
    return AdCopyGenerationResult(
        ads=tuple(ads),
        warnings=tuple(warnings),
        target_mode=target_mode,
    )


def _ad_from_opportunity(
    opportunity: Mapping[str, Any],
    *,
    index: int,
    max_headline_chars: int,
    max_primary_text_chars: int,
) -> dict[str, Any] | None:
    if not opportunity:
        return None
    evidence = _first_evidence(opportunity)
    if not evidence:
        return None
    vendor = _clean(opportunity.get("vendor_name") or opportunity.get("vendor"))
    pain = _first_text(opportunity.get("pain_points"))
    source_id = _clean(opportunity.get("source_id") or opportunity.get("target_id"))
    headline_subject = pain or "customer proof"
    headline = (
        f"{vendor} proof: {headline_subject}"
        if vendor
        else f"Customer proof: {headline_subject}"
    )
    primary_parts = []
    if vendor and pain:
        primary_parts.append(f"When {vendor} buyers mention {pain}, use the proof.")
    elif vendor:
        primary_parts.append(f"Use real buyer evidence for {vendor}.")
    elif pain:
        primary_parts.append(f"Use real buyer evidence about {pain}.")
    else:
        primary_parts.append("Use real buyer evidence.")
    primary_parts.append(f'"{evidence}"')
    primary_parts.append("Turn review signal into the next campaign.")
    return {
        "id": source_id or f"ad-copy-{index}",
        "channel": "paid_social",
        "format": "single_image",
        "headline": _truncate(headline, max_headline_chars),
        "primary_text": _truncate(" ".join(primary_parts), max_primary_text_chars),
        "cta": "See the proof",
        "source_id": source_id,
        "source_type": _clean(opportunity.get("source_type")),
        "target_id": _clean(opportunity.get("target_id")),
        "company_name": _clean(opportunity.get("company_name")),
        "vendor_name": vendor,
        "pain_points": list(opportunity.get("pain_points") or ()),
    }


def _drafts_from_ads(
    ads: Sequence[Mapping[str, Any]],
    *,
    target_mode: str,
) -> tuple[AdCopyDraft, ...]:
    drafts: list[AdCopyDraft] = []
    for ad in ads:
        source_id = _clean(ad.get("source_id") or ad.get("id"))
        target_id = _clean(ad.get("target_id")) or source_id
        drafts.append(
            AdCopyDraft(
                target_id=target_id,
                target_mode=target_mode,
                channel=_clean(ad.get("channel")) or "paid_social",
                format=_clean(ad.get("format")) or "single_image",
                headline=_clean(ad.get("headline")),
                primary_text=_clean(ad.get("primary_text")),
                cta=_clean(ad.get("cta")),
                source_id=source_id,
                source_type=_clean(ad.get("source_type")),
                company_name=_clean(ad.get("company_name")),
                vendor_name=_clean(ad.get("vendor_name")),
                pain_points=_pain_points_from_ad(ad.get("pain_points")),
                metadata={"source_ad": dict(ad)},
            )
        )
    return tuple(drafts)


def _pain_points_from_ad(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ()
    return tuple(_clean(item) for item in value if _clean(item))


def _first_evidence(opportunity: Mapping[str, Any]) -> str:
    raw = opportunity.get("evidence")
    if isinstance(raw, Mapping):
        return _clean(raw.get("text"))
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if isinstance(item, Mapping):
                text = _clean(item.get("text"))
                if text:
                    return text
            else:
                text = _clean(item)
                if text:
                    return text
    return _clean(raw)


def _first_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            text = _clean(item)
            if text:
                return text
    return _clean(value)


def _rows_from_source_material(source_material: Any) -> list[Any]:
    if isinstance(source_material, str):
        text = source_material.strip()
        return [{"text": text}] if text else []
    if isinstance(source_material, Mapping):
        for key in _ROW_LIST_KEYS:
            value = source_material.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return list(value)
        return [dict(source_material)]
    if isinstance(source_material, Sequence) and not isinstance(source_material, (bytes, bytearray)):
        return list(source_material)
    return []


def _truncate(value: str, max_chars: int) -> str:
    text = " ".join(value.split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _clean(value: Any) -> str:
    return str(value or "").strip()


__all__ = [
    "AdCopyGenerationConfig",
    "AdCopyGenerationResult",
    "AdCopyGenerationService",
    "AdCopyDraft",
    "AdCopyRepository",
]
