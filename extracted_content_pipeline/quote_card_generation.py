"""Deterministic quote-card drafts from Content Ops source material."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .campaign_customer_data import CampaignOpportunityWarning
from .campaign_ports import TenantScope
from .campaign_source_adapters import (
    source_material_to_source_rows,
    source_row_to_campaign_opportunity,
)


@dataclass(frozen=True)
class QuoteCardGenerationConfig:
    """Config for deterministic source-material quote cards."""

    limit: int = 3
    max_text_chars: int = 600
    max_quote_chars: int = 180
    max_headline_chars: int = 90
    max_supporting_text_chars: int = 160


@dataclass(frozen=True)
class QuoteCardGenerationResult:
    """Generated quote cards plus non-fatal source-material warnings."""

    cards: tuple[dict[str, Any], ...]
    warnings: tuple[CampaignOpportunityWarning, ...] = ()
    target_mode: str = "vendor_retention"

    @property
    def generated(self) -> int:
        return len(self.cards)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated": self.generated,
            "target_mode": self.target_mode,
            "cards": [dict(card) for card in self.cards],
            "warnings": [warning.as_dict() for warning in self.warnings],
        }


class QuoteCardGenerationService:
    """Build short, evidence-backed quote-card drafts from source material."""

    def __init__(self, config: QuoteCardGenerationConfig | None = None) -> None:
        self.config = config or QuoteCardGenerationConfig()

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        source_material: Any,
        limit: int | None = None,
        max_text_chars: int | None = None,
        **kwargs: Any,
    ) -> QuoteCardGenerationResult:
        del scope, kwargs
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
        return _generate_quote_cards(
            _rows_from_source_material(source_material),
            target_mode=target_mode,
            limit=resolved_limit,
            max_text_chars=resolved_max_text_chars,
            max_quote_chars=self.config.max_quote_chars,
            max_headline_chars=self.config.max_headline_chars,
            max_supporting_text_chars=self.config.max_supporting_text_chars,
        )


def _generate_quote_cards(
    rows: Sequence[Any],
    *,
    target_mode: str,
    limit: int,
    max_text_chars: int,
    max_quote_chars: int,
    max_headline_chars: int,
    max_supporting_text_chars: int,
) -> QuoteCardGenerationResult:
    cards: list[dict[str, Any]] = []
    warnings: list[CampaignOpportunityWarning] = []
    for index, row in enumerate(rows, start=1):
        if len(cards) >= limit:
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
        card = _card_from_opportunity(
            opportunity,
            index=len(cards) + 1,
            max_quote_chars=max_quote_chars,
            max_headline_chars=max_headline_chars,
            max_supporting_text_chars=max_supporting_text_chars,
        )
        if card is None:
            warnings.append(CampaignOpportunityWarning(
                code="missing_quote_card_evidence",
                row_index=index,
                message="Skipped source row because it did not contain usable evidence.",
            ))
            continue
        cards.append(card)
    return QuoteCardGenerationResult(
        cards=tuple(cards),
        warnings=tuple(warnings),
        target_mode=target_mode,
    )


def _card_from_opportunity(
    opportunity: Mapping[str, Any],
    *,
    index: int,
    max_quote_chars: int,
    max_headline_chars: int,
    max_supporting_text_chars: int,
) -> dict[str, Any] | None:
    if not opportunity:
        return None
    evidence = _first_evidence(opportunity)
    if not evidence:
        return None
    vendor = _clean(opportunity.get("vendor_name") or opportunity.get("vendor"))
    company = _clean(opportunity.get("company_name"))
    pain = _first_text(opportunity.get("pain_points"))
    source_id = _clean(opportunity.get("source_id") or opportunity.get("target_id"))
    headline = (
        f"Customer proof for {vendor}"
        if vendor
        else "Customer proof"
    )
    supporting_parts = ["Use this quote"]
    if pain:
        supporting_parts.append(f"to frame {pain}")
    elif vendor:
        supporting_parts.append(f"to frame {vendor} buyer evidence")
    else:
        supporting_parts.append("to frame buyer evidence")
    supporting_text = " ".join(supporting_parts) + "."
    return {
        "id": source_id or f"quote-card-{index}",
        "theme": "customer_proof",
        "quote": _truncate(evidence, max_quote_chars),
        "attribution": company or "Customer evidence",
        "headline": _truncate(headline, max_headline_chars),
        "supporting_text": _truncate(
            supporting_text,
            max_supporting_text_chars,
        ),
        "source_id": source_id,
        "source_type": _clean(opportunity.get("source_type")),
        "target_id": _clean(opportunity.get("target_id")),
        "company_name": company,
        "vendor_name": vendor,
        "pain_points": list(opportunity.get("pain_points") or ()),
    }


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
    return source_material_to_source_rows(source_material)


def _truncate(value: str, max_chars: int) -> str:
    text = " ".join(value.split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _clean(value: Any) -> str:
    return str(value or "").strip()


__all__ = [
    "QuoteCardGenerationConfig",
    "QuoteCardGenerationResult",
    "QuoteCardGenerationService",
]
