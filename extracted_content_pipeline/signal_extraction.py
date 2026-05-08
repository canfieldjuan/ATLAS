"""Deterministic source-material extraction for AI Content Ops."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .campaign_customer_data import CampaignOpportunityWarning
from .campaign_ports import TenantScope
from .campaign_source_adapters import source_rows_to_campaign_opportunities

_ROW_LIST_KEYS = ("sources", "documents", "reviews", "transcripts", "complaints", "rows", "data")


@dataclass(frozen=True)
class SignalExtractionConfig:
    """Config for deterministic source-row extraction."""

    limit: int = 1
    max_text_chars: int = 1200


@dataclass(frozen=True)
class SignalExtractionResult:
    """Normalized opportunities extracted from source material."""

    opportunities: tuple[dict[str, Any], ...]
    warnings: tuple[CampaignOpportunityWarning, ...] = ()
    target_mode: str = "vendor_retention"

    @property
    def generated(self) -> int:
        return len(self.opportunities)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated": self.generated,
            "target_mode": self.target_mode,
            "opportunities": [dict(row) for row in self.opportunities],
            "warnings": [warning.as_dict() for warning in self.warnings],
        }


class SignalExtractionService:
    """Convert host-provided source material into campaign opportunities."""

    def __init__(self, config: SignalExtractionConfig | None = None) -> None:
        self.config = config or SignalExtractionConfig()

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        source_material: Any,
        limit: int | None = None,
        max_text_chars: int | None = None,
        **kwargs: Any,
    ) -> SignalExtractionResult:
        del scope
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
        loaded = source_rows_to_campaign_opportunities(
            _rows_from_source_material(source_material),
            target_mode=target_mode,
            max_text_chars=resolved_max_text_chars,
        )
        return SignalExtractionResult(
            opportunities=tuple(loaded.opportunities[:resolved_limit]),
            warnings=loaded.warnings,
            target_mode=target_mode,
        )


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


__all__ = [
    "SignalExtractionConfig",
    "SignalExtractionResult",
    "SignalExtractionService",
]
