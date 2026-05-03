"""Hierarchical abstraction tiers for extracted reasoning."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from typing import Any

logger = logging.getLogger("extracted_reasoning_core.tiers")


class Tier(IntEnum):
    """Reasoning abstraction tiers."""

    VENDOR_STATE = 1
    VENDOR_ARCHETYPE = 2
    CATEGORY_PATTERN = 3
    MARKET_DYNAMICS = 4


@dataclass(frozen=True)
class TierConfig:
    """Configuration for a single tier."""

    tier: Tier
    name: str
    refresh_interval: timedelta
    pattern_sig_prefix: str
    description: str
    inherits_from: tuple[Tier, ...] = field(default_factory=tuple)


TIER_CONFIGS: dict[Tier, TierConfig] = {
    Tier.VENDOR_STATE: TierConfig(
        tier=Tier.VENDOR_STATE,
        name="Vendor State",
        refresh_interval=timedelta(days=1),
        pattern_sig_prefix="t1",
        description="Deterministic metrics: churn density, urgency, review counts. No LLM.",
        inherits_from=(
            Tier.VENDOR_ARCHETYPE,
            Tier.CATEGORY_PATTERN,
            Tier.MARKET_DYNAMICS,
        ),
    ),
    Tier.VENDOR_ARCHETYPE: TierConfig(
        tier=Tier.VENDOR_ARCHETYPE,
        name="Vendor Archetype",
        refresh_interval=timedelta(weeks=1),
        pattern_sig_prefix="t2",
        description="Vendor-level archetype classification with evidence graph.",
        inherits_from=(Tier.CATEGORY_PATTERN, Tier.MARKET_DYNAMICS),
    ),
    Tier.CATEGORY_PATTERN: TierConfig(
        tier=Tier.CATEGORY_PATTERN,
        name="Category Pattern",
        refresh_interval=timedelta(days=30),
        pattern_sig_prefix="t3",
        description="Category-level baselines and archetype distributions.",
        inherits_from=(Tier.MARKET_DYNAMICS,),
    ),
    Tier.MARKET_DYNAMICS: TierConfig(
        tier=Tier.MARKET_DYNAMICS,
        name="Market Dynamics",
        refresh_interval=timedelta(days=90),
        pattern_sig_prefix="t4",
        description="Market-level displacement patterns and structural shifts.",
        inherits_from=(),
    ),
}


def get_tier_config(tier: Tier) -> TierConfig:
    """Get configuration for a tier."""
    return TIER_CONFIGS[tier]


def build_tiered_pattern_sig(tier: Tier, entity_name: str, evidence_hash: str) -> str:
    """Build a pattern signature scoped to a tier."""
    config = TIER_CONFIGS[tier]
    safe = entity_name.lower().replace(" ", "_").replace(".", "")
    if tier == Tier.MARKET_DYNAMICS:
        return f"{config.pattern_sig_prefix}:market:{safe}:{evidence_hash}"
    if tier == Tier.CATEGORY_PATTERN:
        return f"{config.pattern_sig_prefix}:category:{safe}:{evidence_hash}"
    return f"{config.pattern_sig_prefix}:vendor:{safe}:{evidence_hash}"


def needs_refresh(tier: Tier, last_validated_at: datetime | None) -> bool:
    """Check whether a cached entry for this tier needs re-reasoning."""
    if last_validated_at is None:
        return True
    config = TIER_CONFIGS[tier]
    now = datetime.now(timezone.utc)
    if last_validated_at.tzinfo is None:
        last_validated_at = last_validated_at.replace(tzinfo=timezone.utc)
    return now - last_validated_at > config.refresh_interval


async def gather_tier_context(
    cache: Any,
    tier: Tier,
    vendor_name: str = "",
    product_category: str = "",
) -> dict[str, Any]:
    """Gather inherited context from higher tiers through a cache port."""
    config = TIER_CONFIGS[tier]
    context: dict[str, Any] = {
        "tier": tier.value,
        "tier_name": config.name,
    }
    if not config.inherits_from:
        return context

    inherited_priors: list[dict[str, Any]] = []
    for parent_tier in config.inherits_from:
        parent_config = TIER_CONFIGS[parent_tier]
        entries = []
        if parent_tier == Tier.MARKET_DYNAMICS and product_category:
            entries = await cache.lookup_for_tier(
                "market_dynamics",
                product_category=product_category,
                limit=3,
            )
        elif parent_tier == Tier.CATEGORY_PATTERN and product_category:
            entries = await cache.lookup_for_tier(
                "category_pattern",
                product_category=product_category,
                limit=3,
            )
        elif parent_tier == Tier.VENDOR_ARCHETYPE and vendor_name:
            entries = await cache.lookup_for_tier("", vendor_name=vendor_name, limit=1)
            entries = [entry for entry in entries if entry.vendor_name != "__ecosystem__"]

        for entry in entries:
            prior = {
                "tier": parent_tier.value,
                "tier_name": parent_config.name,
                "pattern": entry.pattern_sig,
                "conclusion_type": entry.conclusion_type,
                "confidence": entry.effective_confidence or entry.confidence,
            }
            conclusion = entry.conclusion
            if isinstance(conclusion, dict):
                for tag_key in (
                    "market_pressure",
                    "category_trend",
                    "archetype",
                    "risk_level",
                    "displacement_direction",
                ):
                    if tag_key in conclusion:
                        prior[tag_key] = conclusion[tag_key]
            inherited_priors.append(prior)

    if inherited_priors:
        context["inherited_priors"] = inherited_priors
        logger.debug("Tier %s gathered %d priors", config.name, len(inherited_priors))
    return context


__all__ = [
    "TIER_CONFIGS",
    "Tier",
    "TierConfig",
    "build_tiered_pattern_sig",
    "gather_tier_context",
    "get_tier_config",
    "needs_refresh",
]
