"""Declared cache strategies for core B2B churn/report/blog LLM stages.

This registry is intentionally explicit. New product/report/blog stages should
add an entry here so cache coverage remains a deliberate design choice rather
than an accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CacheMode = Literal["exact", "semantic", "evidence_hash"]


@dataclass(frozen=True)
class CacheStrategy:
    stage_id: str
    file_path: str
    mode: CacheMode
    namespace: str | None = None
    rationale: str = ""


CORE_B2B_CACHE_STRATEGIES: tuple[CacheStrategy, ...] = (
    CacheStrategy(
        stage_id="b2b_enrichment.tier1",
        file_path="atlas_brain/autonomous/tasks/b2b_enrichment.py",
        mode="exact",
        namespace="b2b_enrichment.tier1",
        rationale="Single-review extraction is a stable exact-repeat workload.",
    ),
    CacheStrategy(
        stage_id="b2b_enrichment.tier2",
        file_path="atlas_brain/autonomous/tasks/b2b_enrichment.py",
        mode="exact",
        namespace="b2b_enrichment.tier2",
        rationale="Tier-2 enrichment is payload-deterministic and cheap to reuse exactly.",
    ),
    CacheStrategy(
        stage_id="b2b_enrichment_repair.extraction",
        file_path="atlas_brain/autonomous/tasks/b2b_enrichment_repair.py",
        mode="exact",
        namespace="b2b_enrichment_repair.extraction",
        rationale="Repair prompts should reuse identical review payloads exactly.",
    ),
    CacheStrategy(
        stage_id="b2b_churn_reports.scorecard_narrative",
        file_path="atlas_brain/autonomous/tasks/b2b_churn_reports.py",
        mode="exact",
        namespace="b2b_churn_reports.scorecard_narrative",
        rationale="Scorecard narratives are exact-repeat renders over deterministic packets.",
    ),
    CacheStrategy(
        stage_id="b2b_tenant_report.synthesis_chunk",
        file_path="atlas_brain/autonomous/tasks/b2b_tenant_report.py",
        mode="exact",
        namespace="b2b_tenant_report.synthesis_chunk",
        rationale="Tenant chunks are exact-repeat synthesis passes over fixed packet slices.",
    ),
    CacheStrategy(
        stage_id="b2b_vendor_briefing.analyst_summary",
        file_path="atlas_brain/autonomous/tasks/b2b_vendor_briefing.py",
        mode="exact",
        namespace="b2b_vendor_briefing.analyst_summary",
        rationale="Analyst summaries are deterministic renders over persisted briefing context.",
    ),
    CacheStrategy(
        stage_id="b2b_vendor_briefing.account_card",
        file_path="atlas_brain/autonomous/tasks/b2b_vendor_briefing.py",
        mode="exact",
        namespace="b2b_vendor_briefing.account_card",
        rationale="Account-card synthesis is exact-repeat on stable account payloads.",
    ),
    CacheStrategy(
        stage_id="b2b_campaign_generation.content",
        file_path="atlas_brain/autonomous/tasks/b2b_campaign_generation.py",
        mode="exact",
        namespace="b2b_campaign_generation.content",
        rationale="Campaign copy generation repeats exactly for unchanged payloads.",
    ),
    CacheStrategy(
        stage_id="b2b_blog_post_generation.content",
        file_path="atlas_brain/autonomous/tasks/b2b_blog_post_generation.py",
        mode="exact",
        namespace="b2b_blog_post_generation.content",
        rationale="Blog drafts reuse exact payloads keyed by slug/topic packet.",
    ),
    CacheStrategy(
        stage_id="b2b_product_profiles.synthesis",
        file_path="atlas_brain/autonomous/tasks/b2b_product_profiles.py",
        mode="exact",
        namespace="b2b_product_profiles.synthesis",
        rationale="Profile synthesis is exact-repeat over stable product profile packets.",
    ),
    CacheStrategy(
        stage_id="b2b_battle_cards.sales_copy",
        file_path="atlas_brain/autonomous/tasks/b2b_battle_cards.py",
        mode="semantic",
        rationale="Battle-card render reuses semantically similar evidence bundles.",
    ),
    CacheStrategy(
        stage_id="b2b_reasoning_synthesis.vendor",
        file_path="atlas_brain/autonomous/tasks/b2b_reasoning_synthesis.py",
        mode="evidence_hash",
        rationale="Vendor reasoning should rerun only when packet evidence changes.",
    ),
    CacheStrategy(
        stage_id="b2b_reasoning_synthesis.cross_vendor",
        file_path="atlas_brain/autonomous/tasks/b2b_reasoning_synthesis.py",
        mode="evidence_hash",
        rationale="Cross-vendor synthesis should rerun only when comparison evidence changes.",
    ),
    CacheStrategy(
        stage_id="win_loss.strategy",
        file_path="atlas_brain/api/b2b_win_loss.py",
        mode="exact",
        namespace="win_loss.strategy",
        rationale="Win/loss strategy is deterministic given the same vendor signals. Cache to avoid redundant LLM calls.",
    ),
)


def iter_core_b2b_cache_strategies() -> tuple[CacheStrategy, ...]:
    """Return the declared cache strategies for current B2B product stages."""
    return CORE_B2B_CACHE_STRATEGIES


def get_b2b_cache_strategy(stage_id: str) -> CacheStrategy | None:
    """Return the declared cache strategy for a stage, if any."""
    for strategy in CORE_B2B_CACHE_STRATEGIES:
        if strategy.stage_id == stage_id:
            return strategy
    return None


def require_b2b_cache_strategy(
    stage_id: str,
    *,
    expected_mode: CacheMode | None = None,
) -> CacheStrategy:
    """Return the declared cache strategy or raise for missing/mismatched config."""
    strategy = get_b2b_cache_strategy(stage_id)
    if strategy is None:
        raise KeyError(f"Missing B2B cache strategy for stage '{stage_id}'")
    if expected_mode is not None and strategy.mode != expected_mode:
        raise ValueError(
            f"Stage '{stage_id}' declares cache mode '{strategy.mode}',"
            f" expected '{expected_mode}'"
        )
    return strategy


def exact_cache_namespaces() -> tuple[str, ...]:
    """Return declared exact-cache namespaces for B2B product stages."""
    return tuple(
        strategy.namespace
        for strategy in CORE_B2B_CACHE_STRATEGIES
        if strategy.mode == "exact" and strategy.namespace
    )
