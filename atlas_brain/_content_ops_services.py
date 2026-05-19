"""Factory for the Content Ops execution-services bundle.

Builds a `ContentOpsExecutionServices` populated with the
generators the host has fully wired. Slots that aren't yet
populated remain `None`; the executor's per-step dispatcher
returns `service_not_configured` for unset slots, which the
route layer maps to a per-step error the UI can render.

Currently wired:
- `signal_extraction` (E1, PR #452): deterministic generator
  with no external dependencies.
- `landing_page` (E2, PR #454 + E2.5, PR #455): plugs the
  host LLM + Skill adapters from PR #453 +
  `PostgresLandingPageRepository` backed by the host's
  `DatabasePool`. `scope_provider` (PR #455) ensures drafts
  persist under the authenticated tenant.
- `campaign` / `report` / `sales_brief` (E3, PR #456):
  share an identical wiring shape -- a single
  `PostgresIntelligenceRepository` (campaign opportunities)
  plus the per-output Postgres repo + LLM/Skill adapters.
  All three slots stay `None` when LLM or pool is absent.
- `blog_post` (E4): plugs the
  `PostgresBlogBlueprintRepository` (PR #458) +
  `PostgresBlogPostRepository` + LLM/Skill adapters. Slot
  stays `None` when LLM or pool is absent.
- `faq_markdown`: deterministic ticket FAQ builder with no LLM or
  database dependency, wired by default.

See `plans/PR-Content-Ops-Execution-Services-Wire-4.md`.
"""

from __future__ import annotations

from typing import Any, Callable

from extracted_content_pipeline.blog_blueprint_postgres import (
    PostgresBlogBlueprintRepository,
)
from extracted_content_pipeline.blog_generation import (
    BlogPostGenerationService,
)
from extracted_content_pipeline.blog_post_postgres import (
    PostgresBlogPostRepository,
)
from extracted_content_pipeline.campaign_generation import (
    CampaignGenerationService,
)
from extracted_content_pipeline.campaign_ports import (
    IntelligenceRepository,
    LLMClient,
    SkillStore,
)
from extracted_content_pipeline.campaign_postgres import (
    PostgresCampaignRepository,
    PostgresIntelligenceRepository,
)
from extracted_content_pipeline.content_ops_execution import (
    ContentOpsExecutionServices,
)
from extracted_content_pipeline.landing_page_generation import (
    LandingPageGenerationService,
)
from extracted_content_pipeline.landing_page_postgres import (
    PostgresLandingPageRepository,
)
from extracted_content_pipeline.report_generation import (
    ReportGenerationService,
)
from extracted_content_pipeline.report_postgres import (
    PostgresReportRepository,
)
from extracted_content_pipeline.sales_brief_generation import (
    SalesBriefGenerationService,
)
from extracted_content_pipeline.sales_brief_postgres import (
    PostgresSalesBriefRepository,
)
from extracted_content_pipeline.signal_extraction import (
    SignalExtractionService,
)
from extracted_content_pipeline.ticket_faq_markdown import (
    TicketFAQMarkdownService,
)


# Module-level singleton: SignalExtractionService is stateless, so
# re-creating it per request would just churn allocations.
_SIGNAL_EXTRACTION_SERVICE: SignalExtractionService = SignalExtractionService()
_FAQ_MARKDOWN_SERVICE: TicketFAQMarkdownService = TicketFAQMarkdownService()


def _build_landing_page_service(
    *,
    llm: LLMClient | None,
    skills: SkillStore,
    pool: Any,
) -> LandingPageGenerationService | None:
    """Build a wired `LandingPageGenerationService`, or `None` if
    no host LLM is currently active.

    Returning `None` (rather than wiring a stub) lets the
    bundle's `landing_page` slot stay empty -- the executor
    surfaces `service_not_configured` per output, which the
    catalog endpoint exposes to the UI's Execute enable-state.
    """

    if llm is None or pool is None:
        # Pool can be None during early host startup before
        # `init_database()` has run; treat the same as the
        # no-LLM branch -- skip the slot rather than build a
        # repo against a dead pool that would fail at first
        # query.
        return None
    return LandingPageGenerationService(
        landing_pages=PostgresLandingPageRepository(pool=pool),
        llm=llm,
        skills=skills,
    )


def _build_campaign_service(
    *,
    intelligence: IntelligenceRepository | None,
    llm: LLMClient | None,
    skills: SkillStore,
    pool: Any,
) -> CampaignGenerationService | None:
    """E3: campaign drafts. Same short-circuit shape as
    `_build_landing_page_service`."""

    if llm is None or pool is None or intelligence is None:
        return None
    return CampaignGenerationService(
        intelligence=intelligence,
        campaigns=PostgresCampaignRepository(pool=pool),
        llm=llm,
        skills=skills,
    )


def _build_report_service(
    *,
    intelligence: IntelligenceRepository | None,
    llm: LLMClient | None,
    skills: SkillStore,
    pool: Any,
) -> ReportGenerationService | None:
    """E3: structured report drafts."""

    if llm is None or pool is None or intelligence is None:
        return None
    return ReportGenerationService(
        intelligence=intelligence,
        reports=PostgresReportRepository(pool=pool),
        llm=llm,
        skills=skills,
    )


def _build_sales_brief_service(
    *,
    intelligence: IntelligenceRepository | None,
    llm: LLMClient | None,
    skills: SkillStore,
    pool: Any,
) -> SalesBriefGenerationService | None:
    """E3: sales-brief drafts."""

    if llm is None or pool is None or intelligence is None:
        return None
    return SalesBriefGenerationService(
        intelligence=intelligence,
        sales_briefs=PostgresSalesBriefRepository(pool=pool),
        llm=llm,
        skills=skills,
    )


def _build_blog_post_service(
    *,
    llm: LLMClient | None,
    skills: SkillStore,
    pool: Any,
) -> BlogPostGenerationService | None:
    """E4: blog-post drafts.

    Same short-circuit shape as `_build_landing_page_service`
    (no `IntelligenceRepository` -- blog_post's data source is
    the blueprint store, not campaign opportunities). An
    empty `blog_blueprints` table just means
    `BlogPostGenerationService.generate()` returns zero
    drafts; the slot is still wired so the catalog endpoint
    advertises `blog_post`.
    """

    if llm is None or pool is None:
        return None
    return BlogPostGenerationService(
        blueprints=PostgresBlogBlueprintRepository(pool=pool),
        blog_posts=PostgresBlogPostRepository(pool=pool),
        llm=llm,
        skills=skills,
    )


def build_content_ops_execution_services(
    *,
    llm_factory: Callable[[], LLMClient | None] | None = None,
    skills_factory: Callable[[], SkillStore] | None = None,
    pool_factory: Callable[[], Any] | None = None,
    enable_db_services: bool = False,
) -> ContentOpsExecutionServices:
    """Return the host's Content Ops execution-services bundle.

    The factory kwargs are dependency injection for tests --
    callers in dev environments without the host's full init
    chain (asyncpg / torch / ollama) pass stubs. Production
    callers omit the kwargs and the factory imports the host
    singletons on demand.

    `enable_db_services` (default `False`) gates DB-backed
    generators (currently `landing_page`). It defaults off
    because the host's content-ops route mount in
    `atlas_brain/api/__init__.py` does not yet pass a
    `scope_provider`, so the executor would fall back to an
    empty `TenantScope` and persist drafts under
    `account_id=""` -- cross-tenant leakage in any
    authenticated B2B deployment. Codex P1 review on PR #454
    flagged this. The follow-up slice (E2.5) wires
    `scope_provider` from the authenticated `AuthUser` and
    flips this flag on. Tests pass `enable_db_services=True`
    explicitly to exercise the wiring path.

    Slots not yet populated remain `None`; the executor
    returns `service_not_configured` per output. As host
    repositories arrive, follow-up slices populate the
    remaining slots.
    """

    if llm_factory is None:
        from atlas_brain._content_ops_infrastructure import (
            build_content_ops_llm_client,
        )

        llm_factory = build_content_ops_llm_client
    if skills_factory is None:
        from atlas_brain._content_ops_infrastructure import (
            build_content_ops_skill_store,
        )

        skills_factory = build_content_ops_skill_store
    if pool_factory is None:
        from atlas_brain.storage.database import get_db_pool

        pool_factory = get_db_pool

    landing_page = None
    campaign = None
    report = None
    sales_brief = None
    blog_post = None
    if enable_db_services:
        llm = llm_factory()
        skills = skills_factory()
        pool = pool_factory()
        # Shared across the three IntelligenceRepository-dependent
        # services -- the dataclass is immutable, the underlying
        # pool is shared anyway, no per-call state.
        intelligence: IntelligenceRepository | None = (
            PostgresIntelligenceRepository(pool=pool) if pool is not None else None
        )
        landing_page = _build_landing_page_service(
            llm=llm,
            skills=skills,
            pool=pool,
        )
        campaign = _build_campaign_service(
            intelligence=intelligence,
            llm=llm,
            skills=skills,
            pool=pool,
        )
        report = _build_report_service(
            intelligence=intelligence,
            llm=llm,
            skills=skills,
            pool=pool,
        )
        sales_brief = _build_sales_brief_service(
            intelligence=intelligence,
            llm=llm,
            skills=skills,
            pool=pool,
        )
        blog_post = _build_blog_post_service(
            llm=llm,
            skills=skills,
            pool=pool,
        )

    return ContentOpsExecutionServices(
        signal_extraction=_SIGNAL_EXTRACTION_SERVICE,
        landing_page=landing_page,
        campaign=campaign,
        report=report,
        sales_brief=sales_brief,
        blog_post=blog_post,
        faq_markdown=_FAQ_MARKDOWN_SERVICE,
    )


__all__ = ["build_content_ops_execution_services"]
