"""Factory for the Content Ops execution-services bundle.

Builds a `ContentOpsExecutionServices` populated with the
generators the host has fully wired. Slots that aren't yet
populated remain `None`; the executor's per-step dispatcher
returns `service_not_configured` for unset slots, which the
route layer maps to a per-step error the UI can render.

Currently wired:
- `signal_extraction` (E1, PR #452): deterministic generator
  with no external dependencies.
- `landing_page` (E2, this slice): plugs the host LLM + Skill
  adapters from PR #453 + `PostgresLandingPageRepository`
  backed by the host's `DatabasePool`. Slot stays `None` when
  no LLM is active.

Follow-up slices (E3+) will plug the remaining 4 generators
(`campaign`, `blog_post`, `report`, `sales_brief`) into the
same bundle once an `IntelligenceRepository` host factory
lands.

See `plans/PR-Content-Ops-Execution-Services-Wire-2.md`.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from extracted_content_pipeline.campaign_ports import LLMClient, SkillStore
from extracted_content_pipeline.content_ops_execution import (
    ContentOpsExecutionServices,
)
from extracted_content_pipeline.landing_page_generation import (
    LandingPageGenerationService,
)
from extracted_content_pipeline.landing_page_postgres import (
    PostgresLandingPageRepository,
)
from extracted_content_pipeline.signal_extraction import (
    SignalExtractionService,
)


# Module-level singleton: SignalExtractionService is stateless, so
# re-creating it per request would just churn allocations.
_SIGNAL_EXTRACTION_SERVICE: SignalExtractionService = SignalExtractionService()


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

    if llm is None:
        return None
    return LandingPageGenerationService(
        landing_pages=PostgresLandingPageRepository(pool=pool),
        llm=llm,
        skills=skills,
    )


def build_content_ops_execution_services(
    *,
    llm_factory: Optional[Callable[[], LLMClient | None]] = None,
    skills_factory: Optional[Callable[[], SkillStore]] = None,
    pool_factory: Optional[Callable[[], Any]] = None,
) -> ContentOpsExecutionServices:
    """Return the host's Content Ops execution-services bundle.

    The factory kwargs are dependency injection for tests --
    callers in dev environments without the host's full init
    chain (asyncpg / torch / ollama) pass stubs. Production
    callers omit the kwargs and the factory imports the host
    singletons on demand.

    Slots not yet populated remain `None`; the executor returns
    `service_not_configured` per output. As host repositories
    arrive, follow-up slices populate the remaining slots.
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

    llm = llm_factory()
    skills = skills_factory()
    pool = pool_factory()

    landing_page = _build_landing_page_service(
        llm=llm,
        skills=skills,
        pool=pool,
    )

    return ContentOpsExecutionServices(
        signal_extraction=_SIGNAL_EXTRACTION_SERVICE,
        landing_page=landing_page,
    )


__all__ = ["build_content_ops_execution_services"]
