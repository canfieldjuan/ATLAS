"""Pin the host's Content Ops execution-services bundle.

`atlas_brain/_content_ops_services.py` builds a
`ContentOpsExecutionServices` populated with the generators
the host has wired. Currently:

- `signal_extraction` (E1, PR #452): always wired (stateless).
- `landing_page` (E2 + E2.5, PRs #454/#455): wired when an
  LLM + pool are active; slot stays `None` otherwise.
- `campaign` / `report` / `sales_brief` (E3, this slice):
  same shape as landing_page but each also takes a
  shared `PostgresIntelligenceRepository`. Skip together
  when LLM or pool is absent.

Tests use the factory's dependency-injection kwargs (`llm_factory`
/ `skills_factory` / `pool_factory`) to stub host
infrastructure -- the canonical singletons trigger the heavy
host init chain (torch / ollama / asyncpg) that dev envs may
not have.

Test inventory (12 tests):

1. `signal_extraction` runs through the full executor.
2. `landing_page` populated when LLM + db enabled (E2 canary).
3. `landing_page` skips when no LLM (E2 fallback).
4. `landing_page` skips when pool is None.
5. `landing_page` skips in production default
   (Codex P1 safety pin).
6. `campaign` populated when LLM + db enabled (E3 canary).
7. `report` populated when LLM + db enabled (E3 canary).
8. `sales_brief` populated when LLM + db enabled (E3 canary).
9. campaign / report / sales_brief skip together when no LLM.
10. campaign / report / sales_brief skip together when pool
    is None.
11. `blog_post` (the only remaining unwired output after E3)
    still returns `service_not_configured`.
12. `configured_outputs()` with LLM + db enabled advertises
    `(email_campaign, report, landing_page, sales_brief,
    signal_extraction)` -- order follows the upstream
    `ContentOpsExecutionServices.configured_outputs` iteration
    (not alphabetical). Without LLM or in production default,
    only `signal_extraction`.

When E4 wires `blog_post`, tests 11 and 12 need updated
expected-sets.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from atlas_brain._content_ops_services import (
    build_content_ops_execution_services,
)
from extracted_content_pipeline.content_ops_execution import (
    execute_content_ops_from_mapping,
)


def _make_llm_stub() -> Any:
    """Return a fake LLM client adequate for satisfying the
    `LandingPageGenerationService` constructor's `llm` arg.

    The bundle factory just stores the reference; the executor
    actually exercises the LLM only when `/execute` runs the
    landing-page generator -- not in the bundle-population
    tests below.
    """

    return SimpleNamespace(complete=lambda *_a, **_k: None)


def _make_skill_store_stub() -> Any:
    return SimpleNamespace(get_prompt=lambda _name: None)


def _make_pool_stub() -> Any:
    """Fake pool for `PostgresLandingPageRepository(pool=...)` --
    the repo only exercises the pool when its methods are called.
    Bundle population doesn't trigger that."""

    return SimpleNamespace()


def _no_llm() -> None:
    return None


@pytest.mark.asyncio
async def test_signal_extraction_runs_through_host_bundle() -> None:
    """`/execute` with `outputs=["signal_extraction"]` returns a
    completed step using the host's wired
    `SignalExtractionService`. Even without an LLM, this output
    works because it's deterministic."""

    services = build_content_ops_execution_services(
        llm_factory=_no_llm,
        skills_factory=_make_skill_store_stub,
        pool_factory=_make_pool_stub,
    )

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["signal_extraction"],
            "inputs": {"source_material": "Acme review export"},
        },
        services=services,
    )

    assert result["status"] == "completed", result
    assert len(result["steps"]) == 1
    step = result["steps"][0]
    assert step["output"] == "signal_extraction"
    assert step["status"] == "completed"
    # SignalExtractionResult.as_dict() shape.
    assert "opportunities" in step["result"]
    assert step["result"]["target_mode"] == "vendor_retention"


def test_landing_page_wired_when_llm_active_and_db_enabled() -> None:
    """E2 canary: when the LLM factory returns a non-None client
    AND `enable_db_services=True`, the bundle's `landing_page`
    slot is populated and `configured_outputs()` advertises
    both `landing_page` and `signal_extraction`."""

    services = build_content_ops_execution_services(
        llm_factory=_make_llm_stub,
        skills_factory=_make_skill_store_stub,
        pool_factory=_make_pool_stub,
        enable_db_services=True,
    )

    assert services.landing_page is not None
    assert services.configured_outputs() == (
        "email_campaign",
        "report",
        "landing_page",
        "sales_brief",
        "signal_extraction",
    )


def test_landing_page_slot_stays_none_when_no_active_llm() -> None:
    """E2 fallback: with `enable_db_services=True` but no active
    LLM, the bundle's `landing_page` slot stays `None` so the
    executor can surface `service_not_configured` to the UI's
    Execute enable-state."""

    services = build_content_ops_execution_services(
        llm_factory=_no_llm,
        skills_factory=_make_skill_store_stub,
        pool_factory=_make_pool_stub,
        enable_db_services=True,
    )

    assert services.landing_page is None
    assert services.configured_outputs() == ("signal_extraction",)


def test_landing_page_slot_stays_none_when_pool_is_none() -> None:
    """Defensive guard: if `get_db_pool()` would return `None`
    during early host startup (before `init_database()` runs),
    `_build_landing_page_service` skips the slot the same way
    it does for an absent LLM. Better than building a
    `PostgresLandingPageRepository(pool=None)` that would
    fail on first query."""

    def _no_pool() -> Any:
        return None

    services = build_content_ops_execution_services(
        llm_factory=_make_llm_stub,
        skills_factory=_make_skill_store_stub,
        pool_factory=_no_pool,
        enable_db_services=True,
    )
    assert services.landing_page is None
    assert services.configured_outputs() == ("signal_extraction",)


def test_landing_page_slot_stays_none_in_production_default() -> None:
    """Codex P1 fix: production callers omit `enable_db_services`
    (default `False`) so DB-backed generators stay unwired
    until the route mount also wires a `scope_provider` (E2.5).
    Without that, drafts would persist under empty
    `account_id` -- cross-tenant leakage. Pin the safe
    default."""

    services = build_content_ops_execution_services(
        llm_factory=_make_llm_stub,
        skills_factory=_make_skill_store_stub,
        pool_factory=_make_pool_stub,
        # enable_db_services intentionally omitted -- defaults False.
    )

    assert services.landing_page is None
    assert services.configured_outputs() == ("signal_extraction",)


def test_campaign_wired_when_llm_active_and_db_enabled() -> None:
    """E3 canary: campaign service slot is populated with the
    full LLM + Skill + Postgres + IntelligenceRepository chain.
    Bundle's `for_output("email_campaign")` returns the
    service."""

    services = build_content_ops_execution_services(
        llm_factory=_make_llm_stub,
        skills_factory=_make_skill_store_stub,
        pool_factory=_make_pool_stub,
        enable_db_services=True,
    )
    assert services.campaign is not None
    assert services.for_output("email_campaign") is services.campaign


def test_report_wired_when_llm_active_and_db_enabled() -> None:
    """E3 canary: report service slot populated."""

    services = build_content_ops_execution_services(
        llm_factory=_make_llm_stub,
        skills_factory=_make_skill_store_stub,
        pool_factory=_make_pool_stub,
        enable_db_services=True,
    )
    assert services.report is not None
    assert services.for_output("report") is services.report


def test_sales_brief_wired_when_llm_active_and_db_enabled() -> None:
    """E3 canary: sales_brief service slot populated."""

    services = build_content_ops_execution_services(
        llm_factory=_make_llm_stub,
        skills_factory=_make_skill_store_stub,
        pool_factory=_make_pool_stub,
        enable_db_services=True,
    )
    assert services.sales_brief is not None
    assert services.for_output("sales_brief") is services.sales_brief


def test_e3_services_skip_together_when_no_active_llm() -> None:
    """E3 fallback: campaign / report / sales_brief slots all
    stay `None` when no LLM is active. Same short-circuit shape
    as landing_page (PR #454). Only signal_extraction remains
    advertised."""

    services = build_content_ops_execution_services(
        llm_factory=_no_llm,
        skills_factory=_make_skill_store_stub,
        pool_factory=_make_pool_stub,
        enable_db_services=True,
    )
    assert services.campaign is None
    assert services.report is None
    assert services.sales_brief is None
    assert services.configured_outputs() == ("signal_extraction",)


def test_e3_services_skip_together_when_pool_is_none() -> None:
    """E3 fallback: campaign / report / sales_brief slots all
    skip when pool is None during early host startup."""

    def _no_pool() -> Any:
        return None

    services = build_content_ops_execution_services(
        llm_factory=_make_llm_stub,
        skills_factory=_make_skill_store_stub,
        pool_factory=_no_pool,
        enable_db_services=True,
    )
    assert services.campaign is None
    assert services.report is None
    assert services.sales_brief is None
    assert services.configured_outputs() == ("signal_extraction",)


@pytest.mark.asyncio
async def test_unwired_blog_post_still_returns_service_not_configured() -> None:
    """After E3 wires campaign / report / sales_brief, the only
    output left in the unwired set is `blog_post` (different
    repo shape -- BlogBlueprintRepository -- E4). The
    executor's per-step dispatcher must surface that as
    `service_not_configured`."""

    services = build_content_ops_execution_services(
        llm_factory=_make_llm_stub,
        skills_factory=_make_skill_store_stub,
        pool_factory=_make_pool_stub,
        enable_db_services=True,
    )

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["blog_post"],
            "inputs": {"topic": "Q3 churn signals"},
        },
        services=services,
    )

    assert result["status"] == "failed", result
    assert len(result["errors"]) == 1
    assert result["errors"][0]["reason"] == "service_not_configured"


def test_bundle_only_advertises_wired_outputs_with_llm_and_db_enabled() -> None:
    """`configured_outputs()` is the source of truth the catalog
    endpoint surfaces in `execution.configured_outputs`. With
    LLM wired AND `enable_db_services=True`, the bundle
    advertises landing_page + signal_extraction."""

    services = build_content_ops_execution_services(
        llm_factory=_make_llm_stub,
        skills_factory=_make_skill_store_stub,
        pool_factory=_make_pool_stub,
        enable_db_services=True,
    )
    assert services.configured_outputs() == (
        "email_campaign",
        "report",
        "landing_page",
        "sales_brief",
        "signal_extraction",
    )


def test_bundle_only_advertises_wired_outputs_without_llm() -> None:
    """Without an active LLM (even with `enable_db_services=True`),
    only `signal_extraction` is advertised."""

    services = build_content_ops_execution_services(
        llm_factory=_no_llm,
        skills_factory=_make_skill_store_stub,
        pool_factory=_make_pool_stub,
        enable_db_services=True,
    )
    assert services.configured_outputs() == ("signal_extraction",)
