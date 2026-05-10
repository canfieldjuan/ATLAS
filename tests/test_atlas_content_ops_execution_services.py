"""Pin the host's Content Ops execution-services bundle.

`atlas_brain/_content_ops_services.py` builds a
`ContentOpsExecutionServices` populated with the generators
the host has wired. Currently:

- `signal_extraction` (E1, PR #452): always wired (stateless).
- `landing_page` (E2, this slice): wired when an LLM is
  active; slot stays `None` otherwise.

Tests use the factory's dependency-injection kwargs (`llm_factory`
/ `skills_factory` / `pool_factory`) to stub host
infrastructure -- the canonical singletons trigger the heavy
host init chain (torch / ollama / asyncpg) that dev envs may
not have.

Test inventory:

1. `signal_extraction` runs through the full executor with the
   bundle attached.
2. `landing_page` is populated when an LLM is wired (E2 canary).
3. `landing_page` slot stays `None` when no LLM is active
   (E2 fallback behavior).
4. Unwired outputs still return `service_not_configured` --
   confirms the bundle doesn't silently mask the remaining 4
   slots (`campaign`, `blog_post`, `report`, `sales_brief`).
5. `configured_outputs()` advertises the right tuple
   depending on LLM presence.

When follow-up slices add `campaign` / `blog_post` / etc.,
tests 4 and 5 need updated expected-sets.
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


def test_landing_page_wired_when_llm_active() -> None:
    """E2 canary: when the LLM factory returns a non-None client,
    the bundle's `landing_page` slot is populated and
    `configured_outputs()` advertises both `landing_page` and
    `signal_extraction`."""

    services = build_content_ops_execution_services(
        llm_factory=_make_llm_stub,
        skills_factory=_make_skill_store_stub,
        pool_factory=_make_pool_stub,
    )

    assert services.landing_page is not None
    assert services.configured_outputs() == (
        "landing_page",
        "signal_extraction",
    )


def test_landing_page_slot_stays_none_when_no_active_llm() -> None:
    """E2 fallback: when `build_content_ops_llm_client` would
    return `None` (no active host model), the bundle's
    `landing_page` slot stays `None` so the executor can
    surface `service_not_configured` to the UI's Execute
    enable-state."""

    services = build_content_ops_execution_services(
        llm_factory=_no_llm,
        skills_factory=_make_skill_store_stub,
        pool_factory=_make_pool_stub,
    )

    assert services.landing_page is None
    assert services.configured_outputs() == ("signal_extraction",)


@pytest.mark.asyncio
async def test_unwired_outputs_still_return_service_not_configured() -> None:
    """The bundle leaves `campaign` / `blog_post` / `report` /
    `sales_brief` slots `None`. The executor's per-step
    dispatcher must surface that as `service_not_configured`
    rather than silently succeeding. Picking `report` since
    `landing_page` is no longer in the unwired set after E2."""

    services = build_content_ops_execution_services(
        llm_factory=_make_llm_stub,
        skills_factory=_make_skill_store_stub,
        pool_factory=_make_pool_stub,
    )

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["report"],
            "inputs": {"opportunity_id": "opp-1"},
        },
        services=services,
    )

    assert result["status"] == "failed", result
    assert len(result["errors"]) == 1
    assert result["errors"][0]["reason"] == "service_not_configured"


def test_bundle_only_advertises_wired_outputs_with_llm() -> None:
    """`configured_outputs()` is the source of truth the catalog
    endpoint surfaces in `execution.configured_outputs`. With
    LLM wired, the bundle advertises landing_page +
    signal_extraction."""

    services = build_content_ops_execution_services(
        llm_factory=_make_llm_stub,
        skills_factory=_make_skill_store_stub,
        pool_factory=_make_pool_stub,
    )
    assert services.configured_outputs() == (
        "landing_page",
        "signal_extraction",
    )


def test_bundle_only_advertises_wired_outputs_without_llm() -> None:
    """Without an active LLM, only `signal_extraction` is
    advertised."""

    services = build_content_ops_execution_services(
        llm_factory=_no_llm,
        skills_factory=_make_skill_store_stub,
        pool_factory=_make_pool_stub,
    )
    assert services.configured_outputs() == ("signal_extraction",)
