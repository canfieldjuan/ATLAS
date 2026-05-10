"""Pin the host's Content Ops execution-services bundle.

`atlas_brain/_content_ops_services.py` builds a
`ContentOpsExecutionServices` with `signal_extraction` populated
and the other 5 slots left `None`. Three regression tests:

1. `signal_extraction` runs through the full executor with the
   bundle attached -- proves the wiring closes the prior 503
   "execution services not configured" gap.

2. The other 5 outputs still fall through with
   `service_not_configured` -- confirms the bundle doesn't
   silently mask outputs we haven't wired yet.

3. `configured_outputs()` advertises only the wired output --
   pins the source-of-truth the catalog endpoint exposes in
   `execution.configured_outputs`.

When follow-up slices add `campaign` / `blog_post` / etc.,
tests 2 and 3 need updated expected-sets; treat them as the
canary that the bundle's slot population matches the plan.
"""

from __future__ import annotations

import pytest

from atlas_brain._content_ops_services import (
    build_content_ops_execution_services,
)
from extracted_content_pipeline.content_ops_execution import (
    execute_content_ops_from_mapping,
)


@pytest.mark.asyncio
async def test_signal_extraction_runs_through_host_bundle() -> None:
    """`/execute` with `outputs=["signal_extraction"]` returns a
    completed step using the host's wired
    `SignalExtractionService`."""

    services = build_content_ops_execution_services()

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
    # SignalExtractionResult.as_dict() shape:
    #   {"opportunities": [...], "warnings": [...], "target_mode": ...}
    assert "opportunities" in step["result"]
    assert step["result"]["target_mode"] == "vendor_retention"


@pytest.mark.asyncio
async def test_unwired_outputs_still_return_service_not_configured() -> None:
    """The bundle leaves `campaign` / `blog_post` / `report` /
    `landing_page` / `sales_brief` slots `None`. The executor's
    per-step dispatcher must surface that as
    `service_not_configured` rather than silently succeeding."""

    services = build_content_ops_execution_services()

    # Pick an output the bundle does NOT wire, ensure the request
    # gets through preview / plan and lands on the executor's
    # per-step service-resolution path.
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


def test_bundle_only_advertises_wired_outputs() -> None:
    """The bundle's `configured_outputs()` is the source of truth
    the catalog endpoint surfaces in
    `execution.configured_outputs`. Today only
    `signal_extraction` is wired."""

    services = build_content_ops_execution_services()
    assert services.configured_outputs() == ("signal_extraction",)
