from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_gate_a_live_quality.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_gate_a_live_quality",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_build_gate_a_payload_sets_top_level_variants_and_brand_voice() -> None:
    payload = smoke.build_gate_a_payload(
        account_id="acct-gate-a",
        support_ticket_rows=[
            {
                "Ticket ID": "ticket-1",
                "Subject": "Export confusion",
                "Description": "How do I export renewal reports?",
            }
        ],
        target_mode="vendor_retention",
        variant_count=3,
        quality_repair_attempts=1,
        max_cost_usd=1.25,
    )

    assert payload["outputs"] == ["landing_page", "blog_post", "sales_brief"]
    assert payload["variant_count"] == 3
    assert payload["max_cost_usd"] == 1.25
    assert payload["inputs"]["target_account"] == "SaaS support team with repeat ticket backlog"
    assert payload["inputs"]["brand_voice"]["account_id"] == "acct-gate-a"
    assert payload["inputs"]["brand_voice"]["preferred_pov"] == "second_person"
    assert payload["inputs"]["source_material"][0]["Ticket ID"] == "ticket-1"


def test_build_gate_a_payload_requires_multiple_variants() -> None:
    with pytest.raises(ValueError, match="variant_count"):
        smoke.build_gate_a_payload(
            account_id="acct-gate-a",
            support_ticket_rows=[],
            target_mode="vendor_retention",
            variant_count=1,
            quality_repair_attempts=1,
        )


def test_saved_ids_by_output_reads_aggregate_variant_saved_ids() -> None:
    result = {
        "status": "completed",
        "steps": [
            {
                "output": "landing_page",
                "status": "completed",
                "result": {
                    "variant_count": 3,
                    "saved_ids": ["lp-1", "lp-2", "lp-3"],
                    "variant_results": [
                        {"variant_angle": {"id": "pain_led"}, "saved_ids": ["lp-1"]},
                        {"variant_angle": {"id": "outcome_led"}, "saved_ids": ["lp-2"]},
                    ],
                },
            },
            {
                "output": "blog_post",
                "status": "completed",
                "result": {"variant_count": 3, "saved_ids": ["blog-1"]},
            },
            {
                "output": "sales_brief",
                "status": "completed",
                "result": {"variant_count": 3, "saved_ids": ["brief-1"]},
            },
        ],
    }

    assert smoke.saved_ids_by_output(result) == {
        "landing_page": ["lp-1", "lp-2", "lp-3"],
        "blog_post": ["blog-1"],
        "sales_brief": ["brief-1"],
    }
    assert smoke.variant_summary(result)["landing_page"]["variants"][0][
        "variant_angle"
    ] == "pain_led"


@pytest.mark.asyncio
async def test_review_saved_ids_reports_missing_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_update_statuses(
        asset: str,
        pool: Any,
        *,
        asset_ids: list[str],
        status: str,
        scope: Any,
    ) -> list[str]:
        del asset, pool, status, scope
        return asset_ids[:1]

    monkeypatch.setattr(
        "extracted_content_pipeline.api.generated_assets._update_asset_statuses",
        _fake_update_statuses,
    )

    reviewed = await smoke.review_saved_ids(
        object(),
        object(),
        {"landing_page": ["lp-1", "lp-2"]},
    )

    assert reviewed["landing_page"]["updated_ids"] == ["lp-1"]
    assert reviewed["landing_page"]["missing_ids"] == ["lp-2"]
    assert smoke._review_errors(reviewed) == [
        "landing_page review update missed ids: lp-2"
    ]


def test_filter_saved_draft_export_rows_fails_closed_on_missing_id() -> None:
    with pytest.raises(ValueError, match="missing-id"):
        smoke._filter_saved_draft_export_rows(
            {"count": 1, "filters": {}, "rows": [{"id": "present-id"}]},
            ["present-id", "missing-id"],
        )


def test_variant_persistence_errors_fail_on_duplicate_saved_ids() -> None:
    result = {
        "status": "completed",
        "steps": [
            {
                "output": "blog_post",
                "status": "completed",
                "result": {
                    "variant_count": 3,
                    "variant_results": [
                        {"generated": 0, "saved_ids": []},
                        {"generated": 1, "saved_ids": ["blog-1"]},
                        {"generated": 1, "saved_ids": ["blog-1"]},
                    ],
                },
            },
        ],
    }

    assert smoke.variant_persistence_errors(
        result,
        saved_ids={"blog_post": ["blog-1", "blog-1"]},
        exports={"blog_post": {"count": 1}},
    ) == [
        "blog_post variant persistence collapsed: "
        "2 successful variant(s), 2 saved id entries, "
        "1 unique saved id(s), 1 exported row(s)"
    ]
