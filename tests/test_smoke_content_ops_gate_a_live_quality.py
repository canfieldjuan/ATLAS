from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from extracted_content_pipeline.support_ticket_input_package import (
    build_support_ticket_input_package,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_gate_a_live_quality.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_gate_a_live_quality",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_messy_grounding_fixture_exercises_noisy_lopsided_ticket_shape() -> None:
    rows = smoke._load_csv_rows(
        ROOT
        / "extracted_content_pipeline"
        / "examples"
        / "support_ticket_messy_grounding_sources.csv"
    )
    package = build_support_ticket_input_package(rows)

    assert package.inputs["source_row_count"] == 44
    assert package.inputs["included_ticket_row_count"] == 42
    assert package.inputs["skipped_ticket_row_count"] == 2
    assert package.inputs["question_like_ticket_count"] == 39
    assert package.inputs["has_dated_window"] is False
    assert [warning["code"] for warning in package.warnings] == [
        "ticket_row_missing_text",
        "ticket_row_missing_text",
    ]
    assert package.inputs["top_ticket_clusters"][:4] == [
        {"label": "reporting export", "count": 11},
        {"label": "dashboard freshness", "count": 7},
        {"label": "sso setup", "count": 6},
        {"label": "billing and plan management", "count": 4},
    ]


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

    assert payload["outputs"] == [
        "email_campaign",
        "landing_page",
        "blog_post",
        "sales_brief",
        "report",
    ]
    assert payload["variant_count"] == 3
    assert payload["max_cost_usd"] == 1.25
    assert payload["inputs"]["target_account"] == "SaaS support team with repeat ticket backlog"
    assert payload["inputs"]["opportunity_id"] == "ticket-1"
    assert payload["inputs"]["selling"] == {
        "product_name": "Support Ticket FAQ Gap Audit",
        "affiliate_url": "https://finetunelab.ai/systems/ai-content-ops/intake",
        "sender_name": "FineTune Lab",
        "sender_title": "Content Ops Team",
        "sender_company": "FineTune Lab",
    }
    assert payload["inputs"]["brand_voice"]["account_id"] == "acct-gate-a"
    assert payload["inputs"]["brand_voice"]["preferred_pov"] == "second_person"
    assert payload["inputs"]["source_material"][0]["Ticket ID"] == "ticket-1"


def test_build_gate_a_payload_accepts_selected_outputs() -> None:
    payload = smoke.build_gate_a_payload(
        account_id="acct-gate-a",
        support_ticket_rows=[],
        target_mode="vendor_retention",
        variant_count=3,
        quality_repair_attempts=1,
        outputs=("landing_page", "blog_post", "sales_brief", "report"),
    )

    assert payload["outputs"] == ["landing_page", "blog_post", "sales_brief", "report"]


def test_bind_report_opportunity_fails_when_requested_id_was_not_imported() -> None:
    payload = {
        "inputs": {
            "opportunity_id": "ticket-2",
            "filters": {"topic_type": "support_ticket_faq_gap_live_gate_a"},
        }
    }

    with pytest.raises(ValueError, match="was not imported"):
        smoke._bind_report_opportunity(payload, {"target_ids": ["ticket-1"]})


def test_bind_report_opportunity_requires_explicit_id_for_multiple_targets() -> None:
    payload = {"inputs": {"filters": {"topic_type": "support_ticket_faq_gap_live_gate_a"}}}

    with pytest.raises(ValueError, match="multiple target_ids"):
        smoke._bind_report_opportunity(
            payload,
            {"target_ids": ["ticket-1", "ticket-2"]},
        )


@pytest.mark.asyncio
async def test_prepare_gate_a_output_dependencies_keeps_email_filters_without_blog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    async def _fake_provider(payload: Any, *, scope: Any) -> dict[str, Any]:
        captured["provider_scope"] = scope
        return dict(payload)

    async def _fake_seed_blog(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("email-only runs must not seed or align blog filters")

    async def _fake_seed_email(pool: Any, scope: Any, **kwargs: Any) -> dict[str, Any]:
        captured["email_pool"] = pool
        captured["email_scope"] = scope
        captured["email_kwargs"] = kwargs
        return {"inserted": 1}

    monkeypatch.setattr(smoke, "_payload_with_support_ticket_provider", _fake_provider)
    monkeypatch.setattr(smoke, "_seed_default_blog_blueprint", _fake_seed_blog)
    monkeypatch.setattr(smoke, "seed_email_campaign_opportunities", _fake_seed_email)
    monkeypatch.setattr(smoke, "_write_json", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        account_id="acct-gate-a",
        target_mode="vendor_retention",
        variant_count=3,
    )
    payload = smoke.build_gate_a_payload(
        account_id="acct-gate-a",
        support_ticket_rows=[],
        target_mode="vendor_retention",
        variant_count=3,
        quality_repair_attempts=1,
        outputs=("email_campaign",),
    )
    prepared, seeded_blog, imported = await smoke.prepare_gate_a_output_dependencies(
        args,
        "scope",
        pool="pool",
        payload=payload,
        selected_outputs=("email_campaign",),
        source_rows=({"Ticket ID": "ticket-1"},),
        output_dir=tmp_path,
    )

    assert seeded_blog is None
    assert imported == {"inserted": 1}
    assert prepared["inputs"]["filters"] == {
        "topic_type": "support_ticket_faq_gap_live_gate_a"
    }
    assert captured["email_pool"] == "pool"
    assert captured["email_scope"] == "scope"
    assert captured["email_kwargs"]["filters"] == {
        "topic_type": "support_ticket_faq_gap_live_gate_a"
    }


@pytest.mark.asyncio
async def test_prepare_gate_a_output_dependencies_imports_after_blog_alignment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    async def _fake_provider(payload: Any, *, scope: Any) -> dict[str, Any]:
        return dict(payload)

    async def _fake_seed_blog(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "topic_type": "content_ops_support_ticket_faq",
            "slug": "support-ticket-faq-blueprint",
            "topic": "Support-ticket questions customers keep asking",
        }

    async def _fake_seed_email(pool: Any, scope: Any, **kwargs: Any) -> dict[str, Any]:
        captured["email_kwargs"] = kwargs
        return {"inserted": 1}

    monkeypatch.setattr(smoke, "_payload_with_support_ticket_provider", _fake_provider)
    monkeypatch.setattr(smoke, "_seed_default_blog_blueprint", _fake_seed_blog)
    monkeypatch.setattr(smoke, "seed_email_campaign_opportunities", _fake_seed_email)
    monkeypatch.setattr(smoke, "_write_json", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        account_id="acct-gate-a",
        target_mode="vendor_retention",
        variant_count=3,
    )
    payload = smoke.build_gate_a_payload(
        account_id="acct-gate-a",
        support_ticket_rows=[],
        target_mode="vendor_retention",
        variant_count=3,
        quality_repair_attempts=1,
        outputs=("email_campaign", "blog_post"),
    )
    await smoke.prepare_gate_a_output_dependencies(
        args,
        "scope",
        pool="pool",
        payload=payload,
        selected_outputs=("email_campaign", "blog_post"),
        source_rows=({"Ticket ID": "ticket-1"},),
        output_dir=tmp_path,
    )

    assert captured["email_kwargs"]["filters"] == {
        "topic_type": "content_ops_support_ticket_faq",
        "slug": "support-ticket-faq-blueprint",
    }


@pytest.mark.asyncio
async def test_prepare_gate_a_output_dependencies_imports_report_opportunity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    async def _fake_provider(payload: Any, *, scope: Any) -> dict[str, Any]:
        del scope
        return dict(payload)

    async def _fake_seed(pool: Any, scope: Any, **kwargs: Any) -> dict[str, Any]:
        captured["pool"] = pool
        captured["scope"] = scope
        captured["kwargs"] = kwargs
        return {"inserted": 1, "skipped": 0, "target_ids": ["ticket-1"]}

    monkeypatch.setattr(smoke, "_payload_with_support_ticket_provider", _fake_provider)
    monkeypatch.setattr(smoke, "seed_email_campaign_opportunities", _fake_seed)
    monkeypatch.setattr(smoke, "_write_json", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        account_id="acct-gate-a",
        target_mode="vendor_retention",
        variant_count=3,
    )
    payload = smoke.build_gate_a_payload(
        account_id="acct-gate-a",
        support_ticket_rows=[],
        target_mode="vendor_retention",
        variant_count=3,
        quality_repair_attempts=1,
        outputs=("report",),
    )
    prepared, seeded_blog, imported = await smoke.prepare_gate_a_output_dependencies(
        args,
        "scope",
        pool="pool",
        payload=payload,
        selected_outputs=("report",),
        source_rows=({"Ticket ID": "ticket-1"},),
        output_dir=tmp_path,
    )

    assert seeded_blog is None
    assert imported == {"inserted": 1, "skipped": 0, "target_ids": ["ticket-1"]}
    assert captured["pool"] == "pool"
    assert captured["scope"] == "scope"
    assert captured["kwargs"]["filters"] == {
        "topic_type": "support_ticket_faq_gap_live_gate_a"
    }
    assert prepared["inputs"]["opportunity_id"] == "ticket-1"
    assert prepared["inputs"]["filters"] == {
        "topic_type": "support_ticket_faq_gap_live_gate_a",
        "target_id": "ticket-1",
    }


@pytest.mark.asyncio
async def test_prepare_gate_a_output_dependencies_fails_closed_for_report_without_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def _fake_provider(payload: Any, *, scope: Any) -> dict[str, Any]:
        del scope
        return dict(payload)

    async def _fake_seed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {"inserted": 0, "skipped": 1, "target_ids": []}

    monkeypatch.setattr(smoke, "_payload_with_support_ticket_provider", _fake_provider)
    monkeypatch.setattr(smoke, "seed_email_campaign_opportunities", _fake_seed)
    monkeypatch.setattr(smoke, "_write_json", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        account_id="acct-gate-a",
        target_mode="vendor_retention",
        variant_count=3,
    )
    payload = smoke.build_gate_a_payload(
        account_id="acct-gate-a",
        support_ticket_rows=[],
        target_mode="vendor_retention",
        variant_count=3,
        quality_repair_attempts=1,
        outputs=("report",),
    )

    with pytest.raises(ValueError, match="returned no target_ids"):
        await smoke.prepare_gate_a_output_dependencies(
            args,
            "scope",
            pool="pool",
            payload=payload,
            selected_outputs=("report",),
            source_rows=(),
            output_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_seed_email_campaign_opportunities_imports_support_ticket_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class _ImportResult:
        def as_dict(self) -> dict[str, Any]:
            return {
                "inserted": 1,
                "skipped": 0,
                "target_ids": ["ticket-1"],
            }

    async def _fake_import_campaign_opportunities(pool: Any, rows: Any, **kwargs: Any):
        calls.append({"pool": pool, "rows": list(rows), **kwargs})
        return _ImportResult()

    monkeypatch.setattr(
        "extracted_content_pipeline.campaign_postgres_import.import_campaign_opportunities",
        _fake_import_campaign_opportunities,
    )

    pool = object()
    scope = object()
    result = await smoke.seed_email_campaign_opportunities(
        pool,
        scope,
        target_mode="vendor_retention",
        filters={"topic_type": "support_ticket_faq_gap_live_gate_a"},
        source_rows=[
            {
                "Ticket ID": "ticket-1",
                "Account Name": "Acme",
                "Vendor Name": "FlowPilot",
                "Contact Email": "ops@acme.example",
                "Subject": "Export blocked",
                "Description": "How do I export the report?",
                "Pain Category": "reporting export",
                "Source Type": "support_ticket",
            }
        ],
    )

    assert result == {"inserted": 1, "skipped": 0, "target_ids": ["ticket-1"]}
    assert calls[0]["pool"] is pool
    assert calls[0]["scope"] is scope
    assert calls[0]["target_mode"] == "vendor_retention"
    assert calls[0]["replace_existing"] is True
    assert calls[0]["normalize"] is False
    assert calls[0]["source"] == "gate_a_support_ticket_csv"
    assert calls[0]["rows"][0]["target_id"] == "ticket-1"
    assert calls[0]["rows"][0]["company_name"] == "Acme"
    assert calls[0]["rows"][0]["vendor_name"] == "FlowPilot"
    assert calls[0]["rows"][0]["topic_type"] == "support_ticket_faq_gap_live_gate_a"
    assert calls[0]["rows"][0]["pain_points"] == ["reporting export"]
    assert calls[0]["rows"][0]["evidence"] == [{
        "text": "How do I export the report?",
        "source_id": "ticket-1",
        "source_type": "support_ticket",
        "source_title": "Export blocked",
    }]


def test_resolve_outputs_rejects_empty_or_unsupported_selection() -> None:
    with pytest.raises(ValueError, match="at least one output"):
        smoke._resolve_outputs(" , ")
    with pytest.raises(ValueError, match="unsupported output"):
        smoke._resolve_outputs("landing_page,unknown")


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
                "output": "email_campaign",
                "status": "completed",
                "result": {"saved_ids": ["campaign-1", "campaign-2"]},
            },
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
        "email_campaign": ["campaign-1", "campaign-2"],
        "landing_page": ["lp-1", "lp-2", "lp-3"],
        "blog_post": ["blog-1"],
        "sales_brief": ["brief-1"],
        "report": [],
    }
    assert smoke.variant_summary(result)["landing_page"]["variants"][0][
        "variant_angle"
    ] == "pain_led"
    assert "email_campaign did not report multiple variants" not in (
        smoke._execution_errors(result, smoke.saved_ids_by_output(result))
    )


def test_execution_errors_respect_selected_outputs() -> None:
    result = {
        "status": "completed",
        "steps": [
            {
                "output": "blog_post",
                "status": "completed",
                "result": {
                    "variant_count": 3,
                    "saved_ids": ["blog-1", "blog-2", "blog-3"],
                },
            },
        ],
    }

    saved_ids = smoke.saved_ids_by_output(result, outputs=("blog_post",))

    assert saved_ids == {"blog_post": ["blog-1", "blog-2", "blog-3"]}
    assert smoke._execution_errors(result, saved_ids, outputs=("blog_post",)) == []
    assert smoke.variant_summary(result, outputs=("blog_post",)) == {
        "blog_post": {"variant_count": 3, "variants": []}
    }


def test_execution_errors_allow_single_report_run_without_variants() -> None:
    result = {
        "status": "completed",
        "steps": [
            {
                "output": "report",
                "status": "completed",
                "result": {
                    "requested": 1,
                    "generated": 1,
                    "skipped": 0,
                    "saved_ids": ["report-1"],
                },
            },
        ],
    }

    saved_ids = smoke.saved_ids_by_output(result, outputs=("report",))

    assert saved_ids == {"report": ["report-1"]}
    assert smoke._execution_errors(result, saved_ids, outputs=("report",)) == []
    assert smoke.variant_summary(result, outputs=("report",)) == {
        "report": {"variant_count": 0, "variants": []}
    }


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


@pytest.mark.asyncio
async def test_review_saved_ids_threads_email_campaign_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def _fake_update_statuses(
        asset: str,
        pool: Any,
        *,
        asset_ids: list[str],
        status: str,
        scope: Any,
    ) -> list[str]:
        calls.append({
            "asset": asset,
            "pool": pool,
            "asset_ids": asset_ids,
            "status": status,
            "scope": scope,
        })
        return list(asset_ids)

    monkeypatch.setattr(
        "extracted_content_pipeline.api.generated_assets._update_asset_statuses",
        _fake_update_statuses,
    )

    pool = object()
    scope = object()
    reviewed = await smoke.review_saved_ids(
        pool,
        scope,
        {"email_campaign": ["campaign-1", "campaign-2"]},
    )

    assert reviewed["email_campaign"]["updated_ids"] == [
        "campaign-1",
        "campaign-2",
    ]
    assert reviewed["email_campaign"]["missing_ids"] == []
    assert calls == [{
        "asset": "email_campaign",
        "pool": pool,
        "asset_ids": ["campaign-1", "campaign-2"],
        "status": "approved",
        "scope": scope,
    }]


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


class _ExportResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def as_dict(self) -> dict[str, Any]:
        return {"count": len(self.rows), "filters": {}, "rows": list(self.rows)}


@pytest.mark.asyncio
async def test_export_saved_drafts_exports_email_campaign_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def _fake_list_campaign_drafts(pool: Any, **kwargs: Any) -> _ExportResult:
        calls.append({"pool": pool, **kwargs})
        return _ExportResult([
            {"id": "campaign-1", "subject": "First"},
            {"id": "campaign-2", "subject": "Second"},
            {"id": "other-campaign", "subject": "Other"},
        ])

    monkeypatch.setattr(
        "extracted_content_pipeline.campaign_postgres_export.list_campaign_drafts",
        _fake_list_campaign_drafts,
    )

    pool = object()
    scope = object()
    exports = await smoke.export_saved_drafts(
        pool,
        scope=scope,
        target_mode="vendor_retention",
        saved_ids={"email_campaign": ["campaign-1", "campaign-2"]},
    )

    assert calls == [{
        "pool": pool,
        "scope": scope,
        "statuses": ("approved",),
        "target_mode": "vendor_retention",
        "limit": 100,
    }]
    assert exports["email_campaign"]["count"] == 2
    assert [row["id"] for row in exports["email_campaign"]["rows"]] == [
        "campaign-1",
        "campaign-2",
    ]


@pytest.mark.asyncio
async def test_export_saved_drafts_exports_report_by_saved_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def _fake_export_report_drafts(repository: Any, **kwargs: Any) -> _ExportResult:
        calls.append({"repository": repository, **kwargs})
        return _ExportResult([
            {"id": "report-1", "title": "Ticket Gap Report"},
            {"id": "other-report", "title": "Other"},
        ])

    monkeypatch.setattr(
        "extracted_content_pipeline.report_export.export_report_drafts",
        _fake_export_report_drafts,
    )

    pool = object()
    scope = object()
    exports = await smoke.export_saved_drafts(
        pool,
        scope=scope,
        target_mode="vendor_retention",
        saved_ids={"report": ["report-1"]},
    )

    assert calls[0]["repository"].pool is pool
    assert calls[0]["scope"] is scope
    assert calls[0]["status"] is None
    assert calls[0]["target_mode"] == "vendor_retention"
    assert calls[0]["limit"] == 100
    assert exports["report"]["count"] == 1
    assert exports["report"]["rows"] == [
        {"id": "report-1", "title": "Ticket Gap Report"}
    ]


def test_variant_persistence_errors_fail_on_duplicate_email_sequence_ids() -> None:
    assert smoke.variant_persistence_errors(
        {"status": "completed"},
        saved_ids={"email_campaign": ["campaign-1", "campaign-1"]},
        exports={
            "email_campaign": {
                "count": 1,
                "rows": [
                    {"id": "campaign-1", "channel": "email_cold"},
                    {"id": "campaign-1", "channel": "email_followup"},
                ],
            }
        },
    ) == [
        "email_campaign sequence persistence collapsed: "
        "2 saved id entries, 1 unique saved id(s), 1 exported row(s)"
    ]


def test_variant_persistence_errors_fail_on_missing_email_sequence_channel() -> None:
    assert smoke.variant_persistence_errors(
        {"status": "completed"},
        saved_ids={"email_campaign": ["campaign-1"]},
        exports={
            "email_campaign": {
                "count": 1,
                "rows": [{"id": "campaign-1", "channel": "email_followup"}],
            }
        },
    ) == ["email_campaign sequence missing required channel: email_cold"]
