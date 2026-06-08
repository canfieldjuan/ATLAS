#!/usr/bin/env python3
"""Gate A live quality proof for brand voice and output variations."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smoke_content_ops_live_generation import (  # noqa: E402
    DEFAULT_LANDING_PAGE_INPUTS,
    _align_blog_payload_to_seed,
    _filter_saved_draft_export_rows,
    _load_csv_rows,
    _load_dotenv_files,
    _payload_with_support_ticket_provider,
    _resolve_runtime_dependencies,
    _seed_default_blog_blueprint,
)


DEFAULT_OUTPUTS = (
    "email_campaign",
    "landing_page",
    "blog_post",
    "sales_brief",
    "report",
)
SEQUENCE_OUTPUTS = frozenset({"email_campaign"})
SINGLE_RUN_OUTPUTS = frozenset({"report"})
OPPORTUNITY_BACKED_OUTPUTS = frozenset({"email_campaign", "report"})
SEQUENCE_REQUIRED_CHANNELS = {"email_campaign": ("email_cold", "email_followup")}
DEFAULT_SUPPORT_TICKET_CSV = (
    ROOT / "extracted_content_pipeline" / "examples" / "support_ticket_saas_demo_sources.csv"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run issue #1357 Gate A live quality validation through the real "
            "Content Ops service builder, Postgres persistence, review queue, "
            "and exact draft export."
        )
    )
    parser.add_argument(
        "--account-id",
        required=True,
        help="Tenant/account id used to scope generated drafts.",
    )
    parser.add_argument("--user-id", default=None)
    parser.add_argument(
        "--target-mode",
        default="vendor_retention",
        help="Content Ops target_mode for the live run.",
    )
    parser.add_argument(
        "--outputs",
        default=",".join(DEFAULT_OUTPUTS),
        help="Comma-separated outputs to run. Defaults to all Gate A outputs.",
    )
    parser.add_argument(
        "--support-ticket-csv",
        type=Path,
        default=DEFAULT_SUPPORT_TICKET_CSV,
        help=(
            "Support-ticket CSV used as the real source material. Defaults "
            "to the packaged SaaS demo ticket export."
        ),
    )
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        type=Path,
        help="Additional dotenv file to load. Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the live result, exports, and summary are written.",
    )
    parser.add_argument(
        "--variant-count",
        type=int,
        default=3,
        help="Requested output variant count. Must be greater than 1.",
    )
    parser.add_argument(
        "--quality-repair-attempts",
        type=int,
        default=1,
        help="Landing-page quality repair attempts for each variant.",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="Optional per-request cost gate passed through to Content Ops.",
    )
    parser.add_argument(
        "--account-usage-budget-usd",
        type=float,
        default=None,
        help="Optional account usage budget passed through to Content Ops.",
    )
    parser.add_argument(
        "--account-usage-budget-days",
        type=int,
        default=7,
        help="Window for --account-usage-budget-usd.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine JSON.")
    return parser.parse_args(argv)


def _default_brand_voice(account_id: str) -> dict[str, Any]:
    return {
        "id": "gate-a-sharp-support-operator",
        "account_id": account_id,
        "name": "Sharp Support Operator",
        "descriptors": [
            "plainspoken",
            "skeptical",
            "operator-led",
            "specific",
            "short sentences",
        ],
        "preferred_pov": "second_person",
        "reading_level": "concise",
        "banned_terms": [
            "revolutionary",
            "transformative",
            "game-changing",
            "seamless",
            "synergy",
        ],
        "exemplars": [
            (
                "Show the repeated ticket pattern first. Name the risk. "
                "Then give the support lead the next action."
            ),
            (
                "Do not dress up a weak claim. If the source does not prove "
                "it, leave it out."
            ),
            (
                "Support leaders do not need theater. They need the repeat "
                "questions, the exposure, and the fix."
            ),
        ],
        "metadata": {"validation_gate": "issue-1357"},
    }


def _report_opportunity_id_from_rows(
    support_ticket_rows: Sequence[Mapping[str, Any]],
) -> str:
    for row in support_ticket_rows:
        for key in ("target_id", "Ticket ID", "ticket_id", "id"):
            value = str(row.get(key) or "").strip()
            if value:
                return value
    return ""


def _resolve_outputs(raw_outputs: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw_outputs, str):
        candidates = raw_outputs.split(",")
    else:
        candidates = list(raw_outputs)
    outputs = tuple(
        dict.fromkeys(str(output).strip() for output in candidates if str(output).strip())
    )
    if not outputs:
        raise ValueError("at least one output is required")
    unsupported = [output for output in outputs if output not in DEFAULT_OUTPUTS]
    if unsupported:
        raise ValueError(
            "unsupported output(s): "
            + ", ".join(unsupported)
            + "; expected one of "
            + ", ".join(DEFAULT_OUTPUTS)
        )
    return outputs


def build_gate_a_payload(
    *,
    account_id: str,
    support_ticket_rows: Sequence[Mapping[str, Any]],
    target_mode: str,
    variant_count: int,
    quality_repair_attempts: int,
    outputs: Sequence[str] = DEFAULT_OUTPUTS,
    max_cost_usd: float | None = None,
    account_usage_budget_usd: float | None = None,
    account_usage_budget_days: int = 7,
) -> dict[str, Any]:
    if variant_count <= 1:
        raise ValueError("variant_count must be greater than 1")
    if quality_repair_attempts < 0:
        raise ValueError("quality_repair_attempts must be >= 0")
    selected_outputs = _resolve_outputs(outputs)

    inputs = {
        **dict(DEFAULT_LANDING_PAGE_INPUTS),
        "source_material": [dict(row) for row in support_ticket_rows],
        "brand_voice": _default_brand_voice(account_id),
        "topic": "Support ticket FAQ gaps: what repeat tickets reveal before renewal",
        "filters": {"topic_type": "support_ticket_faq_gap_live_gate_a"},
        "brief_type": "renewal",
        "landing_page_quality_repair_attempts": quality_repair_attempts,
        "campaign_name": "Support Ticket FAQ Gap Audit",
        "target_account": "SaaS support team with repeat ticket backlog",
        "offer": "Turn repeat support tickets into approved FAQ answers",
        "audience": "SaaS support leaders carrying a repeat-ticket backlog",
        "selling": {
            "product_name": "Support Ticket FAQ Gap Audit",
            "affiliate_url": "https://finetunelab.ai/systems/ai-content-ops/intake",
            "sender_name": "FineTune Lab",
            "sender_title": "Content Ops Team",
            "sender_company": "FineTune Lab",
        },
        "target_keyword": "support ticket FAQ gaps",
        "search_intent": (
            "Find the highest-risk repeat support questions and turn them "
            "into review-ready customer answers."
        ),
        "primary_entity": "Support Ticket FAQ Gap Audit",
        "audience_entity": "SaaS support leader",
        "cta_label": "Upload Ticket CSV -- Get the Gap Audit",
        "cta_url": "/systems/ai-content-ops/intake",
    }
    if "report" in selected_outputs:
        report_opportunity_id = _report_opportunity_id_from_rows(support_ticket_rows)
        if report_opportunity_id:
            inputs["opportunity_id"] = report_opportunity_id
    payload: dict[str, Any] = {
        "outputs": list(selected_outputs),
        "target_mode": target_mode,
        "limit": 1,
        "variant_count": variant_count,
        "require_quality_gates": True,
        "inputs": inputs,
    }
    if max_cost_usd is not None:
        payload["max_cost_usd"] = max_cost_usd
    if account_usage_budget_usd is not None:
        payload["account_usage_budget_usd"] = account_usage_budget_usd
        payload["account_usage_budget_days"] = account_usage_budget_days
    return payload


async def run_gate_a_live_quality(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if int(args.variant_count) <= 1:
        raise SystemExit("--variant-count must be greater than 1")
    try:
        selected_outputs = _resolve_outputs(str(args.outputs or ""))
    except ValueError as exc:
        raise SystemExit(f"--outputs: {exc}") from None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _load_dotenv_files(list(args.env_file or []))
    source_rows = _load_csv_rows(Path(args.support_ticket_csv))
    payload = build_gate_a_payload(
        account_id=str(args.account_id or "").strip(),
        support_ticket_rows=source_rows,
        target_mode=str(args.target_mode or "vendor_retention").strip()
        or "vendor_retention",
        variant_count=int(args.variant_count),
        quality_repair_attempts=int(args.quality_repair_attempts),
        outputs=selected_outputs,
        max_cost_usd=args.max_cost_usd,
        account_usage_budget_usd=args.account_usage_budget_usd,
        account_usage_budget_days=int(args.account_usage_budget_days),
    )

    init_database, close_database, services_factory, executor, tenant_scope_cls = (
        _resolve_runtime_dependencies()
    )
    execution_result: Mapping[str, Any] | None = None
    reviewed: dict[str, Any] = {}
    exports: dict[str, Any] = {}
    seeded_blog_blueprint: Mapping[str, Any] | None = None
    opportunity_import: Mapping[str, Any] | None = None
    errors: list[str] = []
    configured_outputs: tuple[str, ...] = ()
    scope = tenant_scope_cls(
        account_id=str(args.account_id or "").strip(),
        user_id=str(args.user_id or "").strip(),
    )

    try:
        await init_database()
        services = services_factory()
        configured_outputs = tuple(services.configured_outputs())
        missing_outputs = [
            output for output in selected_outputs if output not in configured_outputs
        ]
        if missing_outputs:
            errors.append(
                "services not configured for required outputs: "
                + ", ".join(missing_outputs)
            )
        else:
            pool = _current_db_pool()
            (
                payload,
                seeded_blog_blueprint,
                opportunity_import,
            ) = await prepare_gate_a_output_dependencies(
                args,
                scope,
                pool=pool,
                payload=payload,
                selected_outputs=selected_outputs,
                source_rows=source_rows,
                output_dir=output_dir,
            )
            execution_result = await executor(
                payload,
                services=services,
                scope=scope,
                trace_metadata={
                    "validation_gate": "issue-1357",
                    "validation_slice": "gate_a_live_output_quality",
                },
            )
            _write_json(output_dir / "execution-result.json", execution_result)
            saved_ids = saved_ids_by_output(execution_result, outputs=selected_outputs)
            errors.extend(
                _execution_errors(execution_result, saved_ids, outputs=selected_outputs)
            )
            if not errors:
                reviewed = await review_saved_ids(pool, scope, saved_ids)
                _write_json(output_dir / "review-results.json", reviewed)
                errors.extend(_review_errors(reviewed, outputs=selected_outputs))
                exports = await export_saved_drafts(
                    pool,
                    scope=scope,
                    target_mode=str(args.target_mode or "vendor_retention").strip()
                    or "vendor_retention",
                    saved_ids=saved_ids,
                )
                for output, export in exports.items():
                    _write_json(output_dir / f"export-{output}.json", export)
                errors.extend(
                    variant_persistence_errors(
                        execution_result,
                        saved_ids=saved_ids,
                        exports=exports,
                        outputs=selected_outputs,
                    )
                )
    except Exception as exc:  # live proof must preserve failure details.
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        await close_database()

    summary = {
        "ok": not errors,
        "status": "passed" if not errors else "failed",
        "issue": 1357,
        "account_id": str(args.account_id or "").strip(),
        "support_ticket_csv": str(Path(args.support_ticket_csv)),
        "output_dir": str(output_dir),
        "required_outputs": list(selected_outputs),
        "configured_outputs": list(configured_outputs),
        "variant_count": int(args.variant_count),
        "opportunity_import": dict(opportunity_import or {}),
        "seeded_blog_blueprint": dict(seeded_blog_blueprint or {}),
        "saved_ids": saved_ids_by_output(execution_result, outputs=selected_outputs),
        "variant_summary": variant_summary(execution_result, outputs=selected_outputs),
        "review": reviewed,
        "export_counts": {
            output: int(_mapping(export).get("count") or 0)
            for output, export in exports.items()
        },
        "budget": {
            "max_cost_usd": args.max_cost_usd,
            "account_usage_budget_usd": args.account_usage_budget_usd,
            "account_usage_budget_days": int(args.account_usage_budget_days),
            "triggered": bool(_budget_errors(execution_result)),
            "messages": _budget_errors(execution_result),
        },
        "errors": errors,
    }
    _write_json(output_dir / "summary.json", summary)
    return (0 if summary["ok"] else 1), summary


async def prepare_gate_a_output_dependencies(
    args: argparse.Namespace,
    scope: Any,
    *,
    pool: Any,
    payload: Mapping[str, Any],
    selected_outputs: Sequence[str],
    source_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, Any], Mapping[str, Any] | None, Mapping[str, Any] | None]:
    prepared_payload = await _payload_with_support_ticket_provider(payload, scope=scope)
    _reassert_gate_a_controls(
        prepared_payload,
        account_id=str(args.account_id or "").strip(),
        variant_count=int(args.variant_count),
        outputs=selected_outputs,
    )
    seeded_blog_blueprint: Mapping[str, Any] | None = None
    if "blog_post" in selected_outputs:
        seeded_blog_blueprint = await _seed_default_blog_blueprint(args, scope)
        _align_blog_payload_to_seed(prepared_payload, seeded_blog_blueprint)

    opportunity_import: Mapping[str, Any] | None = None
    if any(output in OPPORTUNITY_BACKED_OUTPUTS for output in selected_outputs):
        opportunity_import = await seed_email_campaign_opportunities(
            pool,
            scope,
            target_mode=str(args.target_mode or "vendor_retention").strip()
            or "vendor_retention",
            source_rows=source_rows,
            filters=_payload_filters(prepared_payload),
        )
        _write_json(output_dir / "opportunity-import.json", opportunity_import)
        if "report" in selected_outputs:
            _bind_report_opportunity(prepared_payload, opportunity_import)
    return prepared_payload, seeded_blog_blueprint, opportunity_import


def _payload_filters(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    filters = inputs.get("filters")
    return filters if isinstance(filters, Mapping) else None


def _bind_report_opportunity(
    payload: dict[str, Any],
    opportunity_import: Mapping[str, Any],
) -> None:
    inputs = payload.setdefault("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError("payload inputs must remain an object before report binding")
    requested_target_id = str(inputs.get("opportunity_id") or "").strip()
    target_ids = opportunity_import.get("target_ids") or ()
    if isinstance(target_ids, (str, bytes)) or not isinstance(target_ids, Sequence):
        target_ids = ()
    imported_target_ids = tuple(
        dict.fromkeys(str(item).strip() for item in target_ids if str(item).strip())
    )
    if requested_target_id:
        if requested_target_id not in imported_target_ids:
            raise ValueError(
                "report opportunity_id was not imported: " + requested_target_id
            )
        target_id = requested_target_id
    elif len(imported_target_ids) == 1:
        target_id = imported_target_ids[0]
    elif not imported_target_ids:
        raise ValueError("report opportunity import returned no target_ids")
    else:
        raise ValueError(
            "report opportunity import returned multiple target_ids without "
            "an explicit opportunity_id"
        )
    filters = inputs.get("filters")
    filters = dict(filters) if isinstance(filters, Mapping) else {}
    filters["target_id"] = target_id
    inputs["filters"] = filters
    inputs["opportunity_id"] = target_id


async def seed_email_campaign_opportunities(
    pool: Any,
    scope: Any,
    *,
    target_mode: str,
    source_rows: Sequence[Mapping[str, Any]],
    filters: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    from extracted_content_pipeline.campaign_postgres_import import (  # noqa: PLC0415
        import_campaign_opportunities,
    )
    from extracted_content_pipeline.campaign_source_adapters import (  # noqa: PLC0415
        source_rows_to_campaign_opportunities,
    )

    loaded = source_rows_to_campaign_opportunities(
        source_rows,
        target_mode=target_mode,
        default_fields=filters,
    )
    imported = await import_campaign_opportunities(
        pool,
        loaded.opportunities,
        scope=scope,
        target_mode=target_mode,
        replace_existing=True,
        normalize=False,
        warnings=loaded.warnings,
        source="gate_a_support_ticket_csv",
    )
    return imported.as_dict()


def _reassert_gate_a_controls(
    payload: dict[str, Any],
    *,
    account_id: str,
    variant_count: int,
    outputs: Sequence[str],
) -> None:
    payload["outputs"] = list(outputs)
    payload["limit"] = 1
    payload["variant_count"] = variant_count
    inputs = payload.setdefault("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError("payload inputs must remain an object after provider merge")
    inputs["brand_voice"] = _default_brand_voice(account_id)
    inputs.setdefault("target_account", "SaaS support team with repeat ticket backlog")


def saved_ids_by_output(
    result: Mapping[str, Any] | None,
    *,
    outputs: Sequence[str] = DEFAULT_OUTPUTS,
) -> dict[str, list[str]]:
    ids_by_output: dict[str, list[str]] = {output: [] for output in outputs}
    if not isinstance(result, Mapping):
        return ids_by_output
    for output in outputs:
        payload = _step_payload(result, output)
        raw_ids = payload.get("saved_ids") or ()
        if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
            continue
        ids_by_output[output] = [
            str(item).strip() for item in raw_ids if str(item).strip()
        ]
    return ids_by_output


def variant_summary(
    result: Mapping[str, Any] | None,
    *,
    outputs: Sequence[str] = DEFAULT_OUTPUTS,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for output in outputs:
        payload = _step_payload(result, output)
        variants = payload.get("variant_results") or ()
        if isinstance(variants, (str, bytes)) or not isinstance(variants, Sequence):
            variants = ()
        output_summary: list[dict[str, Any]] = []
        for variant in variants:
            if not isinstance(variant, Mapping):
                continue
            angle = _mapping(variant.get("variant_angle"))
            output_summary.append({
                "variant_angle": angle.get("id") or angle.get("label"),
                "generated": int(variant.get("generated") or 0),
                "saved_ids": [
                    str(item)
                    for item in variant.get("saved_ids") or ()
                    if str(item).strip()
                ],
                "errors": list(variant.get("errors") or ()),
            })
        summary[output] = {
            "variant_count": int(payload.get("variant_count") or len(output_summary)),
            "variants": output_summary,
        }
    return summary


async def review_saved_ids(
    pool: Any,
    scope: Any,
    saved_ids: Mapping[str, Sequence[str]],
    *,
    status: str = "approved",
) -> dict[str, Any]:
    from extracted_content_pipeline.api.generated_assets import (  # noqa: PLC0415
        _update_asset_statuses,
    )

    reviewed: dict[str, Any] = {}
    for output, ids in saved_ids.items():
        clean_ids = [str(item).strip() for item in ids if str(item).strip()]
        if not clean_ids:
            reviewed[output] = {
                "requested_ids": [],
                "updated_ids": [],
                "missing_ids": [],
            }
            continue
        updated = await _update_asset_statuses(
            output,
            pool,
            asset_ids=clean_ids,
            status=status,
            scope=scope,
        )
        updated_ids = [str(item).strip() for item in updated if str(item).strip()]
        reviewed[output] = {
            "requested_ids": clean_ids,
            "updated_ids": updated_ids,
            "missing_ids": [item for item in clean_ids if item not in set(updated_ids)],
        }
    return reviewed


async def export_saved_drafts(
    pool: Any,
    *,
    scope: Any,
    target_mode: str,
    saved_ids: Mapping[str, Sequence[str]],
) -> dict[str, Mapping[str, Any]]:
    from extracted_content_pipeline.blog_post_export import (  # noqa: PLC0415
        export_blog_post_drafts,
    )
    from extracted_content_pipeline.blog_post_postgres import (  # noqa: PLC0415
        PostgresBlogPostRepository,
    )
    from extracted_content_pipeline.campaign_postgres_export import (  # noqa: PLC0415
        list_campaign_drafts,
    )
    from extracted_content_pipeline.landing_page_export import (  # noqa: PLC0415
        export_landing_page_drafts,
    )
    from extracted_content_pipeline.landing_page_postgres import (  # noqa: PLC0415
        PostgresLandingPageRepository,
    )
    from extracted_content_pipeline.report_export import (  # noqa: PLC0415
        export_report_drafts,
    )
    from extracted_content_pipeline.report_postgres import (  # noqa: PLC0415
        PostgresReportRepository,
    )
    from extracted_content_pipeline.sales_brief_export import (  # noqa: PLC0415
        export_sales_brief_drafts,
    )
    from extracted_content_pipeline.sales_brief_postgres import (  # noqa: PLC0415
        PostgresSalesBriefRepository,
    )

    exports: dict[str, Mapping[str, Any]] = {}
    campaign_ids = saved_ids.get("email_campaign") or ()
    if campaign_ids:
        export = await list_campaign_drafts(
            pool,
            scope=scope,
            statuses=("approved",),
            target_mode=target_mode,
            limit=max(100, len(campaign_ids)),
        )
        exports["email_campaign"] = _filter_saved_draft_export_rows(
            export.as_dict(),
            campaign_ids,
        )
    landing_ids = saved_ids.get("landing_page") or ()
    if landing_ids:
        export = await export_landing_page_drafts(
            PostgresLandingPageRepository(pool),
            scope=scope,
            status=None,
            ids=landing_ids,
            limit=len(landing_ids),
        )
        exports["landing_page"] = _filter_saved_draft_export_rows(
            export.as_dict(),
            landing_ids,
        )
    blog_ids = saved_ids.get("blog_post") or ()
    if blog_ids:
        export = await export_blog_post_drafts(
            PostgresBlogPostRepository(pool),
            scope=scope,
            status=None,
            ids=blog_ids,
            limit=len(blog_ids),
        )
        exports["blog_post"] = _filter_saved_draft_export_rows(
            export.as_dict(),
            blog_ids,
        )
    sales_ids = saved_ids.get("sales_brief") or ()
    if sales_ids:
        export = await export_sales_brief_drafts(
            PostgresSalesBriefRepository(pool),
            scope=scope,
            status=None,
            target_mode=target_mode,
            limit=max(100, len(sales_ids)),
        )
        exports["sales_brief"] = _filter_saved_draft_export_rows(
            export.as_dict(),
            sales_ids,
        )
    report_ids = saved_ids.get("report") or ()
    if report_ids:
        export = await export_report_drafts(
            PostgresReportRepository(pool),
            scope=scope,
            status=None,
            target_mode=target_mode,
            limit=max(100, len(report_ids)),
        )
        exports["report"] = _filter_saved_draft_export_rows(
            export.as_dict(),
            report_ids,
        )
    return exports


def _execution_errors(
    result: Mapping[str, Any] | None,
    saved_ids: Mapping[str, Sequence[str]],
    *,
    outputs: Sequence[str] = DEFAULT_OUTPUTS,
) -> list[str]:
    if not isinstance(result, Mapping):
        return ["execution returned no result"]
    errors: list[str] = []
    if result.get("status") != "completed":
        errors.append(f"execution status was {result.get('status')!r}, not 'completed'")
    for output in outputs:
        step = _step(result, output)
        if not step:
            errors.append(f"execution result did not include {output}")
            continue
        if step.get("status") != "completed":
            errors.append(f"{output} status was {step.get('status')!r}")
        if not saved_ids.get(output):
            errors.append(f"{output} returned no saved_ids")
        payload = _step_payload(result, output)
        if (
            output not in SEQUENCE_OUTPUTS
            and output not in SINGLE_RUN_OUTPUTS
            and int(payload.get("variant_count") or 0) <= 1
        ):
            errors.append(f"{output} did not report multiple variants")
    return errors


def _review_errors(
    reviewed: Mapping[str, Any],
    *,
    outputs: Sequence[str] = DEFAULT_OUTPUTS,
) -> list[str]:
    errors: list[str] = []
    for output in outputs:
        review = _mapping(reviewed.get(output))
        missing = list(review.get("missing_ids") or ())
        if missing:
            errors.append(f"{output} review update missed ids: {', '.join(missing)}")
    return errors


def variant_persistence_errors(
    result: Mapping[str, Any] | None,
    *,
    saved_ids: Mapping[str, Sequence[str]],
    exports: Mapping[str, Mapping[str, Any]],
    outputs: Sequence[str] = DEFAULT_OUTPUTS,
) -> list[str]:
    errors: list[str] = []
    for output in outputs:
        if output in SEQUENCE_OUTPUTS:
            errors.extend(_sequence_persistence_errors(output, saved_ids, exports))
            continue
        successful_variants = _successful_variant_count(result, output)
        if successful_variants <= 1:
            continue
        raw_ids = [
            str(item).strip()
            for item in saved_ids.get(output, ())
            if str(item).strip()
        ]
        unique_ids = tuple(dict.fromkeys(raw_ids))
        export_count = int(_mapping(exports.get(output)).get("count") or 0)
        if len(unique_ids) < successful_variants or export_count < successful_variants:
            errors.append(
                f"{output} variant persistence collapsed: "
                f"{successful_variants} successful variant(s), "
                f"{len(raw_ids)} saved id entr{'y' if len(raw_ids) == 1 else 'ies'}, "
                f"{len(unique_ids)} unique saved id(s), "
                f"{export_count} exported row(s)"
            )
    return errors


def _sequence_persistence_errors(
    output: str,
    saved_ids: Mapping[str, Sequence[str]],
    exports: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    raw_ids = [
        str(item).strip()
        for item in saved_ids.get(output, ())
        if str(item).strip()
    ]
    if not raw_ids:
        return []
    unique_ids = tuple(dict.fromkeys(raw_ids))
    export = _mapping(exports.get(output))
    export_count = int(export.get("count") or 0)
    rows = export.get("rows")
    exported_channels = {
        str(row.get("channel") or "").strip()
        for row in rows
        if isinstance(row, Mapping)
    } if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else set()
    required_channels = set(SEQUENCE_REQUIRED_CHANNELS.get(output, ()))
    missing_channels = sorted(required_channels - exported_channels)
    if missing_channels:
        return [
            f"{output} sequence missing required channel"
            f"{'s' if len(missing_channels) > 1 else ''}: "
            + ", ".join(missing_channels)
        ]
    if len(unique_ids) == len(raw_ids) and export_count >= len(raw_ids):
        return []
    return [
        f"{output} sequence persistence collapsed: "
        f"{len(raw_ids)} saved id entr{'y' if len(raw_ids) == 1 else 'ies'}, "
        f"{len(unique_ids)} unique saved id(s), "
        f"{export_count} exported row(s)"
    ]


def _successful_variant_count(result: Mapping[str, Any] | None, output: str) -> int:
    payload = _step_payload(result, output)
    variants = payload.get("variant_results") or ()
    if isinstance(variants, (str, bytes)) or not isinstance(variants, Sequence):
        return 0
    count = 0
    for variant in variants:
        if isinstance(variant, Mapping) and int(variant.get("generated") or 0) > 0:
            count += 1
    return count


def _budget_errors(result: Mapping[str, Any] | None) -> list[str]:
    messages: list[str] = []
    if not isinstance(result, Mapping):
        return messages
    for error in result.get("errors") or ():
        if not isinstance(error, Mapping):
            continue
        text = json.dumps(error, sort_keys=True, default=str).lower()
        if "budget" in text or "cost" in text:
            messages.append(text)
    return messages


def _step(result: Mapping[str, Any] | None, output: str) -> Mapping[str, Any]:
    if not isinstance(result, Mapping):
        return {}
    for step in result.get("steps") or ():
        if isinstance(step, Mapping) and step.get("output") == output:
            return step
    return {}


def _step_payload(result: Mapping[str, Any] | None, output: str) -> Mapping[str, Any]:
    payload = _step(result, output).get("result")
    return payload if isinstance(payload, Mapping) else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _current_db_pool() -> Any:
    from atlas_brain.storage.database import get_db_pool  # noqa: PLC0415

    return get_db_pool()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


async def _amain(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    code, result = await run_gate_a_live_quality(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    else:
        status = "PASS" if result.get("ok") else "FAIL"
        print(f"Gate A live quality proof: {status}")
        for output, ids in result.get("saved_ids", {}).items():
            print(f"{output}: {len(ids)} saved id(s)")
        for error in result.get("errors") or ():
            print(f"ERROR: {error}")
        print(f"Artifacts: {result.get('output_dir')}")
    return code


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
