#!/usr/bin/env python3
"""Smoke-test live Content Ops generation through host wiring."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Awaitable, Callable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional host dependency
    load_dotenv = None

from extracted_content_pipeline.support_ticket_context_contract import (
    SUPPORT_TICKET_CATEGORY,
    SUPPORT_TICKET_DEFAULT_TOPIC,
    SUPPORT_TICKET_LAST_90_DAYS_REVIEW_PERIOD,
    SUPPORT_TICKET_SOURCE,
    SUPPORT_TICKET_TOPIC_TYPE,
    UPLOADED_SUPPORT_TICKETS_SOURCE_PERIOD,
    UPLOADED_TICKETS_REVIEW_PERIOD,
)


DEFAULT_LANDING_PAGE_INPUTS: Mapping[str, Any] = {
    "campaign_name": "FAQ Report",
    "offer": "Turn repeat support tickets into customer-ready FAQ answers",
    "audience": "10-50 person SaaS support team",
    "target_keyword": "support ticket FAQ",
    "secondary_keywords": [
        "reduce repeat support tickets",
        "help center answers",
    ],
    "search_intent": (
        "Find a low-friction way to turn old support tickets into answers "
        "customers can use."
    ),
    "primary_entity": "FAQ Report",
    "audience_entity": "small SaaS support team",
    "objections": [
        "Will this publish automatically?",
        "Do we need a full-time docs person?",
    ],
    "faq_questions": [
        "What happens after I upload the CSV?",
        "Does FAQ Report publish automatically?",
    ],
    "source_period": "Last 90 days of support tickets",
    "internal_links": ["/systems/ai-content-ops/intake"],
    "cta_label": "Upload Ticket CSV -- Free Analysis",
    "cta_url": "/systems/ai-content-ops/intake",
}
DEFAULT_BLOG_TOPIC = (
    "Support ticket FAQ gaps: what 90 days of repeat tickets reveal"
)
DEFAULT_BLOG_TOPIC_TYPE = "content_ops_live_smoke"
DEFAULT_SUPPORT_TICKET_CSV = (
    ROOT / "extracted_content_pipeline" / "examples" / "support_ticket_sources.csv"
)
SUPPORT_TICKET_BLOG_TOPIC = SUPPORT_TICKET_DEFAULT_TOPIC
SUPPORT_TICKET_BLOG_TOPIC_TYPE = SUPPORT_TICKET_TOPIC_TYPE

AsyncCallable = Callable[[], Awaitable[None]]
ServicesFactory = Callable[[], Any]
Executor = Callable[..., Awaitable[dict[str, Any]]]
BlogBlueprintSeeder = Callable[[argparse.Namespace, Any], Awaitable[Mapping[str, Any]]]
DraftExporter = Callable[[str, Sequence[str], Any], Awaitable[Mapping[str, Any]]]
GeneratedContentEvaluator = Callable[..., Mapping[str, Any]]


def _load_dotenv_files(extra_env_files: list[Path] | None = None) -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
        load_dotenv(ROOT / ".env.local", override=True)
        for path in extra_env_files or []:
            if not path.exists():
                raise SystemExit(f"--env-file does not exist: {path}")
            load_dotenv(path, override=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test live AI Content Ops generation through "
            "the host DB pool, packaged skills, and pipeline-routed LLM."
        )
    )
    parser.add_argument(
        "--output",
        choices=("landing_page", "blog_post"),
        default="landing_page",
        help="Content Ops output to smoke-test. Defaults to landing_page.",
    )
    parser.add_argument(
        "--account-id",
        required=True,
        help="Tenant/account id used to scope the generated draft.",
    )
    parser.add_argument("--user-id", default=None)
    parser.add_argument(
        "--target-mode",
        default="vendor_retention",
        help="Content Ops target_mode for blog blueprints and execution.",
    )
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        type=Path,
        help=(
            "Additional .env file to load before resolving Atlas DB and LLM "
            "settings. Repeatable."
        ),
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Optional JSON object merged over the default inputs.",
    )
    parser.add_argument(
        "--support-ticket-csv",
        nargs="?",
        const=DEFAULT_SUPPORT_TICKET_CSV,
        type=Path,
        help=(
            "Load support-ticket source rows from CSV and package them through "
            "the Atlas support-ticket input provider before execution. When "
            "no path is provided, uses the packaged support_ticket_sources.csv. "
            "CSV rows must include support-ticket-shaped fields such as "
            "ticket id, subject, description/message, or source_type."
        ),
    )
    parser.add_argument(
        "--blog-blueprint-json",
        type=Path,
        help=(
            "Optional custom blog blueprint JSON file for --output blog_post. "
            "The file must normalize to exactly one blueprint."
        ),
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help=(
            "Input override as key=value. JSON values are accepted. Repeatable. "
            "Example: --input cta_url=/systems/ai-content-ops/intake"
        ),
    )
    parser.add_argument(
        "--quality-repair-attempts",
        type=int,
        default=1,
        help="Landing-page quality repair attempts for the generated draft.",
    )
    parser.add_argument(
        "--no-quality-gates",
        action="store_true",
        help="Disable landing-page quality gates for this smoke run.",
    )
    parser.add_argument(
        "--output-result",
        type=Path,
        help="Write the smoke result JSON to this path.",
    )
    parser.add_argument(
        "--export-saved-draft",
        type=Path,
        help=(
            "Write a JSON export of the exact saved landing_page/blog_post "
            "draft ids produced by this smoke run."
        ),
    )
    parser.add_argument(
        "--evaluate-generated-content",
        action="store_true",
        help=(
            "Run deterministic support-ticket generated-content evaluation "
            "against the saved draft export. Requires --support-ticket-csv "
            "and --export-saved-draft."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print machine JSON.")
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Unable to read --input-json: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--input-json must contain a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--input-json must contain a JSON object")
    return parsed


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise SystemExit(f"Unable to read --support-ticket-csv: {exc}") from exc


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise SystemExit("--input overrides must use key=value")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise SystemExit("--input override key cannot be empty")
    value = value.strip()
    if not value:
        return key, ""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return key, parsed


def _payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.quality_repair_attempts) < 0:
        raise SystemExit("--quality-repair-attempts must be >= 0")
    output = str(args.output or "landing_page").strip() or "landing_page"
    if _uses_support_ticket_csv(args):
        inputs = {"source_material": _load_csv_rows(Path(args.support_ticket_csv))}
    elif output == "landing_page":
        inputs = dict(DEFAULT_LANDING_PAGE_INPUTS)
    elif output == "blog_post":
        inputs = {
            "topic": DEFAULT_BLOG_TOPIC,
            "filters": {"topic_type": DEFAULT_BLOG_TOPIC_TYPE},
        }
    else:  # argparse enforces this for CLI callers; keep injected tests honest.
        raise SystemExit(f"unsupported --output: {output}")
    if args.input_json:
        inputs.update(_load_json_object(args.input_json))
    for raw in args.input:
        key, value = _parse_override(str(raw))
        inputs[key] = value
    if output == "landing_page":
        inputs["landing_page_quality_repair_attempts"] = int(args.quality_repair_attempts)
    return {
        "outputs": [output],
        "target_mode": str(args.target_mode or "vendor_retention").strip()
        or "vendor_retention",
        "limit": 1,
        "require_quality_gates": not bool(args.no_quality_gates),
        "inputs": inputs,
    }


def _uses_support_ticket_csv(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "support_ticket_csv", None))


def _evaluates_generated_content(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "evaluate_generated_content", False))


def _generated_content_evaluation_preflight_errors(
    args: argparse.Namespace,
) -> list[str]:
    if not _evaluates_generated_content(args):
        return []
    errors: list[str] = []
    if not _uses_support_ticket_csv(args):
        errors.append(
            "--evaluate-generated-content requires --support-ticket-csv"
        )
    if not getattr(args, "export_saved_draft", None):
        errors.append(
            "--evaluate-generated-content requires --export-saved-draft"
        )
    return errors


def _support_ticket_rows_from_args(args: argparse.Namespace) -> list[dict[str, str]]:
    if not _uses_support_ticket_csv(args):
        return []
    return _load_csv_rows(Path(args.support_ticket_csv))


async def _payload_with_support_ticket_provider(
    payload: Mapping[str, Any],
    *,
    scope: Any,
) -> dict[str, Any]:
    from atlas_brain._content_ops_input_provider import (  # noqa: PLC0415
        build_content_ops_input_provider,
    )
    from extracted_content_pipeline.content_ops_input_provider import (  # noqa: PLC0415
        merge_content_ops_input_package,
    )

    package = build_content_ops_input_provider().build_content_ops_input_package(
        scope=scope,
        request=payload,
    )
    if hasattr(package, "__await__"):
        package = await package
    if _provider_package_is_noop(package):
        raise ValueError(
            "--support-ticket-csv did not contain support-ticket-shaped rows; "
            "include ticket id, subject, and description/message fields."
        )
    return merge_content_ops_input_package(payload, package)


def _provider_package_is_noop(package: Any) -> bool:
    metadata = getattr(package, "metadata", None)
    inputs = getattr(package, "inputs", None)
    outputs = getattr(package, "outputs", None)
    if isinstance(package, Mapping):
        metadata = package.get("metadata")
        inputs = package.get("inputs")
        outputs = package.get("outputs")
    return (
        not inputs
        and not outputs
        and isinstance(metadata, Mapping)
        and metadata.get("mode") == "noop"
    )


def _resolve_runtime_dependencies() -> tuple[AsyncCallable, AsyncCallable, ServicesFactory, Executor, Any]:
    from atlas_brain._content_ops_services import (  # noqa: PLC0415
        build_content_ops_execution_services,
    )
    from atlas_brain.storage.database import (  # noqa: PLC0415
        close_database,
        init_database,
    )
    from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: PLC0415
    from extracted_content_pipeline.content_ops_execution import (  # noqa: PLC0415
        execute_content_ops_from_mapping,
    )

    return (
        init_database,
        close_database,
        lambda: build_content_ops_execution_services(enable_db_services=True),
        execute_content_ops_from_mapping,
        TenantScope,
    )


def _step_result(result: Mapping[str, Any], output: str) -> Mapping[str, Any] | None:
    for step in result.get("steps") or ():
        if isinstance(step, Mapping) and step.get("output") == output:
            return step
    return None


def _saved_ids_for_output(
    result: Mapping[str, Any] | None,
    output: str,
) -> tuple[str, ...]:
    if not isinstance(result, Mapping):
        return ()
    step = _step_result(result, output)
    step_payload = step.get("result") if isinstance(step, Mapping) else {}
    if not isinstance(step_payload, Mapping):
        return ()
    raw_ids = step_payload.get("saved_ids") or ()
    if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
        return ()
    return tuple(str(item).strip() for item in raw_ids if str(item).strip())


async def _export_saved_drafts(
    output: str,
    saved_ids: Sequence[str],
    scope: Any,
) -> Mapping[str, Any]:
    if output not in {"landing_page", "blog_post"}:
        raise ValueError(f"saved draft export is unsupported for output: {output}")

    from atlas_brain.storage.database import get_db_pool  # noqa: PLC0415
    from extracted_content_pipeline.blog_post_export import (  # noqa: PLC0415
        export_blog_post_drafts,
    )
    from extracted_content_pipeline.blog_post_postgres import (  # noqa: PLC0415
        PostgresBlogPostRepository,
    )
    from extracted_content_pipeline.landing_page_export import (  # noqa: PLC0415
        export_landing_page_drafts,
    )
    from extracted_content_pipeline.landing_page_postgres import (  # noqa: PLC0415
        PostgresLandingPageRepository,
    )

    pool = get_db_pool()
    if output == "landing_page":
        return (
            await export_landing_page_drafts(
                PostgresLandingPageRepository(pool),
                scope=scope,
                status=None,
                ids=saved_ids,
                limit=len(saved_ids),
            )
        ).as_dict()
    if output == "blog_post":
        return (
            await export_blog_post_drafts(
                PostgresBlogPostRepository(pool),
                scope=scope,
                status=None,
                ids=saved_ids,
                limit=len(saved_ids),
            )
        ).as_dict()


_LANDING_PAGE_SUPPORT_TICKET_EXPORT_CONTEXT_KEYS = (
    "source_row_count",
    "included_ticket_row_count",
    "skipped_ticket_row_count",
    "truncated_ticket_row_count",
    "question_like_ticket_count",
)
_BLOG_POST_SUPPORT_TICKET_EXPORT_CONTEXT_KEYS = (
    "source_row_count",
    "included_ticket_row_count",
    "question_like_ticket_count",
)


def _support_ticket_export_context_errors(
    *,
    output: str,
    payload: Mapping[str, Any],
    saved_draft_export: Mapping[str, Any] | None,
) -> list[str]:
    expected = _expected_support_ticket_export_context(output=output, payload=payload)
    if not expected:
        return []
    rows = _export_rows(saved_draft_export)
    if not rows:
        return [f"exported {output} draft did not include rows"]
    context = _exported_support_ticket_context(output=output, row=rows[0])
    if not context:
        return [f"exported {output} draft missing support-ticket source context"]
    errors: list[str] = []
    for key, expected_value in expected.items():
        actual_value = context.get(key)
        if actual_value != expected_value:
            errors.append(
                f"exported {output} draft support-ticket context mismatch for "
                f"{key}: expected {expected_value!r}, got {actual_value!r}"
            )
    return errors


def _expected_support_ticket_export_context(
    *,
    output: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    inputs = _as_mapping(payload.get("inputs"))
    context_keys = (
        _BLOG_POST_SUPPORT_TICKET_EXPORT_CONTEXT_KEYS
        if output == "blog_post"
        else _LANDING_PAGE_SUPPORT_TICKET_EXPORT_CONTEXT_KEYS
    )
    expected = {
        key: inputs[key]
        for key in context_keys
        if key in inputs
    }
    clusters = inputs.get("top_ticket_clusters")
    if clusters:
        expected[
            "top_clusters" if output == "blog_post" else "top_ticket_clusters"
        ] = clusters
    if output == "blog_post" and inputs.get("source_period"):
        expected["source_period"] = inputs["source_period"]
    return expected


def _export_rows(
    saved_draft_export: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    export = _as_mapping(saved_draft_export)
    rows = export.get("rows") or ()
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        return ()
    return tuple(row for row in rows if isinstance(row, Mapping))


def _exported_support_ticket_context(
    *,
    output: str,
    row: Mapping[str, Any],
) -> Mapping[str, Any]:
    if output == "landing_page":
        metadata = _as_mapping(row.get("metadata"))
        return _as_mapping(metadata.get("source_context"))
    if output == "blog_post":
        return _as_mapping(row.get("data_context"))
    return {}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _evaluate_generated_content_export(
    saved_draft_export: Mapping[str, Any],
    *,
    output: str,
) -> Mapping[str, Any]:
    from evaluate_support_ticket_generated_content import (  # noqa: PLC0415
        evaluate_support_ticket_generated_content,
    )

    return evaluate_support_ticket_generated_content(
        saved_draft_export,
        output=output,
    )


def _smoke_errors(
    *,
    output: str,
    configured_outputs: tuple[str, ...],
    result: Mapping[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    if output not in configured_outputs:
        errors.append(
            f"{output} service is not configured; check Atlas DB initialization "
            "and pipeline LLM/OpenRouter credentials"
        )
        return errors
    if result is None:
        return errors
    if result.get("status") != "completed":
        errors.append(f"execution status was {result.get('status')!r}, not 'completed'")
    step = _step_result(result, output)
    if step is None:
        errors.append(f"execution result did not include a {output} step")
        return errors
    if step.get("status") != "completed":
        errors.append(f"{output} step status was {step.get('status')!r}")
    step_payload = step.get("result") if isinstance(step.get("result"), Mapping) else {}
    if not step_payload.get("saved_ids"):
        errors.append(f"{output} generation did not return saved draft ids")
    history = step_payload.get("quality_repair_history") or ()
    if history and isinstance(history[-1], Mapping) and not history[-1].get("passed"):
        errors.append(f"{output} quality gate did not pass")
    return errors


def _account_slug(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return slug[:60].strip("-") or "acct"


def _default_blog_blueprint_payload() -> dict[str, Any]:
    return {
        "topic": DEFAULT_BLOG_TOPIC,
        "data_context": {
            "review_period": "last 90 days",
            "report_date": "2026-05-23",
            "total_reviews_analyzed": 186,
            "deep_enriched_count": 186,
            "category": "support tickets",
            "topic": "support tickets",
            "_known_vendors": [],
        },
        "sections": [
            {
                "id": "repeat-ticket-patterns",
                "heading": "What do repeat support tickets reveal?",
                "goal": (
                    "Explain why repeated setup, billing, and account-change "
                    "questions show missing customer-facing answers."
                ),
                "key_stats": {
                    "tickets_analyzed": 186,
                    "repeat_question_count": 78,
                    "repeat_question_rate": "42%",
                },
                "chart_ids": [],
                "data_summary": (
                    "Across 186 support tickets from the last 90 days, 78 "
                    "tickets repeated a question another customer had already "
                    "asked. Setup, billing, and account-change issues were the "
                    "largest repeat clusters."
                ),
            },
            {
                "id": "faq-gap-priorities",
                "heading": "Which FAQ gaps should a small team fix first?",
                "goal": (
                    "Rank the repeat-question clusters by volume and explain "
                    "why the highest-volume issues should become FAQ entries."
                ),
                "key_stats": {
                    "setup_questions": 31,
                    "billing_questions": 24,
                    "account_change_questions": 23,
                },
                "chart_ids": [],
                "data_summary": (
                    "Setup questions appeared 31 times, billing questions 24 "
                    "times, and account-change questions 23 times. The first "
                    "FAQ pass should start with those three clusters."
                ),
            },
            {
                "id": "publishable-answer-process",
                "heading": "How should old tickets become customer-ready answers?",
                "goal": (
                    "Show the operational path from CSV upload to clustered "
                    "questions, customer wording, draft answers, and review."
                ),
                "key_stats": {
                    "source_window_days": 90,
                    "draft_faq_entries": 12,
                    "review_steps": 3,
                },
                "chart_ids": [],
                "data_summary": (
                    "The 90-day ticket CSV produces 12 first-pass FAQ entries. "
                    "Each answer should preserve customer wording, summarize "
                    "the support team's answer, and stay in review until the "
                    "team approves it."
                ),
            },
        ],
        "available_charts": [],
        "related_posts": [],
        "grounded_vendors": [
            "Support Tickets",
            "Support Ticket FAQ",
            "Support Ticket FAQ Gaps",
            "FAQ Report",
            "Help Center",
        ],
        "required_vendors": [],
    }


def _support_ticket_blog_blueprint_payload(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from extracted_content_pipeline.blog_generation import (  # noqa: PLC0415
        support_ticket_descriptive_blog_contract,
    )
    from extracted_content_pipeline.support_ticket_input_package import (  # noqa: PLC0415
        build_support_ticket_input_package,
    )

    package = build_support_ticket_input_package(rows)
    inputs = dict(package.inputs)
    source_row_count = int(inputs.get("source_row_count") or 0)
    included_row_count = int(inputs.get("included_ticket_row_count") or 0)
    question_like_count = int(inputs.get("question_like_ticket_count") or 0)
    top_clusters = list(inputs.get("top_ticket_clusters") or [])
    cluster_summary = _cluster_summary(top_clusters)
    faq_questions = list(inputs.get("faq_questions") or [])
    draft_faq_entries = min(12, max(1, len(faq_questions) or included_row_count))
    source_period = str(
        inputs.get("source_period") or UPLOADED_SUPPORT_TICKETS_SOURCE_PERIOD
    )
    has_valid_date_window = bool(inputs.get("has_dated_window"))
    resolution_evidence_present = bool(
        inputs.get("support_ticket_resolution_evidence_present")
    )
    review_period = (
        SUPPORT_TICKET_LAST_90_DAYS_REVIEW_PERIOD
        if has_valid_date_window
        else UPLOADED_TICKETS_REVIEW_PERIOD
    )
    data_context = {
        "review_period": review_period,
        "source_row_count": source_row_count,
        "included_ticket_row_count": included_row_count,
        "question_like_ticket_count": question_like_count,
        "top_clusters": top_clusters,
        "faq_questions": faq_questions,
        "customer_wording_examples": list(inputs.get("customer_wording_examples") or []),
        "_known_vendors": [],
        "total_reviews_analyzed": included_row_count,
        "deep_enriched_count": included_row_count,
        "category": SUPPORT_TICKET_CATEGORY,
        "topic": SUPPORT_TICKET_BLOG_TOPIC,
        "source_period": source_period,
        "source": SUPPORT_TICKET_SOURCE,
        "has_dated_window": has_valid_date_window,
        "has_measured_outcomes": bool(inputs.get("has_measured_outcomes")),
        "measured_outcome_count": int(inputs.get("measured_outcome_count") or 0),
        "measured_outcome_examples": list(inputs.get("measured_outcome_examples") or []),
        "support_ticket_resolution_evidence_present": resolution_evidence_present,
        "support_ticket_resolution_evidence_count": int(
            inputs.get("support_ticket_resolution_evidence_count") or 0
        ),
        "support_ticket_resolution_examples": list(
            inputs.get("support_ticket_resolution_examples") or []
        ),
    }
    data_context.update(support_ticket_descriptive_blog_contract(data_context))
    if has_valid_date_window:
        data_context["report_date"] = "2026-05-23"
    return {
        "topic": SUPPORT_TICKET_BLOG_TOPIC,
        "data_context": data_context,
        "sections": [
            {
                "id": "repeat-ticket-patterns",
                "heading": "What do repeat support tickets reveal?",
                "goal": (
                    "Explain what the uploaded support-ticket rows show about "
                    "customer questions and missing customer-facing answers."
                ),
                "key_stats": {
                    "support_ticket_rows": source_row_count,
                    "included_ticket_rows": included_row_count,
                    "question_like_rows": question_like_count,
                    "cluster_count": len(top_clusters),
                },
                "chart_ids": [],
                "data_summary": (
                    f"The uploaded CSV contains {source_row_count} support-ticket "
                    f"rows; {included_row_count} rows were included for generation. "
                    f"{question_like_count} rows include direct customer questions. "
                    f"Top observed clusters: {cluster_summary}."
                ),
            },
            {
                "id": "faq-gap-priorities",
                "heading": "Which FAQ gaps should a small team fix first?",
                "goal": (
                    "Rank the observed ticket clusters by volume and explain "
                    "why the highest-volume issues should become FAQ entries."
                ),
                "key_stats": _cluster_key_stats(top_clusters),
                "chart_ids": [],
                "data_summary": (
                    "Prioritize FAQ work by observed ticket volume. In this "
                    f"CSV, the highest-volume clusters are: {cluster_summary}. "
                    "Those clusters should be reviewed before lower-volume or "
                    "one-off questions."
                ),
            },
            {
                "id": "publishable-answer-process",
                "heading": "How should old tickets become review-ready FAQ shells?",
                "goal": (
                    "Show the operational path from CSV upload to clustered "
                    "questions, customer wording, review-needed draft answer "
                    "shells, and support-team verification."
                ),
                "key_stats": {
                    "source_rows": source_row_count,
                    "included_rows": included_row_count,
                    "draft_faq_entries": draft_faq_entries,
                    "review_steps": 3,
                    **({"source_window_days": 90} if has_valid_date_window else {}),
                },
                "chart_ids": [],
                "data_summary": (
                    f"The uploaded ticket CSV can produce up to {draft_faq_entries} "
                    "first-pass FAQ entries from observed customer wording. "
                    + (
                        "Each answer should preserve customer language, use the "
                        "support team's verified resolution evidence, and stay in "
                        "review until the team approves it."
                        if resolution_evidence_present
                        else "Without resolution evidence, each answer should preserve "
                        "customer language, stay as a review-needed shell, and wait "
                        "for the support team to add the verified resolution."
                    )
                ),
            },
        ],
        "available_charts": [],
        "related_posts": [],
        "grounded_vendors": [
            "Support Tickets",
            "Support Ticket FAQ",
            "Support Ticket FAQ Gaps",
            "FAQ Report",
            "Help Center",
        ],
        "required_vendors": [],
    }


def _cluster_summary(clusters: Sequence[Mapping[str, Any]]) -> str:
    if not clusters:
        return "no categorized ticket clusters"
    return ", ".join(
        f"{cluster.get('label')} ({int(cluster.get('count') or 0)})"
        for cluster in clusters
    )


def _cluster_key_stats(clusters: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    stats: dict[str, int] = {}
    for index, cluster in enumerate(clusters[:5], start=1):
        stats[f"cluster_{index}_count"] = int(cluster.get("count") or 0)
    return stats or {"cluster_count": 0}


def _load_single_blog_blueprint_from_file(
    path: Path,
    *,
    target_mode: str,
    topic_type: str = DEFAULT_BLOG_TOPIC_TYPE,
) -> tuple[Any, list[dict[str, Any]]]:
    from extracted_content_pipeline.blog_blueprint_ingest import (  # noqa: PLC0415
        load_blog_blueprints_from_file,
    )

    try:
        loaded = load_blog_blueprints_from_file(
            path,
            target_mode=target_mode,
            topic_type=topic_type,
        )
    except OSError as exc:
        raise ValueError(f"unable to read --blog-blueprint-json: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid --blog-blueprint-json: {exc}") from exc
    warnings = loaded.warning_dicts()
    if len(loaded.blueprints) != 1:
        warning_text = f"; warnings={json.dumps(warnings, sort_keys=True)}" if warnings else ""
        raise ValueError(
            "--blog-blueprint-json must load exactly one blueprint; "
            f"loaded {len(loaded.blueprints)}{warning_text}"
        )
    blueprint = loaded.blueprints[0]
    if str(blueprint.target_mode or "").strip() != target_mode:
        raise ValueError(
            "--blog-blueprint-json target_mode "
            f"{blueprint.target_mode!r} does not match --target-mode {target_mode!r}"
        )
    return blueprint, warnings


def _align_blog_payload_to_seed(
    payload: dict[str, Any],
    seeded_blog_blueprint: Mapping[str, Any],
) -> None:
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        return
    topic_type = str(seeded_blog_blueprint.get("topic_type") or "").strip()
    if topic_type:
        filters = inputs.get("filters")
        if not isinstance(filters, dict):
            filters = {}
            inputs["filters"] = filters
        filters["topic_type"] = topic_type
    slug = str(seeded_blog_blueprint.get("slug") or "").strip()
    if slug:
        filters = inputs.get("filters")
        if not isinstance(filters, dict):
            filters = {}
            inputs["filters"] = filters
        filters["slug"] = slug
    seeded_topic = str(seeded_blog_blueprint.get("topic") or "").strip()
    if seeded_topic and inputs.get("topic") in {
        DEFAULT_BLOG_TOPIC,
        SUPPORT_TICKET_BLOG_TOPIC,
    }:
        inputs["topic"] = seeded_topic


async def _seed_default_blog_blueprint(
    args: argparse.Namespace,
    scope: Any,
) -> Mapping[str, Any]:
    from atlas_brain.storage.database import get_db_pool  # noqa: PLC0415
    from extracted_content_pipeline.blog_blueprint_postgres import (  # noqa: PLC0415
        PostgresBlogBlueprintRepository,
    )
    from extracted_content_pipeline.blog_ports import BlogBlueprint  # noqa: PLC0415

    account_slug = _account_slug(args.account_id)
    support_ticket_mode = _uses_support_ticket_csv(args)
    topic_type = (
        SUPPORT_TICKET_BLOG_TOPIC_TYPE
        if support_ticket_mode
        else DEFAULT_BLOG_TOPIC_TYPE
    )
    slug_prefix = (
        "content-ops-support-ticket-live-smoke"
        if support_ticket_mode
        else "content-ops-blog-live-smoke"
    )
    slug = f"{slug_prefix}-{account_slug}"
    target_mode = str(args.target_mode or "vendor_retention").strip() or "vendor_retention"
    custom_path = getattr(args, "blog_blueprint_json", None)
    custom_warnings: list[dict[str, Any]] = []
    custom_source = ""
    if custom_path:
        blueprint, custom_warnings = _load_single_blog_blueprint_from_file(
            Path(custom_path),
            target_mode=target_mode,
            topic_type=topic_type,
        )
        custom_source = str(custom_path)
    else:
        blueprint = BlogBlueprint(
            target_mode=target_mode,
            topic_type=topic_type,
            slug=slug,
            suggested_title=(
                SUPPORT_TICKET_BLOG_TOPIC
                if support_ticket_mode
                else "Support Tickets: FAQ Gaps From 90 Days of Tickets"
            ),
            payload=(
                _support_ticket_blog_blueprint_payload(
                    _support_ticket_rows_from_args(args)
                )
                if support_ticket_mode
                else _default_blog_blueprint_payload()
            ),
        )
    saved_ids = await PostgresBlogBlueprintRepository(
        pool=get_db_pool()
    ).save_blueprints((blueprint,), scope=scope)
    seeded = {
        "saved_ids": list(saved_ids),
        "slug": blueprint.slug,
        "target_mode": blueprint.target_mode,
        "topic_type": blueprint.topic_type,
        "topic": blueprint.suggested_title,
    }
    if custom_source:
        seeded["source"] = custom_source
    if custom_warnings:
        seeded["warnings"] = custom_warnings
    return seeded


async def run_content_ops_live_generation_smoke(
    args: argparse.Namespace,
    *,
    init_database_fn: AsyncCallable | None = None,
    close_database_fn: AsyncCallable | None = None,
    services_factory: ServicesFactory | None = None,
    executor: Executor | None = None,
    tenant_scope_cls: Any = None,
    blog_blueprint_seed_fn: BlogBlueprintSeeder | None = None,
    draft_export_fn: DraftExporter | None = None,
    generated_content_evaluator: GeneratedContentEvaluator | None = None,
) -> tuple[int, dict[str, Any]]:
    _load_dotenv_files(list(args.env_file or []))
    if (
        init_database_fn is None
        or close_database_fn is None
        or services_factory is None
        or executor is None
        or tenant_scope_cls is None
    ):
        (
            init_database_fn,
            close_database_fn,
            services_factory,
            executor,
            tenant_scope_cls,
        ) = _resolve_runtime_dependencies()

    payload = _payload_from_args(args)
    output = str(payload["outputs"][0])
    configured_outputs: tuple[str, ...] = ()
    execution_result: dict[str, Any] | None = None
    seeded_blog_blueprint: Mapping[str, Any] | None = None
    saved_draft_export: Mapping[str, Any] | None = None
    generated_content_evaluation: Mapping[str, Any] | None = None
    errors: list[str] = _generated_content_evaluation_preflight_errors(args)
    if errors:
        return 1, {
            "ok": False,
            "errors": errors,
            "configured_outputs": [],
            "seeded_blog_blueprint": {},
            "payload": payload,
            "execution": None,
            "saved_draft_export": None,
            "generated_content_evaluation": None,
        }
    try:
        await init_database_fn()
        services = services_factory()
        configured_outputs = tuple(str(item) for item in services.configured_outputs())
        errors.extend(
            _smoke_errors(
                output=output,
                configured_outputs=configured_outputs,
                result=None,
            )
        )
        if not errors:
            scope = tenant_scope_cls(
                account_id=str(args.account_id or "").strip(),
                user_id=str(args.user_id or "").strip(),
            )
            if _uses_support_ticket_csv(args):
                payload = await _payload_with_support_ticket_provider(
                    payload,
                    scope=scope,
                )
            if output == "blog_post":
                seeder = blog_blueprint_seed_fn or _seed_default_blog_blueprint
                seeded_blog_blueprint = await seeder(args, scope)
                _align_blog_payload_to_seed(payload, seeded_blog_blueprint)
            execution_result = await executor(payload, services=services, scope=scope)
            errors.extend(
                _smoke_errors(
                    output=output,
                    configured_outputs=configured_outputs,
                    result=execution_result,
                )
            )
            if not errors and getattr(args, "export_saved_draft", None):
                saved_ids = _saved_ids_for_output(execution_result, output)
                if not saved_ids:
                    errors.append(
                        "--export-saved-draft requested, but generation returned no saved_ids"
                    )
                else:
                    exporter = draft_export_fn or _export_saved_drafts
                    saved_draft_export = await exporter(output, saved_ids, scope)
                    if _uses_support_ticket_csv(args):
                        errors.extend(
                            _support_ticket_export_context_errors(
                                output=output,
                                payload=payload,
                                saved_draft_export=saved_draft_export,
                            )
                        )
                    if _evaluates_generated_content(args) and not errors:
                        evaluator = (
                            generated_content_evaluator
                            or _evaluate_generated_content_export
                        )
                        generated_content_evaluation = evaluator(
                            saved_draft_export,
                            output=output,
                        )
                        evaluation_errors = [
                            str(item)
                            for item in generated_content_evaluation.get("errors", ())
                        ]
                        if not generated_content_evaluation.get("ok") and not evaluation_errors:
                            evaluation_errors = ["unknown generated-content failure"]
                        errors.extend(
                            f"generated content evaluation failed: {error}"
                            for error in evaluation_errors
                        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            await close_database_fn()
        except Exception as exc:
            errors.append(f"close_database failed: {type(exc).__name__}: {exc}")

    result = {
        "ok": not errors,
        "errors": errors,
        "configured_outputs": list(configured_outputs),
        "seeded_blog_blueprint": dict(seeded_blog_blueprint or {}),
        "payload": payload,
        "execution": execution_result,
        "saved_draft_export": (
            dict(saved_draft_export) if isinstance(saved_draft_export, Mapping) else None
        ),
        "generated_content_evaluation": (
            dict(generated_content_evaluation)
            if isinstance(generated_content_evaluation, Mapping)
            else None
        ),
    }
    return (0 if result["ok"] else 1), result


def _print_result(result: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    status = "passed" if result.get("ok") else "failed"
    print(f"Content Ops live generation smoke {status}")
    print(f"configured_outputs: {', '.join(result.get('configured_outputs') or [])}")
    for error in result.get("errors") or ():
        print(f"error: {error}")
    execution = result.get("execution")
    if isinstance(execution, Mapping):
        payload = result.get("payload") if isinstance(result.get("payload"), Mapping) else {}
        outputs = payload.get("outputs") if isinstance(payload, Mapping) else ()
        output = str(outputs[0]) if outputs else "landing_page"
        step = _step_result(execution, output)
        step_payload = step.get("result") if isinstance(step, Mapping) else {}
        if isinstance(step_payload, Mapping):
            saved_ids = step_payload.get("saved_ids") or []
            if saved_ids:
                print(f"saved_ids: {', '.join(str(item) for item in saved_ids)}")


async def _amain(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    code, result = await run_content_ops_live_generation_smoke(args)
    if args.output_result:
        args.output_result.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.export_saved_draft and result.get("saved_draft_export") is not None:
        args.export_saved_draft.write_text(
            json.dumps(result["saved_draft_export"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    _print_result(result, as_json=bool(args.json))
    return code


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_amain(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
