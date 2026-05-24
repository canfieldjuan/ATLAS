#!/usr/bin/env python3
"""Smoke-test live Content Ops generation through host wiring."""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import date, datetime
import json
from pathlib import Path
import re
import sys
from typing import Any, Awaitable, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional host dependency
    load_dotenv = None


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
SUPPORT_TICKET_BLOG_TOPIC = "Support-ticket questions customers keep asking"
SUPPORT_TICKET_BLOG_TOPIC_TYPE = "content_ops_support_ticket_faq"

AsyncCallable = Callable[[], Awaitable[None]]
ServicesFactory = Callable[[], Any]
Executor = Callable[..., Awaitable[dict[str, Any]]]
BlogBlueprintSeeder = Callable[[argparse.Namespace, Any], Awaitable[Mapping[str, Any]]]
DraftExporter = Callable[[str, Sequence[str], Any], Awaitable[Mapping[str, Any]]]


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
    source_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    row_count = len(source_rows)
    question_like_count = sum(1 for row in source_rows if "?" in _ticket_row_text(row))
    top_clusters = _top_ticket_clusters(source_rows)
    cluster_summary = _cluster_summary(top_clusters)
    draft_faq_entries = min(12, max(1, row_count))
    has_valid_date_window = _support_ticket_rows_have_dates(source_rows)
    review_period = "last 90 days" if has_valid_date_window else "uploaded tickets"
    source_period = (
        "Last 90 days of support tickets"
        if has_valid_date_window
        else "Uploaded support tickets"
    )
    return {
        "topic": SUPPORT_TICKET_BLOG_TOPIC,
        "data_context": {
            "review_period": review_period,
            "source_row_count": row_count,
            "question_like_ticket_count": question_like_count,
            "top_clusters": top_clusters,
            "_known_vendors": [],
            "report_date": "2026-05-23",
            "total_reviews_analyzed": row_count,
            "deep_enriched_count": row_count,
            "category": "support tickets",
            "topic": SUPPORT_TICKET_BLOG_TOPIC,
            "source_period": source_period,
            "source": "support_ticket_provider",
        },
        "sections": [
            {
                "id": "repeat-ticket-patterns",
                "heading": "What do repeat support tickets reveal?",
                "goal": (
                    "Explain what the uploaded support-ticket rows show about "
                    "customer questions and missing customer-facing answers."
                ),
                "key_stats": {
                    "support_ticket_rows": row_count,
                    "question_like_rows": question_like_count,
                    "cluster_count": len(top_clusters),
                },
                "chart_ids": [],
                "data_summary": (
                    f"The uploaded CSV contains {row_count} support-ticket "
                    f"rows. {question_like_count} rows include direct customer "
                    f"questions. Top observed clusters: {cluster_summary}."
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
                "heading": "How should old tickets become customer-ready answers?",
                "goal": (
                    "Show the operational path from CSV upload to clustered "
                    "questions, customer wording, draft answers, and review."
                ),
                "key_stats": {
                    "source_rows": row_count,
                    "draft_faq_entries": draft_faq_entries,
                    "review_steps": 3,
                    **({"source_window_days": 90} if has_valid_date_window else {}),
                },
                "chart_ids": [],
                "data_summary": (
                    f"The uploaded ticket CSV can produce up to {draft_faq_entries} "
                    "first-pass FAQ entries from observed customer wording. "
                    "Each answer should preserve customer language, summarize "
                    "the support team's resolution, and stay in review until "
                    "the team approves it."
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


def _ticket_row_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "").strip()
        for key in (
            "Description",
            "description",
            "Message",
            "message",
            "Text",
            "text",
            "Subject",
            "subject",
            "Title",
            "title",
        )
        if str(row.get(key) or "").strip()
    )


def _support_ticket_rows_have_dates(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(_ticket_row_date(row) is not None for row in rows)


def _ticket_row_date(row: Mapping[str, Any]) -> date | None:
    for key in (
        "Created At",
        "created_at",
        "Submitted At",
        "submitted_at",
        "Date",
        "date",
    ):
        value = row.get(key)
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        text = str(value or "").strip()
        if not text:
            continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            pass
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            continue
    return None


def _ticket_cluster_label(row: Mapping[str, Any]) -> str:
    for key in ("Pain Category", "pain_category", "Category", "category", "intent"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return "uncategorized"


def _top_ticket_clusters(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        label = _ticket_cluster_label(row)
        counts[label] = counts.get(label, 0) + 1
    return [
        {"label": label, "count": count}
        for label, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )[:5]
    ]


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
    errors: list[str] = []
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
