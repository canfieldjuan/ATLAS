#!/usr/bin/env python3
"""Build a grounded FAQ Markdown file from ticket-like source rows."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from extracted_content_pipeline.campaign_source_adapters import (  # noqa: E402
    load_source_campaign_opportunities_from_file,
    parse_default_fields_or_exit,
)
from extracted_content_pipeline.ticket_faq_markdown import (  # noqa: E402
    DEFAULT_INTENT_RULES,
    DEFAULT_TITLE,
    build_ticket_faq_markdown,
    is_zero_result_search_row,
    source_row_weight,
    weighted_source_volume_by_group,
)
from content_ops_faq_cli_rules import (  # noqa: E402
    DOCUMENTATION_TERM_FORMATS,
    load_rule_files,
    parse_documentation_terms,
    parse_intent_rules,
    parse_vocabulary_gap_rules,
)

_SOURCE_CHANNELS = {
    # Mirrors the generator's accepted FAQ source types at channel granularity;
    # unknown future types intentionally bucket to "other" until classified.
    "case": "support_tickets",
    "support_ticket": "support_tickets",
    "ticket": "support_tickets",
    "chat": "chats",
    "chat_transcript": "chats",
    "conversation": "chats",
    "transcript": "chats",
    "search_log": "search_logs",
    "search_query": "search_logs",
    "objection": "sales_inputs",
    "sales_call": "sales_inputs",
    "sales_objection": "sales_inputs",
    "meeting": "sales_inputs",
    "complaint": "complaints",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ticket-like source rows into a grounded Markdown FAQ."
        )
    )
    parser.add_argument("path", type=Path, help="Source JSON, JSONL, or CSV file.")
    parser.add_argument(
        "--source-format",
        choices=("auto", "json", "jsonl", "csv"),
        default="auto",
        help="Source file format. Defaults to suffix-based detection.",
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="Markdown H1 title.",
    )
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--max-evidence-per-item", type=int, default=3)
    parser.add_argument(
        "--window-days",
        type=int,
        help="Keep only rows dated within this many days of --as-of-date or today.",
    )
    parser.add_argument(
        "--as-of-date",
        help="YYYY-MM-DD date used with --window-days for reproducible windows.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=1200,
        help="Maximum source text characters copied into each evidence row.",
    )
    parser.add_argument(
        "--default-field",
        action="append",
        default=[],
        help="Fallback metadata applied to every source row. Repeat as key=value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write FAQ Markdown to this path instead of stdout.",
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        help="Write compact JSON run diagnostics to this path.",
    )
    parser.add_argument(
        "--support-contact",
        help="Phone, email, or URL shown when the FAQ tells users to contact support.",
    )
    parser.add_argument(
        "--documentation-term",
        action="append",
        default=[],
        help=(
            "Existing documentation term or heading used for vocabulary-gap "
            "suggestions. Repeat to provide multiple terms."
        ),
    )
    parser.add_argument(
        "--documentation-term-file",
        action="append",
        default=[],
        type=Path,
        help=(
            "UTF-8 text file with one documentation term or heading per line. "
            "Repeat to load multiple files."
        ),
    )
    parser.add_argument(
        "--documentation-term-format",
        choices=DOCUMENTATION_TERM_FORMATS,
        default="auto",
        help=(
            "Format for documentation-term files. Defaults to suffix-based "
            "auto detection."
        ),
    )
    parser.add_argument(
        "--vocabulary-gap-rule",
        action="append",
        default=[],
        help=(
            "Comma-separated customer/documentation aliases used for "
            "vocabulary-gap suggestions. Repeat to provide multiple rules."
        ),
    )
    parser.add_argument(
        "--rule-file",
        action="append",
        default=[],
        type=Path,
        help=(
            "JSON file with intent_rules and/or vocabulary_gap_rules. "
            "Repeat to load multiple files."
        ),
    )
    parser.add_argument(
        "--intent-rule",
        action="append",
        default=[],
        help=(
            "Custom FAQ intent rule as topic=keyword,keyword. Repeat to "
            "provide multiple rules."
        ),
    )
    parser.add_argument(
        "--require-output-checks",
        action="store_true",
        help="Fail when generated FAQ output checks are not all true.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.max_items < 1:
        raise SystemExit("--max-items must be positive")
    if args.max_evidence_per_item < 1:
        raise SystemExit("--max-evidence-per-item must be positive")
    if args.max_text_chars < 1:
        raise SystemExit("--max-text-chars must be positive")
    if args.window_days is not None and args.window_days < 1:
        raise SystemExit("--window-days must be positive")
    if args.as_of_date and args.window_days is None:
        raise SystemExit("--as-of-date requires --window-days")
    if args.as_of_date:
        try:
            date.fromisoformat(args.as_of_date)
        except ValueError:
            raise SystemExit("--as-of-date must use YYYY-MM-DD format") from None
    documentation_terms = parse_documentation_terms(
        args.documentation_term,
        args.documentation_term_file,
        args.documentation_term_format,
    )
    args.documentation_terms = documentation_terms
    file_rules = load_rule_files(args.rule_file)
    vocabulary_gap_rules = (
        *parse_vocabulary_gap_rules(args.vocabulary_gap_rule),
        *file_rules["vocabulary_gap_rules"],
    )
    args.vocabulary_gap_rules = vocabulary_gap_rules
    custom_intent_rules = (
        *parse_intent_rules(args.intent_rule),
        *file_rules["intent_rules"],
    )
    args.custom_intent_rules = custom_intent_rules
    intent_rules = (*custom_intent_rules, *DEFAULT_INTENT_RULES)

    loaded = load_source_campaign_opportunities_from_file(
        args.path,
        file_format=args.source_format,
        max_text_chars=args.max_text_chars,
        default_fields=parse_default_fields_or_exit(args.default_field),
    )
    result = build_ticket_faq_markdown(
        loaded.opportunities,
        title=args.title,
        max_items=args.max_items,
        max_evidence_per_item=args.max_evidence_per_item,
        window_days=args.window_days,
        as_of_date=args.as_of_date,
        support_contact=args.support_contact,
        intent_rules=intent_rules,
        documentation_terms=documentation_terms,
        vocabulary_gap_rules=vocabulary_gap_rules,
    )
    failed_checks = _failed_output_checks(result.output_checks)
    if args.result_output:
        args.result_output.write_text(
            json.dumps(
                _result_payload(
                    args,
                    result,
                    loaded.warning_dicts(),
                    failed_checks,
                    loaded.opportunities,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.require_output_checks and failed_checks:
        raise SystemExit(f"FAQ output checks failed: {', '.join(failed_checks)}")
    if args.output:
        args.output.write_text(result.markdown, encoding="utf-8")
    else:
        print(result.markdown, end="")
    return 0


def _failed_output_checks(output_checks: dict[str, bool]) -> list[str]:
    return [name for name, passed in sorted(output_checks.items()) if passed is not True]


def _result_payload(
    args: argparse.Namespace,
    result: Any,
    load_warnings: list[dict[str, Any]],
    failed_checks: list[str],
    opportunities: list[dict[str, Any]],
) -> dict[str, Any]:
    items = [dict(item) for item in result.items]
    ticket_counts = [int(item.get("ticket_count") or 0) for item in items]
    question_source_counts = _count_by(items, "question_source")
    rendered_ticket_source_count = _rendered_ticket_source_count(items)
    warnings = load_warnings + [dict(warning) for warning in result.warnings]
    source_mix = _source_mix_diagnostics(opportunities)
    item_summaries = [
        _item_summary(index, item) for index, item in enumerate(items, start=1)
    ]
    term_mappings = _term_mapping_summaries(items)
    return {
        "status": "failed_output_checks" if failed_checks else "ok",
        "input": {
            "path": str(args.path),
            "source_format": args.source_format,
        },
        "output": {
            "markdown_path": str(args.output) if args.output else None,
            "result_path": str(args.result_output) if args.result_output else None,
        },
        "config": {
            "title": args.title,
            "max_items": args.max_items,
            "max_evidence_per_item": args.max_evidence_per_item,
            "max_text_chars": args.max_text_chars,
            "window_days": args.window_days,
            "as_of_date": args.as_of_date,
            "require_output_checks": bool(args.require_output_checks),
            "support_contact": args.support_contact,
            "rule_files": [str(path) for path in args.rule_file],
            "documentation_term_files": [
                str(path) for path in args.documentation_term_file
            ],
            "documentation_term_format": args.documentation_term_format,
            "documentation_terms": list(args.documentation_terms),
            "vocabulary_gap_rules": [
                list(rule) for rule in args.vocabulary_gap_rules
            ],
            "custom_intent_rules": [
                {"topic": topic, "keywords": list(keywords)}
                for topic, keywords in args.custom_intent_rules
            ],
        },
        "source_count": result.source_count,
        "ticket_source_count": result.ticket_source_count,
        "generated": len(items),
        "output_checks": dict(result.output_checks),
        "failed_output_checks": list(failed_checks),
        "diagnostics": {
            "output_check_details": _output_check_details(result, failed_checks, rendered_ticket_source_count),
            "question_source_counts": question_source_counts,
            "run_summary": _run_summary(
                result=result,
                failed_checks=failed_checks,
                source_mix=source_mix,
                item_summaries=item_summaries,
                term_mappings=term_mappings,
                warnings=warnings,
            ),
            "source_mix": source_mix,
            "ticket_counts": ticket_counts,
            "rendered_ticket_source_count": rendered_ticket_source_count,
            "unrepresented_ticket_sources": max(result.ticket_source_count - rendered_ticket_source_count, 0),
            "term_mapping_count": len(term_mappings),
            "term_mappings": term_mappings,
            "warning_count": len(warnings),
            "warnings": warnings[:50],
            "warnings_truncated": len(warnings) > 50,
            "items": item_summaries,
        },
    }


def _run_summary(
    *,
    result: Any,
    failed_checks: list[str],
    source_mix: dict[str, Any],
    item_summaries: list[dict[str, Any]],
    term_mappings: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    output_checks = dict(result.output_checks)
    return {
        "status": "failed_output_checks" if failed_checks else "ok",
        "source_count": result.source_count,
        "ticket_source_count": result.ticket_source_count,
        "generated": len(item_summaries),
        "weighted_source_volume": int(source_mix.get("weighted_source_volume") or 0),
        "source_channel_counts": _clean_count_mapping(
            source_mix.get("source_channel_counts")
        ),
        "zero_result_search_source_count": int(
            source_mix.get("zero_result_search_source_count") or 0
        ),
        "output_checks": {
            "passed": sum(1 for passed in output_checks.values() if passed is True),
            "failed": len(failed_checks),
            "total": len(output_checks),
            "failed_checks": list(failed_checks),
        },
        "vocabulary_gaps": _vocabulary_gap_summary(term_mappings),
        "item_score_distribution": _item_score_distribution(item_summaries),
        "warning_count": len(warnings),
    }


def _vocabulary_gap_summary(
    term_mappings: list[dict[str, Any]],
) -> dict[str, Any]:
    topics = {
        str(mapping.get("topic"))
        for mapping in term_mappings
        if mapping.get("topic")
    }
    top_customer_terms: list[str] = []
    seen_terms: set[str] = set()
    zero_result_mapping_count = 0
    max_opportunity_score = 0
    for mapping in term_mappings:
        if _integer_or_zero(mapping.get("zero_result_source_count")) > 0:
            zero_result_mapping_count += 1
        max_opportunity_score = max(
            max_opportunity_score,
            _integer_or_zero(mapping.get("opportunity_score")),
        )
        term = str(mapping.get("customer_term") or "").strip()
        key = term.lower()
        if term and key not in seen_terms and len(top_customer_terms) < 3:
            seen_terms.add(key)
            top_customer_terms.append(term)
    return {
        "term_mapping_count": len(term_mappings),
        "mapped_topic_count": len(topics),
        "zero_result_mapping_count": zero_result_mapping_count,
        "max_opportunity_score": max_opportunity_score,
        "top_customer_terms": top_customer_terms,
    }


def _item_score_distribution(item_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [
        _integer_or_zero(item.get("opportunity_score")) for item in item_summaries
    ]
    bands = {
        "zero": 0,
        "low_1_to_4": 0,
        "medium_5_to_9": 0,
        "high_10_plus": 0,
    }
    for score in scores:
        if score <= 0:
            bands["zero"] += 1
        elif score <= 4:
            bands["low_1_to_4"] += 1
        elif score <= 9:
            bands["medium_5_to_9"] += 1
        else:
            bands["high_10_plus"] += 1
    if not scores:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "average": 0.0,
            "bands": bands,
        }
    return {
        "count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "average": round(sum(scores) / len(scores), 2),
        "bands": bands,
    }


def _source_mix_diagnostics(opportunities: list[dict[str, Any]]) -> dict[str, Any]:
    source_type_counts: dict[str, int] = {}
    source_channel_counts: dict[str, int] = {}
    zero_result_search_sources: set[str] = set()
    for index, opportunity in enumerate(opportunities, start=1):
        source_type = _diagnostic_source_type(opportunity)
        source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
        channel = _SOURCE_CHANNELS.get(source_type, "other")
        source_channel_counts[channel] = source_channel_counts.get(channel, 0) + 1
        source_key = _diagnostic_source_key(opportunity, index)
        if _is_zero_result_search_source(opportunity):
            zero_result_search_sources.add(source_key)
    source_weights = weighted_source_volume_by_group(
        opportunities,
        group_key=lambda _opportunity: "all",
        source_key=_diagnostic_source_key,
        row_weight=_diagnostic_source_weight,
    )
    return {
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "source_channel_counts": dict(sorted(source_channel_counts.items())),
        "weighted_source_volume": source_weights.get("all", 0),
        "weighted_source_volume_by_type": weighted_source_volume_by_group(
            opportunities,
            group_key=_diagnostic_source_type,
            source_key=_diagnostic_source_key,
            row_weight=_diagnostic_source_weight,
        ),
        "weighted_source_volume_by_channel": weighted_source_volume_by_group(
            opportunities,
            group_key=lambda opportunity: _SOURCE_CHANNELS.get(
                _diagnostic_source_type(opportunity),
                "other",
            ),
            source_key=_diagnostic_source_key,
            row_weight=_diagnostic_source_weight,
        ),
        "zero_result_search_source_count": len(zero_result_search_sources),
    }


def _diagnostic_source_type(opportunity: dict[str, Any]) -> str:
    source_type = _clean_diagnostic_text(opportunity.get("source_type")).lower()
    if source_type:
        return source_type
    for evidence in _diagnostic_evidence_rows(opportunity):
        source_type = _clean_diagnostic_text(evidence.get("source_type")).lower()
        if source_type:
            return source_type
    return "unknown"


def _diagnostic_source_key(opportunity: dict[str, Any], index: int) -> str:
    for key in ("source_id", "target_id", "id"):
        value = _clean_diagnostic_text(opportunity.get(key))
        if value:
            return value
    for evidence in _diagnostic_evidence_rows(opportunity):
        value = _clean_diagnostic_text(evidence.get("source_id"))
        if value:
            return value
    return f"row:{index}"


def _diagnostic_evidence_rows(opportunity: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    evidence = opportunity.get("evidence")
    if isinstance(evidence, list):
        return tuple(row for row in evidence if isinstance(row, dict))
    return ()


def _diagnostic_source_weight(opportunity: dict[str, Any]) -> int:
    return source_row_weight(opportunity, *_diagnostic_evidence_rows(opportunity))


def _is_zero_result_search_source(
    opportunity: dict[str, Any],
) -> bool:
    rows = (opportunity, *_diagnostic_evidence_rows(opportunity))
    return any(is_zero_result_search_row(row) for row in rows)


def _clean_diagnostic_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _output_check_details(
    result: Any,
    failed_checks: list[str],
    rendered_ticket_source_count: int,
) -> list[dict[str, Any]]:
    details = []
    for name, passed in sorted(result.output_checks.items()):
        detail: dict[str, Any] = {"check": name, "passed": bool(passed)}
        if name in failed_checks:
            detail["why"] = _output_check_hint(name, result, rendered_ticket_source_count)
        details.append(detail)
    return details


def _output_check_hint(name: str, result: Any, rendered_ticket_source_count: int) -> str:
    if name == "condensed":
        if rendered_ticket_source_count != result.ticket_source_count:
            return (
                "Some ticket sources were not represented in generated FAQ items. "
                f"ticket_source_count={result.ticket_source_count}, "
                f"rendered_ticket_source_count={rendered_ticket_source_count}."
            )
        return (
            "The FAQ produced one item per ticket source, so the output was not condensed. "
            f"ticket_source_count={result.ticket_source_count}, generated={len(result.items)}."
        )
    if name == "uses_user_vocabulary":
        return "One or more FAQ questions did not come from customer wording or source policy."
    if name == "has_action_items":
        return "One or more FAQ items did not include actionable next steps."
    return "Output check failed."


def _item_summary(index: int, item: dict[str, Any]) -> dict[str, Any]:
    source_ids = item.get("source_ids") or ()
    steps = item.get("steps") or ()
    term_mappings = item.get("term_mappings") or ()
    source_type_counts = _clean_count_mapping(item.get("source_type_counts"))
    weighted_source_volume_by_type = _clean_count_mapping(
        item.get("weighted_source_volume_by_type")
    )
    return {
        "rank": index,
        "topic": item.get("topic"),
        "question": item.get("question"),
        "question_source": item.get("question_source"),
        "frequency": item.get("frequency"),
        "weighted_frequency": item.get("weighted_frequency"),
        "failure_risk_score": item.get("failure_risk_score"),
        "failure_risk_signals": list(item.get("failure_risk_signals") or ()),
        "opportunity_score": item.get("opportunity_score"),
        "ticket_count": item.get("ticket_count"),
        "evidence_count": item.get("evidence_count"),
        "source_id_count": len(source_ids),
        "first_source_id": source_ids[0] if source_ids else None,
        "source_type_counts": source_type_counts,
        "source_channel_counts": _source_channel_counts_from_types(source_type_counts),
        "weighted_source_volume_by_type": weighted_source_volume_by_type,
        "weighted_source_volume_by_channel": _source_channel_counts_from_types(
            weighted_source_volume_by_type
        ),
        "step_count": len(steps),
        "term_mapping_count": len(term_mappings),
    }


def _clean_count_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, count in value.items():
        try:
            amount = int(count)
        except (TypeError, ValueError):
            continue
        out[str(key)] = amount
    return dict(sorted(out.items()))


def _integer_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _source_channel_counts_from_types(source_type_counts: dict[str, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source_type, count in source_type_counts.items():
        channel = _SOURCE_CHANNELS.get(source_type, "other")
        counts[channel] = counts.get(channel, 0) + count
    return dict(sorted(counts.items()))


def _term_mapping_count(items: list[dict[str, Any]]) -> int:
    return sum(len(item.get("term_mappings") or ()) for item in items)


def _term_mapping_summaries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        for mapping in item.get("term_mappings") or ():
            if not isinstance(mapping, dict):
                continue
            out.append({
                "rank": index,
                "topic": item.get("topic"),
                "customer_term": mapping.get("customer_term"),
                "documentation_term": mapping.get("documentation_term"),
                "source_id_count": mapping.get("source_id_count"),
                "zero_result_source_count": mapping.get("zero_result_source_count"),
                "failure_risk_score": mapping.get("failure_risk_score"),
                "failure_risk_signals": list(mapping.get("failure_risk_signals") or ()),
                "opportunity_score": mapping.get("opportunity_score"),
                "first_source_id": mapping.get("first_source_id"),
            })
    return sorted(out, key=_term_mapping_sort_key)


def _term_mapping_sort_key(mapping: dict[str, Any]) -> tuple[int, int, int, int, str, str]:
    return (
        -int(mapping.get("opportunity_score") or 0),
        -int(mapping.get("source_id_count") or 0),
        -int(mapping.get("zero_result_source_count") or 0),
        int(mapping.get("rank") or 0),
        str(mapping.get("topic") or ""),
        str(mapping.get("customer_term") or ""),
    )


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _rendered_ticket_source_count(items: list[dict[str, Any]]) -> int:
    source_ids: set[str] = set()
    for item in items:
        values = item.get("source_ids") or ()
        source_ids.update(str(value) for value in values if str(value))
    return len(source_ids)


if __name__ == "__main__":
    raise SystemExit(main())
