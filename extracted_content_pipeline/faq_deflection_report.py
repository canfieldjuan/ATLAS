"""Customer-facing support-ticket deflection report renderer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from .campaign_ports import TenantScope
from .ticket_faq_markdown import TicketFAQMarkdownResult, TicketFAQMarkdownService


_RESOLUTION_EVIDENCE_STATUS = "resolution_evidence"
_RESOLUTION_EVIDENCE_SCOPE_SCOPED = "scoped"
_DRAFT_NEEDS_REVIEW_STATUS = "draft_needs_review"
DEFAULT_DEFLECTION_SNAPSHOT_TOP_N = 5
DEFAULT_DEFLECTION_TEASER_PREVIEW_COUNT = 3
_UNCAPPED_REPORT_MAX_ITEMS = 0
_ASSISTED_CONTACT_COST = 13.50
_ASSISTED_CONTACT_COST_LABEL = "$13.50"
_SOURCE_EXAMPLE_LIMIT = 3


@dataclass(frozen=True)
class DeflectionSnapshot:
    """Free preview projection that excludes paid answer/evidence fields."""

    summary: dict[str, Any]
    top_questions: tuple[dict[str, Any], ...]
    locked_questions: tuple[dict[str, int], ...]
    teaser: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": dict(self.summary),
            "top_questions": [dict(question) for question in self.top_questions],
            "locked_questions": [
                dict(question) for question in self.locked_questions
            ],
            "teaser": {
                "full_answer": (
                    dict(self.teaser["full_answer"])
                    if isinstance(self.teaser.get("full_answer"), Mapping)
                    else None
                ),
                "previews": [
                    dict(preview)
                    for preview in self.teaser.get("previews", ())
                    if isinstance(preview, Mapping)
                ],
            },
        }


@dataclass(frozen=True)
class DeflectionReportArtifact:
    """Rendered deflection report plus compact proof metadata."""

    markdown: str
    summary: dict[str, Any]
    faq_result: TicketFAQMarkdownResult

    def as_dict(self) -> dict[str, Any]:
        return {
            "markdown": self.markdown,
            "summary": dict(self.summary),
            "faq_result": self.faq_result.as_dict(),
        }

    def snapshot(self, *, top_n: int = DEFAULT_DEFLECTION_SNAPSHOT_TOP_N) -> DeflectionSnapshot:
        return build_deflection_snapshot(self, top_n=top_n)


class FAQDeflectionReportService:
    """Service-shaped generator for customer-facing FAQ deflection reports."""

    def __init__(self, faq_markdown: TicketFAQMarkdownService | None = None) -> None:
        self._faq_markdown = faq_markdown or TicketFAQMarkdownService()

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        source_material: Any,
        report_title: str | None = None,
        faq_title: str | None = None,
        max_items: int | None = None,
        max_evidence_per_item: int | None = None,
        source_types: Sequence[str] | None = None,
        max_text_chars: int | None = None,
        window_days: int | None = None,
        as_of_date: Any = None,
        support_contact: str | None = None,
        intent_rules: Sequence[tuple[str, Sequence[str]]] | None = None,
        documentation_terms: Sequence[str] | None = None,
        vocabulary_gap_rules: Sequence[Sequence[str]] | None = None,
        **kwargs: Any,
    ) -> DeflectionReportArtifact:
        del kwargs, max_items
        faq_result = await self._faq_markdown.generate(
            scope=scope,
            target_mode=target_mode,
            source_material=source_material,
            title=faq_title,
            max_items=_UNCAPPED_REPORT_MAX_ITEMS,
            max_evidence_per_item=max_evidence_per_item,
            source_types=source_types,
            max_text_chars=max_text_chars,
            window_days=window_days,
            as_of_date=as_of_date,
            support_contact=support_contact,
            intent_rules=intent_rules,
            documentation_terms=documentation_terms,
            vocabulary_gap_rules=vocabulary_gap_rules,
        )
        return build_deflection_report_artifact(
            faq_result,
            title=report_title or "Support Ticket Deflection Report",
        )


def build_deflection_report_artifact(
    faq_result: TicketFAQMarkdownResult,
    *,
    title: str = "Support Ticket Deflection Report",
    source_label: str | None = None,
) -> DeflectionReportArtifact:
    """Render a customer-facing report from a generated FAQ result."""

    summary = deflection_report_summary(faq_result)
    markdown = render_deflection_report(
        faq_result,
        title=title,
        source_label=source_label,
        summary=summary,
    )
    return DeflectionReportArtifact(
        markdown=markdown,
        summary=summary,
        faq_result=faq_result,
    )


def build_deflection_snapshot(
    artifact: DeflectionReportArtifact | Mapping[str, Any],
    *,
    top_n: int = DEFAULT_DEFLECTION_SNAPSHOT_TOP_N,
    teaser_preview_count: int = DEFAULT_DEFLECTION_TEASER_PREVIEW_COUNT,
) -> DeflectionSnapshot:
    """Project a report into the free snapshot shape.

    The snapshot intentionally omits Markdown, answers, steps, evidence,
    source IDs, term mappings, and nested FAQ item payloads.
    """

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if teaser_preview_count < 0:
        raise ValueError("teaser_preview_count must be non-negative")
    summary = _artifact_summary(artifact)
    items = _artifact_items(artifact)
    snapshot_summary: dict[str, Any] = {
        "generated": _int(summary.get("generated")),
        "drafted_answer_count": _int(summary.get("drafted_answer_count")),
        "no_proven_answer_count": _int(summary.get("no_proven_answer_count")),
        "repeat_ticket_count": sum(_ticket_count(item) for item in items),
    }
    source_date_window = _complete_source_date_window(summary, items)
    if source_date_window:
        snapshot_summary.update(source_date_window)
    top_questions: list[dict[str, Any]] = []
    for rank, item in enumerate(items[:top_n], start=1):
        question = _text(item.get("question"))
        top_questions.append({
            "rank": rank,
            "question": question,
            "ticket_count": _ticket_count(item),
            "weighted_frequency": _int(
                item.get("weighted_frequency") or item.get("frequency")
            ),
            "customer_wording": _snapshot_customer_wording(item, question),
        })
    locked_questions = tuple(
        {
            "rank": rank,
            "ticket_count": _ticket_count(item),
        }
        for rank, item in enumerate(items[top_n:], start=top_n + 1)
    )
    return DeflectionSnapshot(
        summary=snapshot_summary,
        top_questions=tuple(top_questions),
        locked_questions=locked_questions,
        teaser=_snapshot_teaser(items, preview_count=teaser_preview_count),
    )


def deflection_snapshot_content_opportunities(
    snapshot: Mapping[str, Any],
    *,
    limit: int = DEFAULT_DEFLECTION_SNAPSHOT_TOP_N,
) -> tuple[dict[str, Any], ...]:
    """Return structured, unpaid-safe opportunities from a free snapshot."""

    top_questions = snapshot.get("top_questions")
    if not isinstance(top_questions, Sequence) or isinstance(
        top_questions,
        (str, bytes, bytearray),
    ):
        return ()

    out: list[dict[str, Any]] = []
    for index, raw in enumerate(top_questions, start=1):
        if len(out) >= _positive_limit(limit):
            break
        if not isinstance(raw, Mapping):
            continue
        question = _text(raw.get("question"))
        if not question:
            continue
        frequency = _int(raw.get("weighted_frequency") or raw.get("frequency"))
        rank = _int(raw.get("rank")) or index
        out.append(
            {
                "rank": rank,
                "question": question,
                "ticket_count": _int(raw.get("ticket_count")),
                "weighted_frequency": frequency,
                "customer_wording": _text(raw.get("customer_wording")),
                "opportunity_score": _int(raw.get("opportunity_score")) or frequency,
                "coverage_status": _text(raw.get("coverage_status")) or "locked_snapshot",
                "recommended_content_action": "Create or improve an FAQ entry for this repeated customer question.",
                "unlock_hint": "Unlock the full report for detailed source-backed guidance.",
            }
        )
    return tuple(out)


def deflection_report_summary(faq_result: TicketFAQMarkdownResult) -> dict[str, Any]:
    """Return compact counts used by the report and CLI summary JSON."""

    items = tuple(_item(item) for item in faq_result.items)
    proven = tuple(
        item for item in items
        if _text(item.get("answer_evidence_status")) == _RESOLUTION_EVIDENCE_STATUS
    )
    needs_review = tuple(
        item for item in items
        if _text(item.get("answer_evidence_status")) != _RESOLUTION_EVIDENCE_STATUS
    )
    summary = {
        "generated": len(items),
        "source_count": int(faq_result.source_count),
        "ticket_source_count": int(faq_result.ticket_source_count),
        "drafted_answer_count": len(proven),
        "no_proven_answer_count": len(needs_review),
        "output_checks": dict(faq_result.output_checks),
        "top_question": _text(items[0].get("question")) if items else "",
        "top_opportunity_score": _int(items[0].get("opportunity_score")) if items else 0,
    }
    source_date_window = _complete_source_date_window({}, items)
    if source_date_window:
        summary.update(source_date_window)
    return summary


def render_deflection_report(
    faq_result: TicketFAQMarkdownResult,
    *,
    title: str = "Support Ticket Deflection Report",
    source_label: str | None = None,
    summary: Mapping[str, Any] | None = None,
) -> str:
    """Render the customer-facing Markdown report."""

    resolved_summary = dict(summary or deflection_report_summary(faq_result))
    items = tuple(_item(item) for item in faq_result.items)
    proven = tuple(
        item for item in items
        if _text(item.get("answer_evidence_status")) == _RESOLUTION_EVIDENCE_STATUS
    )
    needs_review = tuple(
        item for item in items
        if _text(item.get("answer_evidence_status")) != _RESOLUTION_EVIDENCE_STATUS
    )

    lines: list[str] = [
        f"# {_md(title)}",
        "",
    ]
    lines.extend(_support_tax_section(resolved_summary, items))
    if source_label:
        lines.extend(["**Source file:**", "", _md(source_label), ""])
    lines.extend(_help_desk_seo_targeting_section(items))
    lines.extend(_ranked_opportunity_section(items))
    lines.extend(_drafted_answer_section(proven))
    lines.extend(_no_proven_answer_section(needs_review))
    lines.extend(_vocabulary_gap_section(items))
    lines.extend(_evidence_appendix_section(items))
    return "\n".join(lines).rstrip() + "\n"


def _support_tax_section(
    summary: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> list[str]:
    repeat_ticket_count = _repeat_ticket_count(items)
    batch_cost = _support_cost(repeat_ticket_count)
    source_window = _complete_source_date_window(summary, items)
    lines = [
        "## Support Tax Confirmation",
        "",
        (
            f"ATLAS found {_count(repeat_ticket_count)} repeat-ticket hits across "
            f"{_count(_int(summary.get('generated')))} ranked questions. At the "
            f"Gartner {_ASSISTED_CONTACT_COST_LABEL} assisted-contact benchmark, "
            f"that uploaded work sizes to about {_format_money(batch_cost)} of "
            "assisted-contact handling."
        ),
    ]
    if source_window:
        annualized = _support_cost(
            repeat_ticket_count * 365 / _int(source_window.get("source_window_days"))
        )
        lines.extend([
            "",
            (
                "The source window is "
                f"{_source_window_label(source_window)}. At the same measured daily "
                f"pace, that is about {_format_money(annualized)} over 12 months."
            ),
        ])
    else:
        monthly_pace = _support_cost(repeat_ticket_count * 12)
        lines.extend([
            "",
            (
                "ATLAS did not receive a complete source-date window for every "
                "contributing ticket, so this report does not infer a monthly or "
                "annual reporting period. If this uploaded batch is monthly pace, "
                f"the 12-month run-rate would be about {_format_money(monthly_pace)}."
            ),
        ])
    lines.extend([
        "",
        (
            "Estimate only. This is not a savings guarantee; adjust the "
            f"{_ASSISTED_CONTACT_COST_LABEL} benchmark to your own loaded support "
            "cost."
        ),
        "",
        (
            "The full unlocked report below gives you every ranked repeat question, "
            "the estimated support cost by question, publishable help-center copy "
            "where your uploaded resolutions prove the answer, the no-proven-answer "
            "roadmap, and the complete evidence appendix."
        ),
        "",
        f"- Publishable answers drafted from proven resolutions: {_count(_int(summary.get('drafted_answer_count')))}",
        f"- Questions still needing an approved resolution: {_count(_int(summary.get('no_proven_answer_count')))}",
        f"- Ticket sources represented: {_count(_int(summary.get('ticket_source_count')))}",
        "",
    ])
    return lines


def _help_desk_seo_targeting_section(
    items: Sequence[Mapping[str, Any]],
) -> list[str]:
    phrases = _customer_phrase_list(items)
    lines = [
        "## Your Help-Desk SEO Targeting List",
        "",
        (
            "Use these source-backed phrases as help-center headings, "
            "internal-search synonyms, and FAQ wording. ATLAS mined them from "
            "the tickets you uploaded; it does not claim keyword volume, search "
            "rank, or traffic."
        ),
        "",
    ]
    if not phrases:
        return [*lines, "No customer phrase targets were generated for this run.", ""]
    lines.extend(
        f"{index}. {_md(phrase)}"
        for index, phrase in enumerate(phrases, start=1)
    )
    lines.append("")
    return lines


def _ranked_opportunity_section(items: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "## Ranked Question Opportunities",
        "",
        "| Rank | Customer question | Tickets | Estimated support cost | Opportunity | Answer status | Source proof |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    if not items:
        return [
            *lines,
            "| - | No ranked FAQ opportunities were generated. | 0 | $0 | 0 | - | - |",
            "",
        ]
    for index, item in enumerate(items, start=1):
        ticket_count = _ticket_count(item)
        lines.append(
            "| "
            f"{index} | {_cell(item.get('question'))} | "
            f"{ticket_count} | "
            f"{_format_money(_support_cost(ticket_count))} | "
            f"{_int(item.get('opportunity_score'))} | "
            f"{_status_label(item)} | "
            f"{_cell(_source_count_label(item))} |"
        )
    return [*lines, ""]


def _drafted_answer_section(items: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["## Publishable Help-Center Copy From Proven Resolutions", ""]
    if not items:
        return [
            *lines,
            "No FAQ gap in this run included uploaded resolution evidence. Keep every answer in review until support supplies a verified solution.",
            "",
        ]
    for index, item in enumerate(items, start=1):
        steps = _texts(item.get("steps"))
        answer = _text(item.get("answer")) or (
            "This draft answer is backed by uploaded resolution evidence."
        )
        lines.extend([
            f"### {index}. {_md(item.get('question'))}",
            "",
            _md(answer),
            "",
            "**Draft answer steps:**",
            "",
            *[
                f"{step_index}. {_md(step)}"
                for step_index, step in enumerate(steps, start=1)
            ],
            "",
            f"**Evidence backing:** {_source_backing_summary(item, resolved=True)}",
            "",
        ])
    return lines


def _no_proven_answer_section(items: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["## No Proven Answer Yet", ""]
    if not items:
        return [*lines, "Every generated FAQ opportunity included uploaded resolution evidence.", ""]
    for index, item in enumerate(items, start=1):
        lines.extend([
            f"### {index}. {_md(item.get('question'))}",
            "",
            (
                "Customers repeatedly asked this question, but the uploaded data "
                "did not include verified resolution evidence for a publishable answer."
            ),
            "",
            (
                "No verified support resolution was present in the uploaded data. "
                "Support should add the approved answer before this FAQ is published."
            ),
            "",
            f"**Ticket backing:** {_source_backing_summary(item, resolved=False)}",
            "",
        ])
    return lines


def _vocabulary_gap_section(items: Sequence[Mapping[str, Any]]) -> list[str]:
    mappings: list[Mapping[str, Any]] = []
    for item in items:
        for mapping in item.get("term_mappings") or ():
            if isinstance(mapping, Mapping):
                mappings.append(mapping)
    lines = [
        "## Vocabulary Gaps",
        "",
        "| Customer wording | Documentation term | Suggested update | Source count |",
        "|---|---|---|---:|",
    ]
    if not mappings:
        return [*lines, "| - | - | No vocabulary-gap mappings were generated. | 0 |", ""]
    for mapping in mappings:
        lines.append(
            "| "
            f"{_cell(mapping.get('customer_term'))} | "
            f"{_cell(mapping.get('documentation_term'))} | "
            f"{_cell(mapping.get('suggestion'))} | "
            f"{_int(mapping.get('source_id_count'))} |"
        )
    return [*lines, ""]


def _evidence_appendix_section(items: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["## Evidence Appendix", ""]
    for index, item in enumerate(items, start=1):
        quotes = _texts(item.get("evidence_quotes"))
        lines.extend([f"### {index}. {_md(item.get('question'))}", ""])
        source_ids = _texts(item.get("source_ids"))
        if source_ids:
            lines.extend([
                f"**Source IDs (full list):** {_md(', '.join(source_ids))}",
                "",
            ])
        elif _source_count(item) > 0:
            lines.extend([
                (
                    "**Source IDs (full list):** Not available in this export; "
                    f"source count is {_source_count(item)}."
                ),
                "",
            ])
        if not quotes:
            lines.extend(["No source excerpts were rendered for this item.", ""])
            continue
        for quote in quotes:
            lines.append(f"- {_md(quote)}")
        lines.append("")
    if len(lines) == 2:
        lines.append("No evidence excerpts were rendered.")
        lines.append("")
    return lines


def _status_label(item: Mapping[str, Any]) -> str:
    status = _text(item.get("answer_evidence_status"))
    if status == _RESOLUTION_EVIDENCE_STATUS:
        return "drafted from resolution evidence"
    if status == _DRAFT_NEEDS_REVIEW_STATUS:
        return "no proven answer yet"
    return status or "unknown"


def _item(value: Mapping[str, Any]) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _artifact_summary(
    artifact: DeflectionReportArtifact | Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(artifact, DeflectionReportArtifact):
        return artifact.summary
    value = artifact.get("summary")
    return value if isinstance(value, Mapping) else {}


def _artifact_items(
    artifact: DeflectionReportArtifact | Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(artifact, DeflectionReportArtifact):
        return tuple(_item(item) for item in artifact.faq_result.items)
    faq_result = artifact.get("faq_result")
    if not isinstance(faq_result, Mapping):
        return ()
    items = faq_result.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        return ()
    return tuple(_item(item) for item in items if isinstance(item, Mapping))


def _snapshot_customer_wording(item: Mapping[str, Any], question: str) -> str:
    explicit = _text(item.get("customer_wording"))
    if explicit:
        return explicit
    if _text(item.get("question_source")) == "customer_wording":
        return question
    return ""


def _snapshot_teaser(
    items: Sequence[Mapping[str, Any]],
    *,
    preview_count: int,
) -> dict[str, Any]:
    eligible = tuple(
        (rank, item)
        for rank, item in enumerate(items, start=1)
        if _is_teaser_eligible(item)
    )
    if not eligible:
        return {"full_answer": None, "previews": []}
    full_rank, full_item = _select_full_teaser_item(eligible)
    previews = [
        _teaser_preview(rank, item)
        for rank, item in eligible
        if rank != full_rank
    ][:preview_count]
    return {
        "full_answer": _teaser_full_answer(full_rank, full_item),
        "previews": previews,
    }


def _is_teaser_eligible(item: Mapping[str, Any]) -> bool:
    return (
        _text(item.get("answer_evidence_status")) == _RESOLUTION_EVIDENCE_STATUS
        and _text(item.get("resolution_evidence_scope")) == _RESOLUTION_EVIDENCE_SCOPE_SCOPED
        and bool(_text(item.get("answer")))
    )


def _select_full_teaser_item(
    eligible: Sequence[tuple[int, Mapping[str, Any]]],
) -> tuple[int, Mapping[str, Any]]:
    for rank, item in eligible:
        if rank >= 4:
            return rank, item
    return eligible[-1]


def _teaser_full_answer(rank: int, item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rank": rank,
        "question": _text(item.get("question")),
        "answer": _text(item.get("answer")),
        "steps": _texts(item.get("steps")),
        "answer_evidence_status": _RESOLUTION_EVIDENCE_STATUS,
        "resolution_evidence_scope": _RESOLUTION_EVIDENCE_SCOPE_SCOPED,
        "weighted_frequency": _int(item.get("weighted_frequency") or item.get("frequency")),
        "source_count": _source_count(item),
    }


def _teaser_preview(rank: int, item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rank": rank,
        "question": _text(item.get("question")),
        "answer_evidence_status": _RESOLUTION_EVIDENCE_STATUS,
        "resolution_evidence_scope": _RESOLUTION_EVIDENCE_SCOPE_SCOPED,
        "weighted_frequency": _int(item.get("weighted_frequency") or item.get("frequency")),
        "step_count": len(_texts(item.get("steps"))),
        "source_count": _source_count(item),
        "body_withheld": True,
    }


def _source_count(item: Mapping[str, Any]) -> int:
    source_count = len(_texts(item.get("source_ids")))
    return source_count or _int(item.get("ticket_count"))


def _repeat_ticket_count(items: Sequence[Mapping[str, Any]]) -> int:
    return sum(_ticket_count(item) for item in items)


def _ticket_count(item: Mapping[str, Any]) -> int:
    ticket_count = _int(item.get("ticket_count"))
    if ticket_count > 0:
        return ticket_count
    source_count = len(_texts(item.get("source_ids")))
    return source_count if source_count > 0 else 0


def _complete_source_date_window(
    summary: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary_window = _summary_source_date_window(summary)
    if summary_window:
        return summary_window
    return _items_source_date_window(items)


def _summary_source_date_window(summary: Mapping[str, Any]) -> dict[str, Any]:
    start = _iso_date_text(summary.get("source_date_start"))
    end = _iso_date_text(summary.get("source_date_end"))
    days = _int(summary.get("source_window_days"))
    if not start or not end or days < 1:
        return {}
    parsed_start = date.fromisoformat(start)
    parsed_end = date.fromisoformat(end)
    if parsed_end < parsed_start:
        return {}
    expected_days = (parsed_end - parsed_start).days + 1
    if days != expected_days:
        return {}
    return {
        "source_date_start": start,
        "source_date_end": end,
        "source_window_days": days,
    }


def _items_source_date_window(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    starts: list[date] = []
    ends: list[date] = []
    missing_source_count = 0
    saw_dated_span = False
    for item in items:
        raw_span = item.get("source_date_span")
        if not isinstance(raw_span, Mapping):
            if _ticket_count(item) > 0:
                missing_source_count += _ticket_count(item)
            continue
        start = _iso_date(raw_span.get("start"))
        end = _iso_date(raw_span.get("end"))
        if start is None or end is None or end < start:
            missing_source_count += 1
            continue
        saw_dated_span = True
        starts.append(start)
        ends.append(end)
        missing_source_count += _int(raw_span.get("missing_source_count"))
    if not saw_dated_span or missing_source_count > 0 or not starts or not ends:
        return {}
    start = min(starts)
    end = max(ends)
    return {
        "source_date_start": start.isoformat(),
        "source_date_end": end.isoformat(),
        "source_window_days": (end - start).days + 1,
    }


def _customer_phrase_list(items: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    seen: set[str] = set()
    phrases: list[str] = []

    def add(value: Any) -> None:
        phrase = _text(value)
        if not phrase:
            return
        key = phrase.casefold()
        if key in seen:
            return
        seen.add(key)
        phrases.append(phrase)

    for item in items:
        add(
            item.get("customer_wording")
            or item.get("question")
        )
        for mapping in item.get("term_mappings") or ():
            if isinstance(mapping, Mapping):
                add(mapping.get("customer_term"))
    return tuple(phrases)


def _source_count_label(item: Mapping[str, Any]) -> str:
    count = _source_count(item)
    if count <= 0:
        return "No source tickets"
    if count == 1:
        return "1 source ticket"
    return f"{_count(count)} source tickets"


def _source_backing_summary(item: Mapping[str, Any], *, resolved: bool) -> str:
    count = _source_count(item)
    source_ids = _texts(item.get("source_ids"))
    examples = source_ids[:_SOURCE_EXAMPLE_LIMIT]
    if resolved:
        prefix = (
            f"Backed by {_count(count)} resolved ticket"
            f"{'' if count == 1 else 's'}"
        )
    else:
        prefix = (
            f"Seen in {_count(count)} repeated ticket"
            f"{'' if count == 1 else 's'}"
        )
    if not examples:
        return f"{prefix}. Full source details are in the Evidence Appendix."
    more_count = max(len(source_ids) - len(examples), 0)
    if more_count > 0:
        example_text = f"{', '.join(examples)}, +{_count(more_count)} more"
    else:
        example_text = ", ".join(examples)
    return (
        f"{prefix} ({_md(example_text)}). Full source IDs are in the Evidence Appendix."
    )


def _source_window_label(source_window: Mapping[str, Any]) -> str:
    start = _text(source_window.get("source_date_start"))
    end = _text(source_window.get("source_date_end"))
    days = _int(source_window.get("source_window_days"))
    return f"{start} to {end} ({_count(days)} days)"


def _support_cost(ticket_count: float | int) -> float:
    return max(0.0, float(ticket_count) * _ASSISTED_CONTACT_COST)


def _format_money(value: float | int) -> str:
    rounded = int(float(value) + 0.5)
    return f"${rounded:,}"


def _count(value: int) -> str:
    return f"{value:,}"


def _iso_date_text(value: Any) -> str:
    parsed = _iso_date(value)
    return parsed.isoformat() if parsed is not None else ""


def _iso_date(value: Any) -> date | None:
    text = _text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _texts(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        values = (value,)
    out: list[str] = []
    for item in values:
        text = _text(item)
        if text:
            out.append(text)
    return out


def _text(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _positive_limit(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_DEFLECTION_SNAPSHOT_TOP_N
    return max(1, min(parsed, 25))


def _cell(value: Any) -> str:
    text = _md(value).replace("\n", " ")
    return text.replace("|", "\\|")


def _md(value: Any) -> str:
    return _text(value).replace("\r", " ").strip()


__all__ = [
    "DEFAULT_DEFLECTION_SNAPSHOT_TOP_N",
    "DEFAULT_DEFLECTION_TEASER_PREVIEW_COUNT",
    "DeflectionSnapshot",
    "DeflectionReportArtifact",
    "FAQDeflectionReportService",
    "build_deflection_snapshot",
    "build_deflection_report_artifact",
    "deflection_report_summary",
    "deflection_snapshot_content_opportunities",
    "render_deflection_report",
]
