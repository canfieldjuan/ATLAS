"""Customer-facing support-ticket deflection report renderer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .ticket_faq_markdown import TicketFAQMarkdownResult


_RESOLUTION_EVIDENCE_STATUS = "resolution_evidence"
_DRAFT_NEEDS_REVIEW_STATUS = "draft_needs_review"


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
    return {
        "generated": len(items),
        "source_count": int(faq_result.source_count),
        "ticket_source_count": int(faq_result.ticket_source_count),
        "drafted_answer_count": len(proven),
        "no_proven_answer_count": len(needs_review),
        "output_checks": dict(faq_result.output_checks),
        "top_question": _text(items[0].get("question")) if items else "",
        "top_opportunity_score": _int(items[0].get("opportunity_score")) if items else 0,
    }


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
        "## Executive Summary",
        "",
        (
            f"This report analyzed {resolved_summary.get('source_count', 0)} source rows "
            f"and produced {resolved_summary.get('generated', 0)} ranked FAQ "
            "opportunities from the supplied support data."
        ),
        "",
        (
            f"- Drafted answers with proven solutions: "
            f"{resolved_summary.get('drafted_answer_count', 0)}"
        ),
        (
            f"- No proven answer yet: "
            f"{resolved_summary.get('no_proven_answer_count', 0)}"
        ),
        f"- Ticket sources represented: {resolved_summary.get('ticket_source_count', 0)}",
        "",
    ]
    if source_label:
        lines.extend(["**Source file:**", "", _md(source_label), ""])
    lines.extend(_ranked_opportunity_section(items))
    lines.extend(_drafted_answer_section(proven))
    lines.extend(_no_proven_answer_section(needs_review))
    lines.extend(_vocabulary_gap_section(items))
    lines.extend(_evidence_appendix_section(items))
    return "\n".join(lines).rstrip() + "\n"


def _ranked_opportunity_section(items: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "## Ranked Question Opportunities",
        "",
        "| Rank | Customer question | Frequency | Opportunity | Answer status | Source IDs |",
        "|---:|---|---:|---:|---|---|",
    ]
    if not items:
        return [*lines, "| - | No ranked FAQ opportunities were generated. | 0 | 0 | - | - |", ""]
    for index, item in enumerate(items, start=1):
        lines.append(
            "| "
            f"{index} | {_cell(item.get('question'))} | "
            f"{_int(item.get('weighted_frequency') or item.get('frequency'))} | "
            f"{_int(item.get('opportunity_score'))} | "
            f"{_status_label(item)} | "
            f"{_cell(', '.join(_texts(item.get('source_ids'))[:5]))} |"
        )
    return [*lines, ""]


def _drafted_answer_section(items: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["## Drafted Answers With Proven Solutions", ""]
    if not items:
        return [
            *lines,
            "No FAQ gap in this run included uploaded resolution evidence. Keep every answer in review until support supplies a verified solution.",
            "",
        ]
    for index, item in enumerate(items, start=1):
        steps = _texts(item.get("steps"))
        lines.extend([
            f"### {index}. {_md(item.get('question'))}",
            "",
            "Uploaded resolution evidence supports this draft answer.",
            "",
            "**Draft answer steps:**",
            "",
            *[
                f"{step_index}. {_md(step)}"
                for step_index, step in enumerate(steps, start=1)
            ],
            "",
            f"**Sources:** {_md(', '.join(_texts(item.get('source_ids'))))}",
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
            f"**Sources:** {_md(', '.join(_texts(item.get('source_ids'))))}",
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


def _cell(value: Any) -> str:
    text = _md(value).replace("\n", " ")
    return text.replace("|", "\\|")


def _md(value: Any) -> str:
    return _text(value).replace("\r", " ").strip()


__all__ = [
    "DeflectionReportArtifact",
    "build_deflection_report_artifact",
    "deflection_report_summary",
    "render_deflection_report",
]
