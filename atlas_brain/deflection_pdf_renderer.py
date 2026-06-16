"""PDF renderer for paid Content Ops deflection reports."""

from __future__ import annotations

from collections.abc import Sequence
import io
import re
from typing import Any, Mapping

from extracted_content_pipeline.deflection_report_access import (
    stored_deflection_report_model,
)
from fpdf import FPDF


_CLR_DARK = (32, 38, 46)
_CLR_MUTED = (93, 104, 119)
_CLR_BORDER = (218, 224, 231)
_CLR_ACCENT = (24, 105, 178)
_CLR_WHITE = (255, 255, 255)
_CLR_HEADER = (18, 27, 39)
PDF_RANKED_TABLE_LIMIT = 25
PDF_QUESTION_DETAIL_LIMIT = 10
_COMPLETE_EVIDENCE_MARKER = "**Complete evidence:**"
_EVIDENCE_EXPORT_POINTER = (
    "Complete source IDs and evidence quotes are available in the complete "
    "evidence export JSON linked from the paid result page."
)
_RANKED_TABLE_CAP_NOTE = (
    f"Ranked question table capped at {PDF_RANKED_TABLE_LIMIT} rows for PDF "
    "readability; download the complete evidence export for every ranked "
    "question and source row."
)
_QUESTION_DETAIL_CAP_NOTE = (
    f"Question detail blocks capped at {PDF_QUESTION_DETAIL_LIMIT} questions "
    "for PDF readability; download the complete evidence export for the full "
    "uncapped evidence archive."
)

_UNICODE_MAP = str.maketrans({
    "\u2014": "--",
    "\u2013": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u2022": "*",
    "\u00a0": " ",
    "\ufeff": "",
})


class DeflectionReportPDF(FPDF):
    def header(self) -> None:
        self.set_fill_color(*_CLR_HEADER)
        self.rect(0, 0, self.w, 28, "F")
        self.set_y(8)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*_CLR_WHITE)
        self.cell(
            0,
            8,
            "Support Ticket Deflection Report",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.set_font("Helvetica", "", 8)
        self.cell(0, 5, "Paid full report PDF", new_x="LMARGIN", new_y="NEXT")
        self.set_y(34)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*_CLR_MUTED)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")

    def section_title(self, text: str, *, level: int) -> None:
        self.ln(2)
        self.set_text_color(*_CLR_ACCENT if level <= 2 else _CLR_DARK)
        self.set_font("Helvetica", "B", 13 if level <= 2 else 11)
        self.multi_cell(0, 7, _safe(text), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*_CLR_BORDER)
        if level <= 2:
            self.line(
                self.l_margin,
                self.get_y(),
                self.w - self.r_margin,
                self.get_y(),
            )
            self.ln(2)

    def body_text(self, text: str) -> None:
        self.set_text_color(*_CLR_DARK)
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 5, _safe(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def table_row(self, cells: list[str]) -> None:
        self.set_text_color(*_CLR_DARK)
        self.set_font("Helvetica", "", 8)
        self.multi_cell(
            0,
            4.5,
            _safe(" | ".join(cell for cell in cells if cell)),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.ln(0.5)


def render_deflection_full_report_pdf(
    artifact: Mapping[str, Any],
    *,
    fallback_title: str = "Support Ticket Deflection Report",
) -> bytes:
    """Render a curated/shareable paid deflection report PDF."""

    model_markdown = _artifact_report_model_pdf_markdown(artifact)
    if model_markdown:
        title = _markdown_title(model_markdown) or fallback_title
        curated_markdown = model_markdown
    else:
        markdown = _artifact_markdown(artifact)
        if not markdown:
            raise ValueError("deflection report artifact report_model or markdown is required")
        title = _markdown_title(markdown) or fallback_title
        curated_markdown = _curate_markdown_for_pdf(markdown)

    pdf = DeflectionReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.alias_nb_pages()
    pdf.add_page()
    _render_pdf_intro(pdf, title=title, toc_entries=_toc_entries(curated_markdown))
    _render_markdown(
        pdf,
        _drop_first_title_heading(curated_markdown),
        title=title,
        render_missing_title=False,
    )

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def _artifact_report_model_pdf_markdown(artifact: Mapping[str, Any]) -> str:
    model = stored_deflection_report_model(artifact)
    if model is None:
        return ""
    return _report_model_pdf_markdown(model)


def _report_model_pdf_markdown(model: Mapping[str, Any]) -> str:
    title = _model_text(model.get("title")) or "Support Ticket Deflection Report"
    lines: list[str] = [f"# {title}", ""]
    rendered_section = False
    for section in _model_pdf_sections(model):
        section_lines = _report_model_section_pdf_lines(section)
        if not section_lines:
            continue
        lines.extend(section_lines)
        rendered_section = True
    if not rendered_section:
        return ""
    return "\n".join(lines).strip()


def _model_pdf_sections(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_sections = model.get("sections")
    if not _is_sequence(raw_sections):
        return []
    sections = [
        dict(section)
        for section in raw_sections
        if isinstance(section, Mapping) and "pdf" in _model_texts(section.get("surfaces"))
    ]
    sections.sort(key=lambda section: _model_int(section.get("priority")))
    return sections


def _report_model_section_pdf_lines(section: Mapping[str, Any]) -> list[str]:
    section_id = _model_text(section.get("id"))
    if section_id == "support_tax":
        return _support_tax_model_pdf_lines(section)
    if section_id == "source_file":
        return _source_file_model_pdf_lines(section)
    if section_id == "seo_targets":
        return _seo_targets_model_pdf_lines(section)
    if section_id == "ranked_questions":
        return _ranked_questions_model_pdf_lines(section)
    if section_id == "outcome_diagnostics":
        return _outcome_diagnostics_model_pdf_lines(section)
    if section_id == "question_details":
        return _question_details_model_pdf_lines(section)
    return []


def _support_tax_model_pdf_lines(section: Mapping[str, Any]) -> list[str]:
    data = _section_data(section)
    repeat_ticket_count = _model_int(data.get("repeat_ticket_count"))
    generated_question_count = _model_int(data.get("generated_question_count"))
    non_repeat_ticket_count = _model_int(data.get("non_repeat_ticket_count"))
    assisted_contact_cost = _model_money(data.get("assisted_contact_cost"))
    estimated_support_cost = _model_money(data.get("estimated_support_cost"))
    source_window = data.get("source_date_window")
    lines = [
        "## Support Tax Confirmation",
        "",
        (
            f"This report found {_model_count(repeat_ticket_count)} question-level "
            f"repeat tickets across {_model_count(generated_question_count)} ranked "
            f"questions. At the Gartner {assisted_contact_cost} assisted-contact "
            "benchmark, that repeated-question work sizes to about "
            f"{estimated_support_cost} of assisted-contact handling."
        ),
    ]
    if non_repeat_ticket_count:
        lines.extend([
            "",
            (
                f"{_model_count(non_repeat_ticket_count)} tickets asked a question "
                "that appeared only once in this upload; they are excluded from the "
                "repeat counts and cost sizing above."
            ),
        ])
    if isinstance(source_window, Mapping):
        window_label = _source_window_label(source_window)
        annualized = _model_money(data.get("annualized_support_cost"))
        if window_label:
            lines.extend([
                "",
                (
                    f"The source window is {window_label}. At the same measured "
                    f"daily pace, that is about {annualized} over 12 months."
                ),
            ])
    elif "annualized_run_rate_support_cost" in data:
        lines.extend([
            "",
            (
                "This report did not receive a complete source-date window for every "
                "contributing ticket. If this uploaded batch is monthly pace, the "
                "12-month run-rate would be about "
                f"{_model_money(data.get('annualized_run_rate_support_cost'))}."
            ),
        ])
    lines.extend([
        "",
        (
            "Estimate only. This is not a savings guarantee; adjust the "
            f"{assisted_contact_cost} benchmark to your own loaded support cost."
        ),
        "",
        (
            "The PDF is a curated/shareable report. Download the complete evidence "
            "export for every source row and quote."
        ),
        "",
        (
            "- Publishable answers drafted from proven resolutions: "
            f"{_model_count(_model_int(data.get('drafted_answer_count')))}"
        ),
        (
            "- Questions still needing an approved resolution: "
            f"{_model_count(_model_int(data.get('no_proven_answer_count')))}"
        ),
        (
            "- Ticket sources represented: "
            f"{_model_count(_model_int(data.get('ticket_source_count')))}"
        ),
        "",
    ])
    return lines


def _source_file_model_pdf_lines(section: Mapping[str, Any]) -> list[str]:
    source_label = _model_text(_section_data(section).get("source_label"))
    if not source_label:
        return []
    return ["## Source file", "", source_label, ""]


def _seo_targets_model_pdf_lines(section: Mapping[str, Any]) -> list[str]:
    data = _section_data(section)
    phrases = _model_texts(data.get("phrases"))
    limit = _model_int(data.get("limit")) or len(phrases)
    omitted_count = _model_int(data.get("omitted_phrase_count"))
    lines = [
        "## Your Help-Desk SEO Targeting List",
        "",
        (
            "Use these source-backed phrases as help-center headings, "
            "internal-search synonyms, and FAQ wording. These were mined from "
            "the tickets you uploaded; this report does not claim keyword volume, "
            "search rank, or traffic."
        ),
        "",
    ]
    if not phrases:
        return [*lines, "No customer phrase targets were generated for this run.", ""]
    lines.extend(
        f"{index}. {phrase}"
        for index, phrase in enumerate(phrases[:max(1, limit)], start=1)
    )
    if omitted_count:
        lines.extend([
            "",
            (
                f"SEO phrase index capped at {_model_count(max(1, limit))} entries "
                f"for readability; {_model_count(omitted_count)} additional "
                "source-backed phrases remain represented in the complete evidence "
                "export."
            ),
        ])
    lines.append("")
    return lines


def _ranked_questions_model_pdf_lines(section: Mapping[str, Any]) -> list[str]:
    rows = _model_rows(_section_data(section).get("rows"))
    lines = [
        "## Ranked Question Opportunities",
        "",
        "| Rank | Customer question | Tickets | Estimated support cost | Opportunity | Answer status | Source proof |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    if not rows:
        return [
            *lines,
            "| - | No ranked FAQ opportunities were generated. | 0 | $0 | 0 | - | - |",
            "",
        ]
    for row in rows[:PDF_RANKED_TABLE_LIMIT]:
        lines.append(
            "| "
            f"{_model_int(row.get('rank'))} | "
            f"{_model_cell(row.get('question'))} | "
            f"{_model_count(_model_int(row.get('ticket_count')))} | "
            f"{_model_money(row.get('estimated_support_cost'))} | "
            f"{_model_int(row.get('opportunity_score'))} | "
            f"{_model_cell(row.get('answer_status'))} | "
            f"{_model_cell(row.get('source_proof'))} |"
        )
    if len(rows) > PDF_RANKED_TABLE_LIMIT:
        lines.extend(["", _RANKED_TABLE_CAP_NOTE])
    lines.append("")
    return lines


def _outcome_diagnostics_model_pdf_lines(section: Mapping[str, Any]) -> list[str]:
    data = _section_data(section)
    rows = _model_rows(data.get("rows"))
    if not rows:
        return []
    lines = [
        "## Resolution Outcome Diagnostics",
        "",
        (
            "These status and CSAT signals flag answers that may need review. "
            "They do not prove a publishable answer; only uploaded resolution "
            "evidence can do that."
        ),
        "",
        (
            "- Tickets with outcome diagnostics: "
            f"{_model_count(_model_int(data.get('outcome_diagnostic_ticket_count')))}"
        ),
        (
            "- Tickets with reopened or negative-CSAT risk: "
            f"{_model_count(_model_int(data.get('outcome_risk_ticket_count')))}"
        ),
        f"- Reopened tickets: {_model_count(_model_int(data.get('reopened_ticket_count')))}",
        (
            "- Negative CSAT tickets: "
            f"{_model_count(_model_int(data.get('negative_csat_ticket_count')))}"
        ),
        "",
        "| Customer question | Status mix | Reopened | Negative CSAT | Guidance |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{_model_cell(row.get('question'))} | "
            f"{_model_cell(row.get('status_mix'))} | "
            f"{_model_count(_model_int(row.get('reopened_ticket_count')))} | "
            f"{_model_count(_model_int(row.get('negative_csat_ticket_count')))} | "
            f"{_model_cell(row.get('guidance'))} |"
        )
    lines.append("")
    return lines


def _question_details_model_pdf_lines(section: Mapping[str, Any]) -> list[str]:
    rows = _model_rows(_section_data(section).get("rows"))
    lines = [
        "## Question Details and Evidence",
        "",
        (
            "Each ranked question appears once below with its answer status, "
            "publishable copy or review guidance, vocabulary gaps, and a pointer "
            "to complete source evidence."
        ),
        "",
        (
            "Questions without uploaded resolution evidence stay in review: "
            "outcome/status signals can prioritize them, but only resolution "
            "evidence can make an answer publishable."
        ),
        "",
    ]
    if not rows:
        return [*lines, "No ranked FAQ opportunities were generated.", ""]
    for index, row in enumerate(rows[:PDF_QUESTION_DETAIL_LIMIT], start=1):
        rank = _model_int(row.get("rank")) or index
        source_count = len(_model_texts(row.get("source_ids")))
        lines.extend([
            f"### {rank}. {_model_text(row.get('question'))}",
            "",
        ])
        customer_wording = _model_text(row.get("customer_wording"))
        question = _model_text(row.get("question"))
        if customer_wording and customer_wording != question:
            lines.extend(["**Customer wording:** " + customer_wording, ""])
        lines.extend([
            f"**Answer status:** {_model_text(row.get('answer_status'))}",
            "",
            (
                "**Ticket/support-cost context:** "
                f"{_model_count(_model_int(row.get('ticket_count')))} tickets, "
                f"estimated at {_model_money(row.get('estimated_support_cost'))} "
                "of assisted-contact handling."
            ),
            "",
        ])
        if _model_text(row.get("answer_linkage")) == "publishable_answer":
            lines.extend(_publishable_answer_model_pdf_lines(row, source_count))
        else:
            lines.extend(_no_proven_answer_model_pdf_lines(source_count))
        lines.extend(_term_mapping_model_pdf_lines(row))
    if len(rows) > PDF_QUESTION_DETAIL_LIMIT:
        lines.extend(["", _QUESTION_DETAIL_CAP_NOTE, ""])
    return lines


def _publishable_answer_model_pdf_lines(
    row: Mapping[str, Any],
    source_count: int,
) -> list[str]:
    answer = _model_text(row.get("answer")) or (
        "This draft answer is backed by uploaded resolution evidence."
    )
    steps = _model_texts(row.get("steps"))
    lines = ["**Publishable answer draft:**", "", answer, ""]
    if steps:
        lines.extend(["**Draft answer steps:**", ""])
        lines.extend(f"{index}. {step}" for index, step in enumerate(steps, start=1))
        lines.append("")
    else:
        lines.extend(["**Draft answer steps:**", "", "No step list was generated for this answer.", ""])
    lines.extend([
        (
            "**Evidence backing:** "
            f"{_model_count(source_count)} source tickets; complete source details "
            "are in the complete evidence export."
        ),
        "",
    ])
    return lines


def _no_proven_answer_model_pdf_lines(source_count: int) -> list[str]:
    return [
        "**No proven answer yet:**",
        "",
        "No uploaded resolution evidence was present for this question.",
        "",
        (
            "**Ticket backing:** "
            f"{_model_count(source_count)} source tickets; complete source details "
            "are in the complete evidence export."
        ),
        "",
    ]


def _term_mapping_model_pdf_lines(row: Mapping[str, Any]) -> list[str]:
    mappings = [
        mapping
        for mapping in row.get("term_mappings") or ()
        if isinstance(mapping, Mapping)
    ]
    if not mappings:
        return []
    lines = ["**Vocabulary gaps:**", ""]
    for mapping in mappings:
        source_count = _model_count(_model_int(mapping.get("source_id_count")))
        lines.append(
            "- "
            f"{_model_text(mapping.get('customer_term'))} -> "
            f"{_model_text(mapping.get('documentation_term'))}: "
            f"{_model_text(mapping.get('suggestion'))} "
            f"({source_count} sources)"
        )
    lines.append("")
    return lines


def _render_pdf_intro(
    pdf: DeflectionReportPDF,
    *,
    title: str,
    toc_entries: list[tuple[int, str]],
) -> None:
    pdf.section_title(title, level=1)
    if not toc_entries:
        return
    pdf.section_title("Table of contents", level=2)
    for level, text in toc_entries:
        prefix = "  - " if level >= 3 else "- "
        pdf.body_text(f"{prefix}{text}")
    pdf.ln(2)


def _render_markdown(
    pdf: DeflectionReportPDF,
    markdown: str,
    *,
    title: str,
    render_missing_title: bool = True,
) -> None:
    lines = markdown.splitlines()
    if render_missing_title and not any(line.startswith("# ") for line in lines):
        pdf.section_title(title, level=1)
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            pdf.ln(1.5)
            continue
        heading_level = _heading_level(line)
        if heading_level:
            pdf.section_title(line[heading_level + 1 :].strip(), level=heading_level)
            continue
        if _is_table_rule(line):
            continue
        if line.startswith("|") and line.endswith("|"):
            pdf.table_row(_table_cells(line))
            continue
        if line.startswith("- "):
            pdf.body_text(f"* {_clean_inline(line[2:])}")
            continue
        numbered = re.match(r"^(\d+)\.\s+(.*)$", line)
        if numbered:
            pdf.body_text(f"{numbered.group(1)}. {_clean_inline(numbered.group(2))}")
            continue
        pdf.body_text(_clean_inline(line))


def _curate_markdown_for_pdf(markdown: str) -> str:
    out: list[str] = []
    section = ""
    ranked_rows = 0
    ranked_cap_note_written = False
    detail_count = 0
    detail_cap_note_written = False
    skip_question_detail = False
    skip_complete_evidence = False

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        heading_level = _heading_level(line)
        if skip_complete_evidence:
            if heading_level == 2 or _is_question_detail_heading(line):
                skip_complete_evidence = False
            else:
                continue

        if heading_level == 2:
            section = line[3:].strip()
            skip_question_detail = False
            ranked_rows = 0
            ranked_cap_note_written = False
        elif heading_level == 3 and section == "Question Details and Evidence":
            detail_count += 1
            skip_question_detail = detail_count > PDF_QUESTION_DETAIL_LIMIT
            if skip_question_detail:
                if not detail_cap_note_written:
                    out.extend(["", _QUESTION_DETAIL_CAP_NOTE, ""])
                    detail_cap_note_written = True
                continue

        if skip_question_detail:
            continue

        if section == "Ranked Question Opportunities" and _is_ranked_table_data_row(line):
            ranked_rows += 1
            if ranked_rows > PDF_RANKED_TABLE_LIMIT:
                continue

        if (
            section == "Ranked Question Opportunities"
            and ranked_rows > PDF_RANKED_TABLE_LIMIT
            and not ranked_cap_note_written
            and not line.startswith("|")
        ):
            out.extend(["", _RANKED_TABLE_CAP_NOTE, ""])
            ranked_cap_note_written = True

        if line == _COMPLETE_EVIDENCE_MARKER:
            out.extend([_COMPLETE_EVIDENCE_MARKER, "", _EVIDENCE_EXPORT_POINTER, ""])
            skip_complete_evidence = True
            continue

        out.append(_curate_source_backing_line(raw_line))

    if ranked_rows > PDF_RANKED_TABLE_LIMIT and not ranked_cap_note_written:
        out.extend(["", _RANKED_TABLE_CAP_NOTE, ""])
    return "\n".join(out).strip()


def _toc_entries(markdown: str) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        level = _heading_level(line)
        if level in {2, 3}:
            entries.append((level, _clean_inline(line[level + 1 :])))
    return entries


def _drop_first_title_heading(markdown: str) -> str:
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if line.strip().startswith("# "):
            return "\n".join(lines[:index] + lines[index + 1 :]).strip()
        return markdown.strip()
    return markdown.strip()


def _artifact_markdown(artifact: Mapping[str, Any]) -> str:
    markdown = artifact.get("markdown")
    return str(markdown or "").strip()


def _markdown_title(markdown: str) -> str:
    for line in markdown.splitlines():
        text = line.strip()
        if text.startswith("# "):
            return _clean_inline(text[2:])
    return ""


def _heading_level(line: str) -> int:
    match = re.match(r"^(#{1,6})\s+", line)
    return len(match.group(1)) if match else 0


def _is_table_rule(line: str) -> bool:
    if not (line.startswith("|") and line.endswith("|")):
        return False
    return all(set(cell.strip()) <= {"-", ":"} for cell in line.strip("|").split("|"))


def _is_ranked_table_data_row(line: str) -> bool:
    if not (line.startswith("|") and line.endswith("|")) or _is_table_rule(line):
        return False
    cells = _table_cells(line)
    if not cells:
        return False
    return cells[0].isdigit()


def _is_question_detail_heading(line: str) -> bool:
    return bool(re.match(r"^###\s+\d+\.\s+\S", line))


def _curate_source_backing_line(line: str) -> str:
    stripped = line.strip()
    if not (
        stripped.startswith("**Evidence backing:**")
        or stripped.startswith("**Ticket backing:**")
    ):
        return line
    curated = re.sub(
        r"(resolved tickets?|repeated tickets?)\s*\([^)]*\)",
        r"\1",
        line,
        count=1,
    )
    curated = curated.replace(
        "Complete source IDs are in this question detail block.",
        "Complete source details are in the complete evidence export.",
    )
    curated = curated.replace(
        "Complete source details are in this question detail block.",
        "Complete source details are in the complete evidence export.",
    )
    return curated


def _table_cells(line: str) -> list[str]:
    return [_clean_inline(cell.strip()) for cell in line.strip("|").split("|")]


def _clean_inline(text: Any) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = cleaned.replace("\\|", "|")
    return cleaned.strip()


def _section_data(section: Mapping[str, Any]) -> Mapping[str, Any]:
    data = section.get("data")
    return data if isinstance(data, Mapping) else {}


def _source_window_label(source_window: Mapping[str, Any]) -> str:
    start = _model_text(source_window.get("source_date_start"))
    end = _model_text(source_window.get("source_date_end"))
    if start and end:
        return f"{start} to {end}"
    return start or end


def _model_cell(value: Any) -> str:
    return _model_text(value).replace("|", "\\|")


def _model_text(value: Any) -> str:
    text = _clean_inline(value)
    return re.sub(r"\s+", " ", text).strip()


def _model_texts(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif _is_sequence(value):
        values = value
    else:
        values = (value,)
    return [
        text
        for item in values
        if (text := _model_text(item))
    ]


def _model_rows(value: Any) -> list[dict[str, Any]]:
    if not _is_sequence(value):
        return []
    return [
        dict(row)
        for row in value
        if isinstance(row, Mapping)
    ]


def _model_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _model_count(value: int) -> str:
    return f"{max(0, int(value)):,}"


def _model_money(value: Any) -> str:
    try:
        rounded = int(float(value) + 0.5)
    except (TypeError, ValueError):
        text = _model_text(value)
        return text if text else "$0"
    return f"${max(0, rounded):,}"


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _safe(text: Any) -> str:
    return (
        str(text or "")
        .translate(_UNICODE_MAP)
        .encode("latin-1", errors="replace")
        .decode("latin-1")
    )


__all__ = ["render_deflection_full_report_pdf"]
