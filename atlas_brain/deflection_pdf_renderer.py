"""PDF renderer for paid Content Ops deflection reports."""

from __future__ import annotations

import io
import re
from typing import Any, Mapping

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

    markdown = _artifact_markdown(artifact)
    if not markdown:
        raise ValueError("deflection report artifact markdown is required")
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


def _safe(text: Any) -> str:
    return (
        str(text or "")
        .translate(_UNICODE_MAP)
        .encode("latin-1", errors="replace")
        .decode("latin-1")
    )


__all__ = ["render_deflection_full_report_pdf"]
