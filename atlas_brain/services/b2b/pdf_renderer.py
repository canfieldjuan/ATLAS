"""PDF renderer for B2B intelligence reports.

Uses fpdf2 (pure Python) to generate downloadable PDF reports from
b2b_intelligence rows. Supports all report types with structured sections.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import date
from typing import Any

from fpdf import FPDF

from ..tracing import (
    build_business_trace_context,
    build_reasoning_trace_context,
    tracer,
)

logger = logging.getLogger("atlas.b2b.pdf_renderer")

# -- Brand colors (RGB tuples) ------------------------------------------------
_CLR_PRIMARY = (41, 128, 185)    # #2980b9  header / accent
_CLR_DARK = (44, 62, 80)         # #2c3e50  body text
_CLR_MUTED = (127, 140, 141)     # #7f8c8d  secondary text
_CLR_RED = (231, 76, 60)         # #e74c3c  critical
_CLR_ORANGE = (243, 156, 18)     # #f39c12  warning
_CLR_GREEN = (39, 174, 96)       # #27ae60  healthy
_CLR_BG_LIGHT = (236, 240, 241)  # #ecf0f1  section background
_CLR_WHITE = (255, 255, 255)


def _score_color(score: float) -> tuple[int, int, int]:
    if score >= 70:
        return _CLR_RED
    if score >= 40:
        return _CLR_ORANGE
    return _CLR_GREEN


_UNICODE_MAP = str.maketrans({
    "\u2014": "--",   # em-dash
    "\u2013": "-",    # en-dash
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
    "\u2026": "...",  # ellipsis
    "\u2022": "*",    # bullet
    "\u2122": "(TM)", # trademark
    "\u00ae": "(R)",  # registered
    "\u00a9": "(C)",  # copyright
    "\u200b": "",     # zero-width space
    "\u00a0": " ",    # non-breaking space
    "\ufeff": "",     # BOM
})


def _latin1_safe(text: str) -> str:
    """Replace Unicode characters that Helvetica (latin-1) cannot render."""
    text = text.translate(_UNICODE_MAP)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _safe_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return _latin1_safe(json.dumps(val, default=str))
    return _latin1_safe(str(val))


def _safe_list(val: Any) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


class IntelligenceReportPDF(FPDF):
    """Custom FPDF subclass with Atlas branding."""

    def __init__(self) -> None:
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self) -> None:
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*_CLR_PRIMARY)
        self.cell(0, 8, "Atlas Intelligence", align="L")
        self.set_text_color(*_CLR_MUTED)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 8, "churnsignals.co", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*_CLR_PRIMARY)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_CLR_MUTED)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # -- Helpers ---------------------------------------------------------------

    def section_title(self, title: str) -> None:
        self.ln(4)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*_CLR_PRIMARY)
        self.cell(0, 8, _latin1_safe(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*_CLR_PRIMARY)
        self.line(self.l_margin, self.get_y(), self.l_margin + 40, self.get_y())
        self.ln(3)

    def body_text(self, text: str) -> None:
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_CLR_DARK)
        self.multi_cell(0, 5, _latin1_safe(text))
        self.ln(2)

    def key_value(self, key: str, value: str) -> None:
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*_CLR_MUTED)
        self.cell(55, 5, _latin1_safe(key) + ":")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_CLR_DARK)
        self.cell(0, 5, _latin1_safe(value), new_x="LMARGIN", new_y="NEXT")

    def metric_row(self, label: str, value: str, color: tuple[int, int, int] | None = None) -> None:
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_CLR_DARK)
        self.cell(70, 5, _latin1_safe(label))
        self.set_font("Helvetica", "B", 9)
        if color:
            self.set_text_color(*color)
        self.cell(0, 5, _latin1_safe(value), new_x="LMARGIN", new_y="NEXT")

    def quote_block(self, text: str) -> None:
        self.set_fill_color(*_CLR_BG_LIGHT)
        self.set_draw_color(*_CLR_RED)
        x = self.get_x()
        y_start = self.get_y()
        # Render text first so we know the actual height
        self.set_x(x + 4)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*_CLR_DARK)
        w = self.w - self.r_margin - x - 4
        self.multi_cell(w, 4, _latin1_safe(f'"{text}"'))
        y_end = self.get_y()
        # Draw red left bar spanning the actual text height
        self.set_line_width(0.8)
        self.line(x, y_start, x, y_end)
        self.set_line_width(0.2)
        self.ln(2)

    def simple_table(self, headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None) -> None:
        if not rows:
            return
        usable = self.w - self.l_margin - self.r_margin
        if col_widths is None:
            col_widths = [usable / len(headers)] * len(headers)

        # Header row
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(*_CLR_PRIMARY)
        self.set_text_color(*_CLR_WHITE)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, _latin1_safe(h), border=1, fill=True)
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_CLR_DARK)
        for row_data in rows:
            for i, cell in enumerate(row_data):
                self.cell(col_widths[i], 5, _latin1_safe(cell[:60]), border=1)
            self.ln()
        self.ln(2)


# -- Render functions per report type -----------------------------------------

def _render_churn_feed(pdf: IntelligenceReportPDF, data: Any) -> None:
    """Render weekly_churn_feed report."""
    feed = data if isinstance(data, list) else data.get("weekly_churn_feed", []) if isinstance(data, dict) else []
    if not isinstance(feed, list):
        feed = []

    pdf.section_title("Weekly Churn Feed")
    pdf.body_text(f"{len(feed)} vendors analyzed this period.")

    # Summary table
    headers = ["Vendor", "Score", "Density", "Urgency", "Trend"]
    rows = []
    for entry in feed[:20]:
        if not isinstance(entry, dict):
            continue
        rows.append([
            _safe_str(entry.get("vendor") or entry.get("vendor_name"))[:30],
            f"{float(entry.get('churn_pressure_score', 0)):.0f}",
            f"{float(entry.get('churn_signal_density', 0)):.1f}%",
            f"{float(entry.get('avg_urgency', 0)):.1f}",
            _safe_str(entry.get("trend", "stable")),
        ])
    pdf.simple_table(headers, rows, [55, 25, 25, 25, 30])

    # Per-vendor detail (top 5)
    for entry in feed[:5]:
        if not isinstance(entry, dict):
            continue
        vendor = _safe_str(entry.get("vendor") or entry.get("vendor_name"))
        score = float(entry.get("churn_pressure_score", 0))

        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*_score_color(score))
        pdf.cell(0, 6, f"{vendor} (Score: {score:.0f})", new_x="LMARGIN", new_y="NEXT")

        # Pain breakdown
        pains = _safe_list(entry.get("pain_breakdown"))
        if pains:
            pain_rows = []
            for p in pains[:5]:
                if isinstance(p, dict):
                    pain_rows.append([
                        _safe_str(p.get("category", ""))[:30],
                        str(p.get("count", "")),
                    ])
            if pain_rows:
                pdf.simple_table(["Pain Category", "Count"], pain_rows, [80, 30])

        # Displacement targets
        targets = _safe_list(entry.get("top_displacement_targets"))
        if targets:
            target_rows = []
            for t in targets[:5]:
                if isinstance(t, dict):
                    target_rows.append([
                        _safe_str(t.get("competitor") or t.get("name", ""))[:30],
                        str(t.get("count") or t.get("mentions", 0)),
                    ])
            if target_rows:
                pdf.simple_table(["Competitor", "Mentions"], target_rows, [80, 30])

        # Evidence quotes
        evidence = _safe_list(entry.get("evidence"))
        for e in evidence[:2]:
            text = e.get("quote", e) if isinstance(e, dict) else _safe_str(e)
            if text:
                pdf.quote_block(text[:300])


def _render_vendor_scorecard(pdf: IntelligenceReportPDF, data: dict, exec_summary: str | None) -> None:
    """Render vendor_scorecard or vendor-specific reports."""
    if exec_summary:
        pdf.section_title("Executive Summary")
        pdf.body_text(exec_summary)

    # Key metrics
    pdf.section_title("Key Metrics")
    for key in ("churn_pressure_score", "churn_signal_density", "avg_urgency",
                "review_count", "dm_churn_rate", "trend"):
        val = data.get(key)
        if val is not None:
            color = None
            if key == "churn_pressure_score":
                color = _score_color(float(val))
            pdf.metric_row(key.replace("_", " ").title(), _safe_str(val), color)

    # Pain breakdown
    pains = _safe_list(data.get("pain_breakdown"))
    if pains:
        pdf.section_title("Pain Categories")
        rows = []
        for p in pains[:8]:
            if isinstance(p, dict):
                rows.append([_safe_str(p.get("category", ""))[:30], str(p.get("count", ""))])
        pdf.simple_table(["Category", "Count"], rows, [90, 30])

    # Displacement targets
    targets = _safe_list(data.get("top_displacement_targets"))
    if targets:
        pdf.section_title("Competitive Displacement")
        rows = []
        for t in targets[:8]:
            if isinstance(t, dict):
                rows.append([
                    _safe_str(t.get("competitor") or t.get("name", ""))[:30],
                    str(t.get("count") or t.get("mentions", 0)),
                ])
        pdf.simple_table(["Competitor", "Mentions"], rows, [90, 30])

    # Named accounts
    accounts = _safe_list(data.get("named_accounts"))
    if accounts:
        pdf.section_title("Accounts at Risk")
        rows = []
        for a in accounts[:10]:
            if isinstance(a, dict):
                urg = float(a.get("urgency", 0))
                risk = "Critical" if urg >= 8 else ("High" if urg >= 6 else "Watch")
                rows.append([
                    _safe_str(a.get("company", ""))[:30],
                    f"{urg:.1f}",
                    risk,
                ])
        pdf.simple_table(["Company", "Urgency", "Risk"], rows, [70, 25, 25])

    # Evidence
    evidence = _safe_list(data.get("evidence"))
    if evidence:
        pdf.section_title("Evidence")
        for e in evidence[:5]:
            text = e.get("quote", e) if isinstance(e, dict) else _safe_str(e)
            if text:
                pdf.quote_block(text[:300])

    # Feature gaps
    gaps = _safe_list(data.get("top_feature_gaps"))
    if gaps:
        pdf.section_title("Feature Gaps")
        for g in gaps[:5]:
            txt = g.get("feature", g) if isinstance(g, dict) else _safe_str(g)
            pdf.body_text(f"  - {txt}")


def _render_comparison(pdf: IntelligenceReportPDF, data: dict, exec_summary: str | None) -> None:
    """Render vendor_comparison or account_comparison reports."""
    if exec_summary:
        pdf.section_title("Executive Summary")
        pdf.body_text(exec_summary)

    # The comparison data usually has primary/comparison sub-dicts
    primary = data.get("primary") or data.get("primary_vendor") or {}
    comparison = data.get("comparison") or data.get("comparison_vendor") or {}

    if isinstance(primary, dict) and isinstance(comparison, dict):
        pdf.section_title("Side-by-Side Comparison")
        p_name = _safe_str(primary.get("vendor_name") or primary.get("company_name") or "Primary")
        c_name = _safe_str(comparison.get("vendor_name") or comparison.get("company_name") or "Comparison")

        headers = ["Metric", p_name[:20], c_name[:20]]
        metrics = ["churn_pressure_score", "churn_signal_density", "avg_urgency",
                    "review_count", "dm_churn_rate"]
        rows = []
        for m in metrics:
            pv = primary.get(m)
            cv = comparison.get(m)
            if pv is not None or cv is not None:
                rows.append([
                    m.replace("_", " ").title(),
                    _safe_str(pv),
                    _safe_str(cv),
                ])
        pdf.simple_table(headers, rows, [60, 45, 45])

    # Render each sub-section if present
    for label, sub in [("Primary", primary), ("Comparison", comparison)]:
        if not isinstance(sub, dict):
            continue
        name = _safe_str(sub.get("vendor_name") or sub.get("company_name") or label)

        pains = _safe_list(sub.get("pain_breakdown") or sub.get("top_pain_categories"))
        if pains:
            pdf.section_title(f"{name} - Pain Categories")
            rows = []
            for p in pains[:5]:
                if isinstance(p, dict):
                    rows.append([_safe_str(p.get("category", ""))[:30], str(p.get("count", ""))])
            pdf.simple_table(["Category", "Count"], rows, [90, 30])

    # Common fields at top level
    _render_generic_data(pdf, data, skip_keys={
        "primary", "comparison", "primary_vendor", "comparison_vendor",
    })


def _render_deep_dive(pdf: IntelligenceReportPDF, data: dict, exec_summary: str | None) -> None:
    """Render account_deep_dive reports."""
    if exec_summary:
        pdf.section_title("Executive Summary")
        pdf.body_text(exec_summary)

    company = data.get("company_name") or data.get("account_name") or ""
    if company:
        pdf.section_title(f"Account: {company}")

    # Review summary
    review_data = data.get("review_summary") or data.get("reviews") or {}
    if isinstance(review_data, dict):
        pdf.section_title("Review Summary")
        for k, v in review_data.items():
            if isinstance(v, (str, int, float)):
                pdf.key_value(k.replace("_", " ").title(), _safe_str(v))

    _render_vendor_scorecard(pdf, data, exec_summary=None)


def _render_generic_data(pdf: IntelligenceReportPDF, data: dict, skip_keys: set | None = None) -> None:
    """Render arbitrary intelligence_data as key-value pairs and nested tables."""
    skip = skip_keys or set()
    for key, val in data.items():
        if key in skip or key.startswith("_"):
            continue
        if isinstance(val, (str, int, float, bool)):
            pdf.key_value(key.replace("_", " ").title(), _safe_str(val))
        elif isinstance(val, list) and val:
            pdf.section_title(key.replace("_", " ").title())
            if isinstance(val[0], dict):
                headers = list(val[0].keys())[:5]
                rows = []
                for item in val[:15]:
                    if isinstance(item, dict):
                        rows.append([_safe_str(item.get(h, ""))[:40] for h in headers])
                pdf.simple_table(headers, rows)
            else:
                for item in val[:10]:
                    pdf.body_text(f"  - {_safe_str(item)[:200]}")


# -- Public API ----------------------------------------------------------------

_RENDERERS: dict[str, Any] = {
    "weekly_churn_feed": lambda pdf, d, s: _render_churn_feed(pdf, d),
    "vendor_scorecard": _render_vendor_scorecard,
    "displacement_report": _render_vendor_scorecard,
    "category_overview": _render_vendor_scorecard,
    "exploratory_overview": _render_vendor_scorecard,
    "vendor_comparison": _render_comparison,
    "account_comparison": _render_comparison,
    "account_deep_dive": _render_deep_dive,
    "vendor_retention": _render_vendor_scorecard,
    "challenger_intel": _render_vendor_scorecard,
}


def render_report_pdf(
    *,
    report_type: str,
    vendor_filter: str | None = None,
    category_filter: str | None = None,
    report_date: date | str | None = None,
    executive_summary: str | None = None,
    intelligence_data: dict | list | None = None,
    data_density: dict | None = None,
) -> bytes:
    """Render a b2b_intelligence report row to PDF bytes.

    Parameters match the columns of the ``b2b_intelligence`` table.
    Returns raw PDF bytes suitable for streaming via FastAPI.
    """
    span = tracer.start_span(
        span_name="b2b.report.export_pdf",
        operation_type="business_operation",
        metadata={
            "business": build_business_trace_context(
                workflow="report_export",
                report_type=report_type,
                vendor_name=vendor_filter,
            ),
        },
    )
    pdf = IntelligenceReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # -- Title block -----------------------------------------------------------
    date_str = str(report_date) if report_date else date.today().isoformat()
    title = report_type.replace("_", " ").title()
    if vendor_filter:
        title += f": {vendor_filter}"
    elif category_filter:
        title += f": {category_filter}"

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*_CLR_DARK)
    pdf.cell(0, 10, _latin1_safe(title), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*_CLR_MUTED)
    pdf.cell(0, 5, f"Report Date: {date_str}", new_x="LMARGIN", new_y="NEXT")

    type_label = report_type.replace("_", " ").title()
    pdf.cell(0, 5, f"Report Type: {type_label}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # -- Data density sidebar --------------------------------------------------
    if data_density and isinstance(data_density, dict):
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(*_CLR_MUTED)
        parts = []
        for k, v in list(data_density.items())[:4]:
            parts.append(f"{k}: {v}")
        if parts:
            pdf.cell(0, 4, _latin1_safe("Data: " + " | ".join(parts)), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

    # -- Main content via type-specific renderer --------------------------------
    data = intelligence_data if intelligence_data is not None else {}

    renderer = _RENDERERS.get(report_type)
    if renderer:
        try:
            renderer(pdf, data, executive_summary)
        except Exception:
            logger.exception("Renderer failed for %s, falling back to generic", report_type)
            if executive_summary:
                pdf.section_title("Executive Summary")
                pdf.body_text(executive_summary)
            if isinstance(data, dict):
                _render_generic_data(pdf, data)
    else:
        # Generic fallback
        if executive_summary:
            pdf.section_title("Executive Summary")
            pdf.body_text(executive_summary)
        if isinstance(data, dict):
            _render_generic_data(pdf, data)
        elif isinstance(data, list):
            _render_churn_feed(pdf, data)

    # -- Generate bytes --------------------------------------------------------
    buf = io.BytesIO()
    pdf.output(buf)
    pdf_bytes = buf.getvalue()
    tracer.end_span(
        span,
        status="completed",
        output_data={"size_bytes": len(pdf_bytes)},
        metadata={
            "reasoning": build_reasoning_trace_context(
                decision={"report_type": report_type},
                evidence={
                    "vendor_filter": vendor_filter,
                    "category_filter": category_filter,
                    "data_density_keys": list((data_density or {}).keys())[:10],
                },
                rationale=executive_summary,
            ),
        },
    )
    return pdf_bytes
