"""PDF renderer for consumer competitive intelligence exports.

Uses fpdf2 (pure Python) to generate intelligence report PDFs from
market_intelligence_reports and brand_intelligence data. Part of the
delivery surface layer -- the intelligence is in the data, not the format.
"""

from __future__ import annotations

import io
import json
import logging
import re
from datetime import date
from typing import Any

from fpdf import FPDF

logger = logging.getLogger("atlas.services.consumer_pdf_renderer")

# -- Brand colors (RGB tuples) ------------------------------------------------
_CLR_PRIMARY = (41, 128, 185)    # #2980b9  header / accent
_CLR_DARK = (44, 62, 80)         # #2c3e50  body text
_CLR_MUTED = (127, 140, 141)     # #7f8c8d  secondary text
_CLR_RED = (231, 76, 60)         # #e74c3c  critical
_CLR_ORANGE = (243, 156, 18)     # #f39c12  warning
_CLR_GREEN = (39, 174, 96)       # #27ae60  healthy
_CLR_BG_LIGHT = (236, 240, 241)  # #ecf0f1  section background
_CLR_WHITE = (255, 255, 255)

_UNICODE_MAP = str.maketrans({
    "\u2014": "--",
    "\u2013": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u2022": "*",
    "\u2122": "(TM)",
    "\u00ae": "(R)",
    "\u00a9": "(C)",
    "\u200b": "",
    "\u00a0": " ",
    "\ufeff": "",
})


def _latin1_safe(text: str) -> str:
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


def _health_color(score: float) -> tuple[int, int, int]:
    """Higher health = better (green), lower = worse (red)."""
    if score >= 70:
        return _CLR_GREEN
    if score >= 40:
        return _CLR_ORANGE
    return _CLR_RED


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name.strip())[:80]


class ConsumerReportPDF(FPDF):
    """Custom FPDF subclass with Atlas branding for consumer reports."""

    def __init__(self) -> None:
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self) -> None:
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*_CLR_PRIMARY)
        self.cell(0, 8, "Atlas Consumer Intelligence", align="L")
        self.set_text_color(*_CLR_MUTED)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 8, "Competitive Displacement Intelligence", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*_CLR_PRIMARY)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_CLR_MUTED)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

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

    def simple_table(self, headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None) -> None:
        if not rows:
            return
        usable = self.w - self.l_margin - self.r_margin
        if col_widths is None:
            col_widths = [usable / len(headers)] * len(headers)

        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(*_CLR_PRIMARY)
        self.set_text_color(*_CLR_WHITE)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, _latin1_safe(h), border=1, fill=True)
        self.ln()

        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_CLR_DARK)
        for row_data in rows:
            for i, cell in enumerate(row_data):
                self.cell(col_widths[i], 5, _latin1_safe(cell[:60]), border=1)
            self.ln()
        self.ln(2)


def render_market_report_pdf(report_row: dict) -> tuple[bytes, str]:
    """Render a market_intelligence_reports row as PDF.

    Returns (pdf_bytes, filename).
    """
    pdf = ConsumerReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    report_date = report_row.get("report_date", date.today())
    report_type = report_row.get("report_type", "competitive_intelligence")

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*_CLR_DARK)
    pdf.cell(0, 10, _latin1_safe(f"Market Intelligence Report"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_CLR_MUTED)
    pdf.cell(0, 6, f"Date: {report_date}  |  Type: {report_type}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Executive summary
    exec_summary = report_row.get("analysis_text", "")
    if exec_summary:
        pdf.section_title("Executive Summary")
        pdf.body_text(exec_summary[:3000])

    # Key metrics
    data = report_row.get("report_data")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            data = {}
    if not isinstance(data, dict):
        data = {}

    # Brands overview table
    brands = _safe_list(data.get("brand_scorecards") or data.get("brands"))
    if brands:
        pdf.section_title("Brand Health Overview")
        headers = ["Brand", "Health", "Rating", "Reviews", "Pain Score"]
        rows = []
        for b in brands[:20]:
            if not isinstance(b, dict):
                continue
            rows.append([
                _safe_str(b.get("brand", ""))[:30],
                f"{float(b.get('health_score', 0)):.0f}",
                f"{float(b.get('avg_rating', 0)):.2f}",
                str(b.get("total_reviews", 0)),
                f"{float(b.get('avg_pain_score', 0)):.1f}",
            ])
        pdf.simple_table(headers, rows, [50, 25, 25, 30, 30])

    # Insights
    insights = _safe_list(data.get("insights"))
    if insights:
        pdf.section_title("Key Insights")
        for i, insight in enumerate(insights[:10], 1):
            text = insight if isinstance(insight, str) else _safe_str(insight.get("text", insight))
            pdf.body_text(f"{i}. {text[:500]}")

    # Competitive flows
    flows = _safe_list(data.get("competitive_flows") or data.get("displacement_flows"))
    if flows:
        pdf.section_title("Competitive Flows")
        headers = ["From", "To", "Direction", "Mentions"]
        rows = []
        for f in flows[:15]:
            if not isinstance(f, dict):
                continue
            rows.append([
                _safe_str(f.get("from_brand", f.get("from", "")))[:25],
                _safe_str(f.get("to_brand", f.get("to", "")))[:25],
                _safe_str(f.get("direction", "compared"))[:15],
                str(f.get("mention_count", f.get("count", 0))),
            ])
        pdf.simple_table(headers, rows, [45, 45, 35, 35])

    # Feature gaps
    features = _safe_list(data.get("feature_gaps") or data.get("top_feature_requests"))
    if features:
        pdf.section_title("Top Feature Requests")
        headers = ["Feature", "Mentions", "Category"]
        rows = []
        for f in features[:15]:
            if not isinstance(f, dict):
                continue
            rows.append([
                _safe_str(f.get("feature", f.get("name", "")))[:40],
                str(f.get("count", f.get("mentions", 0))),
                _safe_str(f.get("category", ""))[:20],
            ])
        pdf.simple_table(headers, rows, [60, 30, 40])

    # Recommendations
    recs = _safe_list(data.get("recommendations"))
    if recs:
        pdf.section_title("Recommendations")
        for i, rec in enumerate(recs[:10], 1):
            text = rec if isinstance(rec, str) else _safe_str(rec)
            pdf.body_text(f"{i}. {text[:500]}")

    filename = f"atlas-consumer-report-{_sanitize_filename(str(report_date))}.pdf"
    return bytes(pdf.output()), filename


def render_brand_report_pdf(brand_row: dict, snapshots: list[dict] | None = None) -> tuple[bytes, str]:
    """Render a brand_intelligence row as a single-brand PDF report.

    Returns (pdf_bytes, filename).
    """
    pdf = ConsumerReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    brand = brand_row.get("brand", "Unknown")

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*_CLR_DARK)
    pdf.cell(0, 10, _latin1_safe(f"Brand Intelligence: {brand}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_CLR_MUTED)
    last_computed = brand_row.get("last_computed_at", "")
    pdf.cell(0, 6, f"Last computed: {str(last_computed)[:19]}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Key Metrics
    pdf.section_title("Key Metrics")
    health = float(brand_row.get("health_score", 0) or 0)
    pdf.metric_row("Health Score", f"{health:.0f} / 100", _health_color(health))
    pdf.metric_row("Average Rating", f"{float(brand_row.get('avg_rating', 0) or 0):.2f}")
    pdf.metric_row("Total Reviews", str(brand_row.get("total_reviews", 0)))
    pdf.metric_row("Average Pain Score", f"{float(brand_row.get('avg_pain_score', 0) or 0):.1f}")

    rep_yes = brand_row.get("repurchase_yes", 0) or 0
    rep_no = brand_row.get("repurchase_no", 0) or 0
    rep_total = rep_yes + rep_no
    if rep_total > 0:
        rep_rate = rep_yes / rep_total * 100
        pdf.metric_row("Repurchase Rate", f"{rep_rate:.0f}% ({rep_yes}/{rep_total})")

    confidence = brand_row.get("confidence_score")
    if confidence is not None:
        pdf.metric_row("Confidence Score", f"{float(confidence):.2f}")

    # Top complaints
    complaints = _safe_list(brand_row.get("top_complaints"))
    if complaints:
        pdf.section_title("Top Complaints")
        headers = ["Complaint", "Count"]
        rows = []
        for c in complaints[:10]:
            if isinstance(c, dict):
                rows.append([
                    _safe_str(c.get("complaint", c.get("label", "")))[:50],
                    str(c.get("count", c.get("value", 0))),
                ])
            elif isinstance(c, str):
                rows.append([c[:50], ""])
        pdf.simple_table(headers, rows, [100, 30])

    # Feature requests
    features = _safe_list(brand_row.get("top_feature_requests"))
    if features:
        pdf.section_title("Top Feature Requests")
        headers = ["Feature", "Count"]
        rows = []
        for f in features[:10]:
            if isinstance(f, dict):
                rows.append([
                    _safe_str(f.get("feature", f.get("label", "")))[:50],
                    str(f.get("count", f.get("value", 0))),
                ])
        pdf.simple_table(headers, rows, [100, 30])

    # Competitive flows
    flows = _safe_list(brand_row.get("competitive_flows"))
    if flows:
        pdf.section_title("Competitive Flows")
        headers = ["Competitor", "Direction", "Mentions"]
        rows = []
        for f in flows[:10]:
            if isinstance(f, dict):
                rows.append([
                    _safe_str(f.get("competitor", f.get("brand", "")))[:30],
                    _safe_str(f.get("direction", "compared"))[:15],
                    str(f.get("count", f.get("mentions", 0))),
                ])
        pdf.simple_table(headers, rows, [60, 40, 30])

    # Snapshot trend (if provided)
    if snapshots:
        pdf.section_title(f"Health Trend ({len(snapshots)} days)")
        headers = ["Date", "Health", "Rating", "Pain", "Reviews"]
        rows = []
        for s in snapshots[-15:]:
            rows.append([
                str(s.get("snapshot_date", ""))[:10],
                f"{float(s.get('health_score', 0) or 0):.0f}",
                f"{float(s.get('avg_rating', 0) or 0):.2f}",
                f"{float(s.get('avg_pain_score', 0) or 0):.1f}",
                str(s.get("total_reviews", 0)),
            ])
        pdf.simple_table(headers, rows, [35, 25, 25, 25, 30])

    filename = f"atlas-brand-{_sanitize_filename(brand)}.pdf"
    return bytes(pdf.output()), filename
