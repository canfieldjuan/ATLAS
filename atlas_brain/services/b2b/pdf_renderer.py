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
        self.cell(0, 8, "Churn Signals Intelligence", align="L")
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
    headers = ["Vendor", "Score", "Risk", "Density", "Urgency", "Trend"]
    rows = []
    for entry in feed[:20]:
        if not isinstance(entry, dict):
            continue
        rows.append([
            _safe_str(entry.get("vendor") or entry.get("vendor_name"))[:25],
            f"{float(entry.get('churn_pressure_score', 0)):.0f}",
            _safe_str(entry.get("risk_level", ""))[:6].upper(),
            f"{float(entry.get('churn_signal_density', 0)):.1f}%",
            f"{float(entry.get('avg_urgency', 0)):.1f}",
            _safe_str(entry.get("trend", "stable")),
        ])
    pdf.simple_table(headers, rows, [42, 20, 22, 22, 22, 22])

    # Per-vendor detail (top 5)
    for entry in feed[:5]:
        if not isinstance(entry, dict):
            continue
        vendor = _safe_str(entry.get("vendor") or entry.get("vendor_name"))
        score = float(entry.get("churn_pressure_score", 0))

        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*_score_color(score))
        risk = _safe_str(entry.get("risk_level", ""))
        risk_label = f" | Risk: {risk.upper()}" if risk else ""
        pdf.cell(0, 6, f"{vendor} (Score: {score:.0f}{risk_label})", new_x="LMARGIN", new_y="NEXT")

        # Affected segments
        segments = entry.get("affected_segments") or {}
        if isinstance(segments, dict):
            seg_industries = _safe_list(segments.get("industries"))
            seg_sizes = _safe_list(segments.get("company_sizes"))
            if seg_industries:
                ind_rows = [
                    [_safe_str(s.get("industry", ""))[:25], str(s.get("count", 0))]
                    for s in seg_industries[:3] if isinstance(s, dict)
                ]
                if ind_rows:
                    pdf.simple_table(["Affected Industry", "Count"], ind_rows, [80, 30])
            if seg_sizes:
                sz_rows = [
                    [_safe_str(s.get("size", ""))[:25], str(s.get("count", 0))]
                    for s in seg_sizes[:3] if isinstance(s, dict)
                ]
                if sz_rows:
                    pdf.simple_table(["Company Size", "Count"], sz_rows, [80, 30])

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
    """Render vendor_comparison or account_comparison reports (Competitive Benchmark)."""
    if exec_summary:
        pdf.section_title("Executive Summary")
        pdf.body_text(exec_summary)

    # The comparison data has primary_metrics/comparison_metrics or primary/comparison sub-dicts
    primary = data.get("primary_metrics") or data.get("primary") or data.get("primary_vendor") or {}
    comparison = data.get("comparison_metrics") or data.get("comparison") or data.get("comparison_vendor") or {}

    p_name = _safe_str(
        data.get("primary_vendor")
        or (primary.get("vendor_name") if isinstance(primary, dict) else "")
        or "Primary"
    )
    c_name = _safe_str(
        data.get("comparison_vendor")
        or (comparison.get("vendor_name") if isinstance(comparison, dict) else "")
        or "Comparison"
    )

    if isinstance(primary, dict) and isinstance(comparison, dict):
        pdf.section_title("Side-by-Side Comparison")

        headers = ["Metric", p_name[:20], c_name[:20]]
        metrics = [
            ("signal_count", "Reviews Analyzed"),
            ("churn_signal_density", "Signal Density (%)"),
            ("avg_urgency_score", "Avg Urgency"),
            ("positive_review_pct", "Positive Review (%)"),
            ("recommend_ratio", "Recommend Ratio"),
            ("churn_intent_count", "Churn Intent Count"),
        ]
        rows = []
        for key, label in metrics:
            pv = primary.get(key)
            cv = comparison.get(key)
            if pv is not None or cv is not None:
                rows.append([label, _safe_str(pv), _safe_str(cv)])
        pdf.simple_table(headers, rows, [60, 45, 45])

    # Pain categories per vendor
    for vendor_label, prefix in [(p_name, "primary"), (c_name, "comparison")]:
        pains = _safe_list(data.get(f"{prefix}_top_pains"))
        if pains:
            pdf.section_title(f"{vendor_label} - Pain Categories")
            rows = [
                [_safe_str(p.get("category", ""))[:30], str(p.get("count", ""))]
                for p in pains[:5] if isinstance(p, dict)
            ]
            if rows:
                pdf.simple_table(["Category", "Count"], rows, [90, 30])

    # Strengths/Weaknesses (from product profiles)
    for vendor_label, prefix in [(p_name, "primary"), (c_name, "comparison")]:
        strengths = _safe_list(data.get(f"{prefix}_strengths"))
        weaknesses = _safe_list(data.get(f"{prefix}_weaknesses"))
        if strengths or weaknesses:
            pdf.section_title(f"{vendor_label} - Strengths & Weaknesses")
            if strengths:
                rows = [
                    [_safe_str(s.get("area", ""))[:35],
                     _safe_str(s.get("score", "-"))]
                    for s in strengths[:5] if isinstance(s, dict)
                ]
                if rows:
                    pdf.simple_table(["Strength", "Score"], rows, [100, 30])
            if weaknesses:
                rows = [
                    [_safe_str(w.get("area", ""))[:35],
                     _safe_str(w.get("score", "-"))]
                    for w in weaknesses[:5] if isinstance(w, dict)
                ]
                if rows:
                    pdf.simple_table(["Weakness", "Score"], rows, [100, 30])

    # Switching triggers
    for vendor_label, prefix in [(p_name, "primary"), (c_name, "comparison")]:
        triggers = _safe_list(data.get(f"{prefix}_switching_triggers"))
        if triggers:
            pdf.section_title(f"{vendor_label} - Switching Triggers")
            rows = [
                [_safe_str(t.get("competitor", ""))[:25],
                 _safe_str(t.get("primary_reason", ""))[:25],
                 str(t.get("mention_count", 0))]
                for t in triggers[:5] if isinstance(t, dict)
            ]
            if rows:
                pdf.simple_table(["To Competitor", "Reason", "Mentions"],
                                 rows, [50, 60, 30])

    # Trend analysis
    trend = data.get("trend_analysis")
    if isinstance(trend, dict):
        pdf.section_title("Trend Analysis vs Prior Report")
        prior_date = _safe_str(trend.get("prior_report_date", ""))
        if prior_date:
            pdf.body_text(f"Compared to report from {prior_date}")
        for vendor_label, prefix in [(p_name, "primary"), (c_name, "comparison")]:
            density_change = trend.get(f"{prefix}_churn_density_change")
            urgency_change = trend.get(f"{prefix}_urgency_change")
            if density_change is not None:
                direction = "+" if density_change > 0 else ""
                pdf.metric_row(f"{vendor_label} Density Change",
                               f"{direction}{density_change}pp")
            if urgency_change is not None:
                direction = "+" if urgency_change > 0 else ""
                pdf.metric_row(f"{vendor_label} Urgency Change",
                               f"{direction}{urgency_change}")

    # Quote highlights
    for vendor_label, prefix in [(p_name, "primary"), (c_name, "comparison")]:
        quotes = _safe_list(data.get(f"{prefix}_quote_highlights"))
        if quotes:
            pdf.section_title(f"{vendor_label} - Customer Quotes")
            for q in quotes[:3]:
                text = q if isinstance(q, str) else _safe_str(q)
                if text:
                    pdf.quote_block(text[:300])


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


def _render_vendor_deep_dive(
    pdf: IntelligenceReportPDF, data: dict, exec_summary: str | None,
) -> None:
    """Render enriched vendor deep dive (vendor_scorecard) report."""
    if exec_summary:
        pdf.section_title("Executive Summary")
        pdf.body_text(exec_summary)

    # Handle both list-wrapped and direct dict data
    items = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
    if not items:
        return

    for entry in items[:5]:
        if not isinstance(entry, dict):
            continue
        vendor = _safe_str(entry.get("vendor", "Unknown"))
        score = float(entry.get("churn_pressure_score", 0))
        risk = _safe_str(entry.get("risk_level", ""))

        # Vendor header
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(*_score_color(score))
        risk_label = f" [{risk.upper()}]" if risk else ""
        pdf.cell(0, 8, _latin1_safe(f"{vendor}{risk_label}"),
                 new_x="LMARGIN", new_y="NEXT")

        # Key metrics
        pdf.section_title("Key Metrics")
        for metric_key, label in [
            ("churn_pressure_score", "Churn Pressure Score"),
            ("churn_signal_density", "Signal Density (%)"),
            ("avg_urgency", "Average Urgency"),
            ("dm_churn_rate", "DM Churn Rate"),
            ("price_complaint_rate", "Price Complaint Rate"),
            ("recommend_ratio", "Recommend Ratio"),
            ("trend", "Trend"),
            ("sentiment_direction", "Sentiment Direction"),
        ]:
            val = entry.get(metric_key)
            if val is not None:
                color = _score_color(float(val)) if metric_key == "churn_pressure_score" else None
                pdf.metric_row(label, _safe_str(val), color)

        # Customer profile
        profile = entry.get("customer_profile")
        if isinstance(profile, dict):
            has_content = any(profile.get(k) for k in (
                "typical_industries", "typical_company_size",
                "primary_use_cases", "top_integrations",
            ))
            if has_content:
                pdf.section_title("Customer Profile")
                for field, label in [
                    ("typical_industries", "Industries"),
                    ("typical_company_size", "Company Sizes"),
                    ("primary_use_cases", "Use Cases"),
                    ("top_integrations", "Integrations"),
                ]:
                    items_list = profile.get(field) or []
                    if items_list and isinstance(items_list, list):
                        text = ", ".join(str(i)[:30] for i in items_list[:5])
                        pdf.key_value(label, text)

        # Feature analysis
        features = entry.get("feature_analysis")
        if isinstance(features, dict):
            loved = _safe_list(features.get("loved"))
            hated = _safe_list(features.get("hated"))
            if loved:
                pdf.section_title("Strengths")
                rows = [
                    [_safe_str(f.get("feature", ""))[:35],
                     _safe_str(f.get("score", "-"))]
                    for f in loved[:5] if isinstance(f, dict)
                ]
                if rows:
                    pdf.simple_table(["Feature", "Score"], rows, [100, 30])
            if hated:
                pdf.section_title("Feature Gaps")
                rows = [
                    [_safe_str(f.get("feature", ""))[:35],
                     str(f.get("mentions", 0))]
                    for f in hated[:5] if isinstance(f, dict)
                ]
                if rows:
                    pdf.simple_table(["Feature", "Mentions"], rows, [100, 30])

        # Churn predictors
        predictors = entry.get("churn_predictors")
        if isinstance(predictors, dict):
            high_ind = _safe_list(predictors.get("high_risk_industries"))
            high_sz = _safe_list(predictors.get("high_risk_sizes"))
            if high_ind or high_sz:
                pdf.section_title("Churn Predictors")
                if high_ind:
                    rows = [
                        [_safe_str(i.get("industry", ""))[:30], str(i.get("count", 0))]
                        for i in high_ind[:3] if isinstance(i, dict)
                    ]
                    if rows:
                        pdf.simple_table(["High-Risk Industry", "Count"], rows, [80, 30])
                if high_sz:
                    rows = [
                        [_safe_str(s.get("size", ""))[:30], str(s.get("count", 0))]
                        for s in high_sz[:3] if isinstance(s, dict)
                    ]
                    if rows:
                        pdf.simple_table(["High-Risk Size", "Count"], rows, [80, 30])

        # Competitor overlap
        overlap = _safe_list(entry.get("competitor_overlap"))
        if overlap:
            pdf.section_title("Competitor Overlap")
            rows = [
                [_safe_str(c.get("competitor", ""))[:30], str(c.get("mentions", 0))]
                for c in overlap[:5] if isinstance(c, dict)
            ]
            if rows:
                pdf.simple_table(["Competitor", "Mentions"], rows, [90, 30])

        # Accounts at risk
        accounts = _safe_list(entry.get("named_accounts"))
        if accounts:
            pdf.section_title("Accounts at Risk")
            rows = []
            for a in accounts[:10]:
                if isinstance(a, dict):
                    urg = float(a.get("urgency", 0))
                    risk_label = "Critical" if urg >= 8 else ("High" if urg >= 6 else "Watch")
                    rows.append([
                        _safe_str(a.get("company", ""))[:25],
                        f"{urg:.1f}",
                        risk_label,
                        _safe_str(a.get("industry") or "")[:15],
                    ])
            if rows:
                pdf.simple_table(["Company", "Urgency", "Risk", "Industry"],
                                 rows, [45, 25, 25, 45])

        # Evidence
        evidence = _safe_list(entry.get("evidence"))
        if evidence:
            pdf.section_title("Evidence")
            for e in evidence[:3]:
                text = e.get("quote", e) if isinstance(e, dict) else _safe_str(e)
                if text:
                    pdf.quote_block(str(text)[:300])

        # Expert take (LLM-generated)
        expert = entry.get("expert_take")
        if expert:
            pdf.section_title("Expert Take")
            pdf.body_text(str(expert))


def _render_category_report(
    pdf: IntelligenceReportPDF, data: dict, exec_summary: str | None,
) -> None:
    """Render enriched industry-specific category overview report."""
    if exec_summary:
        pdf.section_title("Executive Summary")
        pdf.body_text(exec_summary)

    categories = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
    if not categories:
        return

    for cat in categories[:12]:
        if not isinstance(cat, dict):
            continue

        category = _safe_str(cat.get("category", "Unknown"))
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*_CLR_PRIMARY)
        pdf.cell(0, 8, _latin1_safe(f"Category: {category}"),
                 new_x="LMARGIN", new_y="NEXT")

        # Key info
        pdf.key_value("Highest Churn Risk", _safe_str(cat.get("highest_churn_risk", "")))
        pdf.key_value("Emerging Challenger", _safe_str(cat.get("emerging_challenger", "")))
        pdf.key_value("Dominant Pain", _safe_str(cat.get("dominant_pain", "")))

        # Market signal
        signal = cat.get("market_shift_signal")
        if signal:
            pdf.ln(2)
            pdf.body_text(str(signal))

        # Vendor rankings
        rankings = _safe_list(cat.get("vendor_rankings"))
        if rankings:
            pdf.section_title("Vendor Rankings")
            rows = []
            for r in rankings[:8]:
                if isinstance(r, dict):
                    rows.append([
                        _safe_str(r.get("vendor", ""))[:25],
                        f"{float(r.get('churn_pressure_score', 0)):.0f}",
                        f"{float(r.get('churn_signal_density', 0)):.1f}%",
                        _safe_str(r.get("risk_level", ""))[:6].upper(),
                    ])
            if rows:
                pdf.simple_table(["Vendor", "Score", "Density", "Risk"],
                                 rows, [50, 25, 30, 25])

        # Feature gaps
        gaps = _safe_list(cat.get("top_feature_gaps"))
        if gaps:
            pdf.section_title("Top Feature Gaps")
            rows = [
                [_safe_str(g.get("feature", ""))[:35], str(g.get("mentions", 0))]
                for g in gaps[:5] if isinstance(g, dict)
            ]
            if rows:
                pdf.simple_table(["Feature", "Mentions"], rows, [100, 30])

        # Case studies
        cases = _safe_list(cat.get("case_studies"))
        if cases:
            pdf.section_title("Case Studies")
            for cs in cases[:3]:
                if isinstance(cs, dict):
                    label_parts = []
                    if cs.get("vendor"):
                        label_parts.append(f"Vendor: {cs['vendor']}")
                    if cs.get("company") and cs["company"] != "Anonymous":
                        label_parts.append(cs["company"])
                    if cs.get("title"):
                        label_parts.append(cs["title"])
                    if label_parts:
                        pdf.set_font("Helvetica", "", 7)
                        pdf.set_text_color(*_CLR_MUTED)
                        pdf.cell(0, 4, _latin1_safe(" | ".join(label_parts)),
                                 new_x="LMARGIN", new_y="NEXT")
                    quote = cs.get("quote", "")
                    if quote:
                        pdf.quote_block(str(quote)[:300])

        # Segment distribution
        ind_dist = _safe_list(cat.get("industry_distribution"))
        size_dist = _safe_list(cat.get("company_size_distribution"))
        if ind_dist or size_dist:
            pdf.section_title("Segment Distribution")
            if ind_dist:
                rows = [
                    [_safe_str(i.get("industry", ""))[:25], str(i.get("count", 0))]
                    for i in ind_dist[:5] if isinstance(i, dict)
                ]
                if rows:
                    pdf.simple_table(["Industry", "Count"], rows, [80, 30])
            if size_dist:
                rows = [
                    [_safe_str(s.get("size", ""))[:25], str(s.get("count", 0))]
                    for s in size_dist[:5] if isinstance(s, dict)
                ]
                if rows:
                    pdf.simple_table(["Company Size", "Count"], rows, [80, 30])


def _render_battle_card(
    pdf: IntelligenceReportPDF, data: Any, exec_summary: str | None,
) -> None:
    """Render a per-vendor battle card report."""
    card = data if isinstance(data, dict) else {}

    if exec_summary:
        pdf.section_title("Executive Summary")
        pdf.body_text(exec_summary)

    vendor = _safe_str(card.get("vendor", "Unknown"))
    score = float(card.get("churn_pressure_score", 0))
    confidence = _safe_str(card.get("confidence", ""))
    total_reviews = card.get("total_reviews", 0)

    # Vendor header with score
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*_score_color(score))
    pdf.cell(0, 8, _latin1_safe(f"Battle Card: {vendor}"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*_CLR_DARK)
    pdf.cell(0, 5, _latin1_safe(
        f"Churn Pressure Score: {score:.0f}/100 | "
        f"Reviews: {total_reviews} | Confidence: {confidence}"
    ), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Section 1: Vendor Weaknesses
    weaknesses = _safe_list(card.get("vendor_weaknesses"))
    if weaknesses:
        pdf.section_title("Vendor Weaknesses")
        rows = []
        for w in weaknesses[:5]:
            if isinstance(w, dict):
                evidence = str(w.get("evidence_count") or w.get("count", ""))
                rows.append([
                    _safe_str(w.get("area", ""))[:35],
                    evidence,
                    _safe_str(w.get("source", ""))[:20],
                ])
        if rows:
            pdf.simple_table(["Weakness", "Evidence", "Source"], rows, [70, 35, 45])

    # Section 2: Customer Pain Quotes
    quotes = _safe_list(card.get("customer_pain_quotes"))
    if quotes:
        pdf.section_title("Customer Pain Points")
        for q in quotes[:4]:
            text = q.get("quote", q) if isinstance(q, dict) else _safe_str(q)
            if text:
                label_parts = []
                if isinstance(q, dict):
                    if q.get("title"):
                        label_parts.append(q["title"])
                    if q.get("industry"):
                        label_parts.append(q["industry"])
                    if q.get("urgency"):
                        label_parts.append(f"urgency: {q['urgency']}")
                if label_parts:
                    pdf.set_font("Helvetica", "", 7)
                    pdf.set_text_color(*_CLR_MUTED)
                    pdf.cell(0, 4, _latin1_safe(" | ".join(label_parts)),
                             new_x="LMARGIN", new_y="NEXT")
                pdf.quote_block(str(text)[:300])

    # Section 3: Competitor Differentiators
    diffs = _safe_list(card.get("competitor_differentiators"))
    if diffs:
        pdf.section_title("Competitor Differentiators")
        rows = []
        for d in diffs[:5]:
            if isinstance(d, dict):
                rows.append([
                    _safe_str(d.get("competitor", ""))[:25],
                    str(d.get("mentions", 0)),
                    _safe_str(d.get("solves_weakness") or "-")[:30],
                    _safe_str(d.get("primary_driver", ""))[:15],
                ])
        if rows:
            pdf.simple_table(
                ["Competitor", "Mentions", "Solves", "Driver"],
                rows, [40, 25, 50, 35],
            )

    # Section 4: Objection Handlers (LLM-generated)
    objections = _safe_list(card.get("objection_handlers"))
    if objections:
        pdf.section_title("Objection Handlers")
        for obj in objections[:4]:
            if isinstance(obj, dict):
                objection_text = obj.get("objection", "")
                response_text = obj.get("response", "")
                backing = obj.get("data_backing", "")
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(*_CLR_DARK)
                pdf.cell(0, 5, _latin1_safe(f'"{objection_text}"'),
                         new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(*_CLR_DARK)
                pdf.multi_cell(0, 5, _latin1_safe(response_text))
                if backing:
                    pdf.set_font("Helvetica", "I", 7)
                    pdf.set_text_color(*_CLR_MUTED)
                    pdf.cell(0, 4, _latin1_safe(f"Data: {backing}"),
                             new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)

    # Section 5: Recommended Plays (LLM-generated)
    plays = _safe_list(card.get("recommended_plays"))
    if plays:
        pdf.section_title("Recommended Plays")
        for play in plays[:3]:
            if isinstance(play, dict):
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(*_CLR_DARK)
                pdf.multi_cell(0, 5, _latin1_safe(play.get("play", "")))
                details = []
                if play.get("target_segment"):
                    details.append(f"Target: {play['target_segment']}")
                if play.get("key_message"):
                    details.append(f"Message: {play['key_message']}")
                if details:
                    pdf.set_font("Helvetica", "", 8)
                    pdf.set_text_color(*_CLR_MUTED)
                    pdf.multi_cell(0, 4, _latin1_safe(" | ".join(details)))
                pdf.ln(2)


# -- Public API ----------------------------------------------------------------

_RENDERERS: dict[str, Any] = {
    "weekly_churn_feed": lambda pdf, d, s: _render_churn_feed(pdf, d),
    "vendor_scorecard": _render_vendor_deep_dive,
    "displacement_report": _render_vendor_scorecard,
    "category_overview": _render_category_report,
    "exploratory_overview": _render_vendor_scorecard,
    "vendor_comparison": _render_comparison,
    "account_comparison": _render_comparison,
    "account_deep_dive": _render_deep_dive,
    "vendor_retention": _render_vendor_scorecard,
    "challenger_intel": _render_vendor_scorecard,
    "battle_card": _render_battle_card,
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


def render_vendor_full_report_pdf(
    vendor_name: str,
    report_data: dict,
    briefing_data: dict | None = None,
) -> bytes:
    """Build a comprehensive single-vendor PDF from the exploratory_overview.

    Extracts the vendor's data from the full report (feed entry, scorecard,
    displacement edges, category insights) and renders all sections.
    If *briefing_data* is provided, it supplements any missing sections.

    Returns raw PDF bytes.
    """
    vn_lower = vendor_name.lower()

    # -- Extract vendor-specific slices from the full report -------------------
    feed = report_data.get("weekly_churn_feed", [])
    vendor_feed: dict = {}
    for entry in (feed if isinstance(feed, list) else []):
        if isinstance(entry, dict):
            ev = (entry.get("vendor") or entry.get("vendor_name") or "").lower()
            if ev == vn_lower:
                vendor_feed = entry
                break

    scorecards = report_data.get("vendor_scorecards", [])
    vendor_sc: dict = {}
    for sc in (scorecards if isinstance(scorecards, list) else []):
        if isinstance(sc, dict):
            sv = (sc.get("vendor") or "").lower()
            if sv == vn_lower:
                vendor_sc = sc
                break

    disp_map = report_data.get("displacement_map", [])
    vendor_disp: list[dict] = []
    for d in (disp_map if isinstance(disp_map, list) else []):
        if isinstance(d, dict):
            fv = (d.get("from_vendor") or "").lower()
            if fv == vn_lower:
                vendor_disp.append(d)

    # Merge: prefer full report data, fall back to briefing_data
    bd = briefing_data or {}
    merged = {**bd, **vendor_feed, **vendor_sc}

    exec_summary = (
        merged.get("executive_summary")
        or report_data.get("executive_summary")
        or ""
    )

    # -- Build PDF -------------------------------------------------------------
    pdf = IntelligenceReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title
    date_str = merged.get("report_date") or date.today().isoformat()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*_CLR_DARK)
    pdf.cell(0, 10, _latin1_safe(f"Churn Intelligence Report: {vendor_name}"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*_CLR_MUTED)
    category = merged.get("category") or merged.get("product_category") or "Software"
    pdf.cell(0, 5, f"Category: {_latin1_safe(category)}  |  Report Date: {date_str}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Executive summary
    if exec_summary:
        pdf.section_title("Executive Summary")
        pdf.body_text(exec_summary)

    # Key metrics
    score = float(merged.get("churn_pressure_score", 0))
    metrics = [
        ("Churn Pressure Score", f"{score:.0f}", _score_color(score)),
        ("Signal Density", f"{float(merged.get('churn_signal_density', 0)):.1f}%", None),
        ("Average Urgency", f"{float(merged.get('avg_urgency', 0)):.1f}", None),
        ("Total Reviews", str(merged.get("total_reviews") or merged.get("review_count", 0)), None),
        ("DM Churn Rate", f"{float(merged.get('dm_churn_rate', 0)):.1f}%", None),
        ("Trend", _safe_str(merged.get("trend", "stable")).title(), None),
    ]
    pdf.section_title("Key Metrics")
    for label, value, color in metrics:
        pdf.metric_row(label, value, color)

    # Action recommendation (from full report)
    action = merged.get("action_recommendation")
    if action:
        pdf.section_title("Action Recommendation")
        pdf.body_text(action)

    # Pain breakdown
    pains = _safe_list(merged.get("pain_breakdown"))
    if pains:
        pdf.section_title("Pain Category Breakdown")
        rows = []
        for p in pains[:10]:
            if isinstance(p, dict):
                cat = _safe_str(p.get("category", ""))[:30]
                count = str(p.get("count", ""))
                urg = p.get("avg_urgency")
                if urg is not None:
                    rows.append([cat, count, f"{float(urg):.1f}"])
                else:
                    rows.append([cat, count, "--"])
        if rows:
            pdf.simple_table(["Category", "Signals", "Avg Urgency"], rows, [60, 30, 30])

    # Competitive displacement (from displacement_map -- full edges)
    if vendor_disp:
        pdf.section_title("Competitive Displacement Analysis")
        rows = []
        for d in vendor_disp[:15]:
            rows.append([
                _safe_str(d.get("to_vendor", ""))[:30],
                str(d.get("mention_count", 0)),
                _safe_str(d.get("primary_driver", ""))[:30],
                _safe_str(d.get("signal_strength", 0)),
            ])
        pdf.simple_table(
            ["Competitor", "Mentions", "Primary Driver", "Strength"],
            rows,
            [45, 25, 55, 25],
        )
        # Key quotes from displacement edges
        for d in vendor_disp[:5]:
            quote = d.get("key_quote")
            if quote:
                pdf.quote_block(_safe_str(quote)[:300])
    else:
        # Fall back to briefing displacement targets
        targets = _safe_list(merged.get("top_displacement_targets"))
        if targets:
            pdf.section_title("Competitive Displacement")
            rows = []
            for t in targets[:10]:
                if isinstance(t, dict):
                    rows.append([
                        _safe_str(t.get("competitor") or t.get("name", ""))[:30],
                        str(t.get("count") or t.get("mentions", 0)),
                    ])
            pdf.simple_table(["Competitor", "Mentions"], rows, [90, 30])

    # Named accounts at risk
    accounts = _safe_list(merged.get("named_accounts"))
    if accounts:
        pdf.section_title("Accounts at Risk")
        rows = []
        for a in accounts[:15]:
            if isinstance(a, dict):
                urg = float(a.get("urgency", 0))
                risk = "Critical" if urg >= 8 else ("High" if urg >= 6 else "Watch")
                rows.append([
                    _safe_str(a.get("company", ""))[:30],
                    f"{urg:.1f}",
                    risk,
                    _safe_str(a.get("primary_pain", ""))[:25],
                ])
        if rows:
            pdf.simple_table(
                ["Company", "Urgency", "Risk", "Primary Pain"],
                rows,
                [50, 20, 25, 50],
            )

    # Feature gaps
    gaps = _safe_list(merged.get("top_feature_gaps"))
    if gaps:
        pdf.section_title("Feature Gaps")
        for g in gaps[:8]:
            txt = g.get("feature", g) if isinstance(g, dict) else _safe_str(g)
            pdf.body_text(f"  - {txt}")

    # Evidence / customer quotes
    evidence = _safe_list(merged.get("evidence"))
    if evidence:
        pdf.section_title("Customer Evidence")
        for e in evidence[:8]:
            text = e.get("quote", e) if isinstance(e, dict) else _safe_str(e)
            if text:
                pdf.quote_block(_safe_str(text)[:400])

    # Budget context
    budget = merged.get("budget_context")
    if isinstance(budget, dict) and budget:
        pdf.section_title("Budget Context")
        _budget_labels = {
            "avg_seat_count": "Avg Seat Count",
            "median_seat_count": "Median Seat Count",
            "max_seat_count": "Max Seat Count",
            "price_increase_rate": "Price Increase Rate",
            "price_increase_count": "Price Increase Mentions",
        }
        for key, label in _budget_labels.items():
            val = budget.get(key)
            if val is None:
                continue
            if key == "price_increase_rate":
                display = f"{float(val) * 100:.1f}%"
            elif isinstance(val, float):
                display = f"{val:,.0f}" if val == int(val) else f"{val:,.1f}"
            else:
                display = str(val)
            pdf.metric_row(label, display)
    elif budget:
        pdf.section_title("Budget Context")
        pdf.body_text(_safe_str(budget))

    # Source distribution from full report
    src_dist = report_data.get("source_distribution")
    if isinstance(src_dist, dict) and src_dist:
        pdf.section_title("Source Distribution")
        rows = []
        for source, stats in src_dist.items():
            if isinstance(stats, dict):
                rows.append([
                    source.replace("_", " ").title(),
                    str(stats.get("reviews", 0)),
                    str(stats.get("high_urgency", 0)),
                ])
        if rows:
            pdf.simple_table(["Source", "Reviews", "High Urgency"], rows, [60, 30, 30])

    # -- Generate bytes --------------------------------------------------------
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
