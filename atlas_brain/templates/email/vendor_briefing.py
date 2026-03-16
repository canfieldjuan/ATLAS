"""
Vendor Intelligence Briefing email template.

Renders a deterministic HTML email showing churn pressure, pain breakdown,
competitive displacement, customer quotes, and a booking CTA for a single
vendor.  Table-based layout with inline CSS for Outlook compatibility.
"""

from __future__ import annotations

import json
from datetime import date
from html import escape
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pressure_color(score: float) -> str:
    """Return hex color for churn pressure score."""
    if score >= 70:
        return "#e74c3c"
    if score >= 40:
        return "#f39c12"
    return "#27ae60"


def _trend_indicator(trend: str | None) -> str:
    """Return arrow + label for trend direction."""
    t = (trend or "").lower()
    if t in ("up", "rising", "increasing", "worsening"):
        return '<span style="color:#e74c3c;">&#9650; Rising</span>'
    if t in ("down", "falling", "decreasing", "improving"):
        return '<span style="color:#27ae60;">&#9660; Improving</span>'
    return '<span style="color:#7f8c8d;">&#9644; Stable</span>'


_PAIN_COLORS = {
    "pricing": "#e74c3c",
    "support": "#e67e22",
    "reliability": "#f39c12",
    "usability": "#3498db",
    "features": "#9b59b6",
    "performance": "#1abc9c",
    "integration": "#2ecc71",
    "security": "#c0392b",
    "onboarding": "#d35400",
    "migration": "#8e44ad",
}


def _pain_color(category: str) -> str:
    """Return color for a pain category, with a default fallback."""
    return _PAIN_COLORS.get((category or "").lower(), "#5b6e7a")


def _bar_width(count: int, max_count: int) -> int:
    """Return proportional bar width as a percentage (10-100)."""
    if max_count <= 0:
        return 10
    pct = int((count / max_count) * 100)
    return max(pct, 10)


def _safe(val: Any, fallback: str = "") -> str:
    """HTML-escape a value, returning fallback if None/empty."""
    if val is None:
        return fallback
    s = str(val)
    return escape(s) if s.strip() else fallback


def _fmt_pct(val: Any) -> str:
    """Format a numeric value as a percentage string."""
    try:
        return f"{float(val):.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_score(val: Any) -> str:
    """Format a numeric score to one decimal."""
    try:
        return f"{float(val):.1f}"
    except (TypeError, ValueError):
        return "N/A"


_ARCHETYPE_LABELS: dict[str, str] = {
    "pricing_shock": "Pricing Shock",
    "feature_gap": "Feature Gap",
    "support_collapse": "Support Collapse",
    "leadership_redesign": "Leadership Redesign",
    "acquisition_decay": "Acquisition Decay",
    "integration_break": "Integration Break",
    "category_disruption": "Category Disruption",
    "compliance_gap": "Compliance Gap",
}


def _render_archetype_section(
    archetype: str | None,
    confidence: float | None,
    archetype_was: str | None,
    archetype_changed: bool,
    falsification: list,
) -> str:
    """Render the archetype intelligence section for vendor briefings."""
    if not archetype:
        return ""

    label = escape(_ARCHETYPE_LABELS.get(archetype, archetype.replace("_", " ").title()))
    conf_pct = f"{confidence * 100:.0f}%" if confidence is not None else "N/A"

    # "What changed" line
    change_html = ""
    if archetype_changed and archetype_was:
        was_label = escape(
            _ARCHETYPE_LABELS.get(archetype_was, archetype_was.replace("_", " ").title())
        )
        change_html = (
            f'<div style="font-size:13px;color:#e67e22;margin-top:8px;">'
            f'<strong>Shifted from:</strong> {was_label}</div>'
        )

    # "What would falsify this" bullets
    falsify_html = ""
    if falsification:
        items = "".join(
            f'<li style="padding:2px 0;font-size:12px;color:#777;">{escape(str(f))}</li>'
            for f in falsification[:3]
        )
        falsify_html = (
            f'<div style="margin-top:10px;">'
            f'<div style="font-size:11px;color:#999;text-transform:uppercase;'
            f'letter-spacing:0.5px;margin-bottom:4px;">Watch for</div>'
            f'<ul style="margin:0;padding-left:16px;">{items}</ul></div>'
        )

    return f"""
  <!-- Archetype Intelligence -->
  <tr>
    <td style="padding:12px 24px 4px;">
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;background:#f8f9fa;border:1px solid #eee;border-radius:6px;overflow:hidden;">
        <tr>
          <td style="padding:14px 18px;">
            <div style="font-size:11px;color:#999;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Churn Pattern Classification</div>
            <div style="font-size:16px;font-weight:700;color:#1a2332;">{label}</div>
            <div style="font-size:13px;color:#888;margin-top:2px;">Confidence: {conf_pct}</div>
            {change_html}
            {falsify_html}
          </td>
        </tr>
      </table>
    </td>
  </tr>
"""


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render_vendor_briefing_html(briefing: dict) -> str:
    """
    Render a full HTML vendor intelligence briefing email.

    Expected briefing keys:
        vendor_name, category, churn_pressure_score, trend,
        churn_signal_density, avg_urgency, review_count, dm_churn_rate,
        pain_breakdown (list of {category, count}),
        top_displacement_targets (list of {competitor, count}),
        evidence (list of quote strings),
        named_accounts (list of {company, urgency}),
        top_feature_gaps (list of strings),
        booking_url, report_date
    """
    vendor = _safe(briefing.get("vendor_name"), "Unknown Vendor")
    category = _safe(briefing.get("category"), "Software")
    try:
        score = float(briefing.get("churn_pressure_score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    score_color = _pressure_color(score)
    trend_html = _trend_indicator(briefing.get("trend"))
    report_date = _safe(
        briefing.get("report_date"), date.today().isoformat()
    )
    booking_url = _safe(
        briefing.get("booking_url"), "https://churnsignals.co"
    )

    # Section 3: key metrics
    density = _fmt_pct(briefing.get("churn_signal_density", 0))
    urgency = _fmt_score(briefing.get("avg_urgency", 0))
    review_count = _safe(briefing.get("review_count", 0), "0")
    dm_rate = _fmt_pct(briefing.get("dm_churn_rate", 0))

    # Analyst enrichment fields
    headline = briefing.get("headline", "")
    executive_summary = briefing.get("executive_summary", "")
    pain_labels = briefing.get("pain_labels") or {}
    cta_hook = briefing.get("cta_hook", "")
    cta_description = briefing.get("cta_description", "")
    displacement_qualifier = briefing.get("displacement_qualifier", "")
    gate_url = briefing.get("gate_url", "")
    is_gated_delivery = briefing.get("is_gated_delivery", False)
    prospect_mode = briefing.get("prospect_mode", False)
    challenger_mode = briefing.get("challenger_mode", False)

    # Archetype context
    archetype = briefing.get("archetype")
    archetype_confidence = briefing.get("archetype_confidence")
    archetype_was = briefing.get("archetype_was")
    archetype_changed = briefing.get("archetype_changed", False)
    falsification_conditions = briefing.get("falsification_conditions") or []

    # Section 4: pain breakdown
    pains = briefing.get("pain_breakdown") or []
    pains = pains[:5]
    def _int_count(p: dict) -> int:
        try:
            return int(p.get("count", 0))
        except (TypeError, ValueError):
            return 0
    max_pain = max((_int_count(p) for p in pains), default=1) if pains else 1

    pain_rows = ""
    for p in pains:
        raw_cat = p.get("category", "Other")
        cat = _safe(pain_labels.get(raw_cat, raw_cat), "Other")
        try:
            cnt = int(p.get("count", 0))
        except (TypeError, ValueError):
            cnt = 0
        color = _pain_color(raw_cat)
        width = _bar_width(cnt, max_pain)
        pain_rows += f"""
        <tr>
          <td style="padding:4px 8px 4px 0;font-size:14px;color:#333;width:180px;white-space:nowrap;">{cat}</td>
          <td style="padding:4px 0;">
            <table cellpadding="0" cellspacing="0" border="0" style="width:100%;">
              <tr>
                <td style="background:{color};width:{width}%;height:22px;border-radius:3px;"></td>
                <td style="width:{100 - width}%;"></td>
              </tr>
            </table>
          </td>
          <td style="padding:4px 0 4px 8px;font-size:14px;color:#555;width:40px;text-align:right;">{cnt}</td>
        </tr>"""

    # Section 5: displacement
    displacements = briefing.get("top_displacement_targets") or []
    displacements = displacements[:5]

    displacement_qual_html = ""
    if displacement_qualifier:
        dq_escaped = escape(displacement_qualifier)
        displacement_qual_html = (
            f'<div style="margin-top:8px;font-size:12px;color:#888;'
            f'font-style:italic;">{dq_escaped}</div>'
        )

    displacement_rows = ""
    for d in displacements:
        raw_comp = d.get("competitor") or d.get("name") or "Unknown"
        # Handle stringified JSON objects like '{"name": "Marketo"}'
        if isinstance(raw_comp, str) and raw_comp.startswith("{"):
            try:
                parsed = json.loads(raw_comp)
                raw_comp = parsed.get("name") or parsed.get("competitor") or raw_comp
            except (json.JSONDecodeError, TypeError):
                pass
        elif isinstance(raw_comp, dict):
            raw_comp = raw_comp.get("name") or raw_comp.get("competitor") or "Unknown"
        comp = _safe(raw_comp, "Unknown")
        cnt = _safe(d.get("count") or d.get("mentions", 0), "0")
        displacement_rows += f"""
        <tr>
          <td style="padding:6px 12px;font-size:14px;color:#333;border-bottom:1px solid #eee;">{comp}</td>
          <td style="padding:6px 12px;font-size:14px;color:#555;text-align:center;border-bottom:1px solid #eee;">{cnt}</td>
        </tr>"""

    # Section 6: quotes
    evidence = briefing.get("evidence") or []
    evidence = evidence[:3]

    quote_blocks = ""
    for q in evidence:
        quote_text = _safe(q) if isinstance(q, str) else _safe(q.get("quote", q.get("text", "")) if isinstance(q, dict) else str(q))
        if quote_text:
            quote_blocks += f"""
            <table cellpadding="0" cellspacing="0" border="0" style="width:100%;margin-bottom:12px;">
              <tr>
                <td style="width:4px;background:#e74c3c;border-radius:2px;"></td>
                <td style="padding:10px 14px;font-size:13px;color:#555;font-style:italic;background:#fafafa;border-radius:0 4px 4px 0;">
                  &ldquo;{quote_text}&rdquo;
                </td>
              </tr>
            </table>"""

    # Section 7: named accounts (conditional)
    named_accounts = briefing.get("named_accounts") or []
    named_accounts_html = ""
    if named_accounts:
        if prospect_mode:
            # Redacted version for sales/demo -- show count + risk levels only
            critical = sum(
                1 for a in named_accounts
                if float(a.get("urgency", 0)) >= 8
            )
            high = sum(
                1 for a in named_accounts
                if 6 <= float(a.get("urgency", 0)) < 8
            )
            watch = len(named_accounts) - critical - high
            risk_parts = []
            if critical:
                risk_parts.append(f"{critical} critical")
            if high:
                risk_parts.append(f"{high} high")
            if watch:
                risk_parts.append(f"{watch} watch")
            risk_text = ", ".join(risk_parts)

            _acct_heading = "Accounts In Motion" if challenger_mode else "Accounts at Risk"
            _acct_desc = "enterprise accounts leaving incumbents" if challenger_mode else "enterprise accounts flagged"
            named_accounts_html = f"""
        <!-- Named Accounts (redacted) -->
        <table cellpadding="0" cellspacing="0" border="0" style="width:100%;margin-bottom:28px;">
          <tr><td style="padding:0 24px;">
            <h3 style="margin:0 0 12px;font-size:16px;color:#1a2332;">{_acct_heading}</h3>
            <div style="background:#fafafa;border:1px solid #eee;border-radius:6px;padding:16px 20px;text-align:center;">
              <div style="font-size:28px;font-weight:700;color:#1a2332;">{len(named_accounts)}</div>
              <div style="font-size:13px;color:#888;margin-top:4px;">{_acct_desc}</div>
              <div style="font-size:13px;color:#555;margin-top:8px;">{risk_text}</div>
              <div style="font-size:12px;color:#999;margin-top:10px;font-style:italic;">Full account list available in paid briefing</div>
            </div>
          </td></tr>
        </table>"""
        else:
            # Full version for paying clients
            account_rows = ""
            for acct in named_accounts[:8]:
                company = _safe(acct.get("company"), "Unknown")
                urg = acct.get("urgency", 0)
                try:
                    urg_val = float(urg)
                except (TypeError, ValueError):
                    urg_val = 0
                if urg_val >= 8:
                    badge_color = "#e74c3c"
                    badge_label = "Critical"
                elif urg_val >= 6:
                    badge_color = "#f39c12"
                    badge_label = "High"
                else:
                    badge_color = "#3498db"
                    badge_label = "Watch"
                account_rows += f"""
            <tr>
              <td style="padding:6px 12px;font-size:14px;color:#333;border-bottom:1px solid #eee;">{company}</td>
              <td style="padding:6px 12px;text-align:center;border-bottom:1px solid #eee;">
                <span style="display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;color:#fff;background:{badge_color};">{badge_label}</span>
              </td>
            </tr>"""

            _full_acct_heading = "Accounts In Motion" if challenger_mode else "Accounts at Risk"
            named_accounts_html = f"""
        <!-- Named Accounts -->
        <table cellpadding="0" cellspacing="0" border="0" style="width:100%;margin-bottom:28px;">
          <tr><td style="padding:0 24px;">
            <h3 style="margin:0 0 12px;font-size:16px;color:#1a2332;">{_full_acct_heading}</h3>
            <table cellpadding="0" cellspacing="0" border="0" style="width:100%;border:1px solid #eee;border-radius:6px;overflow:hidden;">
              <tr style="background:#f8f9fa;">
                <th style="padding:8px 12px;text-align:left;font-size:12px;color:#888;font-weight:600;text-transform:uppercase;">Company</th>
                <th style="padding:8px 12px;text-align:center;font-size:12px;color:#888;font-weight:600;text-transform:uppercase;">Risk</th>
              </tr>
              {account_rows}
            </table>
          </td></tr>
        </table>"""

    # Section 8: feature gaps
    feature_gaps = briefing.get("top_feature_gaps") or []
    feature_gaps = feature_gaps[:3]

    feature_gap_html = ""
    if feature_gaps:
        gap_items = ""
        for gap in feature_gaps:
            gap_text = _safe(gap) if isinstance(gap, str) else _safe(str(gap))
            if gap_text:
                gap_items += f'<li style="padding:3px 0;font-size:14px;color:#555;">{gap_text}</li>'

        if gap_items:
            feature_gap_html = f"""
            <!-- Feature Gaps -->
            <table cellpadding="0" cellspacing="0" border="0" style="width:100%;margin-bottom:28px;">
              <tr><td style="padding:0 24px;">
                <h3 style="margin:0 0 12px;font-size:16px;color:#1a2332;">Top Feature Gaps</h3>
                <ul style="margin:0;padding-left:20px;">
                  {gap_items}
                </ul>
              </td></tr>
            </table>"""

    # Section 8b: correlated articles
    correlated_articles = briefing.get("correlated_articles") or []
    correlated_articles_html = ""
    if correlated_articles:
        article_items = ""
        for art in correlated_articles[:3]:
            art_title = _safe(art.get("title", ""), "Untitled")
            art_url = art.get("url", "")
            art_source = _safe(art.get("source", ""), "")
            source_tag = f' <span style="color:#999;">({art_source})</span>' if art_source else ""
            if art_url:
                article_items += (
                    f'<li style="padding:4px 0;font-size:13px;line-height:1.4;">'
                    f'<a href="{escape(art_url)}" style="color:#2980b9;text-decoration:none;">'
                    f'{art_title}</a>{source_tag}</li>'
                )
            else:
                article_items += (
                    f'<li style="padding:4px 0;font-size:13px;color:#555;line-height:1.4;">'
                    f'{art_title}{source_tag}</li>'
                )

        if article_items:
            correlated_articles_html = f"""
            <!-- Correlated Articles -->
            <table cellpadding="0" cellspacing="0" border="0" style="width:100%;margin-bottom:28px;">
              <tr><td style="padding:0 24px;">
                <h3 style="margin:0 0 12px;font-size:16px;color:#1a2332;">Related Market Coverage</h3>
                <ul style="margin:0;padding-left:20px;">
                  {article_items}
                </ul>
              </td></tr>
            </table>"""

    # Section 9: CTA content
    cta_title = f"There&#39;s more in the full report"
    if challenger_mode:
        cta_description_html = (
            '<div style="font-size:14px;color:#555;margin-bottom:16px;">'
            f"This briefing is a summary. The full sales intelligence report for {vendor} "
            "includes detailed account movement data, incumbent pain analysis, "
            "and actionable prospecting insights."
            "</div>"
        )
    else:
        cta_description_html = (
            '<div style="font-size:14px;color:#555;margin-bottom:16px;">'
            f"This briefing is a summary. The full intelligence report for {vendor} "
            "includes detailed competitive analysis, displacement flow data, "
            "feature gap breakdowns, and actionable retention insights."
            "</div>"
        )

    if prospect_mode or is_gated_delivery:
        if challenger_mode:
            cta_button_label = "See Which Accounts Are In Motion"
        else:
            cta_button_label = "See the Full Report"
        cta_button_sub = (
            '<div style="font-size:13px;color:#555;margin-top:10px;">'
            + ("The full report includes account movement data, "
               "incumbent pain analysis, and more."
               if challenger_mode else
               "The full report includes competitive displacement maps, "
               "customer pain analysis, and more.")
            + "</div>"
        )
        cta_link = gate_url or booking_url
    else:
        if challenger_mode:
            cta_button_label = "See Which Accounts Are In Motion"
        else:
            cta_button_label = "See the Full Report"
        cta_button_sub = ""
        cta_link = booking_url

    cta_button_html = (
        "<!--[if mso]>"
        f'<v:roundrect xmlns:v="urn:schemas-microsoft-com:vml" href="{cta_link}"'
        ' style="height:44px;v-text-anchor:middle;width:280px;" arcsize="12%"'
        ' stroke="f" fillcolor="#2980b9">'
        "<w:anchorlock/>"
        f'<center style="color:#ffffff;font-family:Arial,sans-serif;font-size:15px;'
        f'font-weight:bold;">{cta_button_label}</center>'
        "</v:roundrect>"
        "<![endif]-->"
        "<!--[if !mso]><!-->"
        f'<a href="{cta_link}" style="display:inline-block;padding:12px 32px;'
        "background:#2980b9;color:#ffffff;text-decoration:none;border-radius:6px;"
        f'font-size:15px;font-weight:600;">{cta_button_label}</a>'
        "<!--<![endif]-->"
        f"{cta_button_sub}"
    )

    # -----------------------------------------------------------------------
    # Assemble full document
    # -----------------------------------------------------------------------
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>{"Sales Intelligence Briefing" if challenger_mode else "Churn Intelligence Briefing"} - {vendor}</title>
  <!--[if mso]>
  <style>table {{border-collapse:collapse;}}</style>
  <![endif]-->
</head>
<body style="margin:0;padding:0;background:#f4f5f7;font-family:Arial,Helvetica,sans-serif;-webkit-text-size-adjust:100%;">
<center>
<table cellpadding="0" cellspacing="0" border="0" style="width:100%;max-width:640px;margin:0 auto;background:#ffffff;">

  <!-- Section 1: Header Bar -->
  <tr>
    <td style="background:#1a2332;padding:20px 24px;">
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;">
        <tr>
          <td style="font-size:18px;font-weight:700;color:#ffffff;letter-spacing:0.5px;">Atlas Intelligence</td>
          <td style="text-align:right;font-size:13px;color:#8899aa;">{report_date}</td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- Context line -->
  <tr>
    <td style="padding:20px 24px 0;">
      <p style="margin:0;font-size:13px;color:#666;line-height:1.5;">{"We track accounts in motion across your competitive landscape so your sales team doesn&#39;t have to. Here&#39;s what we found this week." if challenger_mode else "We monitor public customer signals for " + vendor + " so your team doesn&#39;t have to. Here&#39;s what we found this week."}</p>
    </td>
  </tr>

  <!-- Section 2: Vendor Headline -->
  <tr>
    <td style="padding:20px 24px 20px;">
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;">
        <tr>
          <td>
            <h1 style="margin:0 0 6px;font-size:24px;color:#1a2332;">{vendor}</h1>
            <span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;background:#5b6e7a;margin-right:8px;">{category}</span>
          </td>
          <td style="text-align:right;vertical-align:top;">
            <table cellpadding="0" cellspacing="0" border="0">
              <tr>
                <td style="text-align:center;">
                  <div style="font-size:36px;font-weight:700;color:{score_color};line-height:1;">{score:.0f}</div>
                  <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.5px;">{"Market Momentum" if challenger_mode else "Churn Pressure"}</div>
                  <div style="margin-top:4px;font-size:13px;">{trend_html}</div>
                </td>
              </tr>
            </table>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  {"" if not headline else f'''
  <!-- Headline -->
  <tr>
    <td style="padding:16px 24px 0;">
      <div style="font-size:17px;font-weight:700;color:#2c3e50;line-height:1.3;">{escape(headline)}</div>
    </td>
  </tr>
  '''}

  {"" if not executive_summary else f'''
  <!-- Executive Summary -->
  <tr>
    <td style="padding:12px 24px 4px;">
      <div style="font-size:14px;color:#444;line-height:1.6;border-left:3px solid #2980b9;padding-left:14px;">{escape(executive_summary)}</div>
    </td>
  </tr>
  '''}

  {_render_archetype_section(archetype, archetype_confidence, archetype_was, archetype_changed, falsification_conditions)}

  <!-- Divider -->
  <tr><td style="padding:0 24px;"><hr style="border:none;border-top:1px solid #eee;margin:0;"></td></tr>

  <!-- Section 3: Key Metrics -->
  <tr>
    <td style="padding:20px 24px;">
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;">
        <tr>
          <td style="width:25%;text-align:center;padding:8px 4px;">
            <div style="font-size:22px;font-weight:700;color:#1a2332;">{density}</div>
            <div style="font-size:11px;color:#888;text-transform:uppercase;">{"Signal Density" if challenger_mode else "Churn Density"}</div>
          </td>
          <td style="width:25%;text-align:center;padding:8px 4px;">
            <div style="font-size:22px;font-weight:700;color:#1a2332;">{urgency}</div>
            <div style="font-size:11px;color:#888;text-transform:uppercase;">Urgency /10</div>
          </td>
          <td style="width:25%;text-align:center;padding:8px 4px;">
            <div style="font-size:22px;font-weight:700;color:#1a2332;">{review_count}</div>
            <div style="font-size:11px;color:#888;text-transform:uppercase;">Reviews</div>
          </td>
          <td style="width:25%;text-align:center;padding:8px 4px;">
            <div style="font-size:22px;font-weight:700;color:#1a2332;">{dm_rate}</div>
            <div style="font-size:11px;color:#888;text-transform:uppercase;">{"DM Switch Rate" if challenger_mode else "DM Churn Rate"}</div>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- Divider -->
  <tr><td style="padding:0 24px;"><hr style="border:none;border-top:1px solid #eee;margin:0;"></td></tr>

  <!-- Section 4: Pain Breakdown -->
  {"" if not pain_rows else f'''
  <tr>
    <td style="padding:20px 24px;">
      <h3 style="margin:0 0 14px;font-size:16px;color:#1a2332;">{"What's Driving The Switch" if challenger_mode else "What's Driving Churn"}</h3>
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;">
        {pain_rows}
      </table>
    </td>
  </tr>
  '''}

  <!-- Section 5: Displacement -->
  {"" if not displacement_rows else f'''
  <tr>
    <td style="padding:20px 24px;">
      <h3 style="margin:0 0 14px;font-size:16px;color:#1a2332;">{"Incumbents Losing Accounts" if challenger_mode else "Where They&#39;re Going"}</h3>
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;border:1px solid #eee;border-radius:6px;overflow:hidden;">
        <tr style="background:#f8f9fa;">
          <th style="padding:8px 12px;text-align:left;font-size:12px;color:#888;font-weight:600;text-transform:uppercase;">Competitor</th>
          <th style="padding:8px 12px;text-align:center;font-size:12px;color:#888;font-weight:600;text-transform:uppercase;">Mentions</th>
        </tr>
        {displacement_rows}
      </table>
      {displacement_qual_html}
    </td>
  </tr>
  '''}

  <!-- Section 6: Quotes -->
  {"" if not quote_blocks else f'''
  <tr>
    <td style="padding:20px 24px;">
      <h3 style="margin:0 0 14px;font-size:16px;color:#1a2332;">What Customers Are Saying</h3>
      {quote_blocks}
    </td>
  </tr>
  '''}

  {named_accounts_html}

  {feature_gap_html}

  {correlated_articles_html}

  <!-- Section 9: CTA -->
  <tr>
    <td style="padding:8px 24px 28px;">
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;background:#f0f4f8;border-radius:8px;">
        <tr>
          <td style="padding:24px;text-align:center;">
            <div style="font-size:16px;color:#1a2332;font-weight:600;margin-bottom:8px;">{cta_title}</div>
            {cta_description_html}
            {cta_button_html}
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- Section 10: Footer -->
  <tr>
    <td style="background:#f8f9fa;padding:20px 24px;border-top:1px solid #eee;">
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;">
        <tr>
          <td style="font-size:12px;color:#999;line-height:1.5;">
            <strong>Atlas Intelligence</strong><br>
            Data sourced from public review platforms, aggregated and analyzed by Atlas.<br>
            <a href="#" style="color:#999;text-decoration:underline;">Unsubscribe</a>
          </td>
        </tr>
      </table>
    </td>
  </tr>

</table>
</center>
</body>
</html>"""
