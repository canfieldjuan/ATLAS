"""
Vendor Intelligence Briefing email template.

Renders a deterministic HTML email showing churn pressure, pain breakdown,
competitive displacement, customer quotes, and a booking CTA for a single
vendor.  Table-based layout with inline CSS for Outlook compatibility.
"""

from __future__ import annotations

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
    return _PAIN_COLORS.get(category.lower(), "#5b6e7a")


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
        briefing.get("booking_url"), "https://cal.com/atlas-intel/15min"
    )

    # Section 3: key metrics
    density = _fmt_pct(briefing.get("churn_signal_density", 0))
    urgency = _fmt_score(briefing.get("avg_urgency", 0))
    review_count = _safe(briefing.get("review_count", 0), "0")
    dm_rate = _fmt_pct(briefing.get("dm_churn_rate", 0))

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
        cat = _safe(p.get("category"), "Other")
        try:
            cnt = int(p.get("count", 0))
        except (TypeError, ValueError):
            cnt = 0
        color = _pain_color(cat)
        width = _bar_width(cnt, max_pain)
        pain_rows += f"""
        <tr>
          <td style="padding:4px 8px 4px 0;font-size:14px;color:#333;width:120px;white-space:nowrap;">{cat}</td>
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

    displacement_rows = ""
    for d in displacements:
        comp = _safe(d.get("competitor"), "Unknown")
        cnt = _safe(d.get("count", 0), "0")
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

        named_accounts_html = f"""
        <!-- Named Accounts -->
        <table cellpadding="0" cellspacing="0" border="0" style="width:100%;margin-bottom:28px;">
          <tr><td style="padding:0 24px;">
            <h3 style="margin:0 0 12px;font-size:16px;color:#1a2332;">Accounts at Risk</h3>
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

    # -----------------------------------------------------------------------
    # Assemble full document
    # -----------------------------------------------------------------------
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Churn Intelligence Briefing - {vendor}</title>
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

  <!-- Section 2: Vendor Headline -->
  <tr>
    <td style="padding:28px 24px 20px;">
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
                  <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Churn Pressure</div>
                  <div style="margin-top:4px;font-size:13px;">{trend_html}</div>
                </td>
              </tr>
            </table>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- Divider -->
  <tr><td style="padding:0 24px;"><hr style="border:none;border-top:1px solid #eee;margin:0;"></td></tr>

  <!-- Section 3: Key Metrics -->
  <tr>
    <td style="padding:20px 24px;">
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;">
        <tr>
          <td style="width:25%;text-align:center;padding:8px 4px;">
            <div style="font-size:22px;font-weight:700;color:#1a2332;">{density}</div>
            <div style="font-size:11px;color:#888;text-transform:uppercase;">Churn Density</div>
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
            <div style="font-size:11px;color:#888;text-transform:uppercase;">DM Churn Rate</div>
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
      <h3 style="margin:0 0 14px;font-size:16px;color:#1a2332;">What's Driving Churn</h3>
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
      <h3 style="margin:0 0 14px;font-size:16px;color:#1a2332;">Where They're Going</h3>
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;border:1px solid #eee;border-radius:6px;overflow:hidden;">
        <tr style="background:#f8f9fa;">
          <th style="padding:8px 12px;text-align:left;font-size:12px;color:#888;font-weight:600;text-transform:uppercase;">Competitor</th>
          <th style="padding:8px 12px;text-align:center;font-size:12px;color:#888;font-weight:600;text-transform:uppercase;">Mentions</th>
        </tr>
        {displacement_rows}
      </table>
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

  <!-- Section 9: CTA -->
  <tr>
    <td style="padding:8px 24px 28px;">
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;background:#f0f4f8;border-radius:8px;">
        <tr>
          <td style="padding:24px;text-align:center;">
            <div style="font-size:16px;color:#1a2332;font-weight:600;margin-bottom:8px;">Get weekly intelligence for {vendor}</div>
            <div style="font-size:14px;color:#555;margin-bottom:16px;">See who's churning, why, and where they're going -- every week.</div>
            <!--[if mso]>
            <v:roundrect xmlns:v="urn:schemas-microsoft-com:vml" href="{booking_url}" style="height:44px;v-text-anchor:middle;width:260px;" arcsize="12%" stroke="f" fillcolor="#2980b9">
              <w:anchorlock/>
              <center style="color:#ffffff;font-family:Arial,sans-serif;font-size:15px;font-weight:bold;">Book a 15-Minute Strategy Call</center>
            </v:roundrect>
            <![endif]-->
            <!--[if !mso]><!-->
            <a href="{booking_url}" style="display:inline-block;padding:12px 32px;background:#2980b9;color:#ffffff;text-decoration:none;border-radius:6px;font-size:15px;font-weight:600;">Book a 15-Minute Strategy Call</a>
            <!--<![endif]-->
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
