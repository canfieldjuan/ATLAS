"""
Full report delivery email for gated briefing flow.

Sent when a prospect enters their email at the landing page gate.
Short message plus PDF attachment. Table-based layout with inline CSS
for Outlook compatibility.
"""

from __future__ import annotations

from html import escape


def render_report_delivery_html(vendor_name: str) -> str:
    """Render the report delivery cover email as HTML."""
    vendor = escape(vendor_name)

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f4f5f7;font-family:Arial,Helvetica,sans-serif;">
<table cellpadding="0" cellspacing="0" border="0" style="width:100%;background:#f4f5f7;">
<tr><td align="center" style="padding:24px 12px;">
<table cellpadding="0" cellspacing="0" border="0" style="width:100%;max-width:600px;background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.06);">

  <tr>
    <td style="background:#1a2332;padding:24px 24px 20px;text-align:center;">
      <div style="font-size:20px;font-weight:700;color:#ffffff;margin-bottom:4px;">Churn Signals</div>
      <div style="font-size:13px;color:#94a3b8;">Displacement Intelligence Report</div>
    </td>
  </tr>

  <tr>
    <td style="padding:28px 24px 20px;">
      <div style="font-size:22px;font-weight:700;color:#1a2332;margin-bottom:12px;">Your {vendor} Displacement Intelligence Report</div>
      <p style="margin:0 0 16px;font-size:15px;color:#444;line-height:1.6;">
        Your full report is attached as a PDF. It includes:
      </p>

      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;">
        <tr>
          <td style="padding:6px 0;font-size:14px;color:#333;line-height:1.5;">
            <strong style="color:#2980b9;">1.</strong> Churn pressure score and vendor risk profile
          </td>
        </tr>
        <tr>
          <td style="padding:6px 0;font-size:14px;color:#333;line-height:1.5;">
            <strong style="color:#2980b9;">2.</strong> Pain category breakdown with evidence-backed signal counts
          </td>
        </tr>
        <tr>
          <td style="padding:6px 0;font-size:14px;color:#333;line-height:1.5;">
            <strong style="color:#2980b9;">3.</strong> Competitive displacement map with directional flows
          </td>
        </tr>
        <tr>
          <td style="padding:6px 0;font-size:14px;color:#333;line-height:1.5;">
            <strong style="color:#2980b9;">4.</strong> Customer evidence and feature gap intelligence
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <tr>
    <td style="padding:0 24px 24px;">
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;background:#f8fafc;border-radius:6px;border:1px solid #e2e8f0;">
        <tr>
          <td style="padding:16px 20px;">
            <div style="font-size:13px;font-weight:700;color:#2980b9;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">Want this every week?</div>
            <p style="margin:0;font-size:14px;color:#333;line-height:1.5;">
              Subscribe to get fresh {vendor} displacement intelligence delivered to your inbox every Monday.
            </p>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <tr>
    <td style="padding:0 24px 28px;">
      <p style="margin:0;font-size:14px;color:#555;line-height:1.6;">
        Questions? Reply to this email or reach us at
        <a href="mailto:outreach@churnsignals.co" style="color:#2980b9;text-decoration:none;">outreach@churnsignals.co</a>.
      </p>
    </td>
  </tr>

  <tr>
    <td style="background:#f8fafc;padding:16px 24px;border-top:1px solid #e2e8f0;">
      <p style="margin:0;font-size:12px;color:#94a3b8;line-height:1.5;text-align:center;">
        Churn Signals by Atlas Business Intelligence<br>
        You received this because you requested a report on churnsignals.co.
      </p>
    </td>
  </tr>

</table>
</td></tr>
</table>
</body>
</html>"""


def render_report_delivery_text(vendor_name: str) -> str:
    """Plain text fallback for the report delivery email."""
    return f"""Your {vendor_name} Displacement Intelligence Report

Your full report is attached as a PDF. It includes:

1. Churn pressure score and vendor risk profile
2. Pain category breakdown with evidence-backed signal counts
3. Competitive displacement map with directional flows
4. Customer evidence and feature gap intelligence

WANT THIS EVERY WEEK?
Subscribe to get fresh {vendor_name} displacement intelligence delivered
to your inbox every Monday.

QUESTIONS?
Reply to this email or reach us at outreach@churnsignals.co.

--
Churn Signals by Atlas Business Intelligence
"""
