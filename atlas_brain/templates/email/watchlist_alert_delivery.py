"""Saved-view watchlist alert delivery email templates."""

from __future__ import annotations

from html import escape


def render_watchlist_alert_delivery_html(
    *,
    account_name: str,
    view_name: str,
    summary_line: str,
    manage_url: str | None,
    events: list[dict[str, object]],
) -> str:
    safe_account_name = escape(account_name or "your account")
    safe_view_name = escape(view_name or "Saved view alerts")
    safe_summary_line = escape(summary_line or "Open watchlist alerts are ready.")
    safe_manage_url = escape((manage_url or "").strip(), quote=True)

    event_rows = ""
    for event in events:
      label = escape(str(event.get("label") or "Alert"))
      summary = escape(str(event.get("summary") or "Watchlist alert"))
      detail = escape(str(event.get("detail") or ""))
      event_rows += f"""
      <tr>
        <td style="padding:14px 16px;border:1px solid #e2e8f0;border-radius:14px;background:#ffffff;">
          <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#0f766e;">{label}</div>
          <div style="margin-top:8px;font-size:15px;font-weight:700;color:#0f172a;">{summary}</div>
          {f'<div style="margin-top:8px;font-size:13px;line-height:1.6;color:#475569;">{detail}</div>' if detail else ''}
        </td>
      </tr>
      """

    manage_block = ""
    if safe_manage_url:
        manage_block = f"""
        <tr>
          <td style="padding:0 24px 24px;">
            <a href="{safe_manage_url}" style="display:inline-block;background:#0f172a;color:#f8fafc;text-decoration:none;padding:12px 16px;border-radius:12px;font-size:14px;font-weight:700;">
              Open watchlists
            </a>
          </td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#e2e8f0;font-family:Arial,Helvetica,sans-serif;">
<table cellpadding="0" cellspacing="0" border="0" style="width:100%;background:#e2e8f0;">
<tr><td align="center" style="padding:24px 12px;">
<table cellpadding="0" cellspacing="0" border="0" style="width:100%;max-width:680px;background:#f8fafc;border-radius:18px;overflow:hidden;">
  <tr>
    <td style="background:#0f172a;padding:26px 24px 22px;">
      <div style="font-size:12px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#67e8f9;">Churn Signals</div>
      <div style="margin-top:8px;font-size:26px;font-weight:700;color:#f8fafc;">{safe_view_name}</div>
      <p style="margin:10px 0 0;font-size:15px;line-height:1.7;color:#cbd5e1;">{safe_summary_line}</p>
      <p style="margin:12px 0 0;font-size:13px;line-height:1.6;color:#94a3b8;">{safe_account_name} | saved view alert delivery</p>
    </td>
  </tr>
  <tr>
    <td style="padding:0 24px 20px;">
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;border-collapse:separate;border-spacing:0 12px;">
        {event_rows}
      </table>
    </td>
  </tr>
  {manage_block}
  <tr>
    <td style="background:#e2e8f0;padding:16px 24px;">
      <p style="margin:0;font-size:12px;line-height:1.6;color:#64748b;text-align:center;">
        This delivery came from your saved watchlist alert policy.
      </p>
    </td>
  </tr>
</table>
</td></tr>
</table>
</body>
</html>"""
