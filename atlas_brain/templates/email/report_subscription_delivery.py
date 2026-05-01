"""Recurring report-subscription delivery email templates."""

from __future__ import annotations

from html import escape


def _coverage_line(label: str, count: object, sections: object | None = None) -> str:
    try:
        normalized_count = int(count or 0)
    except Exception:
        normalized_count = 0
    text = f"{label}: {normalized_count}"
    if isinstance(sections, list):
        named_sections = [str(item or "").strip() for item in sections if str(item or "").strip()]
        if named_sections:
            text += f" ({', '.join(named_sections[:3])})"
    return text


def _artifact_section_html(artifact: dict[str, object]) -> str:
    title = escape(str(artifact.get("title") or "Persisted artifact"))
    type_label = escape(str(artifact.get("type_label") or "Report"))
    trust_label = escape(str(artifact.get("trust_label") or "Persisted artifact"))
    artifact_label = escape(str(artifact.get("artifact_label") or "Status unknown"))
    review_label = escape(str(artifact.get("review_label") or "Review unknown"))
    freshness_label = escape(str(artifact.get("freshness_label") or "Unknown"))
    freshness_detail = escape(str(artifact.get("freshness_detail") or ""))
    executive_summary = escape(str(artifact.get("executive_summary") or "No executive summary is attached to this artifact yet."))
    report_url = escape(str(artifact.get("report_url") or ""), quote=True)
    evidence_highlights = artifact.get("evidence_highlights") or []
    account_pressure_summary = str(artifact.get("account_pressure_summary") or "").strip()
    account_pressure_disclaimer = str(artifact.get("account_pressure_disclaimer") or "").strip()
    priority_account_names = artifact.get("priority_account_names") or []
    section_evidence_summary = artifact.get("section_evidence_summary") or {}

    highlight_items = ""
    if isinstance(evidence_highlights, list):
        for highlight in evidence_highlights[:3]:
            text = str(highlight or "").strip()
            if not text:
                continue
            highlight_items += (
                "<li style=\"margin:0 0 8px;color:#334155;font-size:14px;line-height:1.6;\">"
                f"{escape(text)}"
                "</li>"
            )

    highlight_block = ""
    if highlight_items:
        highlight_block = (
            "<div style=\"margin-top:14px;\">"
            "<div style=\"font-size:12px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#0f766e;margin-bottom:8px;\">"
            "Witness Highlights"
            "</div>"
            f"<ul style=\"padding-left:18px;margin:0;\">{highlight_items}</ul>"
            "</div>"
        )

    priority_accounts_line = ""
    if isinstance(priority_account_names, list):
        cleaned_priority_names = [str(item or "").strip() for item in priority_account_names if str(item or "").strip()]
        if cleaned_priority_names:
            priority_accounts_line = (
                "<div style=\"margin-top:8px;font-size:13px;color:#334155;line-height:1.6;\">"
                f"<strong>Priority accounts:</strong> {escape(', '.join(cleaned_priority_names[:3]))}"
                "</div>"
            )

    account_pressure_block = ""
    if account_pressure_summary or priority_accounts_line or account_pressure_disclaimer:
        disclaimer_line = ""
        if account_pressure_disclaimer:
            disclaimer_line = (
                "<div style=\"margin-top:8px;font-size:12px;color:#64748b;line-height:1.6;\">"
                f"{escape(account_pressure_disclaimer)}"
                "</div>"
            )
        account_pressure_block = (
            "<div style=\"margin-top:14px;\">"
            "<div style=\"font-size:12px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#0f766e;margin-bottom:8px;\">"
            "Account Pressure"
            "</div>"
            f"<div style=\"font-size:14px;color:#334155;line-height:1.7;\">{escape(account_pressure_summary)}</div>"
            f"{priority_accounts_line}"
            f"{disclaimer_line}"
            "</div>"
        )

    section_coverage_items = ""
    if isinstance(section_evidence_summary, dict):
        section_coverage_lines = [
            _coverage_line(
                "Witness-backed sections",
                section_evidence_summary.get("witness_backed_count"),
            ),
            _coverage_line(
                "Partial evidence",
                section_evidence_summary.get("partial_count"),
                section_evidence_summary.get("partial_sections"),
            ),
            _coverage_line(
                "Thin evidence",
                section_evidence_summary.get("thin_count"),
                section_evidence_summary.get("thin_sections"),
            ),
        ]
        for line in section_coverage_lines:
            section_coverage_items += (
                "<li style=\"margin:0 0 8px;color:#334155;font-size:14px;line-height:1.6;\">"
                f"{escape(line)}"
                "</li>"
            )
    section_coverage_block = ""
    if section_coverage_items:
        section_coverage_block = (
            "<div style=\"margin-top:14px;\">"
            "<div style=\"font-size:12px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#0f766e;margin-bottom:8px;\">"
            "Section Coverage"
            "</div>"
            f"<ul style=\"padding-left:18px;margin:0;\">{section_coverage_items}</ul>"
            "</div>"
        )

    cta_block = ""
    if report_url:
        cta_block = (
            "<div style=\"margin-top:16px;\">"
            f"<a href=\"{report_url}\" style=\"display:inline-block;background:#0f172a;color:#f8fafc;text-decoration:none;padding:10px 14px;border-radius:10px;font-size:13px;font-weight:700;\">"
            "Open artifact"
            "</a>"
            "</div>"
        )

    freshness_detail_block = (
        f"<br><span style=\"color:#64748b;\">{freshness_detail}</span>"
        if freshness_detail
        else ""
    )

    return f"""<tr>
  <td style="padding:18px 20px;border:1px solid #e2e8f0;border-radius:16px;background:#ffffff;">
    <div style="font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#0891b2;">{type_label}</div>
    <div style="margin-top:8px;font-size:20px;font-weight:700;color:#0f172a;">{title}</div>
    <div style="margin-top:10px;font-size:13px;color:#475569;line-height:1.6;">
      <strong>Trust:</strong> {trust_label}<br>
      <strong>Artifact:</strong> {artifact_label}<br>
      <strong>Review:</strong> {review_label}<br>
      <strong>Freshness:</strong> {freshness_label}
      {freshness_detail_block}
    </div>
    <p style="margin:14px 0 0;font-size:14px;color:#334155;line-height:1.7;">{executive_summary}</p>
    {account_pressure_block}
    {section_coverage_block}
    {highlight_block}
    {cta_block}
  </td>
</tr>"""


def render_report_subscription_delivery_html(
    *,
    account_name: str,
    scope_label: str,
    summary_line: str,
    frequency_label: str,
    manage_url: str | None,
    delivery_note: str,
    artifacts: list[dict[str, object]],
) -> str:
    """Render the HTML email for recurring report-subscription delivery."""
    safe_account_name = escape(account_name or "your account")
    safe_scope_label = escape(scope_label or "Recurring delivery")
    safe_summary_line = escape(summary_line or "Persisted report delivery is ready.")
    safe_frequency_label = escape(frequency_label or "Recurring")
    safe_delivery_note = escape(delivery_note or "")
    safe_manage_url = escape((manage_url or "").strip(), quote=True)

    note_block = ""
    if safe_delivery_note:
        note_block = f"""
  <tr>
    <td style="padding:0 24px 18px;">
      <div style="border:1px solid #cbd5e1;border-radius:14px;background:#f8fafc;padding:14px 16px;">
        <div style="font-size:12px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#475569;">Saved delivery note</div>
        <p style="margin:8px 0 0;font-size:14px;line-height:1.6;color:#334155;">{safe_delivery_note}</p>
      </div>
    </td>
  </tr>"""

    manage_block = ""
    if safe_manage_url:
        manage_block = f"""
  <tr>
    <td style="padding:0 24px 24px;">
      <a href="{safe_manage_url}" style="display:inline-block;background:#0f766e;color:#f8fafc;text-decoration:none;padding:12px 16px;border-radius:12px;font-size:14px;font-weight:700;">
        Open report library
      </a>
    </td>
  </tr>"""

    artifact_sections = "".join(_artifact_section_html(artifact) for artifact in artifacts)

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
      <div style="margin-top:8px;font-size:26px;font-weight:700;color:#f8fafc;">{safe_scope_label}</div>
      <p style="margin:10px 0 0;font-size:15px;line-height:1.7;color:#cbd5e1;">{safe_summary_line}</p>
      <p style="margin:12px 0 0;font-size:13px;line-height:1.6;color:#94a3b8;">{safe_account_name} | {safe_frequency_label} subscription</p>
    </td>
  </tr>
  {note_block}
  <tr>
    <td style="padding:0 24px 20px;">
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;border-collapse:separate;border-spacing:0 14px;">
        {artifact_sections}
      </table>
    </td>
  </tr>
  {manage_block}
  <tr>
    <td style="background:#e2e8f0;padding:16px 24px;">
      <p style="margin:0;font-size:12px;line-height:1.6;color:#64748b;text-align:center;">
        This delivery came from your saved recurring report subscription.
      </p>
    </td>
  </tr>
</table>
</td></tr>
</table>
</body>
</html>"""


def render_report_subscription_delivery_text(
    *,
    account_name: str,
    scope_label: str,
    summary_line: str,
    frequency_label: str,
    manage_url: str | None,
    delivery_note: str,
    artifacts: list[dict[str, object]],
) -> str:
    """Render the plain-text fallback for recurring report-subscription delivery."""
    lines = [
        scope_label or "Recurring delivery",
        summary_line or "Persisted report delivery is ready.",
        f"Account: {account_name or 'your account'}",
        f"Cadence: {frequency_label or 'Recurring'}",
        "",
    ]

    if delivery_note.strip():
        lines.extend([
            "Saved delivery note:",
            delivery_note.strip(),
            "",
        ])

    for artifact in artifacts:
        lines.append(f"- {artifact.get('title') or 'Persisted artifact'}")
        lines.append(f"  Type: {artifact.get('type_label') or 'Report'}")
        lines.append(f"  Trust: {artifact.get('trust_label') or 'Persisted artifact'}")
        lines.append(f"  Artifact: {artifact.get('artifact_label') or 'Status unknown'}")
        lines.append(f"  Review: {artifact.get('review_label') or 'Review unknown'}")
        lines.append(f"  Freshness: {artifact.get('freshness_label') or 'Unknown'}")
        freshness_detail = str(artifact.get("freshness_detail") or "").strip()
        if freshness_detail:
            lines.append(f"  Detail: {freshness_detail}")
        lines.append(f"  Summary: {artifact.get('executive_summary') or 'No executive summary is attached to this artifact yet.'}")
        evidence_highlights = artifact.get("evidence_highlights") or []
        account_pressure_summary = str(artifact.get("account_pressure_summary") or "").strip()
        account_pressure_disclaimer = str(artifact.get("account_pressure_disclaimer") or "").strip()
        priority_account_names = artifact.get("priority_account_names") or []
        section_evidence_summary = artifact.get("section_evidence_summary") or {}
        if isinstance(section_evidence_summary, dict):
            lines.append("  Section coverage:")
            lines.append(
                f"    {_coverage_line('Witness-backed sections', section_evidence_summary.get('witness_backed_count'))}"
            )
            lines.append(
                f"    {_coverage_line('Partial evidence', section_evidence_summary.get('partial_count'), section_evidence_summary.get('partial_sections'))}"
            )
            lines.append(
                f"    {_coverage_line('Thin evidence', section_evidence_summary.get('thin_count'), section_evidence_summary.get('thin_sections'))}"
            )
        if account_pressure_summary:
            lines.append(f"  Account pressure: {account_pressure_summary}")
        if isinstance(priority_account_names, list):
            cleaned_priority_names = [str(item or "").strip() for item in priority_account_names if str(item or "").strip()]
            if cleaned_priority_names:
                lines.append(f"  Priority accounts: {', '.join(cleaned_priority_names[:3])}")
        if account_pressure_disclaimer:
            lines.append(f"  Note: {account_pressure_disclaimer}")
        if isinstance(evidence_highlights, list):
            for highlight in evidence_highlights[:3]:
                text = str(highlight or "").strip()
                if text:
                    lines.append(f"  Witness: {text}")
        report_url = str(artifact.get("report_url") or "").strip()
        if report_url:
            lines.append(f"  Open: {report_url}")
        lines.append("")

    if manage_url:
        lines.extend([
            f"Manage in library: {manage_url}",
            "",
        ])

    lines.extend([
        "This delivery came from your saved recurring report subscription.",
        "",
        "--",
        "Churn Signals by Atlas Intelligence",
    ])
    return "\n".join(lines)
