"""
HTML and plain-text invoice templates for Effingham Office Maids.

Renders a professional invoice matching the branded PDF layout:
green header bar, business info, bill-to block, line-item table, totals, footer.
"""

from datetime import date
from typing import Optional

BUSINESS_NAME = "Effingham Office Maids"
BUSINESS_ADDRESS = "1901 S. 4th St. STE #1, Effingham IL 62401"
BUSINESS_PHONE = "(217) 207-3097"
BUSINESS_EMAIL = "info@effinghamofficemaids.com"
BUSINESS_WEBSITE = "effinghamofficemaids.com"


def _fmt(val, fallback: str = "") -> str:
    """Return string value or fallback for None/empty."""
    if val is None:
        return fallback
    s = str(val)
    return s if s.strip() else fallback


def _money(val) -> str:
    """Format a numeric value as $X.XX."""
    try:
        return f"${float(val):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _fmt_date(val) -> str:
    """Format a date-like value as MM/DD/YYYY."""
    if val is None:
        return ""
    if isinstance(val, date):
        return val.strftime("%m/%d/%Y")
    s = str(val)
    # Try ISO format
    try:
        return date.fromisoformat(s).strftime("%m/%d/%Y")
    except (ValueError, TypeError):
        return s


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def render_invoice_html(invoice: dict) -> str:
    """
    Render a full HTML invoice email from an invoice dict.

    Expected keys: invoice_number, issue_date, due_date, customer_name,
    customer_email, customer_phone, customer_address, contact_name,
    invoice_for, line_items, subtotal, tax_amount, discount_amount,
    total_amount, notes.
    """
    inv_number = _fmt(invoice.get("invoice_number"), "---")
    issue_date = _fmt_date(invoice.get("issue_date"))
    due_date = _fmt_date(invoice.get("due_date"))
    customer_name = _fmt(invoice.get("customer_name"), "Customer")
    customer_email = _fmt(invoice.get("customer_email"))
    customer_phone = _fmt(invoice.get("customer_phone"))
    customer_address = _fmt(invoice.get("customer_address"))
    contact_name = _fmt(invoice.get("contact_name"))
    invoice_for = _fmt(invoice.get("invoice_for"))

    items = invoice.get("line_items") or []
    subtotal = float(invoice.get("subtotal", 0))
    tax_amount = float(invoice.get("tax_amount", 0))
    discount_amount = float(invoice.get("discount_amount", 0))
    total_amount = float(invoice.get("total_amount", 0))
    notes = _fmt(invoice.get("notes"))

    # Detect optional columns
    has_date_col = any(item.get("date") for item in items)
    has_discount_col = any(float(item.get("discount", 0)) != 0 for item in items)

    # Rate label override
    rate_label = "RATE"
    if items and items[0].get("rate_label"):
        rate_label = str(items[0]["rate_label"]).upper()

    # Build line item rows
    item_rows = ""
    for item in items:
        desc = _fmt(item.get("description"), "")
        qty = item.get("quantity", 1)
        unit_price = float(item.get("unit_price", 0))
        flat_fee = float(item.get("flat_fee", 0))
        item_discount = float(item.get("discount", 0))
        # Amount: explicit or calculated
        if "amount" in item:
            amount = float(item["amount"])
        else:
            amount = (float(qty) * unit_price) + flat_fee - item_discount

        row = "<tr>"
        if has_date_col:
            row += f'<td style="padding:8px 12px;border-bottom:1px solid #e0e0e0;">{_fmt_date(item.get("date"))}</td>'
        row += f'<td style="padding:8px 12px;border-bottom:1px solid #e0e0e0;">{desc}</td>'
        row += f'<td style="padding:8px 12px;border-bottom:1px solid #e0e0e0;text-align:right;">{_money(unit_price)}</td>'
        row += f'<td style="padding:8px 12px;border-bottom:1px solid #e0e0e0;text-align:center;">{qty}</td>'
        row += f'<td style="padding:8px 12px;border-bottom:1px solid #e0e0e0;text-align:right;">{_money(flat_fee) if flat_fee else ""}</td>'
        if has_discount_col:
            row += f'<td style="padding:8px 12px;border-bottom:1px solid #e0e0e0;text-align:right;">{_money(item_discount) if item_discount else ""}</td>'
        row += f'<td style="padding:8px 12px;border-bottom:1px solid #e0e0e0;text-align:right;font-weight:600;">{_money(amount)}</td>'
        row += "</tr>"
        item_rows += row

    # Build header columns
    header_cols = ""
    if has_date_col:
        header_cols += '<th style="padding:10px 12px;text-align:left;color:#ffffff;font-weight:600;">DATE</th>'
    header_cols += '<th style="padding:10px 12px;text-align:left;color:#ffffff;font-weight:600;">DESCRIPTION</th>'
    header_cols += f'<th style="padding:10px 12px;text-align:right;color:#ffffff;font-weight:600;">{rate_label}</th>'
    header_cols += '<th style="padding:10px 12px;text-align:center;color:#ffffff;font-weight:600;">QTY</th>'
    header_cols += '<th style="padding:10px 12px;text-align:right;color:#ffffff;font-weight:600;">FLAT FEE</th>'
    if has_discount_col:
        header_cols += '<th style="padding:10px 12px;text-align:right;color:#ffffff;font-weight:600;">DISCOUNT</th>'
    header_cols += '<th style="padding:10px 12px;text-align:right;color:#ffffff;font-weight:600;">TOTAL</th>'

    # Bill-to details
    bill_to_lines = f'<strong style="font-size:15px;">{customer_name}</strong><br>'
    if customer_address:
        bill_to_lines += f'{customer_address}<br>'
    if customer_phone:
        bill_to_lines += f'{customer_phone}<br>'
    if customer_email:
        bill_to_lines += f'{customer_email}<br>'
    if contact_name:
        bill_to_lines += f'<br><strong>Attn:</strong> {contact_name}<br>'
    if invoice_for:
        bill_to_lines += f'<strong>Invoice For:</strong> {invoice_for}<br>'

    # Totals rows
    totals_html = f"""
    <tr>
      <td style="padding:6px 12px;text-align:right;color:#555;">Subtotal:</td>
      <td style="padding:6px 12px;text-align:right;">{_money(subtotal)}</td>
    </tr>"""
    if tax_amount > 0:
        totals_html += f"""
    <tr>
      <td style="padding:6px 12px;text-align:right;color:#555;">Tax:</td>
      <td style="padding:6px 12px;text-align:right;">{_money(tax_amount)}</td>
    </tr>"""
    if discount_amount > 0:
        totals_html += f"""
    <tr>
      <td style="padding:6px 12px;text-align:right;color:#555;">Discount:</td>
      <td style="padding:6px 12px;text-align:right;">-{_money(discount_amount)}</td>
    </tr>"""
    totals_html += f"""
    <tr>
      <td style="padding:8px 12px;text-align:right;font-weight:700;font-size:16px;border-top:2px solid #4CAF50;">Total Due:</td>
      <td style="padding:8px 12px;text-align:right;font-weight:700;font-size:16px;border-top:2px solid #4CAF50;">{_money(total_amount)}</td>
    </tr>"""

    # Notes section
    notes_html = ""
    if notes:
        notes_html = f"""
    <div style="margin-top:20px;padding:12px 16px;background:#f9f9f9;border-left:3px solid #4CAF50;color:#555;font-size:13px;">
      <strong>Notes:</strong><br>{notes}
    </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;font-family:Arial,Helvetica,sans-serif;color:#333;background:#f4f4f4;">
<div style="max-width:700px;margin:20px auto;background:#ffffff;border-radius:4px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">

  <!-- Green Header -->
  <div style="background:#4CAF50;padding:24px 32px;color:#ffffff;">
    <table width="100%" cellpadding="0" cellspacing="0" style="border:none;">
      <tr>
        <td style="vertical-align:middle;">
          <div style="font-size:24px;font-weight:700;letter-spacing:1px;">SERVICE INVOICE</div>
          <div style="font-size:13px;margin-top:4px;opacity:0.9;">{BUSINESS_NAME}</div>
        </td>
        <td style="text-align:right;vertical-align:middle;font-size:13px;line-height:1.7;">
          <strong>Invoice #:</strong> {inv_number}<br>
          <strong>Date:</strong> {issue_date}<br>
          <strong>Due:</strong> {due_date}
        </td>
      </tr>
    </table>
  </div>

  <!-- Business + Bill To -->
  <div style="padding:24px 32px;">
    <table width="100%" cellpadding="0" cellspacing="0" style="border:none;">
      <tr>
        <td style="vertical-align:top;width:50%;padding-right:16px;">
          <div style="font-size:11px;text-transform:uppercase;color:#888;margin-bottom:6px;letter-spacing:0.5px;">From</div>
          <strong style="font-size:15px;">{BUSINESS_NAME}</strong><br>
          <span style="color:#555;font-size:13px;line-height:1.7;">
            {BUSINESS_ADDRESS}<br>
            {BUSINESS_PHONE}<br>
            {BUSINESS_EMAIL}<br>
            {BUSINESS_WEBSITE}
          </span>
        </td>
        <td style="vertical-align:top;width:50%;padding-left:16px;">
          <div style="font-size:11px;text-transform:uppercase;color:#888;margin-bottom:6px;letter-spacing:0.5px;">Bill To</div>
          <span style="color:#555;font-size:13px;line-height:1.7;">
            {bill_to_lines}
          </span>
        </td>
      </tr>
    </table>
  </div>

  <!-- Line Items Table -->
  <div style="padding:0 32px;">
    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:13px;">
      <thead>
        <tr style="background:#4CAF50;">
          {header_cols}
        </tr>
      </thead>
      <tbody>
        {item_rows}
      </tbody>
    </table>
  </div>

  <!-- Totals -->
  <div style="padding:16px 32px;">
    <table style="margin-left:auto;border-collapse:collapse;font-size:14px;">
      {totals_html}
    </table>
  </div>

  {notes_html}

  <!-- Footer -->
  <div style="padding:20px 32px;border-top:1px solid #e0e0e0;font-size:12px;color:#888;text-align:center;line-height:1.6;">
    Make all checks payable to <strong>{BUSINESS_NAME}</strong><br>
    {BUSINESS_PHONE} &middot; {BUSINESS_EMAIL} &middot; {BUSINESS_WEBSITE}
  </div>

</div>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Plain-text fallback
# ---------------------------------------------------------------------------

def render_invoice_text(invoice: dict) -> str:
    """Render a plain-text version of the invoice for email body fallback."""
    inv_number = _fmt(invoice.get("invoice_number"), "---")
    issue_date = _fmt_date(invoice.get("issue_date"))
    due_date = _fmt_date(invoice.get("due_date"))
    customer_name = _fmt(invoice.get("customer_name"), "Customer")
    contact_name = _fmt(invoice.get("contact_name"))
    invoice_for = _fmt(invoice.get("invoice_for"))

    items = invoice.get("line_items") or []
    subtotal = float(invoice.get("subtotal", 0))
    tax_amount = float(invoice.get("tax_amount", 0))
    discount_amount = float(invoice.get("discount_amount", 0))
    total_amount = float(invoice.get("total_amount", 0))
    notes = _fmt(invoice.get("notes"))

    lines = [
        f"SERVICE INVOICE - {BUSINESS_NAME}",
        f"{'=' * 50}",
        f"Invoice #: {inv_number}",
        f"Date:      {issue_date}",
        f"Due:       {due_date}",
        "",
        f"From: {BUSINESS_NAME}",
        f"      {BUSINESS_ADDRESS}",
        f"      {BUSINESS_PHONE} | {BUSINESS_EMAIL}",
        "",
        f"Bill To: {customer_name}",
    ]
    if invoice.get("customer_address"):
        lines.append(f"         {invoice['customer_address']}")
    if invoice.get("customer_phone"):
        lines.append(f"         {invoice['customer_phone']}")
    if invoice.get("customer_email"):
        lines.append(f"         {invoice['customer_email']}")
    if contact_name:
        lines.append(f"    Attn: {contact_name}")
    if invoice_for:
        lines.append(f"    For:  {invoice_for}")

    lines.append("")
    lines.append(f"{'─' * 50}")
    lines.append("ITEMS:")
    lines.append(f"{'─' * 50}")

    for item in items:
        desc = _fmt(item.get("description"), "")
        qty = item.get("quantity", 1)
        unit_price = float(item.get("unit_price", 0))
        if "amount" in item:
            amount = float(item["amount"])
        else:
            amount = float(qty) * unit_price
        date_str = ""
        if item.get("date"):
            date_str = f"  [{_fmt_date(item['date'])}]"
        lines.append(f"  {desc}{date_str}")
        lines.append(f"    {qty} x {_money(unit_price)} = {_money(amount)}")

    lines.append(f"{'─' * 50}")
    lines.append(f"  Subtotal:  {_money(subtotal)}")
    if tax_amount > 0:
        lines.append(f"  Tax:       {_money(tax_amount)}")
    if discount_amount > 0:
        lines.append(f"  Discount: -{_money(discount_amount)}")
    lines.append(f"  TOTAL DUE: {_money(total_amount)}")
    lines.append(f"{'─' * 50}")

    if notes:
        lines.append("")
        lines.append(f"Notes: {notes}")

    lines.append("")
    lines.append(f"Make all checks payable to {BUSINESS_NAME}")
    lines.append(f"{BUSINESS_PHONE} | {BUSINESS_EMAIL} | {BUSINESS_WEBSITE}")

    return "\n".join(lines)
