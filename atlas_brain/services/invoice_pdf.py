"""PDF renderer for Atlas invoices.

Uses fpdf2 (pure Python, no system deps) to generate downloadable
invoice PDFs matching the branded HTML email layout.
"""

from __future__ import annotations

import io
from datetime import date
from typing import Any

from fpdf import FPDF

from ..templates.email.invoice import (
    BUSINESS_ADDRESS,
    BUSINESS_EMAIL,
    BUSINESS_NAME,
    BUSINESS_PHONE,
    BUSINESS_WEBSITE,
)

# -- Brand colors (RGB) matching the HTML template green --
_GREEN = (76, 175, 80)       # #4CAF50
_DARK = (51, 51, 51)         # #333333
_MUTED = (136, 136, 136)     # #888888
_LIGHT_BG = (244, 244, 244)  # #f4f4f4
_WHITE = (255, 255, 255)
_TABLE_BORDER = (224, 224, 224)  # #e0e0e0

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
})


def _safe(text: Any) -> str:
    if text is None:
        return ""
    s = str(text).translate(_UNICODE_MAP)
    return s.encode("latin-1", errors="replace").decode("latin-1")


def _money(val: Any) -> str:
    try:
        return f"${float(val):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _fmt_date(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, date):
        return val.strftime("%m/%d/%Y")
    try:
        return date.fromisoformat(str(val)[:10]).strftime("%m/%d/%Y")
    except (ValueError, TypeError):
        return str(val)


class InvoicePDF(FPDF):

    def header(self) -> None:
        # Green header bar
        self.set_fill_color(*_GREEN)
        self.rect(0, 0, self.w, 36, "F")
        self.set_y(8)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*_WHITE)
        self.cell(0, 10, "SERVICE INVOICE", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, BUSINESS_NAME, new_x="LMARGIN", new_y="NEXT")
        self.set_y(40)

    def footer(self) -> None:
        self.set_y(-20)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*_MUTED)
        self.cell(
            0, 5,
            f"Make all checks payable to {BUSINESS_NAME}",
            align="C", new_x="LMARGIN", new_y="NEXT",
        )
        self.cell(
            0, 5,
            f"{BUSINESS_PHONE}  |  {BUSINESS_EMAIL}  |  {BUSINESS_WEBSITE}",
            align="C",
        )


def render_invoice_pdf(invoice: dict[str, Any]) -> bytes:
    """Render an invoice dict to PDF bytes."""
    pdf = InvoicePDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.add_page()

    # --- Invoice details (right-aligned) ---
    pdf.set_y(10)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*_WHITE)
    details = [
        ("Invoice #:", _safe(invoice.get("invoice_number"))),
        ("Date:", _fmt_date(invoice.get("issue_date"))),
        ("Due:", _fmt_date(invoice.get("due_date"))),
    ]
    for label, value in details:
        pdf.set_x(130)
        pdf.cell(25, 5, label, align="R")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, f"  {value}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 9)

    # --- From / Bill To ---
    y_start = 44
    pdf.set_y(y_start)

    # From column
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*_MUTED)
    pdf.cell(90, 4, "FROM", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*_DARK)
    pdf.cell(90, 6, BUSINESS_NAME, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*_DARK)
    for line in [BUSINESS_ADDRESS, BUSINESS_PHONE, BUSINESS_EMAIL]:
        pdf.cell(90, 5, line, new_x="LMARGIN", new_y="NEXT")

    # Bill To column
    pdf.set_y(y_start)
    pdf.set_x(110)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*_MUTED)
    pdf.cell(0, 4, "BILL TO", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(110)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*_DARK)
    pdf.cell(0, 6, _safe(invoice.get("customer_name", "")), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    if invoice.get("customer_address"):
        pdf.set_x(110)
        pdf.cell(0, 5, _safe(invoice["customer_address"]), new_x="LMARGIN", new_y="NEXT")
    if invoice.get("customer_phone"):
        pdf.set_x(110)
        pdf.cell(0, 5, _safe(invoice["customer_phone"]), new_x="LMARGIN", new_y="NEXT")
    if invoice.get("customer_email"):
        pdf.set_x(110)
        pdf.cell(0, 5, _safe(invoice["customer_email"]), new_x="LMARGIN", new_y="NEXT")
    if invoice.get("contact_name"):
        pdf.set_x(110)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(12, 5, "Attn:")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, _safe(invoice["contact_name"]), new_x="LMARGIN", new_y="NEXT")
    if invoice.get("invoice_for"):
        pdf.set_x(110)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(12, 5, "For:")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, _safe(invoice["invoice_for"]), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)

    # --- Line items table ---
    # Detect if any line items have dates — if so, include DATE column
    items = invoice.get("line_items") or []
    parsed_items = []
    for item in items:
        if isinstance(item, str):
            import json
            try:
                item = json.loads(item)
            except Exception:
                continue
        if isinstance(item, dict):
            parsed_items.append(item)

    has_dates = any(item.get("date") for item in parsed_items)

    if has_dates:
        col_w = [27, 68, 15, 25, 22, 28]
        headers = ["DATE", "DESCRIPTION", "QTY", "FLAT FEE", "DISCOUNT", "TOTAL"]
    else:
        col_w = [85, 25, 20, 25, 30]
        headers = ["DESCRIPTION", "RATE", "QTY", "FLAT FEE", "TOTAL"]

    # Header row
    pdf.set_fill_color(*_GREEN)
    pdf.set_text_color(*_WHITE)
    pdf.set_font("Helvetica", "B", 8)
    for i, h in enumerate(headers):
        align = "L" if i <= (1 if has_dates else 0) else "R"
        pdf.cell(col_w[i], 7, h, align=align, fill=True)
    pdf.ln()

    # Data rows
    pdf.set_text_color(*_DARK)
    pdf.set_font("Helvetica", "", 9)
    for item in parsed_items:
        desc = _safe(item.get("description", ""))
        qty = item.get("quantity") or 1
        unit = float(item.get("unit_price") or item.get("rate") or 0)
        flat = item.get("flat_fee")
        discount = item.get("discount")
        total = float(item.get("amount") or (float(flat) if flat else qty * unit))
        if discount:
            total -= float(discount)

        if has_dates:
            pdf.cell(col_w[0], 6, _safe(item.get("date", "")), align="L")
            pdf.cell(col_w[1], 6, desc[:45], align="L")
            pdf.cell(col_w[2], 6, str(qty), align="R")
            pdf.cell(col_w[3], 6, _money(unit), align="R")
            pdf.cell(col_w[4], 6, _money(discount) if discount else "", align="R")
        else:
            pdf.cell(col_w[0], 6, desc[:55], align="L")
            pdf.cell(col_w[1], 6, _money(unit) if not flat else "", align="R")
            pdf.cell(col_w[2], 6, str(qty) if not flat else "", align="R")
            pdf.cell(col_w[3], 6, _money(flat) if flat else "", align="R")

        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(col_w[-1], 6, _money(total), align="R")
        pdf.set_font("Helvetica", "", 9)
        pdf.ln()
        pdf.set_draw_color(*_TABLE_BORDER)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())

    pdf.ln(4)

    # --- Totals ---
    x_label = 130
    x_value = 165

    def _total_row(label: str, value: str, bold: bool = False, border_top: bool = False) -> None:
        if border_top:
            pdf.set_draw_color(*_GREEN)
            pdf.set_line_width(0.6)
            pdf.line(x_label, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.set_line_width(0.2)
        size = 12 if bold else 9
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.set_x(x_label)
        pdf.set_text_color(*(_MUTED if not bold else _DARK))
        pdf.cell(x_value - x_label, 6, label, align="R")
        pdf.set_text_color(*_DARK)
        pdf.cell(0, 6, value, align="R", new_x="LMARGIN", new_y="NEXT")

    _total_row("Subtotal:", _money(invoice.get("subtotal")))
    tax = invoice.get("tax_amount")
    if tax and float(tax) > 0:
        meta = invoice.get("metadata") or {}
        tax_label = meta.get("tax_label", "Tax") if isinstance(meta, dict) else "Tax"
        _total_row(f"{tax_label}:", _money(tax))
    discount = invoice.get("discount_amount")
    if discount and float(discount) > 0:
        _total_row("Discount:", f"-{_money(discount)}")
    _total_row("Total Due:", _money(invoice.get("total_amount") or invoice.get("amount_due")), bold=True, border_top=True)

    # --- Notes ---
    notes = invoice.get("notes")
    if notes:
        pdf.ln(6)
        pdf.set_draw_color(*_GREEN)
        y_notes = pdf.get_y()
        pdf.set_x(pdf.l_margin + 4)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*_DARK)
        pdf.cell(0, 5, "Notes:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(pdf.l_margin + 4)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*_MUTED)
        pdf.multi_cell(0, 5, _safe(notes))
        y_end = pdf.get_y()
        pdf.set_line_width(0.8)
        pdf.line(pdf.l_margin, y_notes, pdf.l_margin, y_end)
        pdf.set_line_width(0.2)

    # --- Status badge ---
    status = str(invoice.get("status") or "").upper()
    if status and status != "DRAFT":
        pdf.ln(6)
        if status == "PAID":
            pdf.set_text_color(*_GREEN)
        elif status == "VOID":
            color = (231, 76, 60)
            pdf.set_text_color(*color)
        else:
            pdf.set_text_color(*_MUTED)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, status, align="C")

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
