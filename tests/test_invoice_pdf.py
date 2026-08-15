"""Exact-money regression coverage for branded invoice PDF rendering."""

from __future__ import annotations

import inspect
from datetime import date
from decimal import Decimal

import pytest

from atlas_brain.services import invoice_pdf
from atlas_brain.services.invoice_pdf import InvoicePDF, _line_total, _money, render_invoice_pdf


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("10.005"), "$10.01"),
        ("10.004", "$10.00"),
        (9650, "$9,650.00"),
        ("96.50", "$96.50"),
        ("NaN", "$0.00"),
        ("Infinity", "$0.00"),
        ("1e999999", "$0.00"),
        ("not-money", "$0.00"),
    ],
)
def test_money_uses_finite_decimal_round_half_up(value, expected):
    assert _money(value) == expected


def test_line_total_uses_exact_cent_values_and_preserves_discount_order():
    total = _line_total(
        {
            "amount": "30.015",
            "discount": Decimal("0.005"),
            "unit_price": "10.005",
        },
        "3",
    )

    assert total == Decimal("30.01")


def test_rendered_pdf_uses_exact_money_for_line_tax_and_discount(monkeypatch):
    captured: list[str] = []
    original_cell = InvoicePDF.cell

    def capture_cell(self, *args, **kwargs):
        if "text" in kwargs:
            captured.append(str(kwargs["text"]))
        elif len(args) >= 3:
            captured.append(str(args[2]))
        return original_cell(self, *args, **kwargs)

    monkeypatch.setattr(InvoicePDF, "cell", capture_cell)

    pdf = render_invoice_pdf(
        {
            "invoice_number": "INV-2026-Mar-0001",
            "issue_date": date(2026, 3, 1),
            "due_date": date(2026, 3, 31),
            "customer_name": "Acme Office",
            "customer_email": "billing@example.test",
            "line_items": [
                {
                    "description": "Commercial cleaning",
                    "quantity": "3",
                    "unit_price": "10.005",
                    "amount": "30.015",
                    "discount": "0.005",
                }
            ],
            "subtotal": "30.015",
            "tax_amount": "0.005",
            "discount_amount": "0.005",
            "total_amount": "30.015",
            "metadata": {"tax_label": "Sales tax"},
            "status": "draft",
        }
    )

    assert pdf.startswith(b"%PDF-")
    assert "$10.01" in captured
    assert "$30.01" in captured
    assert "$30.02" in captured
    assert "$0.01" in captured
    assert "Sales tax:" in captured


def test_renderer_has_no_binary_float_money_conversion():
    assert "float(" not in inspect.getsource(invoice_pdf)
