"""Deterministic plain-text payment receipt for Effingham Office Maids."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import UUID

from .invoice import (
    BUSINESS_ADDRESS,
    BUSINESS_EMAIL,
    BUSINESS_NAME,
    BUSINESS_PHONE,
    BUSINESS_WEBSITE,
)

_PAYMENT_METHOD_LABELS = {
    "check": "Check",
    "ach": "ACH",
    "square": "Square",
}


def receipt_number_for_payment(payment_id: UUID) -> str:
    """Return the stable receipt number for one persisted payment."""
    return f"EOM-RCP-{payment_id}"


def render_residential_payment_receipt(
    *,
    payment_id: UUID,
    customer_name: str,
    payer_name: str,
    total_amount: Decimal,
    payment_method: str,
    reference: str | None,
    received_date: date,
) -> tuple[str, str, str]:
    """Render the immutable receipt number, subject, and plain-text body."""
    receipt_number = receipt_number_for_payment(payment_id)
    method = _PAYMENT_METHOD_LABELS[payment_method]
    reference_label = "Check number" if payment_method == "check" else "Reference"
    reference_value = reference or "Not provided"
    check_status = (
        "\nWe have received your check. It has not yet cleared.\n"
        if payment_method == "check"
        else ""
    )
    subject = f"Payment received — receipt {receipt_number}"
    body = (
        f"Thank you. {BUSINESS_NAME} has received your payment.\n\n"
        f"Customer: {customer_name}\n"
        f"Payer: {payer_name}\n"
        f"Receipt number: {receipt_number}\n"
        f"Amount received: ${total_amount:.2f}\n"
        f"Payment method: {method}\n"
        f"{reference_label}: {reference_value}\n"
        f"Date received: {received_date.strftime('%B')} {received_date.day}, "
        f"{received_date.year}\n"
        f"{check_status}\n"
        f"Questions? Contact {BUSINESS_NAME}\n"
        f"{BUSINESS_ADDRESS}\n"
        f"{BUSINESS_PHONE}\n"
        f"{BUSINESS_EMAIL}\n"
        f"{BUSINESS_WEBSITE}\n"
    )
    return receipt_number, subject, body
