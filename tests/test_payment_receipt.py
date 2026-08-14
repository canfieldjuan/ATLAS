"""Direct, deterministic coverage for the EOM residential receipt template."""

from datetime import date
from decimal import Decimal
from uuid import UUID

from atlas_brain.templates.email.invoice import (
    BUSINESS_ADDRESS,
    BUSINESS_EMAIL,
    BUSINESS_NAME,
    BUSINESS_PHONE,
    BUSINESS_WEBSITE,
)
from atlas_brain.templates.email.payment_receipt import (
    receipt_number_for_payment,
    render_residential_payment_receipt,
)


PAYMENT_ID = UUID("12345678-1234-5678-1234-567812345678")


def test_check_receipt_is_deterministic_and_never_claims_clearing() -> None:
    receipt_number, subject, body = render_residential_payment_receipt(
        payment_id=PAYMENT_ID,
        customer_name="Riley Residence",
        payer_name="Riley Customer",
        total_amount=Decimal("125.00"),
        payment_method="check",
        reference="1042",
        received_date=date(2026, 8, 12),
    )

    assert receipt_number == receipt_number_for_payment(PAYMENT_ID)
    assert receipt_number == "EOM-RCP-12345678-1234-5678-1234-567812345678"
    assert subject == f"Payment received — receipt {receipt_number}"
    assert "Customer: Riley Residence" in body
    assert "Payer: Riley Customer" in body
    assert "Amount received: $125.00" in body
    assert "Payment method: Check" in body
    assert "Check number: 1042" in body
    assert "Date received: August 12, 2026" in body
    assert "We have received your check. It has not yet cleared." in body
    for contact_detail in (
        BUSINESS_NAME,
        BUSINESS_ADDRESS,
        BUSINESS_PHONE,
        BUSINESS_EMAIL,
        BUSINESS_WEBSITE,
    ):
        assert contact_detail in body


def test_ach_receipt_uses_reference_without_check_clearing_language() -> None:
    _, _, body = render_residential_payment_receipt(
        payment_id=PAYMENT_ID,
        customer_name="Riley Residence",
        payer_name="Riley Customer",
        total_amount=Decimal("90.50"),
        payment_method="ach",
        reference="ACH-12345",
        received_date=date(2026, 8, 13),
    )

    assert "Amount received: $90.50" in body
    assert "Payment method: ACH" in body
    assert "Reference: ACH-12345" in body
    assert "We have received your check." not in body
    assert "It has not yet cleared." not in body
