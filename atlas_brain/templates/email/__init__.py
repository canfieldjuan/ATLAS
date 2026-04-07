"""Email templates for Effingham Office Maids."""

from .estimate_confirmation import (
    BUSINESS_NAME,
    BUSINESS_PHONE,
    BUSINESS_EMAIL,
    BUSINESS_WEBSITE,
    format_business_email,
    format_residential_email,
)

from .proposal import (
    format_business_proposal,
    format_residential_proposal,
)

from .invoice import (
    render_invoice_html,
    render_invoice_text,
)

from .vendor_briefing import render_vendor_briefing_html
from .vendor_checkout_confirmation import (
    render_checkout_confirmation_html,
    render_checkout_confirmation_text,
)
from .vendor_report_delivery import (
    render_report_delivery_html,
    render_report_delivery_text,
)
from .report_subscription_delivery import (
    render_report_subscription_delivery_html,
    render_report_subscription_delivery_text,
)

__all__ = [
    "BUSINESS_NAME",
    "BUSINESS_PHONE",
    "BUSINESS_EMAIL",
    "BUSINESS_WEBSITE",
    "format_business_email",
    "format_residential_email",
    "format_business_proposal",
    "format_residential_proposal",
    "render_invoice_html",
    "render_invoice_text",
    "render_vendor_briefing_html",
    "render_checkout_confirmation_html",
    "render_checkout_confirmation_text",
    "render_report_delivery_html",
    "render_report_delivery_text",
    "render_report_subscription_delivery_html",
    "render_report_subscription_delivery_text",
]
