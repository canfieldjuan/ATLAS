"""Email templates for Effingham Office Maids."""

from importlib import import_module
from typing import Any

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

_LAZY_EXPORT_MODULES = {
    "render_vendor_briefing_html": ".vendor_briefing",
    "render_checkout_confirmation_html": ".vendor_checkout_confirmation",
    "render_checkout_confirmation_text": ".vendor_checkout_confirmation",
    "render_report_delivery_html": ".vendor_report_delivery",
    "render_report_delivery_text": ".vendor_report_delivery",
    "render_report_subscription_delivery_html": ".report_subscription_delivery",
    "render_report_subscription_delivery_text": ".report_subscription_delivery",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
