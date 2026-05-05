"""Shared delivery helpers for Competitive Intelligence vendor briefings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...config import settings
from ...services.campaign_sender import get_campaign_sender


class VendorBriefingDeliveryNotConfigured(RuntimeError):
    pass


@dataclass(frozen=True)
class VendorBriefingDeliveryResult:
    provider_message_id: str | None


def vendor_briefing_delivery_configured() -> bool:
    cfg = settings.campaign_sequence
    return bool(cfg.resend_api_key and cfg.resend_from_email)


def require_vendor_briefing_delivery_configured() -> None:
    if not vendor_briefing_delivery_configured():
        raise VendorBriefingDeliveryNotConfigured("Resend not configured")


def _format_subject_template(template: str, vendor_name: str) -> str:
    return template.format(vendor_name=vendor_name)


def build_vendor_briefing_subject(
    vendor_name: str,
    briefing_data: dict[str, Any],
) -> str:
    cfg = settings.b2b_churn
    challenger_mode = bool(briefing_data.get("challenger_mode"))
    if briefing_data.get("prospect_mode"):
        template = (
            cfg.vendor_briefing_prospect_sales_subject_template
            if challenger_mode
            else cfg.vendor_briefing_prospect_churn_subject_template
        )
        return _format_subject_template(template, vendor_name)
    if briefing_data.get("is_gated_delivery"):
        template = (
            cfg.vendor_briefing_gated_sales_subject_template
            if challenger_mode
            else cfg.vendor_briefing_gated_churn_subject_template
        )
        return _format_subject_template(template, vendor_name)
    template = (
        cfg.vendor_briefing_standard_sales_subject_template
        if challenger_mode
        else cfg.vendor_briefing_standard_churn_subject_template
    )
    return _format_subject_template(template, vendor_name)


def build_vendor_briefing_from_address() -> str:
    require_vendor_briefing_delivery_configured()
    sender_name = settings.b2b_churn.vendor_briefing_sender_name
    from_email = settings.campaign_sequence.resend_from_email
    return f"{sender_name} <{from_email}>"


def build_vendor_briefing_tags(vendor_name: str) -> list[dict[str, str]]:
    cfg = settings.b2b_churn
    return [
        {
            "name": cfg.vendor_briefing_tag_type_name,
            "value": cfg.vendor_briefing_tag_type_value,
        },
        {"name": cfg.vendor_briefing_tag_vendor_name, "value": vendor_name},
    ]


async def send_vendor_briefing_delivery(
    *,
    to_email: str,
    vendor_name: str,
    subject: str,
    briefing_html: str,
) -> VendorBriefingDeliveryResult:
    sender = get_campaign_sender()
    result = await sender.send(
        to=to_email,
        from_email=build_vendor_briefing_from_address(),
        subject=subject,
        body=briefing_html,
        tags=build_vendor_briefing_tags(vendor_name),
    )
    return VendorBriefingDeliveryResult(
        provider_message_id=result.get("id") if isinstance(result, dict) else None
    )
