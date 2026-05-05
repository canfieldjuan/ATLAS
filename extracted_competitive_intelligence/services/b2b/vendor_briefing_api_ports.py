"""Runtime ports for the extracted vendor briefing API surface."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote

from ...config import settings
from ...services.email_provider import get_email_provider
from ...templates.email.vendor_checkout_confirmation import (
    render_checkout_confirmation_html,
    render_checkout_confirmation_text,
)
from ...templates.email.vendor_report_delivery import render_report_delivery_html
from .pdf_renderer import render_vendor_full_report_pdf

logger = logging.getLogger("atlas.extracted_competitive.vendor_briefing_api_ports")


class VendorBriefingAPIError(RuntimeError):
    pass


class VendorBriefingAPINotConfigured(VendorBriefingAPIError):
    pass


@dataclass(frozen=True)
class VendorCheckoutSession:
    url: str


@dataclass(frozen=True)
class VendorCheckoutSessionInfo:
    customer_email: str
    vendor_name: str
    tier: str
    source: str
    payment_status: str


@dataclass(frozen=True)
class GatedReportDelivery:
    provider_message_id: str | None = None


@dataclass(frozen=True)
class CheckoutConfirmationDelivery:
    provider_message_id: str | None = None


@runtime_checkable
class VendorBriefingAPIPort(Protocol):
    async def create_checkout_session(
        self,
        *,
        vendor_name: str,
        tier: str,
        customer_email: str | None = None,
    ) -> VendorCheckoutSession:
        ...

    async def retrieve_checkout_session(
        self,
        session_id: str,
    ) -> VendorCheckoutSessionInfo:
        ...

    async def send_gated_report_email(
        self,
        *,
        vendor_name: str,
        recipient_email: str,
        report_data: dict[str, Any],
        briefing_data: dict[str, Any],
    ) -> GatedReportDelivery:
        ...

    async def send_checkout_confirmation_email(
        self,
        *,
        vendor_name: str,
        tier: str,
        customer_email: str,
    ) -> CheckoutConfirmationDelivery:
        ...


_configured_port: VendorBriefingAPIPort | None = None


def configure_vendor_briefing_api_port(
    port: VendorBriefingAPIPort | None,
) -> None:
    global _configured_port
    _configured_port = port


def get_vendor_briefing_api_port() -> VendorBriefingAPIPort:
    if _configured_port is not None:
        return _configured_port
    return DefaultVendorBriefingAPIPort()


async def create_vendor_checkout_session(
    *,
    vendor_name: str,
    tier: str,
    customer_email: str | None = None,
) -> VendorCheckoutSession:
    return await get_vendor_briefing_api_port().create_checkout_session(
        vendor_name=vendor_name,
        tier=tier,
        customer_email=customer_email,
    )


async def retrieve_vendor_checkout_session(
    session_id: str,
) -> VendorCheckoutSessionInfo:
    return await get_vendor_briefing_api_port().retrieve_checkout_session(session_id)


async def send_gated_report_email(
    *,
    vendor_name: str,
    recipient_email: str,
    report_data: dict[str, Any],
    briefing_data: dict[str, Any],
) -> GatedReportDelivery:
    return await get_vendor_briefing_api_port().send_gated_report_email(
        vendor_name=vendor_name,
        recipient_email=recipient_email,
        report_data=report_data,
        briefing_data=briefing_data,
    )


async def send_checkout_confirmation_email(
    *,
    vendor_name: str,
    tier: str,
    customer_email: str,
) -> CheckoutConfirmationDelivery:
    return await get_vendor_briefing_api_port().send_checkout_confirmation_email(
        vendor_name=vendor_name,
        tier=tier,
        customer_email=customer_email,
    )


class DefaultVendorBriefingAPIPort:
    def _stripe(self):
        try:
            import stripe  # type: ignore[import-not-found]
        except Exception as exc:
            raise VendorBriefingAPINotConfigured("Stripe SDK is not available") from exc
        return stripe

    def _require_stripe_config(self):
        cfg = settings.saas_auth
        if not cfg.stripe_secret_key:
            raise VendorBriefingAPINotConfigured("Stripe is not configured")
        stripe = self._stripe()
        stripe.api_key = cfg.stripe_secret_key
        return stripe, cfg

    async def create_checkout_session(
        self,
        *,
        vendor_name: str,
        tier: str,
        customer_email: str | None = None,
    ) -> VendorCheckoutSession:
        stripe, cfg = self._require_stripe_config()
        price_id = (
            cfg.stripe_vendor_standard_price_id
            if tier == "standard"
            else cfg.stripe_vendor_pro_price_id
        )
        if not price_id:
            raise VendorBriefingAPINotConfigured(
                f"No Stripe price configured for vendor {tier} tier"
            )

        report_base_url = _configured_report_base_url()
        vendor_encoded = quote(vendor_name)
        session_params: dict[str, Any] = {
            "mode": "subscription",
            "line_items": [{"price": price_id, "quantity": 1}],
            "success_url": (
                f"{report_base_url}?vendor={vendor_encoded}"
                "&checkout=success&session_id={CHECKOUT_SESSION_ID}"
            ),
            "cancel_url": (
                f"{report_base_url}?vendor={vendor_encoded}&checkout=cancelled"
            ),
            "metadata": {
                "vendor_name": vendor_name,
                "tier": tier,
                "source": "vendor_briefing_report",
            },
        }
        if customer_email:
            session_params["customer_email"] = customer_email.lower()

        try:
            session = stripe.checkout.Session.create(**session_params)
        except stripe.StripeError as exc:
            raise VendorBriefingAPIError("Failed to create checkout session") from exc
        return VendorCheckoutSession(url=session.url)

    async def retrieve_checkout_session(
        self,
        session_id: str,
    ) -> VendorCheckoutSessionInfo:
        stripe, _cfg = self._require_stripe_config()
        try:
            session = stripe.checkout.Session.retrieve(session_id)
        except stripe.StripeError as exc:
            raise VendorBriefingAPIError("Invalid session") from exc

        customer_email = (
            session.customer_details.email
            if session.customer_details
            else session.customer_email
        ) or ""
        meta = session.metadata or {}
        return VendorCheckoutSessionInfo(
            customer_email=customer_email,
            vendor_name=meta.get("vendor_name", ""),
            tier=meta.get("tier", "standard"),
            source=meta.get("source", ""),
            payment_status=getattr(session, "payment_status", "") or "",
        )

    async def send_gated_report_email(
        self,
        *,
        vendor_name: str,
        recipient_email: str,
        report_data: dict[str, Any],
        briefing_data: dict[str, Any],
    ) -> GatedReportDelivery:
        cfg = settings.campaign_sequence
        if not cfg.resend_api_key or not cfg.resend_from_email:
            raise VendorBriefingAPINotConfigured("Resend is not configured")

        pdf_bytes = render_vendor_full_report_pdf(
            vendor_name=vendor_name,
            report_data=report_data,
            briefing_data=briefing_data,
        )
        sender_name = settings.b2b_churn.vendor_briefing_sender_name
        subject = f"Your {vendor_name} Churn Intelligence Report"
        slug = vendor_name.lower().replace(" ", "-")

        payload = {
            "from": f"{sender_name} <{cfg.resend_from_email}>",
            "to": [recipient_email],
            "subject": subject,
            "html": render_report_delivery_html(vendor_name),
            "reply_to": _configured_reply_to_email(),
            "attachments": [{
                "filename": f"{slug}-churn-report.pdf",
                "content": base64.b64encode(pdf_bytes).decode("utf-8"),
            }],
        }

        try:
            import httpx as _httpx
        except Exception as exc:
            raise VendorBriefingAPINotConfigured("httpx is not available") from exc

        try:
            async with _httpx.AsyncClient(
                timeout=_configured_resend_timeout_seconds()
            ) as client:
                resp = await client.post(
                    _configured_resend_api_url(),
                    headers={
                        "Authorization": f"Bearer {cfg.resend_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise VendorBriefingAPIError("Failed to send report") from exc

        message_id = data.get("id") if isinstance(data, dict) else None
        logger.info("Gate report email sent via Resend: %s", message_id)
        return GatedReportDelivery(provider_message_id=message_id)

    async def send_checkout_confirmation_email(
        self,
        *,
        vendor_name: str,
        tier: str,
        customer_email: str,
    ) -> CheckoutConfirmationDelivery:
        html = render_checkout_confirmation_html(vendor_name, tier, customer_email)
        text = render_checkout_confirmation_text(vendor_name, tier)
        result = await get_email_provider().send(
            to=[customer_email],
            subject=f"Subscription Confirmed: {vendor_name} Churn Intelligence",
            body=text,
            html=html,
            reply_to=_configured_reply_to_email(),
        )
        message_id = result.get("id") if isinstance(result, dict) else None
        return CheckoutConfirmationDelivery(provider_message_id=message_id)


def _configured_report_base_url() -> str:
    b2b_cfg = settings.b2b_churn
    configured = str(
        getattr(b2b_cfg, "vendor_briefing_report_base_url", "")
        or b2b_cfg.vendor_briefing_gate_base_url
    ).strip()
    if not configured:
        raise VendorBriefingAPINotConfigured("Vendor report base URL is not configured")
    return configured.rstrip("?")


def _configured_reply_to_email() -> str:
    configured = str(
        getattr(settings.b2b_churn, "vendor_briefing_reply_to_email", "")
    ).strip()
    if not configured:
        raise VendorBriefingAPINotConfigured(
            "Vendor briefing reply-to email is not configured"
        )
    return configured


def _configured_resend_api_url() -> str:
    configured = str(
        getattr(settings.campaign_sequence, "resend_api_url", "")
    ).strip()
    if not configured:
        raise VendorBriefingAPINotConfigured("Resend API URL is not configured")
    return configured


def _configured_resend_timeout_seconds() -> float:
    return float(getattr(settings.campaign_sequence, "resend_timeout_seconds"))
