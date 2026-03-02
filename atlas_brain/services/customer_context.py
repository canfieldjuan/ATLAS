"""
Customer Context Service -- unified cross-reference layer.

Pulls together everything Atlas knows about a customer from CRM,
call transcripts, appointments, sent emails, and interaction logs.

Usage:
    from atlas_brain.services.customer_context import get_customer_context_service

    svc = get_customer_context_service()
    ctx = await svc.get_context(contact_id="...")
    ctx = await svc.get_context_by_phone("+16185551234")
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("atlas.services.customer_context")

_FREE_EMAIL_DOMAINS = frozenset({
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "icloud.com", "mail.com", "protonmail.com", "zoho.com", "yandex.com",
    "gmx.com", "live.com",
})


@dataclass
class CustomerContext:
    """Everything Atlas knows about a customer, in one place."""

    contact: dict[str, Any] = field(default_factory=dict)
    interactions: list[dict[str, Any]] = field(default_factory=list)
    appointments: list[dict[str, Any]] = field(default_factory=list)
    call_transcripts: list[dict[str, Any]] = field(default_factory=list)
    sent_emails: list[dict[str, Any]] = field(default_factory=list)
    inbox_emails: list[dict[str, Any]] = field(default_factory=list)
    sms_messages: list[dict[str, Any]] = field(default_factory=list)
    invoices: list[dict[str, Any]] = field(default_factory=list)
    b2b_churn_signals: list[dict[str, Any]] = field(default_factory=list)

    @property
    def contact_id(self) -> Optional[str]:
        cid = self.contact.get("id")
        return str(cid) if cid else None

    @property
    def display_name(self) -> str:
        return self.contact.get("full_name") or "Unknown"

    @property
    def is_empty(self) -> bool:
        return not self.contact


class CustomerContextService:
    """Aggregates customer data from all Atlas data sources."""

    async def get_context(
        self,
        contact_id: str,
        max_interactions: int = 10,
        max_calls: int = 10,
        max_appointments: int = 10,
        max_emails: int = 10,
        max_sms: int = 10,
        max_invoices: int = 10,
    ) -> CustomerContext:
        """Build full customer context by contact_id.

        Fetches all data sources in parallel via asyncio.gather.
        Each source is fail-open -- a single failure doesn't block others.
        """
        from .crm_provider import get_crm_provider

        crm = get_crm_provider()

        contact = await crm.get_contact(contact_id)
        if not contact:
            return CustomerContext()

        return await self._gather(
            contact, contact_id,
            max_interactions, max_calls, max_appointments, max_emails, max_sms,
            max_invoices,
        )

    async def get_context_by_phone(
        self, phone: str, **kwargs,
    ) -> CustomerContext:
        """Resolve a phone number to a contact, then build context."""
        from .crm_provider import get_crm_provider

        results = await get_crm_provider().search_contacts(phone=phone)
        if not results:
            return CustomerContext()

        contact = results[0]
        contact_id = str(contact["id"])
        return await self._gather(contact, contact_id, **kwargs)

    async def get_context_by_email(
        self, email: str, **kwargs,
    ) -> CustomerContext:
        """Resolve an email to a contact, then build context."""
        from .crm_provider import get_crm_provider

        results = await get_crm_provider().search_contacts(email=email)
        if not results:
            return CustomerContext()

        contact = results[0]
        contact_id = str(contact["id"])
        return await self._gather(contact, contact_id, **kwargs)

    async def _gather(
        self,
        contact: dict,
        contact_id: str,
        max_interactions: int = 10,
        max_calls: int = 10,
        max_appointments: int = 10,
        max_emails: int = 10,
        max_sms: int = 10,
        max_invoices: int = 10,
    ) -> CustomerContext:
        """Fetch all supplementary data in parallel."""
        from .crm_provider import get_crm_provider
        from ..storage.repositories.call_transcript import get_call_transcript_repo
        from ..storage.repositories.sms_message import get_sms_message_repo
        from ..storage.repositories.invoice import get_invoice_repo

        crm = get_crm_provider()
        call_repo = get_call_transcript_repo()
        sms_repo = get_sms_message_repo()
        inv_repo = get_invoice_repo()

        async def _safe(coro, label: str, default=None):
            try:
                return await coro
            except Exception as e:
                logger.warning("CustomerContext %s failed: %s", label, e)
                return default if default is not None else []

        interactions_coro = _safe(
            crm.get_interactions(contact_id, limit=max_interactions),
            "interactions",
        )
        appointments_coro = _safe(
            crm.get_contact_appointments(contact_id),
            "appointments",
        )
        calls_coro = _safe(
            call_repo.get_by_contact_id(contact_id, limit=max_calls),
            "call_transcripts",
        )
        emails_coro = _safe(
            self._get_sent_emails(contact, max_emails),
            "sent_emails",
        )
        inbox_coro = _safe(
            self._get_inbox_emails(contact, max_emails),
            "inbox_emails",
        )
        sms_coro = _safe(
            sms_repo.get_by_contact_id(contact_id, limit=max_sms),
            "sms_messages",
        )
        invoices_coro = _safe(
            inv_repo.get_by_contact_id(contact_id, limit=max_invoices),
            "invoices",
        )
        b2b_coro = _safe(
            self._get_b2b_churn_signals(contact),
            "b2b_churn_signals",
        )

        interactions, appointments, calls, emails, inbox, sms, invoices, b2b = await asyncio.gather(
            interactions_coro, appointments_coro, calls_coro,
            emails_coro, inbox_coro, sms_coro, invoices_coro, b2b_coro,
        )

        return CustomerContext(
            contact=contact,
            interactions=interactions,
            appointments=appointments[:max_appointments],
            call_transcripts=calls,
            sent_emails=emails,
            inbox_emails=inbox,
            sms_messages=sms,
            invoices=invoices,
            b2b_churn_signals=b2b,
        )

    async def _get_b2b_churn_signals(self, contact: dict) -> list[dict[str, Any]]:
        """Look up B2B churn signals for the contact's company domain.

        Extracts the email domain, skips free providers, derives a company
        hint, and queries b2b_churn_signals.company_churn_list JSONB.
        Gated by settings.b2b_churn.context_enrichment_enabled.
        """
        from ..config import settings

        if not settings.b2b_churn.context_enrichment_enabled:
            return []

        email_addr = contact.get("email")
        if not email_addr or "@" not in email_addr:
            return []

        domain = email_addr.rsplit("@", 1)[1].lower()
        if domain in _FREE_EMAIL_DOMAINS:
            return []

        # Derive company hint: strip TLD  ("acme.co.uk" -> "acme")
        company_hint = domain.split(".")[0]
        if not company_hint:
            return []

        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return []

        try:
            rows = await pool.fetch(
                """
                SELECT vendor_name, product_category, avg_urgency_score,
                       top_pain_categories, top_competitors,
                       decision_maker_churn_rate, price_complaint_rate
                FROM b2b_churn_signals
                WHERE EXISTS (
                    SELECT 1 FROM jsonb_array_elements(company_churn_list) AS c
                    WHERE c->>'company' ILIKE '%' || $1 || '%'
                )
                ORDER BY avg_urgency_score DESC
                LIMIT 5
                """,
                company_hint,
            )
            results = []
            for r in rows:
                results.append({
                    "vendor_name": r["vendor_name"],
                    "product_category": r["product_category"],
                    "avg_urgency_score": float(r["avg_urgency_score"]) if r["avg_urgency_score"] else 0,
                    "top_pain_categories": json.loads(r["top_pain_categories"]) if isinstance(r["top_pain_categories"], str) else (r["top_pain_categories"] or []),
                    "top_competitors": json.loads(r["top_competitors"]) if isinstance(r["top_competitors"], str) else (r["top_competitors"] or []),
                    "decision_maker_churn_rate": float(r["decision_maker_churn_rate"]) if r["decision_maker_churn_rate"] else None,
                    "price_complaint_rate": float(r["price_complaint_rate"]) if r["price_complaint_rate"] else None,
                })
            return results
        except Exception as e:
            logger.warning("B2B churn signal lookup failed: %s", e)
            return []

    async def _get_sent_emails(
        self, contact: dict, limit: int,
    ) -> list[dict]:
        """Find sent emails addressed to this contact's email."""
        email_addr = contact.get("email")
        if not email_addr:
            return []

        from ..storage.repositories.email import get_email_repo

        repo = get_email_repo()
        results = await repo.query(to_address=email_addr, limit=limit)
        return [self._email_to_dict(e) for e in results]

    async def _get_inbox_emails(
        self, contact: dict, limit: int,
    ) -> list[dict]:
        """
        Find recent inbound emails from this contact via IMAP/Gmail.

        Searches for messages where the sender matches the contact's email address.
        Uses CompositeEmailProvider (IMAP preferred; Gmail API fallback).
        Fail-open: returns [] if email address is missing or provider unavailable.
        """
        email_addr = contact.get("email")
        if not email_addr:
            return []

        try:
            from .email_provider import get_email_provider

            provider = get_email_provider()
            messages = await provider.list_messages(
                query=f"from:{email_addr}",
                max_results=limit,
            )
            return messages
        except Exception as exc:
            logger.warning("_get_inbox_emails failed for %s: %s", email_addr, exc)
            return []

    @staticmethod
    def _email_to_dict(email) -> dict:
        """Convert a SentEmail dataclass/namedtuple to a plain dict."""
        if hasattr(email, "__dict__"):
            return {k: v for k, v in email.__dict__.items() if not k.startswith("_")}
        return dict(email)


_customer_context_service: Optional[CustomerContextService] = None


def get_customer_context_service() -> CustomerContextService:
    """Get the global CustomerContextService singleton."""
    global _customer_context_service
    if _customer_context_service is None:
        _customer_context_service = CustomerContextService()
    return _customer_context_service
