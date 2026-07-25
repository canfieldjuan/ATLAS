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
import logging
import re
from dataclasses import dataclass, field
from email import policy
from email.parser import Parser
from typing import Any, Optional

logger = logging.getLogger("atlas.services.customer_context")

_FREE_EMAIL_DOMAINS = frozenset({
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "icloud.com", "mail.com", "protonmail.com", "zoho.com", "yandex.com",
    "gmx.com", "live.com",
})
_SCOPED_INBOX_CANDIDATE_LIMIT = 50


@dataclass
class CustomerContext:
    """Everything Atlas knows about a customer, in one place."""

    contact: dict[str, Any] = field(default_factory=dict)
    interactions: list[dict[str, Any]] = field(default_factory=list)
    appointments: list[dict[str, Any]] = field(default_factory=list)
    call_transcripts: list[dict[str, Any]] = field(default_factory=list)
    sent_emails: list[dict[str, Any]] = field(default_factory=list)
    inbox_emails: list[dict[str, Any]] = field(default_factory=list)
    inbox_email_source_omitted: bool = False
    inbox_email_query_address: str | None = None
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
        business_context_id: Optional[str] = None,
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
            business_context_id=business_context_id,
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
        business_context_id: Optional[str] = None,
    ) -> CustomerContext:
        """Fetch all supplementary data in parallel.

        ``business_context_id`` scopes the tenant-stamped child sources
        (appointments strictly; call transcripts tenant-plus-NULL) inside
        their SQL, before per-source limits apply.
        """
        inbox_max_emails = (
            max(0, min(max_emails, 50))
            if business_context_id is not None
            else max_emails
        )

        from .crm_provider import get_crm_provider
        from ..storage.repositories.call_transcript import get_call_transcript_repo
        from ..storage.repositories.sms_message import get_sms_message_repo
        from ..storage.repositories.invoice import get_invoice_repo

        crm = get_crm_provider()
        call_repo = get_call_transcript_repo()
        sms_repo = get_sms_message_repo()
        inv_repo = get_invoice_repo()

        inbox_email_source_omitted = False
        inbox_email_query_address = None
        inbox_provider = None
        if business_context_id is not None:
            from .email_provider import (
                UnmappedInboxContextError,
                get_scoped_inbox_provider,
            )

            try:
                inbox_provider = await get_scoped_inbox_provider(
                    business_context_id
                )
            except UnmappedInboxContextError:
                logger.info(
                    "CustomerContext inbox omitted: no mailbox binding for %s",
                    business_context_id,
                )
                inbox_email_source_omitted = True
            except Exception:
                logger.warning(
                    "CustomerContext inbox provider setup failed for %s",
                    business_context_id,
                )
                inbox_email_source_omitted = True
            else:
                inbox_email_query_address = self._normalize_ascii_mailbox(
                    contact.get("email")
                )

        async def _safe(coro, label: str, default=None):
            try:
                return await coro
            except Exception as e:
                logger.warning("CustomerContext %s failed: %s", label, e)
                return default if default is not None else []

        interactions_coro = _safe(
            crm.get_interactions(
                contact_id, limit=max_interactions,
                business_context_id=business_context_id),
            "interactions",
        )
        appointments_coro = _safe(
            crm.get_contact_appointments(
                contact_id, business_context_id=business_context_id),
            "appointments",
        )
        calls_coro = _safe(
            call_repo.get_by_contact_id(
                contact_id, limit=max_calls,
                business_context_id=business_context_id),
            "call_transcripts",
        )
        emails_coro = _safe(
            self._get_sent_emails(
                contact,
                max_emails,
                business_context_id=business_context_id,
                contact_id=(
                    contact_id
                    if business_context_id is not None
                    else None
                ),
            ),
            "sent_emails",
        )
        if business_context_id is not None:
            if inbox_provider is None:
                inbox_coro = asyncio.sleep(0, result=[])
            else:
                inbox_coro = _safe(
                    self._get_inbox_emails(
                        contact,
                        inbox_max_emails,
                        provider=inbox_provider,
                    ),
                    "inbox_emails",
                )
        else:
            inbox_coro = _safe(
                self._get_inbox_emails(contact, inbox_max_emails),
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
        b2b_coro = (
            asyncio.sleep(0, result=[])
            if business_context_id is not None
            else _safe(
                self._get_b2b_churn_signals(contact),
                "b2b_churn_signals",
            )
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
            inbox_email_source_omitted=inbox_email_source_omitted,
            inbox_email_query_address=inbox_email_query_address,
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
            from ..autonomous.tasks._b2b_shared import read_company_churn_context

            return await read_company_churn_context(
                pool,
                company_hint=company_hint,
                limit=5,
            )
        except Exception as e:
            logger.warning("B2B churn signal lookup failed: %s", e)
            return []

    async def _get_sent_emails(
        self,
        contact: dict,
        limit: int,
        business_context_id: Optional[str] = None,
        contact_id: Optional[str] = None,
    ) -> list[dict]:
        """Find sent emails addressed to this contact within an exact tenant."""
        email_addr = contact.get("email")
        if not email_addr and not contact_id:
            return []

        from ..storage.repositories.email import get_email_repo

        repo = get_email_repo()
        results = await repo.query(
            to_address=email_addr,
            limit=limit,
            business_context_id=business_context_id,
            contact_id=contact_id,
        )
        return [self._email_to_dict(e) for e in results]

    async def _get_inbox_emails(
        self,
        contact: dict,
        limit: int,
        provider: Any | None = None,
    ) -> list[dict]:
        """
        Find recent inbound emails from this contact via IMAP/Gmail.

        Searches for messages where the sender matches the contact's email address.
        Scoped callers supply their authorized reader. Unscoped callers use
        CompositeEmailProvider (IMAP preferred; Gmail API fallback).
        Fail-open: returns [] if email address is missing or provider unavailable.
        """
        raw_email = contact.get("email")
        if not raw_email:
            return []

        scoped_reader = provider is not None
        if not scoped_reader:
            try:
                from .email_provider import get_email_provider

                provider = get_email_provider()
                return await provider.list_messages(
                    query=f"from:{raw_email}",
                    max_results=limit,
                )
            except Exception as exc:
                logger.warning(
                    "_get_inbox_emails failed for %s: %s",
                    raw_email,
                    exc,
                )
                return []

        email_addr = self._normalize_ascii_mailbox(raw_email)
        if email_addr is None:
            logger.warning(
                "_get_inbox_emails refused invalid address for contact %s",
                contact.get("id", "unknown"),
            )
            return []
        safe_limit = max(0, min(limit, 50))
        if safe_limit == 0:
            return []
        try:
            messages = await provider.list_messages(
                query=f'from:"{email_addr}"',
                max_results=_SCOPED_INBOX_CANDIDATE_LIMIT,
            )

            admitted: list[dict[str, Any]] = []
            for message in messages[:_SCOPED_INBOX_CANDIDATE_LIMIT]:
                if not isinstance(message, dict):
                    continue
                try:
                    sender = self._strict_sender_mailbox(message)
                except Exception as exc:
                    logger.warning(
                        "_get_inbox_emails refused malformed sender "
                        "candidate: %s",
                        exc,
                    )
                    continue
                if not self._same_ascii_mailbox(sender, email_addr):
                    continue
                public_message = dict(message)
                public_message.pop("_atlas_from_header_values", None)
                admitted.append(public_message)
                if len(admitted) >= safe_limit:
                    break
            return admitted
        except Exception as exc:
            logger.warning("_get_inbox_emails failed for %s: %s", email_addr, exc)
            return []

    @staticmethod
    def _normalize_ascii_mailbox(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        try:
            from email_validator import EmailNotValidError, validate_email

            validated = validate_email(
                value.strip(),
                check_deliverability=False,
                allow_smtputf8=False,
            )
            return validated.ascii_email
        except EmailNotValidError:
            return None

    @staticmethod
    def _same_ascii_mailbox(left: Any, right: Any) -> bool:
        """Compare normalized mailboxes with a case-sensitive local part."""
        if not isinstance(left, str) or not isinstance(right, str):
            return False
        try:
            left_local, left_domain = left.rsplit("@", 1)
            right_local, right_domain = right.rsplit("@", 1)
        except ValueError:
            return False
        return (
            left_local == right_local
            and left_domain.casefold() == right_domain.casefold()
        )

    @classmethod
    def _strict_sender_mailbox(cls, message: dict[str, Any]) -> str | None:
        """Return one exact sender, rejecting ambiguous header provenance."""
        values = message.get("_atlas_from_header_values")
        if (
            not isinstance(values, list)
            or len(values) != 1
            or not isinstance(values[0], str)
        ):
            return None
        return cls._parse_single_author(values[0])

    @classmethod
    def _parse_single_author(cls, value: Any) -> str | None:
        """Parse one structurally valid, non-group RFC From mailbox."""
        if not isinstance(value, str) or not value.strip():
            return None
        unfolded = re.sub(r"\r?\n(?=[ \t])", "", value)
        if "\r" in unfolded or "\n" in unfolded:
            return None
        try:
            parsed_message = Parser(policy=policy.default).parsestr(
                f"From: {unfolded}\n\n"
            )
            header = parsed_message["From"]
            if (
                header is None
                or parsed_message.defects
                or header.defects
                or len(header.addresses) != 1
                or len(header.groups) != 1
                or header.groups[0].display_name is not None
            ):
                return None
            addr_spec = header.addresses[0].addr_spec
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        return cls._normalize_ascii_mailbox(addr_spec)

    @staticmethod
    def _email_to_dict(email) -> dict:
        """Convert email history without exposing internal ownership metadata."""
        if callable(getattr(email, "to_dict", None)):
            result = email.to_dict()
        elif hasattr(email, "__dict__"):
            result = {
                k: v
                for k, v in email.__dict__.items()
                if not k.startswith("_")
            }
        else:
            result = dict(email)
        result.pop("business_context_id", None)
        return result


_customer_context_service: Optional[CustomerContextService] = None


def get_customer_context_service() -> CustomerContextService:
    """Get the global CustomerContextService singleton."""
    global _customer_context_service
    if _customer_context_service is None:
        _customer_context_service = CustomerContextService()
    return _customer_context_service
