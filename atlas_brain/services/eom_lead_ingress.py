"""Shared EOM inbound identity handling.

Inbound call and SMS extraction is untrusted enrichment.  It may create a new
lead, but it must not alter the identity, type, or pipeline state of a matching
contact.  The public website intake has the same rule.
"""

from __future__ import annotations

import re
from typing import Any, Optional


EOM_BUSINESS_CONTEXT_ID = "effingham_maids"
_MIN_MATCH_PHONE_DIGITS = 10


def normalise_eom_phone_digits(value: Any) -> str:
    """Return the shared EOM phone identity digits.

    EOM lead/customer matching uses ASCII digits-only phone identity and exact
    normalized last-10 equality. Callers decide whether fewer than ten digits
    are admissible for their boundary; the normalizer itself only strips
    formatting. Non-ASCII digit glyphs are intentionally not canonicalized here
    because the SQL resolvers use ``[^0-9]``.
    """
    return re.sub(r"[^0-9]", "", str(value or ""))


def preferred_eom_inbound_phone(extracted_phone: Any, transport_phone: Any) -> str:
    """Prefer a full extracted number, then a full transport caller number.

    Call/SMS extraction is enrichment, while the transport's caller number is
    authoritative.  A local fragment is therefore not allowed to mask a usable
    caller number before EOM's full-phone identity admission.
    """
    extracted = str(extracted_phone or "").strip()
    transport = str(transport_phone or "").strip()
    if len(normalise_eom_phone_digits(extracted)) >= _MIN_MATCH_PHONE_DIGITS:
        return extracted
    if len(normalise_eom_phone_digits(transport)) >= _MIN_MATCH_PHONE_DIGITS:
        return transport
    return extracted or transport


async def resolve_or_create_eom_inbound_lead(
    crm: Any,
    *,
    full_name: str,
    phone: Optional[str],
    email: Optional[str],
    address: Optional[str],
    source: str,
    source_ref: Optional[str],
    relay_event_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    interaction: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return a matching EOM contact unchanged or create one as ``lead/new``.

    Exact tenant and claimable-legacy populations are searched directly instead
    of relying on the provider's mutable find-or-create merge.  The provider's
    ``preserve_existing`` option closes the narrow race between the final lookup
    and insert without turning extracted caller data into a CRM overwrite.
    """

    normalized_email = str(email or "").strip().lower()
    phone_digits = normalise_eom_phone_digits(phone)
    normalized_source = str(source or "").strip()
    normalized_source_ref = str(source_ref or "").strip()
    normalized_relay_event_id = str(relay_event_id or "").strip()
    identityless = (
        len(phone_digits) < _MIN_MATCH_PHONE_DIGITS and not normalized_email
    )
    if identityless and not (normalized_source and normalized_relay_event_id):
        raise ValueError(
            "EOM inbound lead requires phone, email, or a stable relay event identity"
        )

    # DatabaseCRMProvider supplies the authoritative transaction + advisory
    # lock implementation.  The class lookup (rather than ``getattr`` on the
    # instance) deliberately leaves lightweight protocol fakes on this safe,
    # read-only fallback path.  Input admission and relay normalization happen
    # above the split so every caller has the same identity contract.
    atomic_resolver = getattr(
        type(crm), "resolve_or_create_eom_inbound_lead_atomic", None
    )
    if atomic_resolver is not None:
        return await atomic_resolver(
            crm,
            full_name=full_name,
            phone=phone,
            email=normalized_email or None,
            address=address,
            source=normalized_source or source,
            source_ref=(
                normalized_relay_event_id
                if identityless
                else normalized_source_ref or None
            ),
            relay_event_id=normalized_relay_event_id or None,
            tags=tags,
            interaction=interaction,
        )

    async def _resolve_readonly(**channel: Any) -> Optional[dict[str, Any]]:
        scoped = await crm.search_contacts(
            business_context_id=EOM_BUSINESS_CONTEXT_ID, **channel
        )
        if scoped:
            return scoped[0]
        legacy = await crm.search_contacts(
            business_context_id_is_null=True, **channel
        )
        return legacy[0] if legacy else None

    existing: Optional[dict[str, Any]] = None
    if len(phone_digits) >= _MIN_MATCH_PHONE_DIGITS:
        existing = await _resolve_readonly(phone=phone_digits)
    if existing is None and normalized_email:
        existing = await _resolve_readonly(email=normalized_email)
    if existing is not None:
        result = dict(existing)
        result["_was_created"] = False
    else:
        create_kwargs: dict[str, Any] = {
            "full_name": full_name.strip() or phone_digits or "Unknown",
            "phone": phone_digits if len(phone_digits) >= _MIN_MATCH_PHONE_DIGITS else None,
            "email": normalized_email or None,
            "address": address or None,
            "business_context_id": EOM_BUSINESS_CONTEXT_ID,
            "contact_type": "lead",
            "lead_stage": "new",
            "source": normalized_source or source,
            "source_ref": (
                normalized_relay_event_id
                if identityless
                else normalized_source_ref or None
            ),
            "preserve_existing": True,
        }
        if tags:
            create_kwargs["tags"] = tags
        result = await crm.find_or_create_contact(
            **create_kwargs,
        )

    if interaction is not None and result.get("id"):
        result = dict(result)
        result["_inbound_interaction"] = await crm.log_interaction(
            contact_id=str(result["id"]),
            **interaction,
        )
    return result


async def resolve_or_create_eom_inbound_lead_and_log_interaction(
    crm: Any,
    *,
    interaction_type: str,
    summary: str,
    occurred_at: Optional[str] = None,
    intent: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    **lead: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run an EOM inbound contact decision and its interaction as one command.

    The database provider owns the transactional path. Lightweight protocol
    fakes retain the same observable call shape through the read-only fallback,
    but production EOM ingress must expose the atomic provider method.
    """
    contact = await resolve_or_create_eom_inbound_lead(
        crm,
        **lead,
        interaction={
            "interaction_type": interaction_type,
            "summary": summary,
            "occurred_at": occurred_at,
            "intent": intent,
            "metadata": metadata,
        },
    )
    result = dict(contact)
    interaction = result.pop("_inbound_interaction", None)
    if interaction is None:
        raise RuntimeError("EOM inbound interaction command returned no interaction")
    return result, interaction
