"""Operator-authored EOM contact mutations.

This is the authoritative Atlas domain boundary for EOM contact creates and
ordinary identity/contact-field edits. It is deliberately separate from
``eom_lead_ingress``: inbound intake is untrusted enrichment and preserves
existing rows, while this path represents authenticated operator intent.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .eom_lead_ingress import (
    EOM_BUSINESS_CONTEXT_ID,
    normalise_eom_phone_digits,
)

EOM_OPERATOR_CONTACT_CREATED = "contact_created"
EOM_OPERATOR_CONTACT_UPDATED = "contact_updated"
EOM_OPERATOR_CONTACT_EVENT_TYPES = (
    EOM_OPERATOR_CONTACT_CREATED,
    EOM_OPERATOR_CONTACT_UPDATED,
)
EOM_OPERATOR_SOURCE_CHANNELS = (
    "time_tracker",
    "operator_portal",
    "mcp_operator",
    "manual_import",
)
EOM_OPERATOR_CONTACT_TYPES = ("lead", "customer")
# Residential/commercial, on the account record. A DIFFERENT axis from
# EOM_OPERATOR_CONTACT_TYPES above (lead vs customer) -- the two answer
# unrelated questions and neither may be inferred from the other.
#
# Must stay identical to the chk_contacts_customer_type CHECK in migration 366.
# The constraint is the enforcement; this tuple only decides what the boundary
# will accept, so a value that passed here and failed there would surface as a
# 500 instead of a 422.
EOM_CUSTOMER_TYPES = ("residential", "commercial", "unknown")
EOM_OPERATOR_CONTACT_FIELDS = (
    "full_name",
    "email",
    "phone",
    "address",
    "city",
    "state",
    "zip",
    "notes",
    "customer_type",
)
_FIELD_LIMITS = {
    "full_name": 256,
    "email": 256,
    "phone": 32,
    "city": 128,
    "state": 64,
    "zip": 16,
}
_EMAIL_MIN_LENGTH = 3
_EMAIL_LOCAL_MAX_LENGTH = 64
_EMAIL_DOMAIN_MAX_LENGTH = 255
_SOURCE_REF_MAX_LENGTH = 220
_CONTACT_SOURCE = "manual"
_EMAIL_LOCAL_RE = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+$")
_EMAIL_DOMAIN_LABEL_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_DATABASE_INVALID_TEXT_CHARS = frozenset({"\x00"})
_PHONE_EXTENSION_RE = re.compile(
    r"(?:^|[\s,;#/()-])(?:ext|extension|x)\.?\s*\d+\s*$", re.I
)
_PHONE_ALLOWED_RE = re.compile(r"^[0-9\s()+.\-]*$")


class EOMOperatorContactMutationError(Exception):
    """HTTP-mappable domain error for the EOM operator mutation boundary."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _reject_database_invalid_text(field: str, value: str) -> None:
    if any(
        char in _DATABASE_INVALID_TEXT_CHARS or 0xD800 <= ord(char) <= 0xDFFF
        for char in value
    ):
        raise EOMOperatorContactMutationError(422, f"{field} must be valid")


def _normalize_email(value: Any) -> str | None:
    normalized = _blank_to_none(value)
    if normalized is None:
        return None
    if not isinstance(normalized, str):
        raise EOMOperatorContactMutationError(422, "email must be a string")
    candidate = normalized.lower()
    if len(candidate) < _EMAIL_MIN_LENGTH or len(candidate) > _FIELD_LIMITS["email"]:
        raise EOMOperatorContactMutationError(422, "email must be valid")
    if any(char.isspace() or ord(char) < 32 for char in candidate):
        raise EOMOperatorContactMutationError(422, "email must be valid")
    parts = candidate.split("@")
    if len(parts) != 2:
        raise EOMOperatorContactMutationError(422, "email must be valid")
    local, domain = parts
    labels = domain.split(".")
    if (
        not local
        or local.startswith(".")
        or local.endswith(".")
        or len(local) > _EMAIL_LOCAL_MAX_LENGTH
        or len(domain) > _EMAIL_DOMAIN_MAX_LENGTH
        or ".." in local
        or not _EMAIL_LOCAL_RE.fullmatch(local)
        or len(labels) < 2
        or any(not _EMAIL_DOMAIN_LABEL_RE.fullmatch(label) for label in labels)
    ):
        raise EOMOperatorContactMutationError(422, "email must be valid")
    return candidate


def _normalize_phone(value: Any) -> str | None:
    normalized = _blank_to_none(value)
    if normalized is None:
        return None
    if not isinstance(normalized, str):
        raise EOMOperatorContactMutationError(422, "phone must be a string")
    if _PHONE_EXTENSION_RE.search(normalized) or not _PHONE_ALLOWED_RE.fullmatch(
        normalized
    ):
        raise EOMOperatorContactMutationError(
            422, "phone must not include an extension"
        )
    digits = normalise_eom_phone_digits(normalized)
    if len(digits) < 10:
        raise EOMOperatorContactMutationError(
            422, "phone must contain at least 10 digits"
        )
    if len(digits) > _FIELD_LIMITS["phone"]:
        raise EOMOperatorContactMutationError(422, "phone is too long")
    return digits


def _normalize_text_field(field: str, value: Any) -> str | None:
    normalized = _blank_to_none(value)
    if normalized is None:
        if field == "full_name":
            raise EOMOperatorContactMutationError(422, "fullName must not be blank")
        return None
    if not isinstance(normalized, str):
        raise EOMOperatorContactMutationError(422, f"{field} must be a string")
    _reject_database_invalid_text(field, normalized)
    limit = _FIELD_LIMITS.get(field)
    if limit is not None and len(normalized) > limit:
        raise EOMOperatorContactMutationError(422, f"{field} is too long")
    return normalized


def _normalize_customer_type(value: Any) -> str:
    """Admit exactly the three account types, case-insensitively.

    Case folding is not politeness: the evidence this field is populated from
    is the tracker's ``locations.location_type``, which stores ``Residential``
    and ``Commercial`` capitalised. Rejecting those would make the boundary
    refuse the very values the backfill reads.

    Blank is refused rather than treated as ``unknown``. Every other text field
    here maps blank to NULL to mean "clear it", but this column is NOT NULL and
    ``unknown`` is a real member of the set, so that mapping would let an empty
    form field silently downgrade a commercial account to unknown -- a value
    change disguised as a no-op. An operator who means unknown can say so.
    """
    if not isinstance(value, str):
        raise EOMOperatorContactMutationError(422, "customerType must be a string")
    candidate = value.strip().lower()
    if not candidate:
        raise EOMOperatorContactMutationError(422, "customerType must not be blank")
    if candidate not in EOM_CUSTOMER_TYPES:
        raise EOMOperatorContactMutationError(422, "customerType is not supported")
    return candidate


def _normalize_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field, value in fields.items():
        if field not in EOM_OPERATOR_CONTACT_FIELDS:
            raise EOMOperatorContactMutationError(
                422, f"{field} is not an operator contact field"
            )
        if field == "email":
            normalized[field] = _normalize_email(value)
        elif field == "phone":
            normalized[field] = _normalize_phone(value)
        elif field == "customer_type":
            normalized[field] = _normalize_customer_type(value)
        else:
            normalized[field] = _normalize_text_field(field, value)
    if not normalized:
        raise EOMOperatorContactMutationError(
            422, "at least one contact field is required"
        )
    return normalized


def _source_ref_for_channel(source_channel: str, source_ref: str) -> str:
    combined = f"{source_channel}:{source_ref}"
    if len(combined) > 256:
        raise EOMOperatorContactMutationError(422, "sourceRef is too long")
    return combined


@dataclass(frozen=True)
class EOMOperatorContactMutation:
    """Normalized command accepted by the operator contact mutation boundary."""

    operation_key: str
    actor_id: int
    actor_name: str
    source_channel: str
    source_ref: str
    fields: Mapping[str, Any]
    contact_id: str | None = None
    contact_type: str | None = None

    @classmethod
    def from_raw(
        cls,
        *,
        operation_key: str,
        actor_id: int,
        actor_name: str,
        source_channel: str,
        source_ref: str,
        fields: Mapping[str, Any],
        contact_id: str | None = None,
        contact_type: str | None = None,
    ) -> "EOMOperatorContactMutation":
        normalized_channel = str(source_channel or "").strip()
        if normalized_channel not in EOM_OPERATOR_SOURCE_CHANNELS:
            raise EOMOperatorContactMutationError(
                422, "sourceChannel is not supported"
            )
        normalized_ref = str(source_ref or "").strip()
        if not normalized_ref:
            raise EOMOperatorContactMutationError(422, "sourceRef is required")
        _reject_database_invalid_text("sourceRef", normalized_ref)
        if len(normalized_ref) > _SOURCE_REF_MAX_LENGTH:
            raise EOMOperatorContactMutationError(422, "sourceRef is too long")
        normalized_type = None
        if contact_type is not None:
            normalized_type = str(contact_type).strip()
            if normalized_type not in EOM_OPERATOR_CONTACT_TYPES:
                raise EOMOperatorContactMutationError(
                    422, "contactType is not supported"
                )
        normalized_fields = _normalize_fields(fields)
        return cls(
            operation_key=operation_key,
            actor_id=actor_id,
            actor_name=str(actor_name),
            source_channel=normalized_channel,
            source_ref=normalized_ref,
            fields=MappingProxyType(normalized_fields),
            contact_id=str(contact_id) if contact_id else None,
            contact_type=normalized_type,
        )

    @property
    def contact_source(self) -> str:
        return _CONTACT_SOURCE

    @property
    def contact_source_ref(self) -> str:
        return _source_ref_for_channel(self.source_channel, self.source_ref)

    @property
    def request_fingerprint(self) -> str:
        payload = {
            "actor_id": self.actor_id,
            "contact_id": self.contact_id,
            "contact_type": self.contact_type,
            "fields": dict(sorted(self.fields.items())),
            "source_channel": self.source_channel,
            "source_ref": self.source_ref,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def lifecycle_metadata(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "source_channel": self.source_channel,
            "source_ref": self.source_ref,
            "request_fingerprint": self.request_fingerprint,
            "field_names": sorted(self.fields.keys()),
        }


async def mutate_eom_operator_contact(
    crm: Any,
    command: EOMOperatorContactMutation,
) -> dict[str, Any]:
    """Execute one authenticated EOM operator contact mutation."""

    atomic_mutation = getattr(
        type(crm), "mutate_eom_operator_contact_atomic", None
    )
    if atomic_mutation is None:
        raise RuntimeError(
            "EOM operator contact mutations require the database CRM provider"
        )
    return await atomic_mutation(crm, command=command)
