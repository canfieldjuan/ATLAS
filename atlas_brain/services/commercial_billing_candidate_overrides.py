"""Pure, exact-cent projections for one-run EOM billing candidate overrides.

The durable writer lives in :mod:`commercial_billing_runs`; this module has no
database or delivery dependency so the same strict projection is used by the
writer, run reader, and approval boundary.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping
from uuid import UUID

from .eom_crm_mutations import normalize_contact_email

MAX_ADJUSTMENT_CENTS = 999_999_999_999
MAX_INVOICE_LINE_DESCRIPTION_LENGTH = 256
MAX_NOTE_LENGTH = 1000
MAX_RATE_CENTS = 999_999_999_999
MAX_RECIPIENT_NAME_LENGTH = 256
MAX_LINE_OVERRIDES = 500
OVERRIDE_REASON_CODES = frozenset(
    {
        "one_time_service_variation",
        "partial_or_missed_service",
        "approved_pricing_exception",
        "customer_credit",
        "additional_charge",
        "source_correction_pending",
        "billing_delivery_exception",
    }
)
OVERRIDE_DELIVERY_METHODS = frozenset({"gmail_pdf", "manual_square"})


class CommercialBillingCandidateOverrideValidationError(ValueError):
    """An override is malformed or would make billing evidence unsafe."""


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CommercialBillingCandidateOverrideValidationError(
            "Commercial billing override evidence is not JSON-safe"
        ) from exc


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _required_text(value: Any, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise CommercialBillingCandidateOverrideValidationError(f"{field} is required")
    normalized = value.strip()
    if not normalized or len(normalized) > limit or "\x00" in normalized:
        raise CommercialBillingCandidateOverrideValidationError(
            f"{field} must contain 1 to {limit} safe characters"
        )
    return normalized


def _optional_text(value: Any, field: str, limit: int) -> str | None:
    if value is None:
        return None
    return _required_text(value, field, limit)


def _positive_int(value: Any, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise CommercialBillingCandidateOverrideValidationError(
            f"{field} must be an integer between 1 and {maximum}"
        )
    return value


def _line_key(
    candidate_key: str, source_fingerprint: str, index: int, line: Mapping[str, Any]
) -> str:
    return fingerprint(
        {
            "candidateKey": candidate_key,
            "sourceFingerprint": source_fingerprint,
            "index": index,
            "line": dict(line),
        }
    )


def decorate_line_keys(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return an operator view with server-derived stable source-line keys."""

    candidate = deepcopy(dict(snapshot))
    candidate_key = candidate.get("candidateKey")
    source_fingerprint = candidate.get("sourceFingerprint")
    lines = candidate.get("lineItems")
    if not isinstance(candidate_key, str) or not isinstance(source_fingerprint, str):
        raise CommercialBillingCandidateOverrideValidationError(
            "Commercial billing candidate identity is invalid"
        )
    if not isinstance(lines, list):
        raise CommercialBillingCandidateOverrideValidationError(
            "Commercial billing candidate line items are invalid"
        )
    decorated: list[dict[str, Any]] = []
    for index, raw_line in enumerate(lines):
        if not isinstance(raw_line, Mapping):
            raise CommercialBillingCandidateOverrideValidationError(
                "Commercial billing candidate line item is invalid"
            )
        line = deepcopy(dict(raw_line))
        line["lineKey"] = _line_key(candidate_key, source_fingerprint, index, raw_line)
        decorated.append(line)
    candidate["lineItems"] = decorated
    return candidate


def decorate_effective_line_keys(
    source_snapshot: Mapping[str, Any], effective_snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    """Expose effective lines with the immutable source-line keys they revise.

    An effective line's description/rate/quantity may differ from the source,
    so deriving its key from the effective value would make a second revision
    target disappear. Adjustment lines are intentionally not line-editable.
    """

    source = decorate_line_keys(source_snapshot)
    effective = deepcopy(dict(effective_snapshot))
    if effective.get("candidateKey") != source.get("candidateKey") or effective.get(
        "sourceFingerprint"
    ) != source.get("sourceFingerprint"):
        raise CommercialBillingCandidateOverrideValidationError(
            "Commercial billing effective candidate identity is invalid"
        )
    source_lines = source["lineItems"]
    effective_lines = effective.get("lineItems")
    if not isinstance(effective_lines, list) or len(effective_lines) < len(
        source_lines
    ):
        raise CommercialBillingCandidateOverrideValidationError(
            "Commercial billing effective candidate line items are invalid"
        )
    decorated: list[dict[str, Any]] = []
    for index, source_line in enumerate(source_lines):
        raw_effective_line = effective_lines[index]
        if not isinstance(raw_effective_line, Mapping):
            raise CommercialBillingCandidateOverrideValidationError(
                "Commercial billing effective candidate line item is invalid"
            )
        line = deepcopy(dict(raw_effective_line))
        line["lineKey"] = source_line["lineKey"]
        decorated.append(line)
    for raw_effective_line in effective_lines[len(source_lines) :]:
        if (
            not isinstance(raw_effective_line, Mapping)
            or raw_effective_line.get("kind") != "adjustment"
        ):
            raise CommercialBillingCandidateOverrideValidationError(
                "Commercial billing effective candidate line items are invalid"
            )
        line = deepcopy(dict(raw_effective_line))
        line.pop("lineKey", None)
        decorated.append(line)
    effective["lineItems"] = decorated
    return effective


def _source_line_unit(line: Mapping[str, Any]) -> str:
    unit = line.get("quantityUnit")
    return unit if isinstance(unit, str) and unit else "unit"


def _hour_text(minutes: int) -> str:
    hours = Decimal(minutes) / Decimal(60)
    text = f"{hours:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _hour_amount(rate_cents: int, minutes: int) -> int:
    return int(
        (Decimal(rate_cents) * Decimal(minutes) / Decimal(60)).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )


def _tax_amount(subtotal_cents: int, tax_rate_basis_points: int) -> int:
    return int(
        (
            Decimal(subtotal_cents) * Decimal(tax_rate_basis_points) / Decimal(10_000)
        ).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )


def _valid_source_rate(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else None
    )


def _valid_source_quantity(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else None
    )


def _line_override_map(
    source_snapshot: Mapping[str, Any], raw_overrides: Any
) -> dict[str, dict[str, Any]]:
    if raw_overrides is None:
        return {}
    if not isinstance(raw_overrides, list) or len(raw_overrides) > MAX_LINE_OVERRIDES:
        raise CommercialBillingCandidateOverrideValidationError(
            "Line overrides must be a bounded list"
        )
    source = decorate_line_keys(source_snapshot)
    admitted = {line["lineKey"]: line for line in source["lineItems"]}
    normalized: dict[str, dict[str, Any]] = {}
    for item in raw_overrides:
        if not isinstance(item, Mapping):
            raise CommercialBillingCandidateOverrideValidationError(
                "Line override is invalid"
            )
        unknown = set(item) - {
            "lineKey",
            "description",
            "rateCents",
            "quantity",
            "quantityMinutes",
        }
        if unknown:
            raise CommercialBillingCandidateOverrideValidationError(
                "Line override contains unsupported fields"
            )
        key = _required_text(item.get("lineKey"), "Line key", 64)
        line = admitted.get(key)
        if line is None or key in normalized:
            raise CommercialBillingCandidateOverrideValidationError(
                "Line override target is invalid"
            )
        unit = _source_line_unit(line)
        has_quantity = "quantity" in item
        has_minutes = "quantityMinutes" in item
        if unit == "hour" and has_quantity:
            raise CommercialBillingCandidateOverrideValidationError(
                "Hourly line overrides use whole quantityMinutes"
            )
        if unit != "hour" and has_minutes:
            raise CommercialBillingCandidateOverrideValidationError(
                "Only hourly line overrides may contain quantityMinutes"
            )
        change: dict[str, Any] = {}
        if "description" in item:
            change["description"] = _required_text(
                item["description"],
                "Line description",
                MAX_INVOICE_LINE_DESCRIPTION_LENGTH,
            )
        if "rateCents" in item:
            change["rateCents"] = _positive_int(
                item["rateCents"], "Rate cents", MAX_RATE_CENTS
            )
        if has_quantity:
            change["quantity"] = _positive_int(
                item["quantity"], "Quantity", MAX_RATE_CENTS
            )
        if has_minutes:
            change["quantityMinutes"] = _positive_int(
                item["quantityMinutes"], "Quantity minutes", MAX_RATE_CENTS
            )
        if not change:
            raise CommercialBillingCandidateOverrideValidationError(
                "Line override must change at least one permitted field"
            )
        normalized[key] = change
    return normalized


def _adjustment(raw: Any, billing_period: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise CommercialBillingCandidateOverrideValidationError("Adjustment is invalid")
    if set(raw) != {"kind", "description", "amountCents"}:
        raise CommercialBillingCandidateOverrideValidationError(
            "Adjustment must contain kind, description, and amountCents"
        )
    kind = raw.get("kind")
    if kind not in {"credit", "charge"}:
        raise CommercialBillingCandidateOverrideValidationError(
            "Adjustment kind must be credit or charge"
        )
    amount = _positive_int(
        raw.get("amountCents"), "Adjustment cents", MAX_ADJUSTMENT_CENTS
    )
    signed = -amount if kind == "credit" else amount
    return {
        "amountCents": signed,
        "description": _required_text(
            raw.get("description"),
            "Adjustment description",
            MAX_INVOICE_LINE_DESCRIPTION_LENGTH,
        ),
        "kind": "adjustment",
        "quantity": 1,
        "quantityUnit": "adjustment",
        "rateCents": signed,
        "sourceDate": f"{billing_period}-01",
    }


def _recipient(base: Mapping[str, Any], raw: Any) -> dict[str, Any]:
    existing = deepcopy(dict(base)) if isinstance(base, Mapping) else {}
    if raw is None:
        return existing
    if not isinstance(raw, Mapping) or set(raw) - {"displayName", "email"}:
        raise CommercialBillingCandidateOverrideValidationError(
            "Billing recipient is invalid"
        )
    email = normalize_contact_email(raw.get("email"))
    if email is None:
        raise CommercialBillingCandidateOverrideValidationError(
            "Billing recipient email is invalid"
        )
    display_name = _optional_text(
        raw.get("displayName"), "Billing recipient name", MAX_RECIPIENT_NAME_LENGTH
    )
    return {
        "contactId": existing.get("contactId"),
        "displayName": display_name or email,
        "email": email,
    }


def _missing_billing_email_blocker() -> dict[str, Any]:
    """Return the canonical blocker when effective Gmail delivery lacks mail."""

    return {
        "code": "missing_billing_email",
        "eventIds": [],
        "message": "The canonical commercial customer has no valid billing email.",
        "serviceId": None,
    }


def _invalid_rate_blocker_is_repaired(
    blocker: Mapping[str, Any],
    *,
    keyed_source_lines: list[dict[str, Any]],
    overrides: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Return whether every targetable unpriced line for one service was fixed.

    The canonical candidate producer omits lines when a service has no valid
    rate.  An override cannot invent that omitted source evidence, so a
    service-level ``invalid_rate`` blocker must remain unless the retained
    snapshot actually contains every affected source line and each one has an
    explicit valid rate override.  Looking only at the effective lines would
    incorrectly make an omitted service disappear whenever other lines happen
    to be valid.
    """

    service_id = blocker.get("serviceId")
    if not isinstance(service_id, str) or not service_id:
        return False
    affected = [
        line
        for line in keyed_source_lines
        if line.get("serviceId") == service_id
        and _valid_source_rate(line.get("rateCents")) is None
    ]
    return bool(affected) and all(
        _valid_source_rate(overrides.get(line["lineKey"], {}).get("rateCents"))
        is not None
        for line in affected
    )


def _build_effective_snapshot(
    source_snapshot: Mapping[str, Any],
    *,
    line_overrides: Any,
    adjustment: Any,
    recipient: Any,
    delivery_method: Any,
    require_change: bool,
) -> dict[str, Any]:
    """Apply the explicitly permitted one-run edits without touching source evidence."""

    if not isinstance(source_snapshot, Mapping):
        raise CommercialBillingCandidateOverrideValidationError(
            "Commercial billing snapshot is invalid"
        )
    effective = deepcopy(dict(source_snapshot))
    billing_period = _required_text(effective.get("billingPeriod"), "Billing period", 7)
    source_lines = effective.get("lineItems")
    if not isinstance(source_lines, list):
        raise CommercialBillingCandidateOverrideValidationError(
            "Commercial billing line items are invalid"
        )
    overrides = _line_override_map(source_snapshot, line_overrides)
    keyed_source = decorate_line_keys(source_snapshot)["lineItems"]
    effective_lines: list[dict[str, Any]] = []
    incomplete_line = False
    for source_line, decorated in zip(source_lines, keyed_source):
        if not isinstance(source_line, Mapping):
            raise CommercialBillingCandidateOverrideValidationError(
                "Commercial billing line item is invalid"
            )
        line = deepcopy(dict(source_line))
        change = overrides.get(decorated["lineKey"], {})
        line.update(change)
        unit = _source_line_unit(line)
        rate = _valid_source_rate(line.get("rateCents"))
        if unit == "hour":
            minutes = _valid_source_quantity(line.get("quantityMinutes"))
            if minutes is None:
                line["quantity"] = None
                line["amountCents"] = None
                incomplete_line = True
            elif rate is None:
                line["amountCents"] = None
                incomplete_line = True
            else:
                line["quantityMinutes"] = minutes
                line["quantity"] = _hour_text(minutes)
                line["amountCents"] = _hour_amount(rate, minutes)
        else:
            quantity = _valid_source_quantity(line.get("quantity"))
            if quantity is None or rate is None:
                line["amountCents"] = None
                incomplete_line = True
            else:
                line["quantity"] = quantity
                line["amountCents"] = quantity * rate
        effective_lines.append(line)

    effective["lineItems"] = effective_lines
    effective_recipient = _recipient(effective.get("recipient"), recipient)
    effective["recipient"] = effective_recipient
    if delivery_method is not None:
        if delivery_method not in OVERRIDE_DELIVERY_METHODS:
            raise CommercialBillingCandidateOverrideValidationError(
                "Billing delivery method must be Gmail PDF or Manual Square"
            )
        effective["deliveryMethod"] = delivery_method
    if effective.get("deliveryMethod") not in OVERRIDE_DELIVERY_METHODS:
        # The source's missing/receipt-only delivery preference remains a blocker.
        pass

    adjustment_line = _adjustment(adjustment, billing_period)
    if adjustment_line is not None:
        effective_lines.append(adjustment_line)

    blockers = effective.get("blockers")
    if not isinstance(blockers, list):
        raise CommercialBillingCandidateOverrideValidationError(
            "Commercial billing blockers are invalid"
        )
    remaining: list[dict[str, Any]] = []
    all_hourly_complete = all(
        _source_line_unit(line) != "hour"
        or _valid_source_quantity(line.get("quantityMinutes")) is not None
        for line in effective_lines
        if line.get("kind") != "adjustment"
    )
    delivery_ok = effective.get("deliveryMethod") in OVERRIDE_DELIVERY_METHODS
    recipient_ok = (
        effective.get("deliveryMethod") != "gmail_pdf"
        or normalize_contact_email(effective_recipient.get("email")) is not None
    )
    for blocker in blockers:
        if not isinstance(blocker, Mapping):
            raise CommercialBillingCandidateOverrideValidationError(
                "Commercial billing blocker is invalid"
            )
        code = blocker.get("code")
        if code == "missing_hours" and all_hourly_complete:
            continue
        if code == "invalid_rate" and _invalid_rate_blocker_is_repaired(
            blocker,
            keyed_source_lines=keyed_source,
            overrides=overrides,
        ):
            continue
        if code == "missing_billing_email" and recipient_ok:
            continue
        if (
            code
            in {"missing_billing_delivery_preference", "no_invoice_delivery_preference"}
            and delivery_ok
        ):
            continue
        if code == "zero_or_invalid_total":
            continue
        remaining.append(deepcopy(dict(blocker)))
    if not recipient_ok and not any(
        blocker.get("code") == "missing_billing_email" for blocker in remaining
    ):
        remaining.append(_missing_billing_email_blocker())

    tax_rate = effective.get("taxRateBasisPoints")
    if (
        isinstance(tax_rate, bool)
        or not isinstance(tax_rate, int)
        or not 0 <= tax_rate <= 10_000
    ):
        raise CommercialBillingCandidateOverrideValidationError(
            "Commercial billing tax evidence is invalid"
        )
    subtotal = sum(
        int(line["amountCents"])
        for line in effective_lines
        if isinstance(line.get("amountCents"), int)
        and not isinstance(line.get("amountCents"), bool)
    )
    can_total = not incomplete_line and not remaining and subtotal > 0
    if can_total:
        tax = _tax_amount(subtotal, tax_rate)
        total = subtotal + tax
        if total <= 0 or total > MAX_ADJUSTMENT_CENTS:
            can_total = False
    if can_total:
        effective["subtotalCents"] = subtotal
        effective["taxCents"] = tax
        effective["totalCents"] = total
    else:
        effective["subtotalCents"] = None
        effective["taxCents"] = None
        effective["totalCents"] = None
        remaining.append(
            {
                "code": "zero_or_invalid_total",
                "eventIds": [],
                "message": "The candidate total is zero or cannot be calculated from effective review evidence.",
                "serviceId": None,
            }
        )
    effective["blockers"] = sorted(
        remaining,
        key=lambda blocker: (
            str(blocker.get("code") or ""),
            str(blocker.get("serviceId") or ""),
            tuple(blocker.get("eventIds") or []),
        ),
    )
    if require_change:
        baseline = _build_effective_snapshot(
            source_snapshot,
            line_overrides=[],
            adjustment=None,
            recipient=None,
            delivery_method=None,
            require_change=False,
        )
        if canonical_json(effective) == canonical_json(baseline):
            raise CommercialBillingCandidateOverrideValidationError(
                "Commercial billing override must change permitted review evidence"
            )
    return effective


def build_effective_snapshot(
    source_snapshot: Mapping[str, Any],
    *,
    line_overrides: Any,
    adjustment: Any,
    recipient: Any,
    delivery_method: Any,
) -> dict[str, Any]:
    """Project one permitted run-only change over immutable source evidence."""

    return _build_effective_snapshot(
        source_snapshot,
        line_overrides=line_overrides,
        adjustment=adjustment,
        recipient=recipient,
        delivery_method=delivery_method,
        require_change=True,
    )


def override_review_fingerprint(
    billing_run_id: UUID,
    source_fingerprint: str,
    revision: int,
    effective_snapshot: Mapping[str, Any],
) -> str:
    if not isinstance(billing_run_id, UUID):
        raise CommercialBillingCandidateOverrideValidationError(
            "Commercial billing override run identity is invalid"
        )
    if not isinstance(revision, int) or revision <= 0:
        raise CommercialBillingCandidateOverrideValidationError(
            "Override revision is invalid"
        )
    return fingerprint(
        {
            "billingRunId": str(billing_run_id),
            "effectiveSnapshot": dict(effective_snapshot),
            "overrideRevision": revision,
            "sourceFingerprint": source_fingerprint,
        }
    )


__all__ = [
    "CommercialBillingCandidateOverrideValidationError",
    "MAX_INVOICE_LINE_DESCRIPTION_LENGTH",
    "MAX_NOTE_LENGTH",
    "OVERRIDE_DELIVERY_METHODS",
    "OVERRIDE_REASON_CODES",
    "build_effective_snapshot",
    "canonical_json",
    "decorate_effective_line_keys",
    "decorate_line_keys",
    "fingerprint",
    "override_review_fingerprint",
]
