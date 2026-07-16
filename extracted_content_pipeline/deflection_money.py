"""Money helpers for Content Ops deflection report support-cost displays."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any


ASSISTED_CONTACT_COST_USD = Decimal("13.50")
ASSISTED_CONTACT_COST_LABEL = "$13.50"

_CENT = Decimal("0.01")
_ZERO = Decimal("0.00")


def support_cost_usd(ticket_count: Any) -> float:
    amount = _support_cost_amount(ticket_count)
    return float(amount)


def annualized_support_cost_usd(ticket_count: Any, window_days: Any) -> float:
    days = _coerce_decimal(window_days)
    if days <= _ZERO:
        return 0.0
    amount = (
        _coerce_decimal(ticket_count)
        * Decimal(365)
        * ASSISTED_CONTACT_COST_USD
        / days
    )
    return float(_quantize_usd(amount))


def signed_support_cost_delta_usd(value: Any) -> float:
    """Quantize a SIGNED money delta with the same ROUND_HALF_UP rule.

    `support_cost_usd` clamps to zero for report displays; deltas must
    keep their sign, rounding half-up by magnitude on both sides.
    """

    amount = _coerce_decimal(value)
    quantized = abs(amount).quantize(_CENT, rounding=ROUND_HALF_UP)
    return float(-quantized if amount < 0 else quantized)


def format_support_cost_usd(value: Any) -> str:
    quantized = _quantize_usd(_coerce_decimal(value))
    return f"${quantized:,.2f}"


def _coerce_decimal(value: Any) -> Decimal:
    if isinstance(value, bool):
        return _ZERO
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return _ZERO


def _support_cost_amount(ticket_count: Any) -> Decimal:
    amount = _coerce_decimal(ticket_count) * ASSISTED_CONTACT_COST_USD
    return _quantize_usd(amount)


def _quantize_usd(amount: Decimal) -> Decimal:
    return max(_ZERO, amount).quantize(_CENT, rounding=ROUND_HALF_UP)
