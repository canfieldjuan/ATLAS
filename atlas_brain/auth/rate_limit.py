"""Per-account API rate-limiting based on plan tier.

slowapi's ``@limiter.limit(callable)`` supports two callable signatures:
  - ``(key: str) -> str``  -- receives the output of key_func
  - ``() -> str``          -- no arguments

We use the first form: _key_func encodes the plan into the key string
(``plan:account_id``), and _dynamic_limit parses it back out.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

# Plan -> rate string (requests per hour)
PLAN_RATE_LIMITS: dict[str, str] = {
    "trial": "100/hour",
    "starter": "100/hour",
    "growth": "1000/hour",
    "pro": "10000/hour",
}

_DEFAULT_LIMIT = "100/hour"
_KEY_SEP = "|"


def _key_func(request: Request) -> str:
    """Build ``plan|account_id`` key from request state (set by middleware).

    Falls back to ``trial|<ip>`` when state is missing (e.g. no JWT).
    The plan prefix is stripped by _dynamic_limit to select the right tier;
    slowapi uses the full string as the rate-limit bucket key.
    """
    plan = getattr(request.state, "rate_limit_plan", None) or "trial"
    identity = getattr(request.state, "rate_limit_key", None) or get_remote_address(request)
    return f"{plan}{_KEY_SEP}{identity}"


def _dynamic_limit(key: str) -> str:
    """Extract plan from the composite key and return the matching rate string.

    ``key`` is the value returned by ``_key_func``, e.g. ``"growth|acct-uuid"``.
    """
    plan = key.split(_KEY_SEP, 1)[0] if _KEY_SEP in key else "trial"
    return PLAN_RATE_LIMITS.get(plan, _DEFAULT_LIMIT)


limiter = Limiter(key_func=_key_func, default_limits=[])
