"""FastAPI dependencies for authentication and plan-based authorization."""

import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request

from ..config import settings
from .jwt import decode_token

PLAN_ORDER = ["trial", "starter", "growth", "pro"]
B2B_PLAN_ORDER = ["b2b_trial", "b2b_starter", "b2b_growth", "b2b_pro"]
LLM_GATEWAY_PLAN_ORDER = ["llm_trial", "llm_starter", "llm_growth", "llm_pro"]


@dataclass
class AuthUser:
    user_id: str
    account_id: str
    plan: str        # trial | starter | growth | pro | b2b_trial | b2b_starter | b2b_growth | b2b_pro
    plan_status: str  # trialing | active | past_due | canceled
    role: str        # owner | admin | member
    product: str = "consumer"  # consumer | b2b_retention | b2b_challenger
    trial_ends_at: Optional[datetime] = field(default=None, repr=False)
    is_admin: bool = False


def _synthetic_admin() -> AuthUser:
    """Return a synthetic admin user when SaaS auth is disabled."""
    return AuthUser(
        user_id="00000000-0000-0000-0000-000000000000",
        account_id="00000000-0000-0000-0000-000000000000",
        plan="pro",
        plan_status="active",
        role="owner",
        product="consumer",
        is_admin=True,
    )


def _effective_is_admin(role: str | None, is_admin_flag: bool | None) -> bool:
    if bool(is_admin_flag):
        return True
    role_normalized = (role or "").strip().lower()
    return role_normalized in {"owner", "admin"}


def _extract_token(request: Request) -> Optional[str]:
    """Extract JWT from Authorization header, cookie, or query param."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    cookie = request.cookies.get("atlas_token")
    if cookie:
        return cookie
    return request.query_params.get("token")


async def require_auth(request: Request) -> AuthUser:
    """Require a valid JWT. Returns AuthUser or raises 401/403."""
    if not settings.saas_auth.enabled:
        return _synthetic_admin()

    token = _extract_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")

    # Cast JWT string IDs to UUID for asyncpg
    try:
        user_uuid = _uuid.UUID(payload["sub"])
        _uuid.UUID(payload["account_id"])  # validate account_id exists and is valid
    except (ValueError, KeyError):
        raise HTTPException(status_code=401, detail="Invalid token payload")

    # Fetch plan_status, role, product, trial_ends_at from DB for freshness
    from ..storage.database import get_db_pool
    pool = get_db_pool()
    row = await pool.fetchrow(
        """
        SELECT sa.plan, sa.plan_status, sa.product, sa.trial_ends_at,
               su.role, COALESCE(su.is_admin, FALSE) AS is_admin
        FROM saas_users su
        JOIN saas_accounts sa ON sa.id = su.account_id
        WHERE su.id = $1 AND su.is_active = TRUE
        """,
        user_uuid,
    )
    if not row:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    if row["plan_status"] == "canceled":
        raise HTTPException(status_code=403, detail="Subscription canceled")

    # Check trial expiration -- includes llm_trial (PR-D2) so LLM
    # Gateway trial accounts also expire after trial_days.
    trial_ends = row["trial_ends_at"]
    if row["plan"] in ("trial", "b2b_trial", "llm_trial") and trial_ends:
        # Ensure timezone-aware comparison
        te = trial_ends if trial_ends.tzinfo else trial_ends.replace(tzinfo=timezone.utc)
        if te < datetime.now(timezone.utc):
            raise HTTPException(status_code=403, detail="Trial expired")

    return AuthUser(
        user_id=payload["sub"],
        account_id=payload["account_id"],
        plan=row["plan"],
        plan_status=row["plan_status"],
        role=row["role"],
        product=row["product"] or "consumer",
        trial_ends_at=trial_ends,
        is_admin=_effective_is_admin(row["role"], row["is_admin"]),
    )


async def optional_auth(request: Request) -> Optional[AuthUser]:
    """Same as require_auth but returns None instead of 401.

    When SaaS auth is disabled (local dev), returns None so endpoints
    behave as unscoped (admin mode) rather than scoping to a synthetic
    account that has no tracked vendors.
    """
    if not settings.saas_auth.enabled:
        return None

    token = _extract_token(request)
    if not token:
        return None

    try:
        payload = decode_token(token)
    except Exception:
        return None

    if payload.get("type") != "access":
        return None

    try:
        user_uuid = _uuid.UUID(payload["sub"])
        _uuid.UUID(payload["account_id"])  # validate account_id exists and is valid
    except (ValueError, KeyError):
        return None

    from ..storage.database import get_db_pool
    pool = get_db_pool()
    try:
        row = await pool.fetchrow(
            """
            SELECT sa.plan, sa.plan_status, sa.product, sa.trial_ends_at,
                   su.role, COALESCE(su.is_admin, FALSE) AS is_admin
            FROM saas_users su
            JOIN saas_accounts sa ON sa.id = su.account_id
            WHERE su.id = $1 AND su.is_active = TRUE
            """,
            user_uuid,
        )
    except Exception:
        return None
    if not row:
        return None

    if row["plan_status"] == "canceled":
        return None

    return AuthUser(
        user_id=payload["sub"],
        account_id=payload["account_id"],
        plan=row["plan"],
        plan_status=row["plan_status"],
        role=row["role"],
        product=row["product"] or "consumer",
        trial_ends_at=row["trial_ends_at"],
        is_admin=_effective_is_admin(row["role"], row["is_admin"]),
    )


def require_plan(min_plan: str):
    """Return a dependency that enforces a minimum plan tier."""
    if min_plan not in PLAN_ORDER:
        raise ValueError(
            f"Invalid consumer plan tier '{min_plan}'. Expected one of {PLAN_ORDER}"
        )
    min_idx = PLAN_ORDER.index(min_plan)

    async def _check(user: AuthUser = Depends(require_auth)) -> AuthUser:
        if user.plan_status == "past_due":
            raise HTTPException(
                status_code=402,
                detail="Payment past due",
            )
        user_idx = PLAN_ORDER.index(user.plan) if user.plan in PLAN_ORDER else -1
        if user_idx < min_idx:
            raise HTTPException(
                status_code=403,
                detail=f"Plan '{min_plan}' or higher required (current: '{user.plan}')",
            )
        return user

    return _check


def require_b2b_plan(min_plan: str):
    """Return a dependency that enforces a B2B product + minimum B2B plan tier."""
    if min_plan not in B2B_PLAN_ORDER:
        raise ValueError(
            f"Invalid B2B plan tier '{min_plan}'. Expected one of {B2B_PLAN_ORDER}"
        )
    min_idx = B2B_PLAN_ORDER.index(min_plan)

    async def _check(user: AuthUser = Depends(require_auth)) -> AuthUser:
        if user.plan_status == "past_due":
            raise HTTPException(
                status_code=402,
                detail="Payment past due",
            )
        if user.product not in ("b2b_retention", "b2b_challenger"):
            raise HTTPException(
                status_code=403,
                detail="B2B product required",
            )
        user_idx = B2B_PLAN_ORDER.index(user.plan) if user.plan in B2B_PLAN_ORDER else -1
        if user_idx < min_idx:
            raise HTTPException(
                status_code=403,
                detail=f"Plan '{min_plan}' or higher required (current: '{user.plan}')",
            )
        return user

    return _check


def require_llm_plan(min_plan: str):
    """Return a dependency that enforces a minimum LLM-Gateway plan tier.

    Mirrors ``require_b2b_plan``: enforces both the product binding
    (``user.product == "llm_gateway"``) and the plan ordering. Used
    by PR-D4's ``/api/v1/llm/*`` router to gate per-tier feature
    access.
    """
    if min_plan not in LLM_GATEWAY_PLAN_ORDER:
        raise ValueError(
            f"Invalid LLM Gateway plan tier '{min_plan}'. Expected one of {LLM_GATEWAY_PLAN_ORDER}"
        )
    min_idx = LLM_GATEWAY_PLAN_ORDER.index(min_plan)

    # PR-D4 fix: depend on the API-key-or-JWT helper so customer scripts
    # using ``atls_live_*`` keys reach the plan check. ``require_auth``
    # (JWT-only) silently rejected them before the plan logic ran.
    async def _check(user: AuthUser = Depends(require_auth_or_api_key)) -> AuthUser:
        if user.plan_status == "past_due":
            raise HTTPException(
                status_code=402,
                detail="Payment past due",
            )
        if user.product != "llm_gateway":
            raise HTTPException(
                status_code=403,
                detail="LLM Gateway product required",
            )
        user_idx = (
            LLM_GATEWAY_PLAN_ORDER.index(user.plan)
            if user.plan in LLM_GATEWAY_PLAN_ORDER
            else -1
        )
        if user_idx < min_idx:
            raise HTTPException(
                status_code=403,
                detail=f"Plan '{min_plan}' or higher required (current: '{user.plan}')",
            )
        return user

    return _check


def _extract_api_key(request: Request) -> Optional[str]:
    """Extract a customer API key from the Authorization header.

    Customer API keys (``atls_live_*``) and JWTs both ride on the same
    ``Authorization: Bearer ...`` header. This helper only returns
    values that look like API keys; JWT bearer tokens fall through to
    ``require_auth``'s extractor.
    """
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:].strip()
    if not token.startswith("atls_live_"):
        return None
    return token


async def require_api_key(request: Request) -> AuthUser:
    """Authenticate via a customer-issued API key (PR-D1).

    Returns an ``AuthUser`` with the same shape as ``require_auth`` so
    downstream endpoints stay auth-method agnostic. The ``user_id``
    field is populated from ``api_keys.user_id`` (creator audit) when
    present, else falls back to ``account_id`` so the endpoint always
    has a non-empty caller identity.

    Raises 401 when the key is missing, malformed, revoked, or
    unrecognized; 403 when the account is canceled or the trial
    expired.
    """
    if not settings.saas_auth.enabled:
        return _synthetic_admin()

    raw_key = _extract_api_key(request)
    if not raw_key:
        raise HTTPException(status_code=401, detail="API key required")

    from ..auth.api_keys import lookup_api_key, touch_api_key
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    row = await lookup_api_key(pool, raw_key)
    if not row:
        raise HTTPException(status_code=401, detail="Invalid API key")

    account_row = await pool.fetchrow(
        """
        SELECT plan, plan_status, product, trial_ends_at
        FROM saas_accounts
        WHERE id = $1
        """,
        row["account_id"],
    )
    if not account_row:
        raise HTTPException(status_code=401, detail="Account not found")

    if account_row["plan_status"] == "canceled":
        raise HTTPException(status_code=403, detail="Subscription canceled")

    # Trial expiration check includes llm_trial (PR-D2). Mirrors the
    # same check in require_auth so JWT and API-key paths are consistent.
    trial_ends = account_row["trial_ends_at"]
    if account_row["plan"] in ("trial", "b2b_trial", "llm_trial") and trial_ends:
        te = trial_ends if trial_ends.tzinfo else trial_ends.replace(tzinfo=timezone.utc)
        if te < datetime.now(timezone.utc):
            raise HTTPException(status_code=403, detail="Trial expired")

    creator_id = row["user_id"]
    if creator_id is None:
        creator_id = row["account_id"]

    user = AuthUser(
        user_id=str(creator_id),
        account_id=str(row["account_id"]),
        plan=account_row["plan"],
        plan_status=account_row["plan_status"],
        role="member",
        product=account_row["product"] or "consumer",
        trial_ends_at=trial_ends,
        is_admin=False,
    )

    client_ip = request.client.host if request.client else None
    await touch_api_key(pool, key_id=row["id"], client_ip=client_ip)
    return user


async def require_auth_or_api_key(request: Request) -> AuthUser:
    """Accept EITHER a JWT bearer (dashboard) OR a customer API key
    (production scripts). PR-D4 review fix: ``require_llm_plan``
    used ``Depends(require_auth)`` which only validates JWTs, so
    ``Authorization: Bearer atls_live_*`` got rejected before plan
    checks. This helper dispatches by token shape so both auth
    methods work.

    Used only by LLM Gateway endpoints (PR-D4); atlas's existing
    products are dashboard-only and keep their JWT-only chains.
    """
    if not settings.saas_auth.enabled:
        return _synthetic_admin()

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    token = auth_header[7:].strip()

    if token.startswith("atls_live_"):
        return await require_api_key(request)
    return await require_auth(request)
