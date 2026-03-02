"""FastAPI dependencies for authentication and plan-based authorization."""

import uuid as _uuid
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request

from ..config import settings
from .jwt import decode_token

PLAN_ORDER = ["trial", "starter", "growth", "pro"]
B2B_PLAN_ORDER = ["b2b_trial", "b2b_starter", "b2b_growth", "b2b_pro"]


@dataclass
class AuthUser:
    user_id: str
    account_id: str
    plan: str        # trial | starter | growth | pro | b2b_trial | b2b_starter | b2b_growth | b2b_pro
    plan_status: str  # trialing | active | past_due | canceled
    role: str        # owner | admin | member
    product: str = "consumer"  # consumer | b2b_retention | b2b_challenger


def _synthetic_admin() -> AuthUser:
    """Return a synthetic admin user when SaaS auth is disabled."""
    return AuthUser(
        user_id="00000000-0000-0000-0000-000000000000",
        account_id="00000000-0000-0000-0000-000000000000",
        plan="pro",
        plan_status="active",
        role="owner",
        product="consumer",
    )


def _extract_token(request: Request) -> Optional[str]:
    """Extract JWT from Authorization header or cookie."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return request.cookies.get("atlas_token")


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

    # Fetch plan_status, role, product from DB for freshness
    from ..storage.database import get_db_pool
    pool = get_db_pool()
    row = await pool.fetchrow(
        """
        SELECT sa.plan, sa.plan_status, sa.product, su.role
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

    return AuthUser(
        user_id=payload["sub"],
        account_id=payload["account_id"],
        plan=row["plan"],
        plan_status=row["plan_status"],
        role=row["role"],
        product=row["product"] or "consumer",
    )


async def optional_auth(request: Request) -> Optional[AuthUser]:
    """Same as require_auth but returns None instead of 401."""
    if not settings.saas_auth.enabled:
        return _synthetic_admin()

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
            SELECT sa.plan, sa.plan_status, sa.product, su.role
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
    )


def require_plan(min_plan: str):
    """Return a dependency that enforces a minimum plan tier."""
    min_idx = PLAN_ORDER.index(min_plan)

    async def _check(user: AuthUser = Depends(require_auth)) -> AuthUser:
        user_idx = PLAN_ORDER.index(user.plan) if user.plan in PLAN_ORDER else -1
        if user_idx < min_idx:
            raise HTTPException(
                status_code=403,
                detail=f"Plan '{min_plan}' or higher required (current: '{user.plan}')",
            )
        return user

    return Depends(_check)


def require_b2b_plan(min_plan: str):
    """Return a dependency that enforces a B2B product + minimum B2B plan tier."""
    min_idx = B2B_PLAN_ORDER.index(min_plan)

    async def _check(user: AuthUser = Depends(require_auth)) -> AuthUser:
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

    return Depends(_check)
