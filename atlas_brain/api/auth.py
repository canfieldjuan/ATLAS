"""Authentication endpoints: register, login, refresh, me, change-password."""

import logging
import uuid as _uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr

from ..auth.dependencies import AuthUser, require_auth
from ..auth.jwt import create_access_token, create_refresh_token, decode_token
from ..auth.passwords import hash_password, verify_password
from ..config import settings
from ..storage.database import get_db_pool

# Import PLAN_LIMITS for default asin limit
from .billing import PLAN_LIMITS

logger = logging.getLogger("atlas.api.auth")

router = APIRouter(prefix="/auth", tags=["auth"])


# -- Request/Response schemas --

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    account_name: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    user_id: str
    email: str
    full_name: str | None
    role: str
    account_id: str
    account_name: str
    plan: str
    plan_status: str
    asin_limit: int
    trial_ends_at: str | None


# -- Endpoints --

@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    """Create a new account and user, return JWT tokens."""
    if not settings.saas_auth.enabled:
        raise HTTPException(status_code=404, detail="Registration not available")

    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    cfg = settings.saas_auth
    trial_ends = datetime.now(timezone.utc) + timedelta(days=cfg.trial_days)
    trial_asin_limit = PLAN_LIMITS.get("trial", {}).get("asins", 5)

    # Use a transaction so account + user are created atomically
    async with pool.transaction() as conn:
        # Check email uniqueness inside transaction to prevent TOCTOU
        existing = await conn.fetchval(
            "SELECT id FROM saas_users WHERE email = $1", req.email.lower()
        )
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered")

        # Create account
        account_id = await conn.fetchval(
            """
            INSERT INTO saas_accounts (name, plan, plan_status, trial_ends_at, asin_limit)
            VALUES ($1, 'trial', 'trialing', $2, $3)
            RETURNING id
            """,
            req.account_name,
            trial_ends,
            trial_asin_limit,
        )

        # Create user
        pw_hash = hash_password(req.password)
        user_id = await conn.fetchval(
            """
            INSERT INTO saas_users (account_id, email, password_hash, full_name, role)
            VALUES ($1, $2, $3, $4, 'owner')
            RETURNING id
            """,
            account_id,
            req.email.lower(),
            pw_hash,
            req.full_name,
        )

    # Create Stripe customer if configured (outside transaction -- non-critical)
    if cfg.stripe_secret_key:
        try:
            import stripe
            stripe.api_key = cfg.stripe_secret_key
            customer = stripe.Customer.create(
                email=req.email,
                name=req.account_name,
                metadata={"account_id": str(account_id)},
            )
            await pool.execute(
                "UPDATE saas_accounts SET stripe_customer_id = $1 WHERE id = $2",
                customer.id,
                account_id,
            )
        except Exception as e:
            logger.warning("Stripe customer creation failed: %s", e)

    access = create_access_token(str(user_id), str(account_id), "trial")
    refresh = create_refresh_token(str(user_id))

    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """Authenticate with email/password, return JWT tokens."""
    if not settings.saas_auth.enabled:
        raise HTTPException(status_code=404, detail="Login not available")

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    row = await pool.fetchrow(
        """
        SELECT su.id, su.password_hash, su.is_active, su.account_id,
               sa.plan
        FROM saas_users su
        JOIN saas_accounts sa ON sa.id = su.account_id
        WHERE su.email = $1
        """,
        req.email.lower(),
    )

    if not row or not verify_password(req.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not row["is_active"]:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    # Update last login
    await pool.execute(
        "UPDATE saas_users SET last_login_at = NOW() WHERE id = $1",
        row["id"],
    )

    access = create_access_token(str(row["id"]), str(row["account_id"]), row["plan"])
    refresh = create_refresh_token(str(row["id"]))

    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(req: RefreshRequest):
    """Exchange a refresh token for a new access token."""
    try:
        payload = decode_token(req.refresh_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    pool = get_db_pool()
    try:
        user_uuid = _uuid.UUID(payload["sub"])
    except (ValueError, KeyError):
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    row = await pool.fetchrow(
        """
        SELECT su.id, su.account_id, su.is_active, sa.plan
        FROM saas_users su
        JOIN saas_accounts sa ON sa.id = su.account_id
        WHERE su.id = $1
        """,
        user_uuid,
    )

    if not row or not row["is_active"]:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    access = create_access_token(str(row["id"]), str(row["account_id"]), row["plan"])
    new_refresh = create_refresh_token(str(row["id"]))

    return TokenResponse(access_token=access, refresh_token=new_refresh)


@router.get("/me", response_model=UserResponse)
async def me(user: AuthUser = Depends(require_auth)):
    """Get current user and account info."""
    # When auth is disabled, return synthetic admin data (no DB query)
    if not settings.saas_auth.enabled:
        return UserResponse(
            user_id=user.user_id,
            email="admin@localhost",
            full_name="Admin",
            role="owner",
            account_id=user.account_id,
            account_name="Local Dev",
            plan="pro",
            plan_status="active",
            asin_limit=9999,
            trial_ends_at=None,
        )

    pool = get_db_pool()
    row = await pool.fetchrow(
        """
        SELECT su.email, su.full_name, su.role,
               sa.id AS account_id, sa.name AS account_name,
               sa.plan, sa.plan_status, sa.asin_limit, sa.trial_ends_at
        FROM saas_users su
        JOIN saas_accounts sa ON sa.id = su.account_id
        WHERE su.id = $1
        """,
        _uuid.UUID(user.user_id),
    )

    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        user_id=user.user_id,
        email=row["email"],
        full_name=row["full_name"],
        role=row["role"],
        account_id=str(row["account_id"]),
        account_name=row["account_name"],
        plan=row["plan"],
        plan_status=row["plan_status"],
        asin_limit=row["asin_limit"],
        trial_ends_at=row["trial_ends_at"].isoformat() if row["trial_ends_at"] else None,
    )


@router.post("/change-password")
async def change_password(req: ChangePasswordRequest, user: AuthUser = Depends(require_auth)):
    """Change password for current user."""
    if not settings.saas_auth.enabled:
        raise HTTPException(status_code=404, detail="Password management not available in local dev mode")

    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

    pool = get_db_pool()
    uid = _uuid.UUID(user.user_id)
    row = await pool.fetchrow(
        "SELECT password_hash FROM saas_users WHERE id = $1",
        uid,
    )

    if not row or not verify_password(req.current_password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    new_hash = hash_password(req.new_password)
    await pool.execute(
        "UPDATE saas_users SET password_hash = $1 WHERE id = $2",
        new_hash,
        uid,
    )

    return {"status": "ok"}
