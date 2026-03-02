"""Stripe billing endpoints: checkout, portal, status, webhook."""

import logging
import uuid as _uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ..auth.dependencies import AuthUser, require_auth
from ..config import settings
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.billing")

router = APIRouter(prefix="/billing", tags=["billing"])
webhook_router = APIRouter(tags=["billing-webhook"])

PLAN_LIMITS = {
    "trial":   {"asins": 5,   "compare": False, "api": False},
    "starter": {"asins": 5,   "compare": False, "api": False},
    "growth":  {"asins": 25,  "compare": True,  "api": True},
    "pro":     {"asins": 100, "compare": True,  "api": True},
}

PRICE_TO_PLAN = {}  # populated at module init from config


def _init_price_map():
    cfg = settings.saas_auth
    if cfg.stripe_starter_price_id:
        PRICE_TO_PLAN[cfg.stripe_starter_price_id] = "starter"
    if cfg.stripe_growth_price_id:
        PRICE_TO_PLAN[cfg.stripe_growth_price_id] = "growth"
    if cfg.stripe_pro_price_id:
        PRICE_TO_PLAN[cfg.stripe_pro_price_id] = "pro"


def _get_stripe():
    import stripe
    cfg = settings.saas_auth
    if not cfg.stripe_secret_key:
        raise HTTPException(status_code=503, detail="Stripe not configured")
    stripe.api_key = cfg.stripe_secret_key
    return stripe


# -- Request/Response schemas --

PLAN_NAME_TO_CONFIG_KEY = {
    "starter": "stripe_starter_price_id",
    "growth": "stripe_growth_price_id",
    "pro": "stripe_pro_price_id",
}


class CheckoutRequest(BaseModel):
    price_id: str = ""
    plan: str = ""  # alternative: pass plan name (starter/growth/pro)
    success_url: str = ""
    cancel_url: str = ""


class CheckoutResponse(BaseModel):
    checkout_url: str


class PortalResponse(BaseModel):
    portal_url: str


class BillingStatus(BaseModel):
    plan: str
    plan_status: str
    asin_limit: int
    trial_ends_at: str | None
    stripe_customer_id: str | None


# -- Endpoints --

@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(req: CheckoutRequest, user: AuthUser = Depends(require_auth)):
    """Create a Stripe Checkout session for plan upgrade."""
    stripe = _get_stripe()
    pool = get_db_pool()
    acct_uuid = _uuid.UUID(user.account_id)
    user_uuid = _uuid.UUID(user.user_id)

    # Resolve price_id from plan name if needed
    price_id = req.price_id
    if not price_id and req.plan:
        cfg_key = PLAN_NAME_TO_CONFIG_KEY.get(req.plan)
        if not cfg_key:
            raise HTTPException(status_code=400, detail=f"Unknown plan: {req.plan}")
        price_id = getattr(settings.saas_auth, cfg_key, "")
        if not price_id:
            raise HTTPException(status_code=400, detail=f"No Stripe price configured for plan '{req.plan}'")
    if not price_id:
        raise HTTPException(status_code=400, detail="Either price_id or plan is required")

    # Validate price_id against configured prices
    _cfg = settings.saas_auth
    valid_prices = {
        v for v in [_cfg.stripe_starter_price_id, _cfg.stripe_growth_price_id, _cfg.stripe_pro_price_id]
        if v
    }
    if valid_prices and price_id not in valid_prices:
        raise HTTPException(status_code=400, detail="Invalid price ID")

    # Get or create Stripe customer
    account = await pool.fetchrow(
        "SELECT stripe_customer_id, name FROM saas_accounts WHERE id = $1",
        acct_uuid,
    )
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    customer_id = account["stripe_customer_id"]
    if not customer_id:
        # Create customer on the fly
        user_row = await pool.fetchrow(
            "SELECT email FROM saas_users WHERE id = $1", user_uuid
        )
        customer = stripe.Customer.create(
            email=user_row["email"] if user_row else "",
            name=account["name"],
            metadata={"account_id": str(user.account_id)},
            timeout=10,
        )
        customer_id = customer.id
        await pool.execute(
            "UPDATE saas_accounts SET stripe_customer_id = $1 WHERE id = $2",
            customer_id,
            acct_uuid,
        )

    if not req.success_url:
        raise HTTPException(status_code=400, detail="success_url is required")
    if not req.cancel_url:
        raise HTTPException(status_code=400, detail="cancel_url is required")

    session = stripe.checkout.Session.create(
        customer=customer_id,
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=req.success_url,
        cancel_url=req.cancel_url,
        metadata={"account_id": str(user.account_id)},
        timeout=10,
    )

    return CheckoutResponse(checkout_url=session.url)


@router.post("/portal", response_model=PortalResponse)
async def create_portal(user: AuthUser = Depends(require_auth)):
    """Create a Stripe Customer Portal session."""
    if user.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only account owner/admin can manage billing")

    stripe = _get_stripe()
    pool = get_db_pool()

    customer_id = await pool.fetchval(
        "SELECT stripe_customer_id FROM saas_accounts WHERE id = $1",
        _uuid.UUID(user.account_id),
    )
    if not customer_id:
        raise HTTPException(status_code=400, detail="No billing account found. Subscribe to a plan first.")

    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        timeout=10,
    )

    return PortalResponse(portal_url=session.url)


@router.get("/status", response_model=BillingStatus)
async def billing_status(user: AuthUser = Depends(require_auth)):
    """Get current billing status for the account."""
    pool = get_db_pool()
    row = await pool.fetchrow(
        """
        SELECT plan, plan_status, asin_limit, trial_ends_at, stripe_customer_id
        FROM saas_accounts WHERE id = $1
        """,
        _uuid.UUID(user.account_id),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Account not found")

    return BillingStatus(
        plan=row["plan"],
        plan_status=row["plan_status"],
        asin_limit=row["asin_limit"],
        trial_ends_at=row["trial_ends_at"].isoformat() if row["trial_ends_at"] else None,
        stripe_customer_id=row["stripe_customer_id"],
    )


# -- Stripe Webhook --

@webhook_router.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events."""
    cfg = settings.saas_auth
    if not cfg.stripe_secret_key:
        raise HTTPException(status_code=503, detail="Stripe not configured")

    import stripe
    stripe.api_key = cfg.stripe_secret_key

    body = await request.body()
    if len(body) > 65536:
        raise HTTPException(status_code=413, detail="Payload too large")

    sig = request.headers.get("stripe-signature", "")
    if not sig:
        raise HTTPException(status_code=400, detail="Missing stripe-signature header")

    if not cfg.stripe_webhook_secret:
        raise HTTPException(status_code=503, detail="Webhook secret not configured")

    try:
        event = stripe.Webhook.construct_event(body, sig, cfg.stripe_webhook_secret)
    except Exception as e:
        logger.warning("Stripe webhook signature verification failed: %s", e)
        raise HTTPException(status_code=400, detail="Invalid signature")

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    # Idempotency check
    existing = await pool.fetchval(
        "SELECT id FROM billing_events WHERE stripe_event_id = $1", event.id
    )
    if existing:
        return {"status": "already_processed"}

    # Initialize price map if needed
    if not PRICE_TO_PLAN:
        _init_price_map()

    event_type = event.type
    obj = event.data.object

    account_id = None

    if event_type == "checkout.session.completed":
        account_id = await _handle_checkout_completed(pool, obj)

    elif event_type == "invoice.paid":
        account_id = await _handle_invoice_paid(pool, obj)

    elif event_type == "invoice.payment_failed":
        account_id = await _handle_invoice_failed(pool, obj)

    elif event_type == "customer.subscription.updated":
        account_id = await _handle_subscription_updated(pool, obj)

    elif event_type == "customer.subscription.deleted":
        account_id = await _handle_subscription_deleted(pool, obj)

    # Log event -- pass dict for JSONB column, add ::jsonb cast for asyncpg
    import json
    payload_str = json.dumps(event.data.object.to_dict() if hasattr(event.data.object, 'to_dict') else {})
    await pool.execute(
        """
        INSERT INTO billing_events (account_id, stripe_event_id, event_type, payload)
        VALUES ($1, $2, $3, $4::jsonb)
        ON CONFLICT (stripe_event_id) DO NOTHING
        """,
        account_id,
        event.id,
        event_type,
        payload_str,
    )

    return {"status": "ok"}


async def _find_account_by_customer(pool, customer_id: str) -> _uuid.UUID | None:
    """Find account ID (UUID) by Stripe customer ID."""
    return await pool.fetchval(
        "SELECT id FROM saas_accounts WHERE stripe_customer_id = $1", customer_id
    )


async def _handle_checkout_completed(pool, session) -> _uuid.UUID | None:
    """Handle checkout.session.completed -- activate subscription."""
    customer_id = session.customer
    subscription_id = session.subscription
    account_id_str = session.metadata.get("account_id") if session.metadata else None

    account_id: _uuid.UUID | None = None
    if account_id_str:
        try:
            account_id = _uuid.UUID(account_id_str)
        except ValueError:
            pass

    if not account_id:
        account_id = await _find_account_by_customer(pool, customer_id)

    if not account_id:
        logger.warning("Checkout completed but no account found for customer %s", customer_id)
        return None

    # Determine plan from line items via subscription
    plan = "starter"
    try:
        import stripe
        sub = stripe.Subscription.retrieve(subscription_id, timeout=10)
        if sub.items and sub.items.data:
            price_id = sub.items.data[0].price.id
            plan = PRICE_TO_PLAN.get(price_id, "starter")
    except Exception as e:
        logger.error("Failed to determine plan from subscription %s: %s", subscription_id, e)

    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["starter"])

    await pool.execute(
        """
        UPDATE saas_accounts
        SET plan = $1, plan_status = 'active',
            stripe_customer_id = $2, stripe_subscription_id = $3,
            asin_limit = $4, updated_at = NOW()
        WHERE id = $5
        """,
        plan,
        customer_id,
        subscription_id,
        limits["asins"],
        account_id,
    )

    logger.info("Account %s upgraded to %s", account_id, plan)
    return account_id


async def _handle_invoice_paid(pool, invoice) -> _uuid.UUID | None:
    """Handle invoice.paid -- ensure plan_status is active."""
    customer_id = invoice.customer
    account_id = await _find_account_by_customer(pool, customer_id)
    if account_id:
        await pool.execute(
            "UPDATE saas_accounts SET plan_status = 'active', updated_at = NOW() WHERE id = $1",
            account_id,
        )
    return account_id


async def _handle_invoice_failed(pool, invoice) -> _uuid.UUID | None:
    """Handle invoice.payment_failed -- set past_due."""
    customer_id = invoice.customer
    account_id = await _find_account_by_customer(pool, customer_id)
    if account_id:
        await pool.execute(
            "UPDATE saas_accounts SET plan_status = 'past_due', updated_at = NOW() WHERE id = $1",
            account_id,
        )
        logger.warning("Account %s payment failed, set to past_due", account_id)
    return account_id


async def _handle_subscription_updated(pool, subscription) -> _uuid.UUID | None:
    """Handle subscription changes (upgrades/downgrades)."""
    customer_id = subscription.customer
    account_id = await _find_account_by_customer(pool, customer_id)
    if not account_id:
        return None

    plan = "starter"
    if subscription.items and subscription.items.data:
        price_id = subscription.items.data[0].price.id
        plan = PRICE_TO_PLAN.get(price_id, "starter")

    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["starter"])

    # Map Stripe subscription status to our plan_status
    status_map = {
        "active": "active",
        "past_due": "past_due",
        "trialing": "trialing",
        "canceled": "canceled",
        "unpaid": "past_due",
        "incomplete": "past_due",
        "incomplete_expired": "canceled",
    }
    sub_status = getattr(subscription, "status", "active")
    plan_status = status_map.get(sub_status, "active")

    await pool.execute(
        """
        UPDATE saas_accounts
        SET plan = $1, plan_status = $2, asin_limit = $3,
            stripe_subscription_id = $4, updated_at = NOW()
        WHERE id = $5
        """,
        plan,
        plan_status,
        limits["asins"],
        subscription.id,
        account_id,
    )

    logger.info("Account %s subscription updated to %s (status=%s)", account_id, plan, plan_status)
    return account_id


async def _handle_subscription_deleted(pool, subscription) -> _uuid.UUID | None:
    """Handle subscription cancellation."""
    customer_id = subscription.customer
    account_id = await _find_account_by_customer(pool, customer_id)
    if account_id:
        await pool.execute(
            "UPDATE saas_accounts SET plan_status = 'canceled', updated_at = NOW() WHERE id = $1",
            account_id,
        )
        logger.info("Account %s subscription canceled", account_id)
    return account_id
