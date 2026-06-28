"""Stripe billing endpoints: checkout, portal, status, webhook."""

import logging
import re
import time
import uuid as _uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ..auth.dependencies import AuthUser, require_auth
from ..config import settings
from ..content_ops_deflection_incidents import emit_deflection_paid_funnel_incident_alert
from ..content_ops_deflection_reconciliation import record_paid_report_missing
from ..storage.database import get_db_pool
from extracted_content_pipeline.deflection_report_access import (
    DELTA_ENTITLEMENT_ACTIVE_STATUSES,
    DeflectionReportAccessRecord,
    DeflectionReportArtifactStore,
    PostgresDeflectionReportArtifactStore,
)

logger = logging.getLogger("atlas.api.billing")

router = APIRouter(prefix="/billing", tags=["billing"])
webhook_router = APIRouter(tags=["billing-webhook"])

PLAN_LIMITS = {
    "trial":   {"asins": 5,   "compare": False, "api": False},
    "starter": {"asins": 5,   "compare": False, "api": False},
    "growth":  {"asins": 25,  "compare": True,  "api": True},
    "pro":     {"asins": 100, "compare": True,  "api": True},
}

B2B_PLAN_LIMITS = {
    "b2b_trial":   {"vendors": 1,  "campaigns": False, "reports": False},
    "b2b_starter": {"vendors": 5,  "campaigns": False, "reports": True},
    "b2b_growth":  {"vendors": 25, "campaigns": True,  "reports": True},
    "b2b_pro":     {"vendors": -1, "campaigns": True,  "reports": True, "api": True},
}

# LLM Gateway plan tiers (PR-D2). monthly_token_limit is advertised
# here but enforced in PR-D4 against per-account llm_usage rows;
# byok_keys_max gates how many provider API keys a customer can store
# (PR-D5). cache_enabled / batch_enabled are feature gates for the
# /api/v1/llm/* router.
LLM_PLAN_LIMITS = {
    "llm_trial":   {"monthly_token_limit": 1_000_000,   "cache_enabled": True, "batch_enabled": False, "byok_keys_max": 2},
    "llm_starter": {"monthly_token_limit": 10_000_000,  "cache_enabled": True, "batch_enabled": True,  "byok_keys_max": 4},
    "llm_growth":  {"monthly_token_limit": 100_000_000, "cache_enabled": True, "batch_enabled": True,  "byok_keys_max": 10},
    "llm_pro":     {"monthly_token_limit": -1,          "cache_enabled": True, "batch_enabled": True,  "byok_keys_max": -1},
}

PRICE_TO_PLAN = {}  # populated at module init from config
STRIPE_API_VERSION = "2026-05-27.dahlia"
DEFLECTION_DELTA_SUBSCRIPTION_SOURCE = "content_ops_deflection_delta_subscription"
_UUID_TEXT_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _init_price_map():
    cfg = settings.saas_auth
    if cfg.stripe_starter_price_id:
        PRICE_TO_PLAN[cfg.stripe_starter_price_id] = "starter"
    if cfg.stripe_growth_price_id:
        PRICE_TO_PLAN[cfg.stripe_growth_price_id] = "growth"
    if cfg.stripe_pro_price_id:
        PRICE_TO_PLAN[cfg.stripe_pro_price_id] = "pro"
    if cfg.stripe_b2b_starter_price_id:
        PRICE_TO_PLAN[cfg.stripe_b2b_starter_price_id] = "b2b_starter"
    if cfg.stripe_b2b_growth_price_id:
        PRICE_TO_PLAN[cfg.stripe_b2b_growth_price_id] = "b2b_growth"
    if cfg.stripe_b2b_pro_price_id:
        PRICE_TO_PLAN[cfg.stripe_b2b_pro_price_id] = "b2b_pro"
    if cfg.stripe_vendor_standard_price_id:
        PRICE_TO_PLAN[cfg.stripe_vendor_standard_price_id] = "vendor_standard"
    if cfg.stripe_vendor_pro_price_id:
        PRICE_TO_PLAN[cfg.stripe_vendor_pro_price_id] = "vendor_pro"
    # LLM Gateway plan tiers (PR-D2)
    if cfg.stripe_llm_starter_price_id:
        PRICE_TO_PLAN[cfg.stripe_llm_starter_price_id] = "llm_starter"
    if cfg.stripe_llm_growth_price_id:
        PRICE_TO_PLAN[cfg.stripe_llm_growth_price_id] = "llm_growth"
    if cfg.stripe_llm_pro_price_id:
        PRICE_TO_PLAN[cfg.stripe_llm_pro_price_id] = "llm_pro"


def _warn_if_unrestricted_stripe_key(secret_key: str) -> None:
    if secret_key.startswith("sk_"):
        logger.warning(
            "ATLAS_SAAS_STRIPE_SECRET_KEY is configured with a full Stripe secret key; "
            "use a restricted rk_ key scoped to Atlas billing operations when possible"
        )


def _configure_stripe_module(stripe_module: Any, secret_key: str) -> Any:
    stripe_module.api_key = secret_key
    stripe_module.api_version = STRIPE_API_VERSION
    _warn_if_unrestricted_stripe_key(secret_key)
    return stripe_module


def _stripe_webhook_secret_candidates(secret_config: str) -> tuple[str, ...]:
    """Return ordered Stripe webhook signing secrets from config.

    Stripe exposes separate signing secrets for live, test, and rotated webhook
    endpoints. A comma-separated value lets ATLAS accept a bounded rotation set
    without weakening the signed-webhook trust boundary.
    """

    return tuple(
        secret.strip()
        for secret in str(secret_config or "").split(",")
        if secret.strip()
    )


def _construct_stripe_webhook_event(
    stripe_module: Any,
    body: bytes,
    signature: str,
    secret_config: str,
) -> Any:
    """Verify a Stripe webhook against one or more configured secrets."""

    secrets = _stripe_webhook_secret_candidates(secret_config)
    if not secrets:
        raise ValueError("Webhook secret not configured")

    last_error: Exception | None = None
    for secret in secrets:
        try:
            return stripe_module.Webhook.construct_event(body, signature, secret)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("Webhook secret not configured")


def _stripe_customer_idempotency_key(account_id: str) -> str:
    return f"atlas-customer:{account_id}"


def _stripe_checkout_idempotency_key(
    *,
    account_id: str,
    user_id: str,
    price_id: str,
    success_url: str,
    cancel_url: str,
) -> str:
    digest = sha256(
        "\0".join((price_id, success_url, cancel_url)).encode("utf-8")
    ).hexdigest()[:32]
    return f"atlas-checkout:{account_id}:{user_id}:{digest}"


def _get_stripe():
    import stripe
    cfg = settings.saas_auth
    if not cfg.stripe_secret_key:
        raise HTTPException(status_code=503, detail="Stripe not configured")
    return _configure_stripe_module(stripe, cfg.stripe_secret_key)


# -- Request/Response schemas --

PLAN_NAME_TO_CONFIG_KEY = {
    "starter": "stripe_starter_price_id",
    "growth": "stripe_growth_price_id",
    "pro": "stripe_pro_price_id",
    "b2b_starter": "stripe_b2b_starter_price_id",
    "b2b_growth": "stripe_b2b_growth_price_id",
    "b2b_pro": "stripe_b2b_pro_price_id",
    # LLM Gateway plan tiers (PR-D2)
    "llm_starter": "stripe_llm_starter_price_id",
    "llm_growth": "stripe_llm_growth_price_id",
    "llm_pro": "stripe_llm_pro_price_id",
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


class PortalRequest(BaseModel):
    return_url: str = ""


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
        v for v in [
            _cfg.stripe_starter_price_id, _cfg.stripe_growth_price_id, _cfg.stripe_pro_price_id,
            _cfg.stripe_b2b_starter_price_id, _cfg.stripe_b2b_growth_price_id, _cfg.stripe_b2b_pro_price_id,
            # LLM Gateway plan tiers (PR-D2)
            _cfg.stripe_llm_starter_price_id, _cfg.stripe_llm_growth_price_id, _cfg.stripe_llm_pro_price_id,
        ]
        if v
    }
    delta_price_ids = set(_configured_deflection_delta_price_ids())
    valid_prices.update(delta_price_ids)
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
            idempotency_key=_stripe_customer_idempotency_key(str(user.account_id)),
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

    checkout_metadata = {"account_id": str(user.account_id)}
    session_params: dict[str, Any] = {}
    if price_id in delta_price_ids:
        checkout_metadata = {
            **checkout_metadata,
            "source": DEFLECTION_DELTA_SUBSCRIPTION_SOURCE,
        }
        session_params["subscription_data"] = {"metadata": checkout_metadata}

    session = stripe.checkout.Session.create(
        customer=customer_id,
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=req.success_url,
        cancel_url=req.cancel_url,
        metadata=checkout_metadata,
        idempotency_key=_stripe_checkout_idempotency_key(
            account_id=str(user.account_id),
            user_id=str(user.user_id),
            price_id=price_id,
            success_url=req.success_url,
            cancel_url=req.cancel_url,
        ),
        timeout=10,
        **session_params,
    )

    return CheckoutResponse(checkout_url=session.url)


@router.post("/portal", response_model=PortalResponse)
async def create_portal(req: PortalRequest, user: AuthUser = Depends(require_auth)):
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
        return_url=req.return_url or None,
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
    _configure_stripe_module(stripe, cfg.stripe_secret_key)

    body = await request.body()
    if len(body) > 65536:
        raise HTTPException(status_code=413, detail="Payload too large")

    sig = request.headers.get("stripe-signature", "")
    if not sig:
        raise HTTPException(status_code=400, detail="Missing stripe-signature header")

    if not _stripe_webhook_secret_candidates(cfg.stripe_webhook_secret):
        raise HTTPException(status_code=503, detail="Webhook secret not configured")

    try:
        event = _construct_stripe_webhook_event(
            stripe,
            body,
            sig,
            cfg.stripe_webhook_secret,
        )
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
    event_created = _stripe_event_created(event)

    account_id = None

    if event_type == "checkout.session.completed":
        # Check if this is a vendor retention checkout (has vendor metadata)
        meta = obj.metadata or {}
        if meta.get("source") == "vendor_briefing_report":
            await _handle_vendor_checkout_completed(pool, obj, meta)
        elif meta.get("source") == "content_ops_deflection_report":
            if _stripe_text(obj, "payment_status") != "paid":
                _log_content_ops_deflection_report_payment_pending(obj, meta)
            else:
                await _handle_content_ops_deflection_report_checkout_completed(
                    pool,
                    obj,
                    meta,
                    event_type=event_type,
                    event_created=getattr(event, "created", None),
                )
        elif _is_deflection_delta_subscription_source(obj) or (
            _deflection_delta_price_id_from_object(obj) is not None
        ):
            if _stripe_text(obj, "payment_status") == "paid":
                account_id = await _handle_deflection_delta_invoice_lifecycle(
                    pool,
                    obj,
                    stripe_subscription_status="active",
                    stripe_event_created=event_created,
                )
            else:
                logger.warning(
                    "Deflection delta checkout completed before funds were available: session=%s payment_status=%s",
                    _stripe_text(obj, "id") or "<missing>",
                    _stripe_text(obj, "payment_status") or "<missing>",
                )
        else:
            account_id = await _handle_checkout_completed(pool, obj)

    elif event_type == "checkout.session.async_payment_succeeded":
        meta = obj.metadata or {}
        if meta.get("source") == "content_ops_deflection_report":
            if _stripe_text(obj, "payment_status") != "paid":
                _log_content_ops_deflection_report_payment_pending(obj, meta)
            else:
                await _handle_content_ops_deflection_report_checkout_completed(
                    pool,
                    obj,
                    meta,
                    event_type=event_type,
                    event_created=getattr(event, "created", None),
                )

    elif event_type == "checkout.session.async_payment_failed":
        meta = obj.metadata or {}
        if meta.get("source") == "content_ops_deflection_report":
            _log_content_ops_deflection_report_async_payment_failed(obj, meta)

    elif event_type in {"charge.refunded", "charge.dispute.created"}:
        await _handle_content_ops_deflection_report_payment_revoked(
            pool,
            stripe,
            obj,
            event_type=event_type,
        )

    elif event_type == "charge.dispute.closed":
        await _handle_content_ops_deflection_report_dispute_closed(
            pool,
            stripe,
            obj,
            event_type=event_type,
        )

    elif event_type == "invoice.paid":
        if _deflection_delta_price_id_from_object(obj) is not None:
            account_id = await _handle_deflection_delta_invoice_lifecycle(
                pool,
                obj,
                stripe_subscription_status="active",
                stripe_event_created=event_created,
            )
        else:
            account_id = await _handle_invoice_paid(pool, obj)

    elif event_type == "invoice.payment_failed":
        if _deflection_delta_price_id_from_object(obj) is not None:
            account_id = await _handle_deflection_delta_invoice_lifecycle(
                pool,
                obj,
                stripe_subscription_status="past_due",
                stripe_event_created=event_created,
            )
        else:
            account_id = await _handle_invoice_failed(pool, obj)

    elif event_type in {"customer.subscription.created", "customer.subscription.updated"}:
        if (
            event_type == "customer.subscription.created"
            and _deflection_delta_price_id_from_object(obj) is None
        ):
            account_id = None
        else:
            account_id = await _handle_subscription_updated(
                pool,
                obj,
                stripe_event_created=event_created,
            )

    elif event_type == "customer.subscription.deleted":
        account_id = await _handle_subscription_deleted(
            pool,
            obj,
            stripe_event_created=event_created,
        )

    # Log event -- pass dict for JSONB column, add ::jsonb cast for asyncpg
    import json
    payload_str = json.dumps(event.data.object.to_dict() if hasattr(event.data.object, 'to_dict') else {})
    try:
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
    except Exception:
        logger.exception(
            "Stripe webhook side effect completed but billing_events audit insert failed: event=%s",
            event.id,
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

    is_b2b = plan.startswith("b2b_")
    if is_b2b:
        b2b_limits = B2B_PLAN_LIMITS.get(plan, B2B_PLAN_LIMITS["b2b_starter"])
        await pool.execute(
            """
            UPDATE saas_accounts
            SET plan = $1, plan_status = 'active',
                stripe_customer_id = $2, stripe_subscription_id = $3,
                vendor_limit = $4, updated_at = NOW()
            WHERE id = $5
            """,
            plan,
            customer_id,
            subscription_id,
            b2b_limits["vendors"],
            account_id,
        )
    else:
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


async def _handle_vendor_checkout_completed(pool, session, meta: dict) -> None:
    """Handle vendor retention checkout -- send confirmation email.

    Deduplicates against the direct-send path in checkout_session_info.
    """
    vendor_name = meta.get("vendor_name", "")
    tier = meta.get("tier", "standard")
    customer_email = ""

    if session.customer_details:
        customer_email = session.customer_details.email or ""
    if not customer_email:
        customer_email = session.customer_email or ""

    if not customer_email:
        logger.warning("Vendor checkout completed but no customer email found")
        return

    # Dedup: check if direct-send path already sent for this session
    dedup_key = f"vendor_checkout_email_{session.id}"
    already_sent = await pool.fetchval(
        "SELECT 1 FROM billing_events WHERE stripe_event_id = $1", dedup_key
    )
    if already_sent:
        logger.info("Vendor checkout confirmation already sent (direct path): %s", session.id)
        return

    # Send confirmation email
    try:
        from ..templates.email.vendor_checkout_confirmation import (
            render_checkout_confirmation_html,
            render_checkout_confirmation_text,
        )
        from ..services.email_provider import get_email_provider

        html = render_checkout_confirmation_html(vendor_name, tier, customer_email)
        text = render_checkout_confirmation_text(vendor_name, tier)

        email_provider = get_email_provider()
        await email_provider.send(
            to=[customer_email],
            subject=f"Subscription Confirmed: {vendor_name} Churn Intelligence",
            body=text,
            html=html,
            reply_to="outreach@churnsignals.co",
        )
        # Mark as sent
        await pool.execute(
            """
            INSERT INTO billing_events (stripe_event_id, event_type, payload)
            VALUES ($1, $2, '{}'::jsonb)
            ON CONFLICT (stripe_event_id) DO NOTHING
            """,
            dedup_key,
            "vendor_checkout_confirmation_email",
        )
        logger.info(
            "Vendor checkout confirmation sent (webhook): email=%s vendor=%s tier=%s",
            customer_email, vendor_name, tier,
        )
    except Exception:
        logger.exception("Failed to send vendor checkout confirmation email")


def _event_age_seconds(event_created: int | None) -> int | None:
    """Seconds since a Stripe event was created, or None if unknown.

    Used to tell a transient write-ordering race (recent event -> retry) from a
    permanent paid-but-missing report (aged event -> reconcile). #1462.
    """

    if not event_created:
        return None
    try:
        return max(0, int(time.time()) - int(event_created))
    except (TypeError, ValueError):
        return None


async def _handle_content_ops_deflection_report_checkout_completed(
    pool: Any,
    session: Any,
    meta: Mapping[str, Any],
    *,
    event_type: str = "checkout.session.completed",
    event_created: int | None = None,
) -> None:
    """Handle one-time deflection report checkout completion."""

    session_id = _stripe_text(session, "id")
    account_id_text = _clean_metadata(meta.get("account_id"))
    request_id = _clean_metadata(meta.get("request_id"))
    if _stripe_text(session, "mode") != "payment":
        logger.warning(
            "Deflection report checkout ignored: non-payment mode session=%s",
            session_id,
        )
        return
    if _stripe_text(session, "payment_status") != "paid":
        logger.warning(
            "Deflection report checkout ignored: unpaid session=%s",
            session_id,
        )
        return
    if not account_id_text or not request_id:
        logger.warning(
            "Deflection report checkout ignored: missing account_id/request_id session=%s",
            session_id,
        )
        return
    try:
        _uuid.UUID(account_id_text)
    except ValueError:
        logger.warning(
            "Deflection report checkout ignored: invalid account_id session=%s",
            session_id,
        )
        return
    paid_terms = _deflection_checkout_paid_terms(session)
    if paid_terms is None:
        await emit_deflection_paid_funnel_incident_alert(
            logger,
            incident_type="paid_report_checkout_terms_mismatch",
            severity="error",
            account_id=account_id_text,
            request_id=request_id,
            event_type=event_type,
            stripe_session_id=session_id,
            amount_total=_stripe_object_value(session, "amount_total"),
            currency=_stripe_text(session, "currency"),
        )
        logger.warning(
            "Deflection report checkout ignored: amount/currency mismatch session=%s",
            session_id,
        )
        return

    paid_amount_cents, paid_currency, allowed_cents = paid_terms
    store = PostgresDeflectionReportArtifactStore(pool=pool)
    marked = await store.mark_paid(
        account_id=account_id_text,
        request_id=request_id,
        payment_reference=session_id or None,
        checkout_amount_cents=paid_amount_cents,
        checkout_currency=paid_currency,
        require_checkout_authorization=len(allowed_cents) > 1,
    )
    if not marked:
        record = await store.get_artifact_record(
            account_id=account_id_text,
            request_id=request_id,
        )
        if record is not None:
            await _emit_deflection_checkout_authorized_terms_mismatch(
                record,
                event_type=event_type,
                session_id=session_id,
                paid_amount_cents=paid_amount_cents,
                paid_currency=paid_currency,
            )
            return
        age = _event_age_seconds(event_created)
        grace = int(
            getattr(
                settings.saas_auth,
                "stripe_content_ops_deflection_report_reconcile_grace_seconds",
                300,
            )
            or 300
        )
        if age is not None and age > grace:
            # Aged past the write-ordering race window: the report row will not
            # appear on retry, so this is a permanent paid-but-missing case.
            # Record it for manual reconciliation and return 2xx so Stripe stops
            # retrying a non-2xx for hours (#1462).
            await record_paid_report_missing(
                pool,
                account_id=account_id_text,
                request_id=request_id,
                # '' (not NULL) so the reconciliation ledger's
                # (account_id, request_id, stripe_session_id) UNIQUE dedups a
                # missing-session retry; NULL would be treated as DISTINCT.
                stripe_session_id=session_id or "",
                event_type=event_type,
            )
            await emit_deflection_paid_funnel_incident_alert(
                logger,
                incident_type="paid_report_missing_after_payment",
                severity="error",
                account_id=account_id_text,
                request_id=request_id,
                event_type=event_type,
                stripe_session_id=session_id,
                disposition="reconciled",
                event_age_seconds=age,
            )
            logger.error(
                "Deflection report checkout completed but report was not found "
                "(permanent, recorded for reconciliation): account=%s request=%s "
                "session=%s age=%ss",
                account_id_text,
                request_id,
                session_id,
                age,
            )
            return
        # Within the race window (or unknown event age): 409 so Stripe retries
        # and finds the report row once its write commits. Real Stripe events
        # always carry `created`, so a genuinely permanent miss ages past the
        # window above; only the transient race (or a malformed/timestampless
        # event) lands here.
        await emit_deflection_paid_funnel_incident_alert(
            logger,
            incident_type="paid_report_missing_after_payment",
            severity="error",
            account_id=account_id_text,
            request_id=request_id,
            event_type=event_type,
            stripe_session_id=session_id,
        )
        logger.error(
            "Deflection report checkout completed but report was not found: account=%s request=%s session=%s",
            account_id_text,
            request_id,
            session_id,
        )
        raise HTTPException(status_code=409, detail="Deflection report not found")
    await _queue_content_ops_deflection_report_delivery(
        pool,
        account_id=account_id_text,
        request_id=request_id,
        payment_reference=session_id or None,
    )
    return


async def _queue_content_ops_deflection_report_delivery(
    pool: Any,
    *,
    account_id: str,
    request_id: str,
    payment_reference: str | None = None,
) -> str:
    """Queue post-purchase delivery after the verified Stripe paid gate."""

    store = PostgresDeflectionReportArtifactStore(pool=pool)
    record = await store.get_artifact_record(
        account_id=account_id,
        request_id=request_id,
    )
    if record is None:
        return "missing_report"
    if not record.delivery_email:
        logger.info(
            "Deflection report delivery queue skipped: no delivery email account=%s request=%s",
            account_id,
            request_id,
        )
        return "missing_delivery_email"
    await pool.execute(
        """
        INSERT INTO content_ops_deflection_report_deliveries (
            account_id, request_id, payment_reference, delivery_status, updated_at
        )
        VALUES ($1, $2, $3, 'pending', NOW())
        ON CONFLICT (account_id, request_id) DO UPDATE
        SET payment_reference = COALESCE(EXCLUDED.payment_reference, content_ops_deflection_report_deliveries.payment_reference),
            delivery_status = CASE
                WHEN content_ops_deflection_report_deliveries.delivery_status IN ('delivered', 'sending')
                    THEN content_ops_deflection_report_deliveries.delivery_status
                ELSE 'pending'
            END,
            updated_at = NOW()
        """,
        account_id,
        request_id,
        payment_reference,
    )
    return "queued"


async def _handle_content_ops_deflection_report_payment_revoked(
    pool: Any,
    stripe_module: Any,
    obj: Any,
    *,
    event_type: str,
) -> None:
    """Relock a deflection report after a refund or dispute webhook."""

    if event_type == "charge.refunded" and not _stripe_charge_refund_is_full(obj):
        logger.info(
            "Deflection report partial refund observed without revocation: "
            "event_type=%s object=%s amount_refunded=%s amount_captured=%s",
            event_type,
            _stripe_text(obj, "id") or "<missing>",
            _stripe_object_value(obj, "amount_refunded"),
            _stripe_object_value(obj, "amount_captured")
            or _stripe_object_value(obj, "amount"),
        )
        return

    meta, payment_reference = _content_ops_deflection_revocation_metadata(
        stripe_module,
        obj,
    )
    if meta.get("source") != "content_ops_deflection_report":
        logger.info(
            "Deflection report payment revocation could not be mapped: "
            "event_type=%s object=%s payment_intent=%s",
            event_type,
            _stripe_text(obj, "id") or "<missing>",
            _payment_intent_from_payment_event(obj) or "<missing>",
        )
        return

    account_id_text = _clean_metadata(meta.get("account_id"))
    request_id = _clean_metadata(meta.get("request_id"))
    if not account_id_text or not request_id:
        logger.error(
            "Deflection report payment revocation missing account_id/request_id: "
            "event_type=%s object=%s",
            event_type,
            _stripe_text(obj, "id") or "<missing>",
        )
        return
    try:
        _uuid.UUID(account_id_text)
    except ValueError:
        logger.error(
            "Deflection report payment revocation has invalid account_id: "
            "event_type=%s account=%s object=%s",
            event_type,
            account_id_text,
            _stripe_text(obj, "id") or "<missing>",
        )
        return

    store = PostgresDeflectionReportArtifactStore(pool=pool)
    revoked = await store.mark_unpaid(
        account_id=account_id_text,
        request_id=request_id,
        payment_reference=payment_reference,
    )
    if not revoked:
        await emit_deflection_paid_funnel_incident_alert(
            logger,
            incident_type="paid_report_revocation_missed_report",
            severity="error",
            account_id=account_id_text,
            request_id=request_id,
            event_type=event_type,
            payment_reference=payment_reference or "",
            stripe_object_id=_stripe_text(obj, "id") or "<missing>",
        )
        logger.error(
            "Deflection report payment revocation missed report: "
            "event_type=%s account=%s request=%s payment_reference=%s object=%s",
            event_type,
            account_id_text,
            request_id,
            payment_reference or "<missing>",
            _stripe_text(obj, "id") or "<missing>",
        )
        return

    await _cancel_content_ops_deflection_report_delivery(
        pool,
        account_id=account_id_text,
        request_id=request_id,
        event_type=event_type,
    )
    logger.warning(
        "Deflection report access revoked after Stripe payment reversal: "
        "event_type=%s account=%s request=%s payment_reference=%s object=%s",
        event_type,
        account_id_text,
        request_id,
        payment_reference or "<missing>",
        _stripe_text(obj, "id") or "<missing>",
    )


async def _cancel_content_ops_deflection_report_delivery(
    pool: Any,
    *,
    account_id: str,
    request_id: str,
    event_type: str,
) -> None:
    await pool.execute(
        """
        UPDATE content_ops_deflection_report_deliveries
        SET delivery_status = 'revoked',
            delivery_error = $3,
            updated_at = NOW()
        WHERE account_id = $1
          AND request_id = $2
          AND delivery_status IN ('pending', 'sending')
        """,
        account_id,
        request_id,
        f"payment_revoked:{event_type}",
    )


async def _handle_content_ops_deflection_report_dispute_closed(
    pool: Any,
    stripe_module: Any,
    obj: Any,
    *,
    event_type: str,
) -> None:
    """Restore a deflection report after Stripe closes a dispute as won."""

    dispute_status = _stripe_text(obj, "status").lower()
    if dispute_status != "won":
        logger.info(
            "Deflection report dispute closed without restore: "
            "event_type=%s object=%s status=%s",
            event_type,
            _stripe_text(obj, "id") or "<missing>",
            dispute_status or "<missing>",
        )
        return

    meta, payment_reference = _content_ops_deflection_revocation_metadata(
        stripe_module,
        obj,
    )
    if meta.get("source") != "content_ops_deflection_report":
        logger.info(
            "Deflection report dispute restore could not be mapped: "
            "event_type=%s object=%s payment_intent=%s",
            event_type,
            _stripe_text(obj, "id") or "<missing>",
            _payment_intent_from_payment_event(obj) or "<missing>",
        )
        return

    account_id_text = _clean_metadata(meta.get("account_id"))
    request_id = _clean_metadata(meta.get("request_id"))
    if not account_id_text or not request_id:
        logger.error(
            "Deflection report dispute restore missing account_id/request_id: "
            "event_type=%s object=%s",
            event_type,
            _stripe_text(obj, "id") or "<missing>",
        )
        return
    try:
        _uuid.UUID(account_id_text)
    except ValueError:
        logger.error(
            "Deflection report dispute restore has invalid account_id: "
            "event_type=%s account=%s object=%s",
            event_type,
            account_id_text,
            _stripe_text(obj, "id") or "<missing>",
        )
        return

    store = PostgresDeflectionReportArtifactStore(pool=pool)
    record = await store.get_artifact_record(
        account_id=account_id_text,
        request_id=request_id,
    )
    if record is None:
        await emit_deflection_paid_funnel_incident_alert(
            logger,
            incident_type="paid_report_restore_missed_report",
            severity="error",
            account_id=account_id_text,
            request_id=request_id,
            event_type=event_type,
            payment_reference=payment_reference or "",
            stripe_object_id=_stripe_text(obj, "id") or "<missing>",
        )
        logger.error(
            "Deflection report dispute restore missed report: "
            "event_type=%s account=%s request=%s payment_reference=%s object=%s",
            event_type,
            account_id_text,
            request_id,
            payment_reference or "<missing>",
            _stripe_text(obj, "id") or "<missing>",
        )
        return

    restore_reference = payment_reference
    if (
        record.payment_reference
        and payment_reference
        and record.payment_reference != payment_reference
    ):
        restore_reference = None
        logger.warning(
            "Deflection report dispute restore preserved newer payment reference: "
            "event_type=%s account=%s request=%s restored_reference=%s existing_reference=%s object=%s",
            event_type,
            account_id_text,
            request_id,
            payment_reference,
            record.payment_reference,
            _stripe_text(obj, "id") or "<missing>",
        )

    restored = await store.mark_paid(
        account_id=account_id_text,
        request_id=request_id,
        payment_reference=restore_reference,
    )
    if not restored:
        await emit_deflection_paid_funnel_incident_alert(
            logger,
            incident_type="paid_report_restore_missed_report",
            severity="error",
            account_id=account_id_text,
            request_id=request_id,
            event_type=event_type,
            payment_reference=payment_reference or "",
            stripe_object_id=_stripe_text(obj, "id") or "<missing>",
        )
        logger.error(
            "Deflection report dispute restore missed report: "
            "event_type=%s account=%s request=%s payment_reference=%s object=%s",
            event_type,
            account_id_text,
            request_id,
            payment_reference or "<missing>",
            _stripe_text(obj, "id") or "<missing>",
        )
        return

    delivery_result = await _queue_content_ops_deflection_report_delivery(
        pool,
        account_id=account_id_text,
        request_id=request_id,
        payment_reference=restore_reference,
    )
    delta_delivery_result = await _requeue_content_ops_deflection_delta_deliveries(
        pool,
        account_id=account_id_text,
        request_id=request_id,
    )
    logger.warning(
        "Deflection report access restored after Stripe dispute win: "
        "event_type=%s account=%s request=%s payment_reference=%s object=%s delivery=%s delta_delivery=%s",
        event_type,
        account_id_text,
        request_id,
        payment_reference or "<missing>",
        _stripe_text(obj, "id") or "<missing>",
        delivery_result,
        delta_delivery_result,
    )


async def _requeue_content_ops_deflection_delta_deliveries(
    pool: Any,
    *,
    account_id: str,
    request_id: str,
) -> str:
    result = await pool.execute(
        """
        UPDATE content_ops_deflection_delta_deliveries
        SET delivery_status = 'pending',
            delivery_error = NULL,
            updated_at = NOW()
        WHERE account_id = $1
          AND (current_request_id = $2 OR baseline_request_id = $2)
          AND (
                delivery_status IN ('pending', 'sending')
             OR (
                    delivery_status = 'failed'
                    AND delivery_error IN (
                        'source_report_not_paid',
                        'delta_no_longer_sendable'
                    )
                )
          )
        """,
        account_id,
        request_id,
    )
    return str(result)


def _content_ops_deflection_revocation_metadata(
    stripe_module: Any,
    obj: Any,
) -> tuple[Mapping[str, Any], str | None]:
    payment_intent = _payment_intent_from_payment_event(obj)
    session = _checkout_session_for_payment_intent(stripe_module, payment_intent)
    if session is not None:
        meta = _stripe_metadata(session)
        if meta.get("source") == "content_ops_deflection_report":
            return meta, _stripe_text(session, "id") or None

    meta = _stripe_metadata(obj)
    if meta.get("source") == "content_ops_deflection_report":
        return meta, None
    return {}, None


def _checkout_session_for_payment_intent(
    stripe_module: Any,
    payment_intent: str,
) -> Any | None:
    if not payment_intent:
        return None
    try:
        sessions = stripe_module.checkout.Session.list(
            payment_intent=payment_intent,
            limit=1,
            timeout=10,
        )
    except Exception:
        logger.exception(
            "Deflection report payment revocation checkout lookup failed: payment_intent=%s",
            payment_intent,
        )
        raise HTTPException(
            status_code=503,
            detail="Deflection report payment revocation lookup failed",
        )
    data = _stripe_object_value(sessions, "data")
    if isinstance(data, Sequence) and data:
        return data[0]
    return None


def _stripe_charge_refund_is_full(obj: Any) -> bool:
    if _stripe_object_value(obj, "refunded") is True:
        return True
    amount_refunded = _stripe_int(obj, "amount_refunded")
    amount_captured = _stripe_int(obj, "amount_captured") or _stripe_int(obj, "amount")
    return (
        amount_refunded is not None
        and amount_captured is not None
        and amount_captured > 0
        and amount_refunded >= amount_captured
    )


def _stripe_int(obj: Any, key: str) -> int | None:
    value = _stripe_object_value(obj, key)
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stripe_event_created(event: Any) -> int | None:
    value = getattr(event, "created", None)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _payment_intent_from_payment_event(obj: Any) -> str:
    payment_intent = _stripe_text(obj, "payment_intent")
    if payment_intent:
        return payment_intent
    charge = _stripe_object_value(obj, "charge")
    if charge is not None and not isinstance(charge, str):
        return _stripe_text(charge, "payment_intent")
    return ""


def _stripe_metadata(obj: Any) -> Mapping[str, Any]:
    metadata = _stripe_object_value(obj, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _configured_deflection_delta_price_ids() -> tuple[str, ...]:
    raw = str(
        getattr(
            settings.saas_auth,
            "stripe_content_ops_deflection_delta_price_ids",
            "",
        )
        or ""
    )
    seen: set[str] = set()
    price_ids: list[str] = []
    for item in raw.split(","):
        price_id = item.strip()
        if not price_id or price_id in seen:
            continue
        seen.add(price_id)
        price_ids.append(price_id)
    return tuple(price_ids)


def _stripe_price_ids_from_object(obj: Any) -> tuple[str, ...]:
    price_ids: list[str] = []

    def add_price_id(value: Any) -> None:
        price_id = _stripe_text(value, "id") if not isinstance(value, str) else value
        if price_id:
            price_ids.append(price_id)

    direct_price = _stripe_object_value(obj, "price")
    if direct_price is not None:
        add_price_id(direct_price)

    items = _stripe_object_value(obj, "items")
    data = _stripe_object_value(items, "data")
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        for item in data:
            add_price_id(_stripe_object_value(item, "price"))

    lines = _stripe_object_value(obj, "lines")
    line_data = _stripe_object_value(lines, "data")
    if isinstance(line_data, Sequence) and not isinstance(line_data, (str, bytes, bytearray)):
        for line in line_data:
            add_price_id(_stripe_object_value(line, "price"))
            pricing = _stripe_object_value(line, "pricing")
            price_details = _stripe_object_value(pricing, "price_details")
            add_price_id(_stripe_object_value(price_details, "price"))

    return tuple(dict.fromkeys(price_ids))


def _deflection_delta_price_id_from_object(obj: Any) -> str | None:
    configured = set(_configured_deflection_delta_price_ids())
    if not configured:
        return None
    for price_id in _stripe_price_ids_from_object(obj):
        if price_id in configured:
            return price_id
    return None


def _is_deflection_delta_subscription_source(obj: Any) -> bool:
    return (
        _clean_metadata(_stripe_metadata(obj).get("source"))
        == DEFLECTION_DELTA_SUBSCRIPTION_SOURCE
    )


def _stripe_subscription_id(obj: Any) -> str:
    subscription_id = _stripe_text(obj, "id")
    object_type = _stripe_text(obj, "object")
    if subscription_id.startswith("sub_") or object_type == "subscription":
        return subscription_id
    legacy_subscription_id = _stripe_text(obj, "subscription")
    if legacy_subscription_id:
        return legacy_subscription_id
    parent = _stripe_object_value(obj, "parent")
    subscription_details = _stripe_object_value(parent, "subscription_details")
    return _stripe_text(subscription_details, "subscription")


def _stripe_current_period_end(obj: Any) -> datetime | None:
    period_end = _stripe_int(obj, "current_period_end")
    if period_end is None or period_end <= 0 or period_end > 4_102_444_800:
        return None
    return datetime.fromtimestamp(period_end, timezone.utc)


async def _account_id_for_stripe_object(pool: Any, obj: Any) -> _uuid.UUID | None:
    meta = _stripe_metadata(obj)
    account_id_str = _clean_metadata(meta.get("account_id"))
    if account_id_str:
        if not _UUID_TEXT_RE.fullmatch(account_id_str):
            return None
        return _uuid.UUID(account_id_str)
    customer_id = _stripe_text(obj, "customer")
    if not customer_id:
        return None
    return await _find_account_by_customer(pool, customer_id)


async def _handle_deflection_delta_subscription_lifecycle(
    pool: Any,
    subscription: Any,
    *,
    stripe_subscription_status: str | None = None,
    stripe_event_created: int | None = None,
    store: DeflectionReportArtifactStore | None = None,
) -> _uuid.UUID | None:
    price_id = _deflection_delta_price_id_from_object(subscription)
    if price_id is None:
        return None
    subscription_id = _stripe_subscription_id(subscription)
    if not subscription_id:
        logger.warning("Deflection delta subscription event missing subscription id")
        return None
    account_id = await _account_id_for_stripe_object(pool, subscription)
    if account_id is None:
        logger.warning(
            "Deflection delta subscription event missing account mapping: customer=%s subscription=%s",
            _stripe_text(subscription, "customer") or "<missing>",
            subscription_id,
        )
        return None
    status = stripe_subscription_status or _stripe_text(subscription, "status")
    if not status:
        logger.warning(
            "Deflection delta subscription event missing status: account=%s subscription=%s",
            account_id,
            subscription_id,
        )
        return account_id
    entitlement_store = store or PostgresDeflectionReportArtifactStore(pool)
    await entitlement_store.upsert_deflection_delta_entitlement(
        account_id=str(account_id),
        stripe_subscription_id=subscription_id,
        stripe_customer_id=_stripe_text(subscription, "customer") or None,
        stripe_price_id=price_id,
        stripe_subscription_status=status,
        current_period_end=_stripe_current_period_end(subscription),
        stripe_event_created=stripe_event_created,
        metadata={
            "source": "stripe_billing_webhook",
            "grants_delta_entitlement": status in DELTA_ENTITLEMENT_ACTIVE_STATUSES,
        },
    )
    return account_id


async def _handle_deflection_delta_invoice_lifecycle(
    pool: Any,
    invoice: Any,
    *,
    stripe_subscription_status: str,
    stripe_event_created: int | None = None,
    store: DeflectionReportArtifactStore | None = None,
) -> _uuid.UUID | None:
    price_id = _deflection_delta_price_id_from_object(invoice)
    if price_id is None:
        return None
    subscription_id = _stripe_subscription_id(invoice)
    if not subscription_id:
        logger.warning("Deflection delta invoice event missing subscription id")
        return None
    account_id = await _account_id_for_stripe_object(pool, invoice)
    if account_id is None:
        logger.warning(
            "Deflection delta invoice event missing account mapping: customer=%s subscription=%s",
            _stripe_text(invoice, "customer") or "<missing>",
            subscription_id,
        )
        return None
    entitlement_store = store or PostgresDeflectionReportArtifactStore(pool)
    await entitlement_store.upsert_deflection_delta_entitlement(
        account_id=str(account_id),
        stripe_subscription_id=subscription_id,
        stripe_customer_id=_stripe_text(invoice, "customer") or None,
        stripe_price_id=price_id,
        stripe_subscription_status=stripe_subscription_status,
        current_period_end=_stripe_current_period_end(invoice),
        stripe_event_created=stripe_event_created,
        metadata={
            "source": "stripe_billing_invoice_webhook",
            "grants_delta_entitlement": stripe_subscription_status
            in DELTA_ENTITLEMENT_ACTIVE_STATUSES,
        },
    )
    return account_id


def _log_content_ops_deflection_report_payment_pending(
    session: Any,
    meta: Mapping[str, Any],
) -> None:
    logger.warning(
        "Deflection report checkout event arrived before funds were available: account=%s request=%s session=%s payment_status=%s",
        _clean_metadata(meta.get("account_id")) or "<missing>",
        _clean_metadata(meta.get("request_id")) or "<missing>",
        _stripe_text(session, "id") or "<missing>",
        _stripe_text(session, "payment_status") or "<missing>",
    )


def _log_content_ops_deflection_report_async_payment_failed(
    session: Any,
    meta: Mapping[str, Any],
) -> None:
    logger.warning(
        "Deflection report async payment failed: account=%s request=%s session=%s payment_status=%s",
        _clean_metadata(meta.get("account_id")) or "<missing>",
        _clean_metadata(meta.get("request_id")) or "<missing>",
        _stripe_text(session, "id") or "<missing>",
        _stripe_text(session, "payment_status") or "<missing>",
    )


def _deflection_checkout_amount_is_valid(session: Any) -> bool:
    return _deflection_checkout_paid_terms(session) is not None


def _deflection_checkout_paid_terms(
    session: Any,
) -> tuple[int, str, tuple[int, ...]] | None:
    cfg = settings.saas_auth
    allowed_cents = _deflection_allowed_checkout_amounts_cents(cfg)
    expected_currency = str(
        getattr(cfg, "stripe_content_ops_deflection_report_currency", "usd") or ""
    ).strip().lower()
    amount_total = _stripe_object_value(session, "amount_total")
    currency = _stripe_text(session, "currency").lower()
    if not allowed_cents:
        logger.error("Deflection report checkout amount gate is misconfigured")
        return None
    if not expected_currency:
        logger.error("Deflection report checkout currency gate is misconfigured")
        return None
    if currency != expected_currency:
        return None
    try:
        actual_cents = int(amount_total)
    except (TypeError, ValueError):
        return None
    if actual_cents not in allowed_cents:
        return None
    return actual_cents, currency, allowed_cents


async def _emit_deflection_checkout_authorized_terms_mismatch(
    record: DeflectionReportAccessRecord,
    *,
    event_type: str,
    session_id: str,
    paid_amount_cents: int,
    paid_currency: str,
) -> None:
    await emit_deflection_paid_funnel_incident_alert(
        logger,
        incident_type="paid_report_checkout_terms_mismatch",
        severity="error",
        account_id=record.account_id,
        request_id=record.request_id,
        event_type=event_type,
        stripe_session_id=session_id,
        amount_total=paid_amount_cents,
        currency=paid_currency,
        expected_amount_cents=record.checkout_amount_cents,
        expected_currency=record.checkout_currency,
        expected_price_variant=record.checkout_price_variant,
    )
    logger.warning(
        "Deflection report checkout ignored: authorized terms mismatch "
        "account=%s request=%s session=%s paid_amount=%s paid_currency=%s "
        "expected_amount=%s expected_currency=%s expected_variant=%s",
        record.account_id,
        record.request_id,
        session_id,
        paid_amount_cents,
        paid_currency,
        record.checkout_amount_cents,
        record.checkout_currency,
        record.checkout_price_variant,
    )


def _deflection_allowed_checkout_amounts_cents(cfg: Any) -> tuple[int, ...]:
    configured = str(
        getattr(
            cfg,
            "stripe_content_ops_deflection_report_allowed_amount_cents",
            "",
        )
        or ""
    ).strip()
    if configured:
        amounts: list[int] = []
        for raw_part in configured.split(","):
            part = raw_part.strip()
            if not part:
                logger.error(
                    "Deflection report checkout allowed amount gate has an empty entry"
                )
                return ()
            try:
                amount_cents = int(part)
            except ValueError:
                logger.error(
                    "Deflection report checkout allowed amount gate has an invalid entry"
                )
                return ()
            if amount_cents <= 0:
                logger.error(
                    "Deflection report checkout allowed amount gate has a non-positive entry"
                )
                return ()
            amounts.append(amount_cents)
        return tuple(dict.fromkeys(amounts))

    expected_cents = int(
        getattr(cfg, "stripe_content_ops_deflection_report_amount_cents", 150000)
        or 0
    )
    if expected_cents <= 0:
        logger.error(
            "Deflection report checkout price gate misconfigured: expected_cents=%s",
            expected_cents,
        )
        return ()
    return (expected_cents,)


def _stripe_text(obj: Any, key: str) -> str:
    return str(_stripe_object_value(obj, key) or "").strip()


def _stripe_object_value(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _clean_metadata(value: Any) -> str:
    return str(value or "").strip()


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


async def _handle_subscription_updated(
    pool,
    subscription,
    *,
    stripe_event_created: int | None = None,
) -> _uuid.UUID | None:
    """Handle subscription changes (upgrades/downgrades)."""
    if _deflection_delta_price_id_from_object(subscription) is not None:
        return await _handle_deflection_delta_subscription_lifecycle(
            pool,
            subscription,
            stripe_event_created=stripe_event_created,
        )

    customer_id = subscription.customer
    account_id = await _find_account_by_customer(pool, customer_id)
    if not account_id:
        return None

    plan = "starter"
    if subscription.items and subscription.items.data:
        price_id = subscription.items.data[0].price.id
        plan = PRICE_TO_PLAN.get(price_id, "starter")

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

    is_b2b = plan.startswith("b2b_")
    if is_b2b:
        b2b_limits = B2B_PLAN_LIMITS.get(plan, B2B_PLAN_LIMITS["b2b_starter"])
        await pool.execute(
            """
            UPDATE saas_accounts
            SET plan = $1, plan_status = $2, vendor_limit = $3,
                stripe_subscription_id = $4, updated_at = NOW()
            WHERE id = $5
            """,
            plan,
            plan_status,
            b2b_limits["vendors"],
            subscription.id,
            account_id,
        )
    else:
        limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["starter"])
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


async def _handle_subscription_deleted(
    pool,
    subscription,
    *,
    stripe_event_created: int | None = None,
) -> _uuid.UUID | None:
    """Handle subscription cancellation."""
    if _deflection_delta_price_id_from_object(subscription) is not None:
        return await _handle_deflection_delta_subscription_lifecycle(
            pool,
            subscription,
            stripe_subscription_status="canceled",
            stripe_event_created=stripe_event_created,
        )

    customer_id = subscription.customer
    account_id = await _find_account_by_customer(pool, customer_id)
    if account_id:
        await pool.execute(
            "UPDATE saas_accounts SET plan_status = 'canceled', updated_at = NOW() WHERE id = $1",
            account_id,
        )
        logger.info("Account %s subscription canceled", account_id)
    return account_id
