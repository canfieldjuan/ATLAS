"""Receivables endpoints for the slim EOM API profile."""

from __future__ import annotations

import errno
import socket
from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Literal, Optional
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..services.receivables import (
    ReceivablesConflictError,
    ReceivablesError,
    ReceivablesNotFoundError,
    ReceivablesSchemaUnavailableError,
    ReceivablesService,
    ReceivablesValidationError,
    get_receivables_service,
)
from ..services.eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID
from ..storage.exceptions import DatabaseUnavailableError
from .auth import require_actor, require_receivables_api
from .config import funnel_settings
from .funnel_database import get_eom_funnel_crm_provider, get_eom_funnel_db_pool

_DATABASE_UNAVAILABLE_ERRORS = (
    DatabaseUnavailableError,
    asyncpg.PostgresConnectionError,
    asyncpg.CannotConnectNowError,
    asyncpg.TooManyConnectionsError,
    asyncpg.AdminShutdownError,
    asyncpg.CrashShutdownError,
)
_UNAVAILABLE_INTERFACE_PREFIXES = (
    "connection is closed",
    "pool is closed",
    "pool is closing",
    "pool is not initialized",
    "pool is being initialized, but not yet ready",
)
_UNAVAILABLE_INTERFACE_SUFFIXES = (
    ": the underlying connection is closed",
)
_NETWORK_UNAVAILABLE_ERRNOS = frozenset(
    {
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTDOWN,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETRESET,
        errno.ENETUNREACH,
        errno.ENOTCONN,
        errno.EPIPE,
        errno.ETIMEDOUT,
    }
)

router = APIRouter(
    prefix="/receivables",
    tags=["receivables"],
    dependencies=[Depends(require_receivables_api)],
)

PositiveCents = Annotated[int, Field(strict=True, gt=0)]


class AllocationRequest(BaseModel):
    invoice_id: UUID
    amount_cents: PositiveCents


class CreatePaymentRequest(BaseModel):
    contact_id: UUID
    payer_name: str = Field(min_length=1, max_length=256)
    total_amount_cents: PositiveCents
    payment_method: Literal["check", "ach", "square"]
    received_date: date
    check_date: Optional[date] = None
    received_through: Optional[str] = Field(default=None, max_length=128)
    reference: Optional[str] = Field(default=None, max_length=256)
    notes: Optional[str] = None
    allocations: list[AllocationRequest] = Field(default_factory=list, max_length=100)


class AdjustAllocationsRequest(BaseModel):
    allocations: list[AllocationRequest] = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=1000)


class PaymentActionRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=1000)


class CreateDepositBatchRequest(BaseModel):
    payment_ids: list[UUID] = Field(min_length=1, max_length=500)
    deposit_date: date
    bank_reference: Optional[str] = Field(default=None, max_length=256)


def _dollars(amount_cents: int) -> Decimal:
    return Decimal(amount_cents) / Decimal(100)


def _allocations(items: list[AllocationRequest]) -> list[dict]:
    return [
        {"invoice_id": item.invoice_id, "amount": _dollars(item.amount_cents)}
        for item in items
    ]


def _is_database_unavailable_error(exc: Exception) -> bool:
    if isinstance(exc, _DATABASE_UNAVAILABLE_ERRORS):
        return True
    if isinstance(exc, asyncpg.InterfaceError):
        message = str(exc).casefold()
        return (
            any(
                message.startswith(prefix)
                for prefix in _UNAVAILABLE_INTERFACE_PREFIXES
            )
            or any(
                message.endswith(suffix)
                for suffix in _UNAVAILABLE_INTERFACE_SUFFIXES
            )
        )
    if isinstance(exc, (ConnectionError, TimeoutError, socket.gaierror)):
        return True
    return isinstance(exc, OSError) and exc.errno in _NETWORK_UNAVAILABLE_ERRNOS


async def _call(awaitable):
    try:
        return await awaitable
    except ReceivablesSchemaUnavailableError as exc:
        raise HTTPException(
            status_code=503, detail={"code": exc.code, "message": str(exc)}
        ) from exc
    except ReceivablesValidationError as exc:
        raise HTTPException(
            status_code=422, detail={"code": exc.code, "message": str(exc)}
        ) from exc
    except ReceivablesNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail={"code": exc.code, "message": str(exc)}
        ) from exc
    except ReceivablesConflictError as exc:
        raise HTTPException(
            status_code=409, detail={"code": exc.code, "message": str(exc)}
        ) from exc
    except ReceivablesError as exc:
        raise HTTPException(
            status_code=400, detail={"code": exc.code, "message": str(exc)}
        ) from exc
    except Exception as exc:
        if not _is_database_unavailable_error(exc):
            raise
        message = (
            str(exc)
            if isinstance(exc, DatabaseUnavailableError)
            else "Receivables database unavailable"
        )
        raise HTTPException(
            status_code=503,
            detail={"code": "database_unavailable", "message": message},
        ) from exc


@router.get("/ready")
async def ready() -> dict:
    service = get_receivables_service()
    try:
        schema_ready = await service.is_ready()
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Receivables database unavailable"
        ) from exc
    if not schema_ready:
        raise HTTPException(status_code=503, detail="Receivables schema unavailable")
    # The billing-recipient routes read a SECOND database -- the pool that owns
    # canonical EOM contacts -- so readiness off the global receivables
    # database alone is not the whole answer.
    #
    # Two different states, deliberately not collapsed:
    #
    #   unconfigured -- no funnel DSN. This is a SUPPORTED deployment: a
    #     profile running receivables without billing recipients. Failing
    #     readiness here would take the whole invoicing service out over a
    #     capability it does not use, which is worse than the gap. Reported,
    #     not fatal.
    #
    #   unavailable -- a DSN IS configured and the pool cannot serve the two
    #     queries: the pool never came up, or the database is reachable but
    #     partially migrated. That is a real misconfiguration, and an
    #     initialized pool alone does not rule it out -- it proves a
    #     connection opened, not that `contacts` has the columns both queries
    #     name. This one fails readiness.
    pool = get_eom_funnel_db_pool()
    if not funnel_settings.db_connection_string.strip():
        return {"status": "ready", "billingRecipients": "unconfigured"}
    schema_ready = False
    if pool.is_initialized:
        try:
            schema_ready = await get_eom_funnel_crm_provider(
            ).billing_recipients_schema_ready()
        except Exception:
            schema_ready = False
    if not schema_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "billing_recipients_unavailable",
                "message": (
                    "A canonical EOM contact database is configured but cannot "
                    "serve billing-recipient reads."
                ),
            },
        )
    return {"status": "ready", "billingRecipients": "ready"}


@router.get("/open-invoices")
async def list_open_invoices(
    contact_id: Optional[UUID] = None,
    search: Optional[str] = Query(default=None, max_length=256),
) -> list[dict]:
    return await _call(
        get_receivables_service().list_open_invoices(
            contact_id=contact_id, search=search
        )
    )


def _billing_crm_dependency(request: Request) -> Any:
    """Resolve the CRM provider pinned to the canonical EOM contact pool.

    Contacts are owned by the dedicated funnel CRM pool. ReceivablesService
    resolves the separate global DSN, which in the deployed slim topology is a
    different database entirely -- reading contacts through it would query the
    wrong one. The credential boundary and the data source are independent:
    these routes stay behind the receivables token while reading contacts from
    the pool that actually owns them.
    """
    # Availability FIRST, before the app-state factory. The real app installs
    # that factory unconditionally (atlas_brain/main_eom.py), so checking it
    # after the factory branch put this guard on a path production never
    # takes: the fetch would raise an untranslated RuntimeError and answer 500
    # instead of failing closed. Startup deliberately does NOT raise when
    # receivables runs without a funnel DSN -- that would stop a profile which
    # never touches billing recipients from booting -- so the cost is borne
    # here, by the routes that actually need the pool. Falling through is worse
    # than a 503: DatabaseCRMProvider._get_pool() defaults to the global pool,
    # which is the wrong database and would answer with silence, not an error.
    if not get_eom_funnel_db_pool().is_initialized:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "billing_recipients_unavailable",
                "message": (
                    "The canonical EOM contact pool is not configured; set "
                    "ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING to use billing "
                    "recipients."
                ),
            },
        )
    provider_factory = getattr(request.app.state, "eom_funnel_crm_provider", None)
    if callable(provider_factory):
        return provider_factory()
    return get_eom_funnel_crm_provider()


@router.get("/billing-recipients")
async def list_billing_recipients(
    request: Request,
    search: Optional[str] = Query(default=None, max_length=256),
    limit: int = Query(default=200, ge=1, le=500),
    crm: Any = Depends(_billing_crm_dependency),
) -> list[dict]:
    """EOM contacts assignable as an invoice recipient. Eligible only.

    Behind the receivables credential rather than the EOM funnel one: this is a
    billing capability, and the funnel token is broad. /eom-funnel/known-contacts
    is deliberately NOT extended for it -- that route's value is that it
    discloses nothing beyond whether an id resolves, and widening it a second
    time would turn link verification into a general contact reader.
    """
    return await _call(crm.list_billing_recipients(search=search, limit=limit))


@router.get("/billing-recipients/{contact_id}")
async def get_billing_recipient(
    contact_id: UUID,
    request: Request,
    crm: Any = Depends(_billing_crm_dependency),
) -> dict:
    """Authoritative answer on whether ONE contact may receive invoices.

    Always 200 with an explicit verdict, never 404: "this contact is not an
    eligible recipient, because X" is a domain answer the caller must handle,
    not a transport failure. An ineligible verdict carries identity and cause
    only -- never a name or address.
    """
    return await _call(crm.get_billing_recipient(contact_id))


@router.get("/allocation-suggestions")
async def allocation_suggestions(
    contact_id: UUID,
    total_amount_cents: int = Query(gt=0),
) -> list[dict]:
    return await _call(
        get_receivables_service().suggest_allocations(
            contact_id=contact_id,
            total_amount=_dollars(total_amount_cents),
        )
    )


@router.post("/payments", status_code=201)
async def create_payment(
    body: CreatePaymentRequest,
    actor: Annotated[str, Depends(require_actor)],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=1, max_length=128)
    ],
    service: ReceivablesService = Depends(get_receivables_service),
) -> dict:
    return await _call(
        service.create_payment(
            contact_id=body.contact_id,
            payer_name=body.payer_name,
            total_amount=_dollars(body.total_amount_cents),
            payment_method=body.payment_method,
            received_date=body.received_date,
            check_date=body.check_date,
            received_through=body.received_through,
            reference=body.reference,
            notes=body.notes,
            allocations=_allocations(body.allocations),
            recorded_by=actor,
            idempotency_key=idempotency_key,
            allow_unapplied=True,
            unapplied_contact_context_id=EOM_BUSINESS_CONTEXT_ID,
        )
    )


@router.get("/payments")
async def list_payments(
    status: Optional[str] = Query(default=None, max_length=16),
    search: Optional[str] = Query(default=None, max_length=256),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    return await _call(
        get_receivables_service().list_payments(
            status=status, search=search, limit=limit, offset=offset
        )
    )


@router.put("/payments/{payment_id}/allocations")
async def adjust_allocations(
    payment_id: UUID,
    body: AdjustAllocationsRequest,
    actor: Annotated[str, Depends(require_actor)],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=1, max_length=128)
    ],
) -> dict:
    return await _call(
        get_receivables_service().adjust_allocations(
            payment_id=payment_id,
            allocations=_allocations(body.allocations),
            reason=body.reason,
            actor=actor,
            idempotency_key=idempotency_key,
        )
    )


@router.post("/payments/{payment_id}/return")
async def return_payment(
    payment_id: UUID,
    body: PaymentActionRequest,
    actor: Annotated[str, Depends(require_actor)],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=1, max_length=128)
    ],
) -> dict:
    return await _call(
        get_receivables_service().return_payment(
            payment_id=payment_id,
            reason=body.reason,
            actor=actor,
            idempotency_key=idempotency_key,
        )
    )


@router.post("/payments/{payment_id}/void")
async def void_payment(
    payment_id: UUID,
    body: PaymentActionRequest,
    actor: Annotated[str, Depends(require_actor)],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=1, max_length=128)
    ],
) -> dict:
    return await _call(
        get_receivables_service().void_payment(
            payment_id=payment_id,
            reason=body.reason,
            actor=actor,
            idempotency_key=idempotency_key,
        )
    )


@router.get("/deposit-batches")
async def list_deposit_batches(
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
    return await _call(get_receivables_service().list_deposit_batches(limit=limit))


@router.post("/deposit-batches", status_code=201)
async def create_deposit_batch(
    body: CreateDepositBatchRequest,
    actor: Annotated[str, Depends(require_actor)],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=1, max_length=128)
    ],
) -> dict:
    return await _call(
        get_receivables_service().create_deposit_batch(
            payment_ids=body.payment_ids,
            deposit_date=body.deposit_date,
            bank_reference=body.bank_reference,
            actor=actor,
            idempotency_key=idempotency_key,
        )
    )


@router.post("/deposit-batches/{batch_id}/clear")
async def clear_deposit_batch(
    batch_id: UUID,
    actor: Annotated[str, Depends(require_actor)],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=1, max_length=128)
    ],
) -> dict:
    return await _call(
        get_receivables_service().clear_deposit_batch(
            batch_id=batch_id,
            actor=actor,
            idempotency_key=idempotency_key,
        )
    )
