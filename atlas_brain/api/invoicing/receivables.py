"""Strict service-to-service HTTP surface for the EOM receivables portal."""

from __future__ import annotations

import errno
import socket
from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Literal, Optional
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from ...services.receivables import (
    ReceivablesConflictError,
    ReceivablesError,
    ReceivablesNotFoundError,
    ReceivablesReceiptContextRequiredError,
    ReceivablesSchemaUnavailableError,
    ReceivablesService,
    ReceivablesValidationError,
    PaymentReceiptRecipient,
    get_receivables_service,
)
from ...services.crm_provider import EOMBillingDeliveryMethod, get_crm_provider
from ...services.commercial_billing_candidates import (
    CommercialBillingCandidateService,
    CommercialBillingCandidatesUnavailableError,
    CommercialBillingCandidatesValidationError,
    get_commercial_billing_candidate_service,
)
from ...services.commercial_billing_runs import (
    CommercialBillingRunConflictError,
    CommercialBillingRunNotFoundError,
    CommercialBillingRunService,
    CommercialBillingRunUnavailableError,
    CommercialBillingRunValidationError,
    get_commercial_billing_run_service,
)
from ...services.commercial_billing_approvals import (
    CommercialBillingApprovalConflictError,
    CommercialBillingApprovalNotFoundError,
    CommercialBillingApprovalService,
    CommercialBillingApprovalStaleError,
    CommercialBillingApprovalUnavailableError,
    CommercialBillingApprovalValidationError,
    get_commercial_billing_approval_service,
)
from ...services.commercial_billing_invoice_pdfs import (
    CommercialBillingInvoicePDFConflictError,
    CommercialBillingInvoicePDFNotFoundError,
    CommercialBillingInvoicePDFRenderError,
    CommercialBillingInvoicePDFService,
    CommercialBillingInvoicePDFUnavailableError,
    CommercialBillingInvoicePDFValidationError,
    get_commercial_billing_invoice_pdf_service,
)
from ...services.commercial_billing_invoice_gmail_drafts import (
    CommercialBillingGmailDraftConflictError,
    CommercialBillingGmailDraftNotFoundError,
    CommercialBillingGmailDraftRecoveryRequiredError,
    CommercialBillingInvoiceGmailDraftService,
    CommercialBillingGmailDraftUnavailableError,
    CommercialBillingGmailDraftValidationError,
    get_commercial_billing_invoice_gmail_draft_service,
)
from ...services.commercial_billing_invoice_gmail_sent_reconciliation import (
    MAX_DELIVERY_STATE_OFFSET,
    CommercialBillingGmailDeliveryStateNotFoundError,
    CommercialBillingGmailSentReconciliationConflictError,
    CommercialBillingGmailSentReconciliationNotFoundError,
    CommercialBillingGmailSentReconciliationUnavailableError,
    CommercialBillingGmailSentReconciliationValidationError,
    CommercialBillingInvoiceGmailSentReconciliationService,
    get_commercial_billing_invoice_gmail_sent_reconciliation_service,
)
from ...services.commercial_billing_manual_square_invoices import (
    CommercialBillingManualSquareInvoiceConflictError,
    CommercialBillingManualSquareInvoiceNotFoundError,
    CommercialBillingManualSquareInvoiceService,
    CommercialBillingManualSquareInvoiceUnavailableError,
    CommercialBillingManualSquareInvoiceValidationError,
    get_commercial_billing_manual_square_invoice_service,
)
from ...services.eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID
from ...storage.exceptions import DatabaseUnavailableError
from .auth import require_actor, require_receivables_api

_DATABASE_UNAVAILABLE_ERRORS = (
    DatabaseUnavailableError,
    asyncpg.PostgresConnectionError,
    asyncpg.CannotConnectNowError,
    asyncpg.TooManyConnectionsError,
    asyncpg.AdminShutdownError,
    asyncpg.CrashShutdownError,
)
_CANONICAL_CUSTOMER_UNAVAILABLE_ERRORS = _DATABASE_UNAVAILABLE_ERRORS + (
    asyncpg.UndefinedTableError,
    asyncpg.UndefinedColumnError,
    asyncpg.InvalidSchemaNameError,
    asyncpg.InsufficientPrivilegeError,
    asyncpg.InvalidAuthorizationSpecificationError,
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


class CreateCommercialBillingRunRequest(BaseModel):
    billing_period: str = Field(
        min_length=7,
        max_length=7,
        pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
    )


class ApproveCommercialBillingCandidateRequest(BaseModel):
    candidate_key: str = Field(min_length=1, max_length=512)
    expected_source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )


class SetCommercialBillingDeliveryPreferenceRequest(BaseModel):
    delivery_method: EOMBillingDeliveryMethod


class RecordCommercialBillingManualSquareReferenceRequest(BaseModel):
    square_invoice_reference: str = Field(min_length=1, max_length=256)


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


def _is_canonical_customer_unavailable_error(exc: Exception) -> bool:
    """Classify canonical-CRM schema and connection failures for payment reads.

    The payment route has a recovery path for a same-key replay when the
    canonical customer read is unavailable.  A partially migrated or
    permission-denied canonical CRM must therefore take that same controlled
    path rather than escaping as an HTTP 500 before the ledger can reconcile
    the existing payment.
    """
    return _is_database_unavailable_error(exc) or isinstance(
        exc, _CANONICAL_CUSTOMER_UNAVAILABLE_ERRORS
    )


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


async def _call_commercial_billing_run(awaitable):
    try:
        return await awaitable
    except CommercialBillingCandidatesValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingCandidatesUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingRunValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingRunNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingRunConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingRunUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc


async def _call_commercial_billing_approval(awaitable):
    try:
        return await awaitable
    except CommercialBillingApprovalValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingApprovalNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except (CommercialBillingApprovalConflictError, CommercialBillingApprovalStaleError) as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingApprovalUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc


async def _call_commercial_billing_invoice_pdf(awaitable):
    try:
        return await awaitable
    except CommercialBillingInvoicePDFValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingInvoicePDFNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingInvoicePDFConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except (
        CommercialBillingInvoicePDFRenderError,
        CommercialBillingInvoicePDFUnavailableError,
    ) as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc


async def _call_commercial_billing_gmail_draft(awaitable):
    try:
        return await awaitable
    except CommercialBillingGmailDraftValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingGmailDraftNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except (
        CommercialBillingGmailDraftConflictError,
        CommercialBillingGmailDraftRecoveryRequiredError,
    ) as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingGmailDraftUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except Exception as exc:
        if not _is_database_unavailable_error(exc):
            raise
        raise HTTPException(
            status_code=503,
            detail={
                "code": "commercial_billing_gmail_draft_unavailable",
                "message": "Commercial billing Gmail draft database unavailable",
            },
        ) from exc


async def _call_commercial_billing_gmail_sent_reconciliation(awaitable):
    try:
        return await awaitable
    except CommercialBillingGmailDeliveryStateNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingGmailSentReconciliationValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingGmailSentReconciliationNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingGmailSentReconciliationConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingGmailSentReconciliationUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except Exception as exc:
        if not _is_database_unavailable_error(exc):
            raise
        raise HTTPException(
            status_code=503,
            detail={
                "code": "commercial_billing_gmail_sent_reconciliation_unavailable",
                "message": "Commercial billing Gmail sent reconciliation database unavailable",
            },
        ) from exc


async def _call_commercial_billing_manual_square_invoice(awaitable):
    try:
        return await awaitable
    except CommercialBillingManualSquareInvoiceValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingManualSquareInvoiceNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingManualSquareInvoiceConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingManualSquareInvoiceUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except Exception as exc:
        if not _is_database_unavailable_error(exc):
            raise
        raise HTTPException(
            status_code=503,
            detail={
                "code": "commercial_billing_manual_square_invoice_unavailable",
                "message": "Commercial billing manual Square database unavailable",
            },
        ) from exc


def _commercial_billing_delivery_preference_crm_dependency() -> Any:
    """Expose the canonical CRM provider through an overrideable route seam."""

    return get_crm_provider()


async def _call_commercial_billing_delivery_preference(awaitable) -> dict:
    """Map canonical profile failures without guessing a delivery policy."""

    try:
        result = await awaitable
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_billing_delivery_preference", "message": str(exc)},
        ) from exc
    except Exception as exc:
        if not _is_canonical_customer_unavailable_error(exc):
            raise
        raise HTTPException(
            status_code=503,
            detail={
                "code": "canonical_customer_unavailable",
                "message": "Canonical EOM customer data is unavailable",
            },
        ) from exc
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "not_found", "message": "Customer not found"},
        )
    return result


@router.get("/ready")
async def ready() -> dict:
    service = get_receivables_service()
    try:
        schema_ready = await service.is_receipt_delivery_ready()
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Receivables database unavailable"
        ) from exc
    if not schema_ready:
        raise HTTPException(status_code=503, detail="Receivables schema unavailable")
    return {"status": "ready"}


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
    """Record a payment from the deployed full EOM provider route.

    The full application owns both the receivables ledger and canonical CRM
    contacts through its primary database.  Resolve a narrow active-customer
    snapshot before a new write, but defer a missing/unavailable CRM result to
    the service's idempotency lookup so an unchanged retry can recover a
    previously committed payment.
    """
    receipt_recipient: PaymentReceiptRecipient | None = None
    canonical_failure: HTTPException | None = None
    try:
        customer = await get_crm_provider().get_eom_payment_customer(
            body.contact_id
        )
        if customer is None:
            canonical_failure = HTTPException(
                status_code=404,
                detail={
                    "code": "not_found",
                    "message": "Customer not found",
                },
            )
        else:
            receipt_recipient = PaymentReceiptRecipient(
                contact_id=UUID(str(customer["contact_id"])),
                customer_name=str(customer["customer_name"]),
                customer_type=str(customer["customer_type"]),
                recipient_email=(
                    str(customer["recipient_email"])
                    if customer["recipient_email"] is not None
                    else None
                ),
            )
    except Exception as exc:
        if not _is_canonical_customer_unavailable_error(exc):
            raise
        canonical_failure = HTTPException(
            status_code=503,
            detail={
                "code": "canonical_customer_unavailable",
                "message": "Canonical EOM customer data is unavailable",
            },
        )

    async def _write_payment() -> dict:
        try:
            return await service.create_payment(
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
                receipt_recipient=receipt_recipient,
                require_receipt_recipient=True,
            )
        except ReceivablesReceiptContextRequiredError:
            if canonical_failure is not None:
                raise canonical_failure
            raise

    return await _call(_write_payment())


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


@router.get("/customers/{contact_id}/ledger")
async def customer_ledger(
    contact_id: UUID,
    payment_status: Optional[str] = Query(default=None, max_length=16),
    payment_method: Optional[str] = Query(default=None, max_length=32),
    search: Optional[str] = Query(default=None, max_length=256),
    from_date: Optional[date] = Query(default=None),
    to_date: Optional[date] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: ReceivablesService = Depends(get_receivables_service),
) -> dict:
    """Return one bounded, receipt-aware financial ledger page for a customer."""
    return await _call(
        service.list_customer_ledger(
            contact_id=contact_id,
            payment_status=payment_status,
            payment_method=payment_method,
            search=search,
            from_date=from_date,
            to_date=to_date,
            limit=limit,
            offset=offset,
        )
    )


@router.get("/commercial-billing-candidates")
async def commercial_billing_candidates(
    billing_period: str = Query(
        ...,
        min_length=7,
        max_length=7,
        pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
    ),
    service: CommercialBillingCandidateService = Depends(
        get_commercial_billing_candidate_service
    ),
) -> dict:
    """Return a pure commercial billing preview; approval lives in a later slice."""

    try:
        return await service.preview(billing_period=billing_period)
    except CommercialBillingCandidatesValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except CommercialBillingCandidatesUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc


@router.get("/commercial-billing-delivery-preferences/{contact_id}")
async def get_commercial_billing_delivery_preference(
    contact_id: UUID,
    crm: Annotated[
        Any,
        Depends(_commercial_billing_delivery_preference_crm_dependency),
    ],
) -> dict:
    """Read one explicit canonical delivery policy; absence is not inferred."""

    return await _call_commercial_billing_delivery_preference(
        crm.get_eom_billing_delivery_preference(contact_id)
    )


@router.put("/commercial-billing-delivery-preferences/{contact_id}")
async def set_commercial_billing_delivery_preference(
    contact_id: UUID,
    body: SetCommercialBillingDeliveryPreferenceRequest,
    actor: Annotated[str, Depends(require_actor)],
    crm: Annotated[
        Any,
        Depends(_commercial_billing_delivery_preference_crm_dependency),
    ],
) -> dict:
    """Store one actor-audited policy without creating any delivery effect."""

    return await _call_commercial_billing_delivery_preference(
        crm.set_eom_billing_delivery_preference(
            contact_id=contact_id,
            delivery_method=body.delivery_method.value,
            actor=actor,
        )
    )


@router.post("/commercial-billing-runs", status_code=201)
async def create_commercial_billing_run(
    body: CreateCommercialBillingRunRequest,
    actor: Annotated[str, Depends(require_actor)],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=1, max_length=128)
    ],
    service: CommercialBillingRunService = Depends(get_commercial_billing_run_service),
) -> dict:
    """Persist immutable review evidence; explicit approval remains a later slice."""

    return await _call_commercial_billing_run(
        service.create_run(
            billing_period=body.billing_period,
            idempotency_key=idempotency_key,
            actor=actor,
        )
    )


@router.post("/commercial-billing-runs/{billing_run_id}/approvals", status_code=201)
async def approve_commercial_billing_candidate(
    billing_run_id: UUID,
    body: ApproveCommercialBillingCandidateRequest,
    actor: Annotated[str, Depends(require_actor)],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=1, max_length=128)
    ],
    service: CommercialBillingApprovalService = Depends(
        get_commercial_billing_approval_service
    ),
) -> dict:
    """Create or reuse one draft invoice only after explicit candidate approval."""

    return await _call_commercial_billing_approval(
        service.approve(
            billing_run_id=billing_run_id,
            candidate_key=body.candidate_key,
            expected_source_fingerprint=body.expected_source_fingerprint,
            idempotency_key=idempotency_key,
            actor=actor,
        )
    )


@router.post("/commercial-billing-approvals/{approval_id}/invoice-pdf", status_code=201)
async def generate_commercial_billing_invoice_pdf(
    approval_id: UUID,
    actor: Annotated[str, Depends(require_actor)],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=1, max_length=128)
    ],
    service: CommercialBillingInvoicePDFService = Depends(
        get_commercial_billing_invoice_pdf_service
    ),
) -> dict:
    """Create or reuse one durable PDF for an explicitly approved draft invoice."""

    return await _call_commercial_billing_invoice_pdf(
        service.generate_or_reuse(
            approval_id=approval_id,
            idempotency_key=idempotency_key,
            actor=actor,
        )
    )


@router.post("/commercial-billing-approvals/{approval_id}/gmail-draft", status_code=201)
async def create_commercial_billing_invoice_gmail_draft(
    approval_id: UUID,
    actor: Annotated[str, Depends(require_actor)],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=1, max_length=128)
    ],
    service: CommercialBillingInvoiceGmailDraftService = Depends(
        get_commercial_billing_invoice_gmail_draft_service
    ),
) -> dict:
    """Create, recover, or reuse one no-send Gmail draft for an approval PDF."""

    return await _call_commercial_billing_gmail_draft(
        service.create_or_reuse(
            approval_id=approval_id,
            idempotency_key=idempotency_key,
            actor=actor,
        )
    )


@router.post("/commercial-billing-approvals/{approval_id}/gmail-draft/reconcile")
async def reconcile_commercial_billing_invoice_gmail_draft_sent_mail(
    approval_id: UUID,
    actor: Annotated[str, Depends(require_actor)],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=1, max_length=128)
    ],
    service: CommercialBillingInvoiceGmailSentReconciliationService = Depends(
        get_commercial_billing_invoice_gmail_sent_reconciliation_service
    ),
) -> dict:
    """Reconcile a manually sent Gmail draft from verifiable Sent-mail evidence."""

    return await _call_commercial_billing_gmail_sent_reconciliation(
        service.reconcile(
            approval_id=approval_id,
            idempotency_key=idempotency_key,
            actor=actor,
        )
    )


@router.get("/commercial-billing-runs/{billing_run_id}/gmail-delivery-state")
async def list_commercial_billing_gmail_delivery_state(
    billing_run_id: UUID,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=MAX_DELIVERY_STATE_OFFSET),
    service: CommercialBillingInvoiceGmailSentReconciliationService = Depends(
        get_commercial_billing_invoice_gmail_sent_reconciliation_service
    ),
) -> dict:
    """Return durable Gmail delivery evidence for one immutable review run."""

    return await _call_commercial_billing_gmail_sent_reconciliation(
        service.list_delivery_state_for_run(
            billing_run_id=billing_run_id,
            limit=limit,
            offset=offset,
        )
    )


@router.get("/commercial-billing/manual-square-invoices")
async def list_commercial_billing_manual_square_invoices(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    service: CommercialBillingManualSquareInvoiceService = Depends(
        get_commercial_billing_manual_square_invoice_service
    ),
) -> dict:
    """List bounded manual-Square delivery work without changing financial state."""

    return await _call_commercial_billing_manual_square_invoice(
        service.list_needs_square_invoices(limit=limit, offset=offset)
    )


@router.post(
    "/commercial-billing-approvals/{approval_id}/manual-square-invoice-reference",
    status_code=201,
)
async def record_commercial_billing_manual_square_invoice_reference(
    approval_id: UUID,
    body: RecordCommercialBillingManualSquareReferenceRequest,
    actor: Annotated[str, Depends(require_actor)],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=1, max_length=128)
    ],
    service: CommercialBillingManualSquareInvoiceService = Depends(
        get_commercial_billing_manual_square_invoice_service
    ),
) -> dict:
    """Record one external Square reference without marking an invoice sent."""

    return await _call_commercial_billing_manual_square_invoice(
        service.record_reference(
            approval_id=approval_id,
            square_invoice_reference=body.square_invoice_reference,
            idempotency_key=idempotency_key,
            actor=actor,
        )
    )


@router.post(
    "/commercial-billing-approvals/{approval_id}/manual-square-invoice/mark-sent"
)
async def mark_commercial_billing_manual_square_invoice_sent(
    approval_id: UUID,
    actor: Annotated[str, Depends(require_actor)],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=1, max_length=128)
    ],
    service: CommercialBillingManualSquareInvoiceService = Depends(
        get_commercial_billing_manual_square_invoice_service
    ),
) -> dict:
    """Explicitly mark one referenced manual-Square invoice sent via Square."""

    return await _call_commercial_billing_manual_square_invoice(
        service.mark_sent(
            approval_id=approval_id,
            idempotency_key=idempotency_key,
            actor=actor,
        )
    )


@router.get("/commercial-billing-runs")
async def list_commercial_billing_runs(
    billing_period: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    service: CommercialBillingRunService = Depends(get_commercial_billing_run_service),
) -> dict:
    """List bounded durable review-run summaries without regenerating sources."""

    return await _call_commercial_billing_run(
        service.list_runs(
            billing_period=billing_period,
            limit=limit,
            offset=offset,
        )
    )


@router.get("/commercial-billing-runs/{billing_run_id}/reconciliation")
async def reconcile_commercial_billing_run(
    billing_run_id: UUID,
    service: CommercialBillingRunService = Depends(get_commercial_billing_run_service),
) -> dict:
    """Compare durable evidence with fresh sources without writing either."""

    return await _call_commercial_billing_run(service.reconcile_run(billing_run_id))


@router.get("/commercial-billing-runs/{billing_run_id}")
async def get_commercial_billing_run(
    billing_run_id: UUID,
    service: CommercialBillingRunService = Depends(get_commercial_billing_run_service),
) -> dict:
    """Return the immutable candidate snapshot retained for operator review."""

    return await _call_commercial_billing_run(service.get_run(billing_run_id))


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
