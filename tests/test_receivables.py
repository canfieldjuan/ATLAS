from __future__ import annotations

import asyncio
import errno
import json
import os
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from atlas_brain.api.invoicing import actions
from atlas_brain.api.invoicing import auth as receivables_auth
from atlas_brain.eom_api import auth as eom_receivables_auth
from atlas_brain.services.receivables import (
    _RECEIPT_DELIVERY_REQUIRED_COLUMNS,
    _RECEIPT_DELIVERY_REQUIRED_INDEXES,
    _RECEIPT_DISPATCH_REQUIRED_COLUMNS,
    _RECEIPT_DISPATCH_REQUIRED_INDEXES,
    _RECEIVABLES_REQUIRED_COLUMNS,
    _RECEIVABLES_REQUIRED_INDEXES,
    ReceivablesConflictError,
    ReceivablesError,
    ReceivablesNotFoundError,
    ReceivablesReceiptContextRequiredError,
    ReceivablesSchemaUnavailableError,
    ReceivablesService,
    ReceivablesValidationError,
    PaymentReceiptRecipient,
    aggregate_cents,
    money,
    request_fingerprint,
)
from atlas_brain.storage.exceptions import DatabaseUnavailableError


class _CreatePaymentConnection:
    def __init__(
        self,
        invoices: list[dict],
        contact_ids: list[UUID] | None = None,
        contact_rows: list[dict] | None = None,
    ) -> None:
        self.invoices = invoices
        self.contacts = {
            contact_id: {
                "business_context_id": "effingham_maids",
                "contact_type": "customer",
                "status": "active",
            }
            for contact_id in contact_ids or []
        }
        self.contacts.update(
            {contact["id"]: contact for contact in contact_rows or []}
        )
        self.contact_lock_count = 0
        self.contact_queries: list[str] = []
        self.parent_args = None
        self.parent_fingerprint_index: int | None = None
        self.parent_has_check_metadata = False
        self.parent_insert_count = 0
        self.receipt_deliveries: dict[UUID, dict] = {}
        self.allocations: list[dict] = []
        self.fetches: list[tuple[str, tuple]] = []
        self.executions: list[tuple[str, tuple]] = []

    async def fetchrow(self, query: str, *args):
        if "WHERE source = $1 AND idempotency_key = $2" in query:
            if self.parent_args is not None:
                return {
                    "id": self.parent_args[0],
                    "request_fingerprint": self.parent_args[
                        self.parent_fingerprint_index
                    ],
                }
            return None
        if "FROM contacts" in query:
            self.contact_lock_count += 1
            self.contact_queries.append(query)
            contact = self.contacts.get(args[0])
            if not contact:
                return None
            if args[1] is not None and (
                contact.get("business_context_id") != args[1]
                or contact.get("contact_type") != "customer"
                or contact.get("status") != "active"
            ):
                return None
            return {"id": args[0]}
        if "FROM payment_events" in query:
            return None
        if "INSERT INTO customer_payments" in query:
            self.parent_insert_count += 1
            self.parent_args = args
            self.parent_has_check_metadata = (
                "check_date, received_through" in query
            )
            self.parent_fingerprint_index = (
                12 if self.parent_has_check_metadata else 10
            )
            return {"id": args[0]}
        if "SELECT cp.*, pdi.batch_id" in query:
            assert self.parent_args is not None
            parent = self.parent_args
            payment = {
                "id": parent[0],
                "contact_id": parent[1],
                "payer_name": parent[2],
                "total_amount": parent[3],
                "payment_method": parent[4],
                "reference": parent[5],
                "received_date": parent[6],
                "metadata": {},
                "batch_id": None,
            }
            if self.parent_has_check_metadata:
                payment.update(
                    {
                        "check_date": parent[7],
                        "received_through": parent[8],
                        "status": parent[9],
                        "source": parent[10],
                        "idempotency_key": parent[11],
                        "request_fingerprint": parent[12],
                        "notes": parent[13],
                        "recorded_by": parent[14],
                    }
                )
            else:
                payment.update(
                    {
                        "status": parent[7],
                        "source": parent[8],
                        "idempotency_key": parent[9],
                        "request_fingerprint": parent[10],
                        "notes": parent[11],
                        "recorded_by": parent[12],
                    }
                )
            return payment
        if "FROM payment_receipt_deliveries" in query:
            stored = self.receipt_deliveries.get(args[0])
            if stored is None:
                return None
            return {
                "receipt_number": stored["receipt_number"],
                "recipient_email": stored["recipient_email"],
                "status": stored["delivery_status"],
                "skip_reason": stored["skip_reason"],
            }
        raise AssertionError(f"Unexpected fetchrow query: {query}")

    async def fetchval(self, query: str, *args):
        if "pg_advisory_xact_lock" in query:
            return None
        if "information_schema.columns" in query:
            return True
        if "FROM payment_receipt_deliveries" in query:
            return args[0] in self.receipt_deliveries
        raise AssertionError(f"Unexpected fetchval query: {query}")

    async def fetch(self, query: str, *args):
        self.fetches.append((query, args))
        if "FROM pg_index AS index_state" in query:
            required_indexes = {
                index_name: (
                    table_name,
                    is_unique,
                    key_columns,
                    predicate,
                    constraint_type,
                )
                for (
                    table_name,
                    index_name,
                    is_unique,
                    key_columns,
                    predicate,
                    constraint_type,
                ) in (
                    *_RECEIVABLES_REQUIRED_INDEXES,
                    *_RECEIPT_DELIVERY_REQUIRED_INDEXES,
                    *_RECEIPT_DISPATCH_REQUIRED_INDEXES,
                )
            }
            return [
                {
                    "index_name": index_name,
                    "table_name": required_indexes[index_name][0],
                    "is_unique": required_indexes[index_name][1],
                    "key_columns": list(required_indexes[index_name][2]),
                    "predicate": required_indexes[index_name][3],
                    "constraint_type": required_indexes[index_name][4],
                    "is_valid": True,
                    "is_ready": True,
                }
                for index_name in args[0]
            ]
        if "FROM invoices" in query and "ORDER BY id FOR UPDATE" in query:
            selected = {str(item) for item in args[0]}
            return [row for row in self.invoices if str(row["id"]) in selected]
        if "FROM invoice_payments ip" in query and "JOIN invoices" in query:
            return self.allocations
        raise AssertionError(f"Unexpected fetch query: {query}")

    async def execute(self, query: str, *args):
        self.executions.append((query, args))
        if "INSERT INTO invoice_payments" in query:
            invoice = next(row for row in self.invoices if row["id"] == args[1])
            self.allocations.append(
                {
                    "payment_id": args[2],
                    "id": args[0],
                    "invoice_id": args[1],
                    "invoice_number": invoice["invoice_number"],
                    "amount": args[3],
                    "reversed_at": None,
                    "reversal_reason": None,
                }
            )
        if "INSERT INTO payment_receipt_deliveries" in query:
            self.receipt_deliveries[args[1]] = {
                "id": args[0],
                "payment_id": args[1],
                "contact_id": args[2],
                "receipt_number": args[3],
                "recipient_email": args[4],
                "delivery_status": args[5],
                "skip_reason": args[6],
                "subject": args[7],
                "body": args[8],
                "created_at": args[9],
            }
        return "OK"


class _CreatePaymentPool:
    is_initialized = True

    def __init__(
        self,
        invoices: list[dict],
        contact_ids: list[UUID] | None = None,
        contact_rows: list[dict] | None = None,
    ) -> None:
        self.conn = _CreatePaymentConnection(invoices, contact_ids, contact_rows)
        self.transaction_count = 0

    @asynccontextmanager
    async def transaction(self):
        self.transaction_count += 1
        yield self.conn


class _TransactionPool:
    is_initialized = True

    def __init__(self, conn) -> None:
        self.conn = conn

    @asynccontextmanager
    async def transaction(self):
        yield self.conn


def _invoice(
    invoice_number: str,
    contact_id: UUID,
    amount_due: str,
) -> dict:
    return {
        "id": uuid4(),
        "invoice_number": invoice_number,
        "contact_id": contact_id,
        "status": "sent",
        "amount_due": Decimal(amount_due),
    }


@pytest.mark.asyncio
async def test_create_payment_records_multi_invoice_receipt_in_one_transaction():
    contact_id = uuid4()
    first = _invoice("INV-1", contact_id, "125.00")
    second = _invoice("INV-2", contact_id, "175.00")
    pool = _CreatePaymentPool([first, second])

    payment = await ReceivablesService(pool).create_payment(
        contact_id=contact_id,
        payer_name="Acme Offices",
        total_amount=Decimal("300.00"),
        payment_method="ach",
        received_date=date(2026, 7, 16),
        allocations=[
            {"invoice_id": first["id"], "amount": Decimal("100.00")},
            {"invoice_id": second["id"], "amount": Decimal("150.00")},
        ],
        idempotency_key="payment-create-1",
        recorded_by="Juan Canfield",
    )

    assert pool.transaction_count == 1
    assert payment["status"] == "cleared"
    assert payment["allocated_amount_cents"] == 25_000
    assert payment["unapplied_amount_cents"] == 5_000
    assert len(payment["allocations"]) == 2
    assert (
        sum(
            "INSERT INTO invoice_payments" in query
            for query, _args in pool.conn.executions
        )
        == 2
    )
    assert any("WITH totals AS" in query for query, _args in pool.conn.executions)
    event = next(
        args
        for query, args in pool.conn.executions
        if "INSERT INTO payment_events" in query
    )
    assert event[8] == "payment-create-1"


@pytest.mark.asyncio
async def test_create_payment_records_unapplied_receipt_and_replays_without_invoice_side_effects():
    contact_id = uuid4()
    pool = _CreatePaymentPool([], [contact_id])
    service = ReceivablesService(pool)
    create_args = {
        "contact_id": contact_id,
        "payer_name": "Residential Customer",
        "total_amount": Decimal("125.00"),
        "payment_method": "check",
        "received_date": date(2026, 8, 12),
        "allocations": [],
        "idempotency_key": "unapplied-payment-create-1",
        "recorded_by": "Juan Canfield",
        "allow_unapplied": True,
        "unapplied_contact_context_id": "effingham_maids",
    }

    first = await service.create_payment_with_outcome(**create_args)
    replay = await service.create_payment_with_outcome(**create_args)

    assert first.replayed is False
    assert replay.replayed is True
    assert first.payment["id"] == replay.payment["id"]
    assert first.payment["status"] == "received"
    assert first.payment["allocated_amount_cents"] == 0
    assert first.payment["unapplied_amount_cents"] == 12_500
    assert first.payment["allocations"] == []
    assert pool.transaction_count == 2
    assert pool.conn.parent_insert_count == 1
    assert pool.conn.contact_lock_count == 1
    assert all("lead_stage" not in query for query in pool.conn.contact_queries)
    assert all("FOR SHARE" in query for query in pool.conn.contact_queries)
    assert all("FOR KEY SHARE" not in query for query in pool.conn.contact_queries)
    assert not any("FROM invoices" in query for query, _args in pool.conn.fetches)
    assert not any(
        "INSERT INTO invoice_payments" in query or "WITH totals AS" in query
        for query, _args in pool.conn.executions
    )
    events = [
        args
        for query, args in pool.conn.executions
        if "INSERT INTO payment_events" in query
    ]
    assert len(events) == 1
    assert json.loads(events[0][10]) == {"allocated_amount": "0"}


@pytest.mark.asyncio
async def test_residential_payment_enqueues_one_pending_receipt_without_sending():
    contact_id = uuid4()
    pool = _CreatePaymentPool([], [contact_id])
    service = ReceivablesService(pool)

    async def ready(_conn=None) -> bool:
        return True

    service.is_receipt_delivery_ready = ready
    create_args = {
        "contact_id": contact_id,
        "payer_name": "Jordan Customer",
        "total_amount": Decimal("125.00"),
        "payment_method": "check",
        "received_date": date(2026, 8, 12),
        "reference": "1042",
        "allocations": [],
        "idempotency_key": "residential-receipt-payment-1",
        "recorded_by": "Juan Canfield",
        "allow_unapplied": True,
        "unapplied_contact_context_id": "effingham_maids",
        "receipt_recipient": PaymentReceiptRecipient(
            contact_id=contact_id,
            customer_name="Jordan Customer",
            customer_type="residential",
            recipient_email="jordan@example.test",
        ),
        "require_receipt_recipient": True,
    }

    first = await service.create_payment_with_outcome(**create_args)
    replay = await service.create_payment_with_outcome(**create_args)

    assert first.replayed is False
    assert replay.replayed is True
    assert first.payment["id"] == replay.payment["id"]
    receipt = first.payment["receipt_delivery"]
    assert receipt == {
        "receipt_number": f"EOM-RCP-{first.payment['id']}",
        "recipient_email": "jordan@example.test",
        "status": "pending",
        "skip_reason": None,
    }
    assert len(pool.conn.receipt_deliveries) == 1
    stored = next(iter(pool.conn.receipt_deliveries.values()))
    assert stored["subject"] == (
        f"Payment received — receipt EOM-RCP-{first.payment['id']}"
    )
    assert "Customer: Jordan Customer" in stored["body"]
    assert "Payer: Jordan Customer" in stored["body"]
    assert "Amount received: $125.00" in stored["body"]
    assert "Payment method: Check" in stored["body"]
    assert "Check number: 1042" in stored["body"]
    assert "Date received: August 12, 2026" in stored["body"]
    assert "We have received your check. It has not yet cleared." in stored["body"]
    assert not any(
        "Gmail" in query or "Resend" in query
        for query, _args in pool.conn.executions
    )


@pytest.mark.asyncio
async def test_residential_receipt_readiness_uses_held_transaction_connection():
    """Receipt readiness must not check out another pooled connection mid-write."""
    contact_id = uuid4()
    pool = _CreatePaymentPool([], [contact_id])

    result = await ReceivablesService(pool).create_payment_with_outcome(
        contact_id=contact_id,
        payer_name="Held Connection Customer",
        total_amount=Decimal("50.00"),
        payment_method="check",
        received_date=date(2026, 8, 12),
        allocations=[],
        idempotency_key="receipt-readiness-held-connection-1",
        recorded_by="Juan Canfield",
        allow_unapplied=True,
        unapplied_contact_context_id="effingham_maids",
        receipt_recipient=PaymentReceiptRecipient(
            contact_id=contact_id,
            customer_name="Held Connection Customer",
            customer_type="residential",
            recipient_email="held-connection@example.test",
        ),
        require_receipt_recipient=True,
    )

    assert result.replayed is False
    assert result.payment["receipt_delivery"]["status"] == "pending"
    # _CreatePaymentPool intentionally has no direct fetch/fetchval methods.
    # Success therefore proves the readiness catalogs used the held `conn`.
    assert pool.transaction_count == 1


@pytest.mark.asyncio
async def test_residential_replay_uses_persisted_receipt_after_reclassification():
    contact_id = uuid4()
    pool = _CreatePaymentPool([], [contact_id])
    service = ReceivablesService(pool)

    async def ready(_conn=None) -> bool:
        return True

    service.is_receipt_delivery_ready = ready
    create_args = {
        "contact_id": contact_id,
        "payer_name": "Reclassified Customer",
        "total_amount": Decimal("50.00"),
        "payment_method": "ach",
        "received_date": date(2026, 8, 12),
        "allocations": [],
        "idempotency_key": "residential-reclassification-replay-1",
        "recorded_by": "Juan Canfield",
        "allow_unapplied": True,
        "unapplied_contact_context_id": "effingham_maids",
        "require_receipt_recipient": True,
    }
    original = await service.create_payment_with_outcome(
        **{
            **create_args,
            "receipt_recipient": PaymentReceiptRecipient(
                contact_id=contact_id,
                customer_name="Reclassified Customer",
                customer_type="residential",
                recipient_email="reclassified@example.test",
            ),
        }
    )
    replay = await service.create_payment_with_outcome(
        **{
            **create_args,
            "receipt_recipient": PaymentReceiptRecipient(
                contact_id=contact_id,
                customer_name="Reclassified Customer",
                customer_type="commercial",
                recipient_email="billing@example.test",
            ),
        }
    )

    assert replay.replayed is True
    assert replay.payment["id"] == original.payment["id"]
    assert replay.payment["receipt_delivery"] == original.payment["receipt_delivery"]
    assert len(pool.conn.receipt_deliveries) == 1


@pytest.mark.asyncio
async def test_residential_payment_without_email_is_skipped_but_financially_valid():
    contact_id = uuid4()
    pool = _CreatePaymentPool([], [contact_id])
    service = ReceivablesService(pool)

    async def ready(_conn=None) -> bool:
        return True

    service.is_receipt_delivery_ready = ready
    result = await service.create_payment_with_outcome(
        contact_id=contact_id,
        payer_name="No Email Customer",
        total_amount=Decimal("50.00"),
        payment_method="ach",
        received_date=date(2026, 8, 12),
        allocations=[],
        idempotency_key="residential-receipt-no-email-1",
        recorded_by="Juan Canfield",
        allow_unapplied=True,
        unapplied_contact_context_id="effingham_maids",
        receipt_recipient=PaymentReceiptRecipient(
            contact_id=contact_id,
            customer_name="No Email Customer",
            customer_type="residential",
            recipient_email=None,
        ),
        require_receipt_recipient=True,
    )

    assert result.payment["status"] == "cleared"
    assert result.payment["receipt_delivery"]["status"] == "skipped"
    assert result.payment["receipt_delivery"]["skip_reason"] == "no_email"
    assert result.payment["receipt_delivery"]["recipient_email"] is None
    assert pool.conn.parent_insert_count == 1
    assert len(pool.conn.receipt_deliveries) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("customer_type", ("commercial", "unknown"))
async def test_non_residential_payment_has_no_residential_receipt_delivery(
    customer_type,
):
    contact_id = uuid4()
    pool = _CreatePaymentPool([], [contact_id])
    service = ReceivablesService(pool)
    result = await service.create_payment_with_outcome(
        contact_id=contact_id,
        payer_name="Commercial Customer",
        total_amount=Decimal("50.00"),
        payment_method="ach",
        received_date=date(2026, 8, 12),
        allocations=[],
        idempotency_key=f"{customer_type}-no-residential-receipt-1",
        recorded_by="Juan Canfield",
        allow_unapplied=True,
        unapplied_contact_context_id="effingham_maids",
        receipt_recipient=PaymentReceiptRecipient(
            contact_id=contact_id,
            customer_name="Commercial Customer",
            customer_type=customer_type,
            recipient_email="billing@example.test",
        ),
        require_receipt_recipient=True,
    )

    assert result.payment["status"] == "cleared"
    assert "receipt_delivery" not in result.payment
    assert pool.conn.receipt_deliveries == {}


@pytest.mark.asyncio
async def test_receipt_context_rejects_noncanonical_email_before_payment_write():
    contact_id = uuid4()
    pool = _CreatePaymentPool([], [contact_id])
    with pytest.raises(ReceivablesValidationError, match="Receipt email is invalid"):
        await ReceivablesService(pool).create_payment_with_outcome(
            contact_id=contact_id,
            payer_name="Residential Customer",
            total_amount=Decimal("50.00"),
            payment_method="ach",
            received_date=date(2026, 8, 12),
            allocations=[],
            idempotency_key="invalid-receipt-email-1",
            allow_unapplied=True,
            unapplied_contact_context_id="effingham_maids",
            receipt_recipient=PaymentReceiptRecipient(
                contact_id=contact_id,
                customer_name="Residential Customer",
                customer_type="residential",
                recipient_email="not-an-email",
            ),
            require_receipt_recipient=True,
        )

    assert pool.transaction_count == 0
    assert pool.conn.parent_insert_count == 0


@pytest.mark.asyncio
async def test_canonical_receipt_context_is_required_only_for_a_new_eom_payment():
    contact_id = uuid4()
    pool = _CreatePaymentPool([], [contact_id])
    service = ReceivablesService(pool)

    async def ready(_conn=None) -> bool:
        return True

    service.is_receipt_delivery_ready = ready
    create_args = {
        "contact_id": contact_id,
        "payer_name": "Replay Customer",
        "total_amount": Decimal("50.00"),
        "payment_method": "ach",
        "received_date": date(2026, 8, 12),
        "allocations": [],
        "idempotency_key": "residential-receipt-replay-1",
        "recorded_by": "Juan Canfield",
        "allow_unapplied": True,
        "unapplied_contact_context_id": "effingham_maids",
        "require_receipt_recipient": True,
    }
    with pytest.raises(ReceivablesReceiptContextRequiredError):
        await service.create_payment_with_outcome(**create_args)
    assert pool.conn.parent_insert_count == 0

    first = await service.create_payment_with_outcome(
        **{
            **create_args,
            "receipt_recipient": PaymentReceiptRecipient(
                contact_id=contact_id,
                customer_name="Replay Customer",
                customer_type="residential",
                recipient_email="replay@example.test",
            ),
        }
    )
    replay = await service.create_payment_with_outcome(**create_args)

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.payment["id"] == first.payment["id"]
    assert pool.conn.parent_insert_count == 1
    assert len(pool.conn.receipt_deliveries) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("received_through", (None, "  "))
async def test_create_payment_without_check_metadata_preserves_legacy_intent(
    received_through,
):
    contact_id = uuid4()
    pool = _CreatePaymentPool([], [contact_id])
    service = ReceivablesService(pool)
    create_args = {
        "contact_id": contact_id,
        "payer_name": "Residential Customer",
        "total_amount": Decimal("125.00"),
        "payment_method": "check",
        "received_date": date(2026, 8, 12),
        "received_through": received_through,
        "allocations": [],
        "idempotency_key": "legacy-payment-intent-1",
        "recorded_by": "Juan Canfield",
        "allow_unapplied": True,
        "unapplied_contact_context_id": "effingham_maids",
    }

    result = await service.create_payment_with_outcome(**create_args)

    assert result.replayed is False
    assert pool.conn.parent_has_check_metadata is False
    assert pool.conn.parent_args[10] == request_fingerprint(
        {
            "contact_id": contact_id,
            "payer_name": "Residential Customer",
            "total_amount": Decimal("125.00"),
            "payment_method": "check",
            "received_date": date(2026, 8, 12),
            "allocations": [],
            "reference": None,
            "notes": None,
        }
    )

    with pytest.raises(ReceivablesValidationError, match="at most 128"):
        await service.create_payment_with_outcome(
            **{**create_args, "received_through": "x" * 129}
        )


@pytest.mark.asyncio
async def test_create_payment_persists_optional_check_metadata_in_payment_intent():
    contact_id = uuid4()
    pool = _CreatePaymentPool([], [contact_id])
    service = ReceivablesService(pool)

    async def ready() -> bool:
        return True

    service.is_ready = ready
    create_args = {
        "contact_id": contact_id,
        "payer_name": "Residential Customer",
        "total_amount": Decimal("125.00"),
        "payment_method": "check",
        "received_date": date(2026, 8, 12),
        "check_date": date(2026, 8, 10),
        "received_through": " Mail ",
        "allocations": [],
        "idempotency_key": "unapplied-payment-check-metadata-1",
        "recorded_by": "Juan Canfield",
        "allow_unapplied": True,
        "unapplied_contact_context_id": "effingham_maids",
    }

    first = await service.create_payment_with_outcome(**create_args)
    replay = await service.create_payment_with_outcome(**create_args)

    assert first.replayed is False
    assert replay.replayed is True
    assert first.payment["id"] == replay.payment["id"]
    assert first.payment["check_date"] == date(2026, 8, 10)
    assert first.payment["received_through"] == "Mail"
    for changed_field, changed_value in (
        ("check_date", date(2026, 8, 11)),
        ("received_through", "Employee handoff"),
    ):
        with pytest.raises(ReceivablesConflictError, match="different request"):
            await service.create_payment_with_outcome(
                **{**create_args, changed_field: changed_value}
            )

    assert pool.conn.parent_insert_count == 1
    assert len(
        [
            query
            for query, _args in pool.conn.executions
            if "INSERT INTO payment_events" in query
        ]
    ) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("payment_method", ("ach", "square"))
async def test_create_payment_rejects_check_metadata_for_non_check_methods(
    payment_method,
):
    contact_id = uuid4()
    pool = _CreatePaymentPool([], [contact_id])

    with pytest.raises(
        ReceivablesValidationError,
        match="Check metadata requires a check payment method",
    ):
        await ReceivablesService(pool).create_payment(
            contact_id=contact_id,
            payer_name="Residential Customer",
            total_amount=Decimal("125.00"),
            payment_method=payment_method,
            received_date=date(2026, 8, 12),
            check_date=date(2026, 8, 10),
            allocations=[],
            idempotency_key=f"{payment_method}-check-metadata",
            allow_unapplied=True,
            unapplied_contact_context_id="effingham_maids",
        )

    assert pool.transaction_count == 0
    assert pool.conn.parent_args is None


@pytest.mark.asyncio
async def test_create_unapplied_payment_requires_opt_in_and_existing_customer_before_insert():
    pool = _CreatePaymentPool([])
    create_args = {
        "contact_id": uuid4(),
        "payer_name": "Residential Customer",
        "total_amount": Decimal("125.00"),
        "payment_method": "check",
        "received_date": date(2026, 8, 12),
        "allocations": [],
        "idempotency_key": "unapplied-payment-unknown-customer",
    }

    with pytest.raises(ReceivablesValidationError, match="At least one"):
        await ReceivablesService(pool).create_payment(**create_args)
    with pytest.raises(ReceivablesValidationError, match="canonical customer context"):
        await ReceivablesService(pool).create_payment(
            **create_args, allow_unapplied=True
        )
    assert pool.transaction_count == 0
    with pytest.raises(ReceivablesNotFoundError, match="Customer not found"):
        await ReceivablesService(pool).create_payment(
            **create_args,
            allow_unapplied=True,
            unapplied_contact_context_id="effingham_maids",
        )

    assert pool.transaction_count == 1
    assert pool.conn.contact_lock_count == 1
    assert pool.conn.parent_args is None
    assert not pool.conn.executions
    assert not any("FROM invoices" in query for query, _args in pool.conn.fetches)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contact",
    [
        {"business_context_id": "other", "contact_type": "customer", "status": "active"},
        {"business_context_id": "effingham_maids", "contact_type": "lead", "status": "active"},
        {"business_context_id": "effingham_maids", "contact_type": "customer", "status": "inactive"},
    ],
)
async def test_create_unapplied_payment_rejects_ineligible_eom_contacts_before_insert(
    contact,
):
    contact_id = uuid4()
    pool = _CreatePaymentPool(
        [],
        contact_rows=[{"id": contact_id, **contact}],
    )

    with pytest.raises(ReceivablesNotFoundError, match="Customer not found"):
        await ReceivablesService(pool).create_payment(
            contact_id=contact_id,
            payer_name="Residential Customer",
            total_amount=Decimal("125.00"),
            payment_method="check",
            received_date=date(2026, 8, 12),
            allocations=[],
            idempotency_key="unapplied-payment-ineligible-customer",
            allow_unapplied=True,
            unapplied_contact_context_id="effingham_maids",
        )

    assert pool.conn.parent_args is None
    assert pool.conn.contact_lock_count == 1


@pytest.mark.asyncio
async def test_create_payment_rejects_cross_customer_allocations_before_insert():
    selected_contact = uuid4()
    first = _invoice("INV-1", selected_contact, "100.00")
    second = _invoice("INV-2", uuid4(), "100.00")
    pool = _CreatePaymentPool([first, second])

    with pytest.raises(ReceivablesValidationError, match="same customer"):
        await ReceivablesService(pool).create_payment(
            contact_id=selected_contact,
            payer_name="Acme Offices",
            total_amount=Decimal("100.00"),
            payment_method="check",
            received_date=date(2026, 7, 16),
            allocations=[
                {"invoice_id": first["id"], "amount": Decimal("50.00")},
                {"invoice_id": second["id"], "amount": Decimal("50.00")},
            ],
            idempotency_key="payment-create-2",
        )

    assert pool.conn.parent_args is None
    assert not pool.conn.executions


@pytest.mark.asyncio
async def test_create_payment_rejects_allocation_above_receipt_before_transaction():
    contact_id = uuid4()
    invoice = _invoice("INV-1", contact_id, "100.00")
    pool = _CreatePaymentPool([invoice])

    with pytest.raises(ReceivablesValidationError, match="cannot exceed"):
        await ReceivablesService(pool).create_payment(
            contact_id=contact_id,
            payer_name="Acme",
            total_amount=Decimal("10.00"),
            payment_method="check",
            received_date=date(2026, 7, 16),
            allocations=[{"invoice_id": invoice["id"], "amount": "10.01"}],
            idempotency_key="over-allocated",
        )

    assert pool.transaction_count == 0


@pytest.mark.parametrize(
    "invalid_key",
    ["", "   ", "x" * 129, "long-prefix-" + "y" * 256],
)
@pytest.mark.asyncio
async def test_create_payment_rejects_invalid_idempotency_key_before_transaction(
    invalid_key,
):
    contact_id = uuid4()
    invoice = _invoice("INV-1", contact_id, "100.00")
    pool = _CreatePaymentPool([invoice])

    with pytest.raises(ReceivablesValidationError, match="1 to 128"):
        await ReceivablesService(pool).create_payment(
            contact_id=contact_id,
            payer_name="Acme",
            total_amount=Decimal("10.00"),
            payment_method="check",
            received_date=date(2026, 7, 16),
            allocations=[{"invoice_id": invoice["id"], "amount": "10.00"}],
            idempotency_key=invalid_key,
        )

    assert pool.transaction_count == 0


@pytest.mark.asyncio
async def test_open_invoice_feed_orders_due_date_before_customer_name():
    class _Pool:
        is_initialized = True

        def __init__(self):
            self.query = ""

        async def fetch(self, query, *_args):
            self.query = " ".join(query.split())
            return []

    pool = _Pool()
    await ReceivablesService(pool).suggest_allocations(
        contact_id=uuid4(), total_amount=Decimal("10.00")
    )

    assert (
        "ORDER BY due_date, issue_date, customer_name, invoice_number"
        in pool.query
    )


def test_money_and_aggregate_cents_preserve_currency_validation():
    with pytest.raises(ReceivablesValidationError, match="finite"):
        money("NaN")
    with pytest.raises(ReceivablesValidationError, match="cent precision"):
        money("10.005")
    with pytest.raises(ReceivablesValidationError, match="finite"):
        aggregate_cents("NaN")
    assert aggregate_cents("10000000000.00") == 1_000_000_000_000

    invoice_id = uuid4()
    with pytest.raises(ReceivablesValidationError, match="only appear once"):
        ReceivablesService._normalize_allocations(
            [
                {"invoice_id": invoice_id, "amount": "1.00"},
                {"invoice_id": invoice_id, "amount": "2.00"},
            ]
        )


def test_empty_allocations_require_explicit_initial_creation_opt_in():
    assert ReceivablesService._normalize_allocations([], allow_empty=True) == []
    with pytest.raises(ReceivablesValidationError, match="At least one"):
        ReceivablesService._normalize_allocations([])
    with pytest.raises(ReceivablesValidationError, match="At least one"):
        ReceivablesService._normalize_allocations(None, allow_empty=True)


def test_idempotency_key_reuse_with_different_request_conflicts():
    with pytest.raises(ReceivablesConflictError, match="different request"):
        ReceivablesService._assert_idempotent(
            {"request_fingerprint": "first"}, "second"
        )


@pytest.mark.asyncio
async def test_payment_event_key_lookup_is_global_across_payments():
    class _Connection:
        def __init__(self):
            self.query = ""
            self.args = ()

        async def fetchrow(self, query, *args):
            self.query = query
            self.args = args
            return {"payment_id": uuid4(), "request_fingerprint": "first"}

    conn = _Connection()
    event = await ReceivablesService._event_for_key(conn, "shared-event-key")

    assert event["request_fingerprint"] == "first"
    assert "payment_id = ANY" not in conn.query
    assert "WHERE idempotency_key = $1" in conn.query
    assert conn.args == ("shared-event-key",)


@pytest.mark.parametrize("status", ["returned", "voided"])
def test_inactive_payment_summary_excludes_historical_allocations(status):
    result = ReceivablesService._compose_payment(
        {"id": uuid4(), "status": status, "total_amount": Decimal("100.00")},
        [
            {
                "id": uuid4(),
                "invoice_id": uuid4(),
                "amount": Decimal("75.00"),
                "reversed_at": None,
            }
        ],
    )

    assert result["allocated_amount_cents"] == 0
    assert result["unapplied_amount_cents"] == 0
    assert result["allocations"][0]["amount_cents"] == 7_500


class _InvoiceLockConnection:
    def __init__(self, rows):
        self.rows = rows
        self.locked_ids = []

    async def fetch(self, _query, invoice_ids):
        self.locked_ids = invoice_ids
        return self.rows


@pytest.mark.asyncio
async def test_adjustment_lock_includes_invoice_removed_from_replacement():
    contact_id = uuid4()
    kept = _invoice("INV-1", contact_id, "20.00")
    removed = _invoice("INV-2", contact_id, "0.00")
    removed["status"] = "paid"
    conn = _InvoiceLockConnection([kept, removed])

    await ReceivablesService._lock_and_validate_invoices(
        ReceivablesService(),
        conn,
        contact_id=contact_id,
        allocations=[{"invoice_id": kept["id"], "amount": Decimal("10.00")}],
        current_allocations={str(removed["id"]): Decimal("10.00")},
        additional_invoice_ids=[removed["id"]],
    )

    assert set(conn.locked_ids) == {kept["id"], removed["id"]}


class _RecalculationConnection:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def fetch(self, query, *_args):
        self.calls.append(query)
        return []

    async def execute(self, query, *_args):
        self.calls.append(query)
        return "UPDATE 1"


@pytest.mark.asyncio
async def test_recalculation_locks_before_reading_totals_and_counts_null_parent_rows():
    conn = _RecalculationConnection()

    await ReceivablesService._recalculate_invoices(conn, [uuid4()])

    assert "FOR UPDATE" in conn.calls[0]
    assert "WITH totals AS" in conn.calls[1]
    assert "ip.payment_id IS NULL" in conn.calls[1]


@pytest.mark.asyncio
async def test_draft_invoice_cannot_receive_an_allocation():
    contact_id = uuid4()
    draft = _invoice("INV-DRAFT", contact_id, "20.00")
    draft["status"] = "draft"

    with pytest.raises(ReceivablesConflictError, match="not open"):
        await ReceivablesService._lock_and_validate_invoices(
            ReceivablesService(),
            _InvoiceLockConnection([draft]),
            contact_id=contact_id,
            allocations=[{"invoice_id": draft["id"], "amount": Decimal("10.00")}],
        )


class _ClearBatchConnection:
    def __init__(self, batch_id, payment_id) -> None:
        self.batch_id = batch_id
        self.payment_id = payment_id
        self.executions = []

    async def fetchrow(self, query, *_args):
        if "WHERE clear_idempotency_key" in query:
            return None
        if "FROM payment_events" in query:
            return None
        return {
            "id": self.batch_id,
            "status": "deposited",
            "clear_idempotency_key": None,
            "clear_request_fingerprint": None,
        }

    async def execute(self, query, *args):
        self.executions.append((query, args))
        return "OK"

    async def fetch(self, _query, *_args):
        return [{"id": self.payment_id, "status": "returned"}]

    async def fetchval(self, query, *_args):
        assert "pg_advisory_xact_lock" in query
        return None


@pytest.mark.asyncio
async def test_clear_batch_rejects_a_returned_check_instead_of_skipping_it():
    batch_id = uuid4()
    conn = _ClearBatchConnection(batch_id, uuid4())

    with pytest.raises(ReceivablesConflictError, match="must still be deposited"):
        await ReceivablesService(_TransactionPool(conn)).clear_deposit_batch(
            batch_id=batch_id,
            actor="Juan",
            idempotency_key="clear-batch-1",
        )

    assert not any(
        "UPDATE customer_payments" in query for query, _args in conn.executions
    )


def test_receivables_migration_keeps_legacy_rows_one_to_one():
    migration = (
        Path(__file__).parents[1]
        / "atlas_brain/storage/migrations/344_receivables_payments.sql"
    ).read_text(encoding="utf-8")

    assert "id, contact_id, payer_name" in migration
    assert "SELECT\n    ip.id," in migration
    assert (
        "GROUP BY"
        not in migration.split("CREATE TABLE IF NOT EXISTS payment_deposit_batches")[0]
    )
    assert "clear_idempotency_key" in migration
    customer_table = migration.split("CREATE TABLE IF NOT EXISTS customer_payments", 1)[
        1
    ]
    customer_table = customer_table.split(");", 1)[0]
    assert "clear_idempotency_key" not in customer_table
    assert "CREATE TRIGGER trg_adopt_legacy_invoice_payment" in migration
    assert "WHEN (NEW.payment_id IS NULL)" in migration
    assert "adopted_rolling_writer" in migration
    assert (
        "COALESCE(NULLIF(lower(btrim(NEW.payment_method)), ''), 'other')" in migration
    )
    assert "NEW.amount > 0 AND lower(btrim(NEW.payment_method)) = 'check'" in migration


def test_global_payment_event_key_lookup_has_a_followup_migration():
    migration = (
        Path(__file__).parents[1]
        / "atlas_brain/storage/migrations/345_receivables_event_key_lookup.sql"
    ).read_text(encoding="utf-8")

    assert "CREATE INDEX IF NOT EXISTS idx_payment_events_key_lookup" in migration
    assert "ON payment_events(idempotency_key)" in migration


@pytest.mark.asyncio
async def test_standalone_mcp_applies_migrations_before_serving_and_closes():
    from atlas_brain.mcp import invoicing_server

    events = []
    pool = SimpleNamespace(is_initialized=True)

    async def initialize():
        events.append("initialize")

    def get_pool():
        events.append("get-pool")
        return pool

    async def migrate(candidate):
        assert candidate is pool
        events.append("migrate")

    async def ready(candidate):
        assert candidate is pool
        events.append("ready")
        return True

    async def close():
        events.append("close")

    async with invoicing_server._database_lifespan(
        init_database_fn=initialize,
        get_db_pool_fn=get_pool,
        run_migrations_fn=migrate,
        receivables_ready_fn=ready,
        close_database_fn=close,
    ):
        events.append("serving")

    assert events == [
        "initialize",
        "get-pool",
        "migrate",
        "ready",
        "serving",
        "close",
    ]


@pytest.mark.asyncio
async def test_standalone_mcp_migration_failure_aborts_startup_and_closes():
    from atlas_brain.mcp import invoicing_server

    events = []
    pool = SimpleNamespace(is_initialized=True)

    async def initialize():
        events.append("initialize")

    async def migrate(candidate):
        assert candidate is pool
        events.append("migrate")
        raise RuntimeError("migration failed")

    async def ready(_candidate):
        raise AssertionError("readiness must not run after migration failure")

    async def close():
        events.append("close")

    with pytest.raises(RuntimeError, match="migration failed"):
        async with invoicing_server._database_lifespan(
            init_database_fn=initialize,
            get_db_pool_fn=lambda: pool,
            run_migrations_fn=migrate,
            receivables_ready_fn=ready,
            close_database_fn=close,
        ):
            events.append("serving")

    assert events == ["initialize", "migrate", "close"]


@pytest.mark.asyncio
async def test_standalone_mcp_incomplete_schema_aborts_startup_and_closes():
    from atlas_brain.mcp import invoicing_server

    events = []
    pool = SimpleNamespace(is_initialized=True)

    async def initialize():
        events.append("initialize")

    async def migrate(candidate):
        assert candidate is pool
        events.append("migrate")

    async def ready(candidate):
        assert candidate is pool
        events.append("ready")
        return False

    async def close():
        events.append("close")

    with pytest.raises(RuntimeError, match="complete receivables schema"):
        async with invoicing_server._database_lifespan(
            init_database_fn=initialize,
            get_db_pool_fn=lambda: pool,
            run_migrations_fn=migrate,
            receivables_ready_fn=ready,
            close_database_fn=close,
        ):
            events.append("serving")

    assert events == ["initialize", "migrate", "ready", "close"]


def _workflow_event_paths(workflow: str, event_name: str) -> tuple[str, ...]:
    event_start = re.search(rf"^  {re.escape(event_name)}:\n", workflow, re.MULTILINE)
    assert event_start is not None
    event_end = re.search(
        r"^  [a-z_]+:\n|^jobs:\n",
        workflow[event_start.end() :],
        re.MULTILINE,
    )
    event_block = (
        workflow[event_start.end() :]
        if event_end is None
        else workflow[event_start.end() : event_start.end() + event_end.start()]
    )
    paths_start = re.search(r"^    paths:\n", event_block, re.MULTILINE)
    assert paths_start is not None
    paths_end = re.search(
        r"^    [a-z_]+:\n", event_block[paths_start.end() :], re.MULTILINE
    )
    paths_block = (
        event_block[paths_start.end() :]
        if paths_end is None
        else event_block[paths_start.end() : paths_start.end() + paths_end.start()]
    )
    return tuple(
        match.group("path")
        for match in re.finditer(
            r'^      - "(?P<path>[^"]+)"\s*$', paths_block, re.MULTILINE
        )
    )


@pytest.mark.parametrize(
    "migration_path",
    (
        "atlas_brain/storage/migrations/345_receivables_event_key_lookup.sql",
        "atlas_brain/storage/migrations/369_receivables_payment_receipt_outbox.sql",
        "atlas_brain/storage/migrations/378_receivables_payment_receipt_delivery.sql",
        "atlas_brain/storage/migrations/379_receivables_payment_receipt_delivery_recovery.sql",
    ),
)
def test_receivables_migrations_are_enrolled_in_invoicing_ci(migration_path):
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/atlas_invoicing_checks.yml"
    ).read_text(encoding="utf-8")
    entry = f'      - "{migration_path}"\n'

    assert migration_path in _workflow_event_paths(workflow, "pull_request")
    assert migration_path in _workflow_event_paths(workflow, "push")

    without_pull_enrollment = workflow.replace(entry, "", 1)
    assert migration_path not in _workflow_event_paths(
        without_pull_enrollment, "pull_request"
    )
    before_push_entry, separator, after_push_entry = workflow.rpartition(entry)
    assert separator == entry
    without_push_enrollment = before_push_entry + after_push_entry
    assert migration_path not in _workflow_event_paths(without_push_enrollment, "push")
    misplaced_under_branches = without_push_enrollment.replace(
        "  push:\n    branches:\n",
        "  push:\n    branches:\n" + entry,
        1,
    )
    assert migration_path not in _workflow_event_paths(
        misplaced_under_branches, "push"
    )


@pytest.mark.parametrize(
    "test_path",
    (
        "tests/test_eom_payment_receipts.py",
        "tests/test_residential_payment_receipt_delivery.py",
    ),
)
def test_payment_receipt_route_tests_are_enrolled_in_invoicing_ci(test_path):
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/atlas_invoicing_checks.yml"
    ).read_text(encoding="utf-8")
    assert test_path in _workflow_event_paths(workflow, "pull_request")
    assert test_path in _workflow_event_paths(workflow, "push")
    assert f"            {test_path} \\\n" in workflow


class _SingleConnectionPool:
    is_initialized = True

    def __init__(self, conn, schema: str) -> None:
        self.conn = conn
        self.schema = schema

    @asynccontextmanager
    async def transaction(self):
        async with self.conn.transaction():
            await self.conn.execute(f'SET LOCAL search_path TO "{self.schema}"')
            yield self.conn

    async def fetch(self, query, *args):
        async with self.conn.transaction():
            await self.conn.execute(f'SET LOCAL search_path TO "{self.schema}"')
            return await self.conn.fetch(query, *args)

    async def fetchrow(self, query, *args):
        async with self.conn.transaction():
            await self.conn.execute(f'SET LOCAL search_path TO "{self.schema}"')
            return await self.conn.fetchrow(query, *args)

    async def fetchval(self, query, *args):
        async with self.conn.transaction():
            await self.conn.execute(f'SET LOCAL search_path TO "{self.schema}"')
            return await self.conn.fetchval(query, *args)


class _PauseAfterContactLockConnection:
    def __init__(self, conn, contact_locked, release_contact_lock) -> None:
        self._conn = conn
        self._contact_locked = contact_locked
        self._release_contact_lock = release_contact_lock

    async def fetchrow(self, query, *args):
        row = await self._conn.fetchrow(query, *args)
        if "FROM contacts" in query:
            self._contact_locked.set()
            await self._release_contact_lock.wait()
        return row

    def __getattr__(self, name):
        return getattr(self._conn, name)


class _PauseAfterContactLockPool(_SingleConnectionPool):
    def __init__(self, conn, schema, contact_locked, release_contact_lock) -> None:
        super().__init__(conn, schema)
        self._contact_locked = contact_locked
        self._release_contact_lock = release_contact_lock

    @asynccontextmanager
    async def transaction(self):
        async with self.conn.transaction():
            await self.conn.execute(f'SET LOCAL search_path TO "{self.schema}"')
            yield _PauseAfterContactLockConnection(
                self.conn,
                self._contact_locked,
                self._release_contact_lock,
            )


class _MigrationConnectionPool:
    is_initialized = True

    def __init__(self, conn, schema: str) -> None:
        self.conn = conn
        self.schema = schema

    async def acquire(self):
        await self.conn.execute(f'SET search_path TO "{self.schema}"')
        return self.conn

    async def release(self, released) -> None:
        assert released is self.conn


async def _create_pre_receivables_schema(conn, schema: str) -> None:
    await conn.execute(f'CREATE SCHEMA "{schema}"')
    await conn.execute(f'SET search_path TO "{schema}"')
    await conn.execute(
        "CREATE TABLE contacts (id uuid PRIMARY KEY, business_context_id VARCHAR(64), "
        "contact_type VARCHAR(32) NOT NULL DEFAULT 'customer', "
        "status VARCHAR(32) NOT NULL DEFAULT 'active')"
    )
    invoice_migration = (
        Path(__file__).parents[1] / "atlas_brain/storage/migrations/045_invoices.sql"
    ).read_text(encoding="utf-8")
    await conn.execute(invoice_migration)


def _receivables_migration_sql(
    *,
    include_check_metadata: bool = True,
    include_receipt_outbox: bool = True,
) -> str:
    migrations = Path(__file__).parents[1] / "atlas_brain/storage/migrations"
    filenames = [
        "344_receivables_payments.sql",
        "345_receivables_event_key_lookup.sql",
    ]
    if include_check_metadata:
        filenames.append("368_receivables_payment_check_metadata.sql")
    if include_receipt_outbox:
        filenames.append("369_receivables_payment_receipt_outbox.sql")
    return "\n".join(
        (migrations / filename).read_text(encoding="utf-8")
        for filename in filenames
    )


def _packaged_migration_sql(migration_stem: str) -> str:
    return (
        Path(__file__).parents[1]
        / "atlas_brain/storage/migrations"
        / f"{migration_stem}.sql"
    ).read_text(encoding="utf-8")


async def _apply_receivables_migrations(
    conn,
    schema: str,
    *,
    include_check_metadata: bool = True,
    include_receipt_outbox: bool = True,
    include_receipt_dispatch: bool = True,
) -> None:
    """Apply the real receipt-aware migration set to an isolated test schema.

    Migration 378 contains concurrent index DDL and therefore must run through
    the production migration runner rather than the legacy raw-SQL fixture.
    Tests that model a pre-378 deployment continue to use
    ``_receivables_migration_sql`` directly and state that boundary explicitly.
    """
    if include_receipt_dispatch and not include_receipt_outbox:
        raise ValueError("Receipt dispatch requires the receipt-outbox migration")

    await conn.execute(
        _receivables_migration_sql(
            include_check_metadata=include_check_metadata,
            include_receipt_outbox=include_receipt_outbox,
        )
    )
    if not include_receipt_dispatch:
        return

    from atlas_brain.storage.migrations import run_migrations

    migrations_dir = Path(__file__).parents[1] / "atlas_brain/storage/migrations"
    await run_migrations(
        _MigrationConnectionPool(conn, schema),
        migrations_dir=migrations_dir,
        only={"378_receivables_payment_receipt_delivery"},
    )


async def _schema_foreign_key_relationships(conn, schema: str) -> set[tuple]:
    rows = await conn.fetch(
        """
        SELECT
            source_table.relname AS source_table,
            ARRAY_AGG(source_column.attname ORDER BY source_key.position)
                AS source_columns,
            target_table.relname AS target_table,
            ARRAY_AGG(target_column.attname ORDER BY target_key.position)
                AS target_columns,
            constraint_state.confdeltype::text AS delete_action
        FROM pg_constraint AS constraint_state
        JOIN pg_class AS source_table
          ON source_table.oid = constraint_state.conrelid
        JOIN pg_namespace AS source_namespace
          ON source_namespace.oid = source_table.relnamespace
        JOIN pg_class AS target_table
          ON target_table.oid = constraint_state.confrelid
        JOIN UNNEST(constraint_state.conkey) WITH ORDINALITY
            AS source_key(attnum, position)
          ON TRUE
        JOIN pg_attribute AS source_column
          ON source_column.attrelid = source_table.oid
         AND source_column.attnum = source_key.attnum
        JOIN UNNEST(constraint_state.confkey) WITH ORDINALITY
            AS target_key(attnum, position)
          ON target_key.position = source_key.position
        JOIN pg_attribute AS target_column
          ON target_column.attrelid = target_table.oid
         AND target_column.attnum = target_key.attnum
        WHERE source_namespace.nspname = $1
          AND constraint_state.contype = 'f'
        GROUP BY
            constraint_state.oid,
            source_table.relname,
            target_table.relname,
            constraint_state.confdeltype
        """,
        schema,
    )
    return {
        (
            row["source_table"],
            tuple(row["source_columns"]),
            row["target_table"],
            tuple(row["target_columns"]),
            row["delete_action"],
        )
        for row in rows
    }


def test_eom_readiness_migration_set_is_closed_over_receivables_dependencies():
    from atlas_brain import main_eom

    positions = {
        migration: index
        for index, migration in enumerate(main_eom.EOM_RECEIVABLES_READINESS_MIGRATIONS)
    }
    combined_sql = "\n".join(
        _packaged_migration_sql(migration)
        for migration in main_eom.EOM_RECEIVABLES_READINESS_MIGRATIONS
    )
    created_tables = set(
        re.findall(r"CREATE TABLE IF NOT EXISTS ([a-z_]+)", combined_sql)
    )

    assert set(_RECEIVABLES_REQUIRED_COLUMNS) <= created_tables
    assert set(_RECEIPT_DELIVERY_REQUIRED_COLUMNS) <= created_tables
    assert set(_RECEIPT_DISPATCH_REQUIRED_COLUMNS) <= created_tables
    assert positions["012_appointments"] < positions["035_contacts"]
    assert positions["035_contacts"] < positions["045_invoices"]
    assert positions["045_invoices"] < positions["344_receivables_payments"]
    assert (
        positions["344_receivables_payments"]
        < positions["345_receivables_event_key_lookup"]
    )
    assert (
        positions["345_receivables_event_key_lookup"]
        < positions["368_receivables_payment_check_metadata"]
    )
    assert (
        positions["368_receivables_payment_check_metadata"]
        < positions["369_receivables_payment_receipt_outbox"]
    )
    assert (
        positions["369_receivables_payment_receipt_outbox"]
        < positions["378_receivables_payment_receipt_delivery"]
    )
    assert (
        positions["378_receivables_payment_receipt_delivery"]
        < positions["379_receivables_payment_receipt_delivery_recovery"]
    )
    assert "ALTER TABLE appointments" in _packaged_migration_sql("035_contacts")
    assert "REFERENCES contacts(id)" in _packaged_migration_sql("045_invoices")
    assert "ALTER TABLE invoice_payments" in _packaged_migration_sql(
        "344_receivables_payments"
    )
    assert "ON payment_events(idempotency_key)" in _packaged_migration_sql(
        "345_receivables_event_key_lookup"
    )
    assert "ADD COLUMN IF NOT EXISTS check_date DATE" in _packaged_migration_sql(
        "368_receivables_payment_check_metadata"
    )
    receipt_outbox_sql = _packaged_migration_sql(
        "369_receivables_payment_receipt_outbox"
    )
    assert "CREATE TABLE IF NOT EXISTS payment_receipt_deliveries" in receipt_outbox_sql
    assert "UNIQUE (payment_id)" in receipt_outbox_sql
    assert "'pending', 'sent', 'failed', 'skipped'" in receipt_outbox_sql


@pytest.mark.asyncio
async def test_eom_receivables_readiness_migration_set_builds_ready_schema():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    from atlas_brain import main_eom
    from atlas_brain.storage.migrations import run_migrations

    schema = f"eom_receivables_readiness_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    migrations_dir = Path(__file__).parents[1] / "atlas_brain/storage/migrations"
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        pool = _MigrationConnectionPool(conn, schema)

        await run_migrations(
            pool,
            migrations_dir=migrations_dir,
            only=main_eom.EOM_RECEIVABLES_READINESS_MIGRATIONS,
        )

        applied_names = {
            row["name"]
            for row in await conn.fetch(
                """
                SELECT name
                FROM schema_migrations
                ORDER BY name
                """
            )
        }
        assert applied_names == set(main_eom.EOM_RECEIVABLES_READINESS_MIGRATIONS)
        assert not any(
            name.startswith(("066_", "068_", "074_", "076_", "083_", "095_"))
            for name in applied_names
        )
        service = ReceivablesService(_SingleConnectionPool(conn, schema))
        assert await service.is_ready() is True
        assert await service.is_receipt_delivery_ready() is True
        for table_name, required_columns in _RECEIPT_DISPATCH_REQUIRED_COLUMNS.items():
            actual_columns = {
                row["column_name"]
                for row in await conn.fetch(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = current_schema() AND table_name = $1
                    """,
                    table_name,
                )
            }
            expected_columns = set(required_columns)
            if table_name == "payment_receipt_deliveries":
                expected_columns |= set(_RECEIPT_DELIVERY_REQUIRED_COLUMNS[table_name])
            assert actual_columns == expected_columns

        async def assert_dispatch_schema_is_not_ready(drop_statement: str) -> None:
            transaction = conn.transaction()
            await transaction.start()
            try:
                await conn.execute(drop_statement)
                assert await service.is_ready() is True
                assert await service.is_receipt_delivery_ready() is False
            finally:
                await transaction.rollback()
            assert await service.is_receipt_delivery_ready() is True

        await assert_dispatch_schema_is_not_ready(
            "ALTER TABLE payment_receipt_deliveries "
            "DROP COLUMN rfc_message_id CASCADE"
        )
        await assert_dispatch_schema_is_not_ready(
            "DROP TABLE payment_receipt_delivery_operations CASCADE"
        )
        await assert_dispatch_schema_is_not_ready(
            "DROP TABLE payment_receipt_delivery_reconciliation_events"
        )
        for table_name, index_name, *_ignored, constraint_type in (
            _RECEIPT_DISPATCH_REQUIRED_INDEXES
        ):
            drop_statement = (
                f'ALTER TABLE "{table_name}" DROP CONSTRAINT "{index_name}" CASCADE'
                if constraint_type is not None
                else f'DROP INDEX "{index_name}"'
            )
            await assert_dispatch_schema_is_not_ready(drop_statement)
        expected_foreign_keys = {
            ("appointments", ("contact_id",), "contacts", ("id",), "n"),
            ("contact_interactions", ("contact_id",), "contacts", ("id",), "c"),
            ("invoices", ("contact_id",), "contacts", ("id",), "n"),
            ("invoice_payments", ("invoice_id",), "invoices", ("id",), "c"),
            ("customer_payments", ("contact_id",), "contacts", ("id",), "n"),
            (
                "invoice_payments",
                ("payment_id",),
                "customer_payments",
                ("id",),
                "r",
            ),
            (
                "payment_deposit_items",
                ("batch_id",),
                "payment_deposit_batches",
                ("id",),
                "r",
            ),
            (
                "payment_deposit_items",
                ("payment_id",),
                "customer_payments",
                ("id",),
                "r",
            ),
            ("payment_events", ("payment_id",), "customer_payments", ("id",), "r"),
            (
                "payment_receipt_deliveries",
                ("payment_id",),
                "customer_payments",
                ("id",),
                "r",
            ),
            (
                "payment_receipt_deliveries",
                ("contact_id",),
                "contacts",
                ("id",),
                "r",
            ),
            (
                "payment_receipt_delivery_operations",
                ("receipt_delivery_id",),
                "payment_receipt_deliveries",
                ("id",),
                "r",
            ),
        }
        assert expected_foreign_keys <= await _schema_foreign_key_relationships(
            conn,
            schema,
        )

        await run_migrations(
            pool,
            migrations_dir=migrations_dir,
            only=main_eom.EOM_RECEIVABLES_READINESS_MIGRATIONS,
        )
        assert {
            row["name"]
            for row in await conn.fetch(
                """
                SELECT name
                FROM schema_migrations
                ORDER BY name
                """
            )
        } == applied_names
        assert await service.is_ready() is True
        assert await service.is_receipt_delivery_ready() is True
        for column, definition in (
            ("check_date", "DATE"),
            ("received_through", "VARCHAR(128)"),
        ):
            await conn.execute(
                f"ALTER TABLE customer_payments DROP COLUMN {column}"
            )
            assert await service.is_ready() is False
            await conn.execute(
                f"ALTER TABLE customer_payments ADD COLUMN {column} {definition}"
            )
            assert await service.is_ready() is True
            assert await service.is_receipt_delivery_ready() is True

        await conn.execute("DROP TABLE payment_receipt_deliveries CASCADE")
        # The global/full-MCP readiness contract deliberately remains intact:
        # it has no outbox writer.  The slim EOM receipt capability alone must
        # fail closed until its additive migration is restored.
        assert await service.is_ready() is True
        assert await service.is_receipt_delivery_ready() is False
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_recorded_378_recovery_restores_receipt_delivery_readiness():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    from atlas_brain import main_eom
    from atlas_brain.storage.migrations import run_migrations

    recovery_migration = "379_receivables_payment_receipt_delivery_recovery"
    pre_recovery_migrations = tuple(
        migration
        for migration in main_eom.EOM_RECEIVABLES_READINESS_MIGRATIONS
        if migration != recovery_migration
    )
    recovery_sql = _packaged_migration_sql(recovery_migration)
    schema = f"eom_receivables_recorded_378_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    migrations_dir = Path(__file__).parents[1] / "atlas_brain/storage/migrations"
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        pool = _MigrationConnectionPool(conn, schema)
        await run_migrations(
            pool,
            migrations_dir=migrations_dir,
            only=pre_recovery_migrations,
        )

        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations WHERE name = $1",
            "378_receivables_payment_receipt_delivery",
        ) == 1
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations WHERE name = $1",
            recovery_migration,
        ) == 0

        contact_id, payment_id, receipt_id, operation_id = (
            uuid4(),
            uuid4(),
            uuid4(),
            uuid4(),
        )
        await conn.execute(
            "INSERT INTO contacts (id, full_name, business_context_id) "
            "VALUES ($1, 'Migration Recovery Customer', 'effingham_maids')",
            contact_id,
        )
        await conn.execute(
            """
            INSERT INTO customer_payments (
                id, contact_id, payer_name, total_amount, payment_method,
                received_date, status, source
            ) VALUES ($1, $2, 'Migration Recovery Customer', 125.25, 'check',
                      DATE '2026-08-16', 'received', 'migration_recovery_test')
            """,
            payment_id,
            contact_id,
        )
        await conn.execute(
            """
            INSERT INTO payment_receipt_deliveries (
                id, payment_id, contact_id, receipt_number, recipient_email,
                delivery_status, skip_reason, subject, body
            ) VALUES ($1, $2, $3, 'RCP-MIGRATION-RECOVERY',
                      'customer@example.test', 'pending', NULL,
                      'Payment receipt', 'Payment receipt body')
            """,
            receipt_id,
            payment_id,
            contact_id,
        )
        await conn.execute(
            """
            INSERT INTO payment_receipt_delivery_operations (
                id, receipt_delivery_id, source, idempotency_key,
                request_fingerprint, state, requested_by
            ) VALUES ($1, $2, 'eom_admin', 'migration-recovery-operation',
                      $3, 'prepared', 'migration-recovery-test')
            """,
            operation_id,
            receipt_id,
            "0" * 64,
        )

        async def fact_snapshot() -> tuple[tuple, tuple, tuple]:
            payment = await conn.fetchrow(
                """
                SELECT id, contact_id, total_amount, status, source, received_date
                FROM customer_payments WHERE id = $1
                """,
                payment_id,
            )
            receipt = await conn.fetchrow(
                """
                SELECT id, payment_id, contact_id, receipt_number, recipient_email,
                       delivery_status, skip_reason, subject, body, rfc_message_id
                FROM payment_receipt_deliveries WHERE id = $1
                """,
                receipt_id,
            )
            operation = await conn.fetchrow(
                """
                SELECT id, receipt_delivery_id, source, idempotency_key,
                       request_fingerprint, state, outcome, requested_by
                FROM payment_receipt_delivery_operations WHERE id = $1
                """,
                operation_id,
            )
            return tuple(payment), tuple(receipt), tuple(operation)

        facts_before = await fact_snapshot()

        # This is the schema shape created by the original 378 revision that
        # was later recorded before the result/reconciliation DDL landed.
        await conn.execute("DROP TABLE payment_receipt_delivery_reconciliation_events")
        await conn.execute(
            "ALTER TABLE payment_receipt_delivery_operations "
            "DROP CONSTRAINT payment_receipt_delivery_operations_result_shape_check"
        )
        await conn.execute(
            "ALTER TABLE payment_receipt_delivery_operations "
            "DROP COLUMN result_delivery_status, DROP COLUMN result_sent_at"
        )

        service = ReceivablesService(_SingleConnectionPool(conn, schema))
        assert await service.is_ready() is True
        assert await service.is_receipt_delivery_ready() is False

        assert recovery_sql.startswith("-- atlas: atomic-bookkeeping")
        assert not re.search(
            r"(?im)^(?:UPDATE|INSERT|DELETE|DROP)\b", recovery_sql
        )

        await run_migrations(
            pool,
            migrations_dir=migrations_dir,
            only=main_eom.EOM_RECEIVABLES_READINESS_MIGRATIONS,
        )

        applied_names = {
            row["name"]
            for row in await conn.fetch("SELECT name FROM schema_migrations")
        }
        assert applied_names == set(main_eom.EOM_RECEIVABLES_READINESS_MIGRATIONS)
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations WHERE name = $1",
            "378_receivables_payment_receipt_delivery",
        ) == 1
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations WHERE name = $1",
            recovery_migration,
        ) == 1
        assert await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conrelid = 'payment_receipt_delivery_operations'::regclass
                  AND conname =
                      'payment_receipt_delivery_operations_result_shape_check'
            )
            """
        ) is True
        assert await service.is_ready() is True
        assert await service.is_receipt_delivery_ready() is True

        assert await fact_snapshot() == facts_before
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM payment_receipt_delivery_reconciliation_events"
        ) == 0

        await run_migrations(
            pool,
            migrations_dir=migrations_dir,
            only=main_eom.EOM_RECEIVABLES_READINESS_MIGRATIONS,
        )
        assert {
            row["name"]
            for row in await conn.fetch("SELECT name FROM schema_migrations")
        } == applied_names
        assert await service.is_receipt_delivery_ready() is True
        assert await fact_snapshot() == facts_before
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "check_metadata",
    (
        {"check_date": date(2026, 8, 10)},
        {"received_through": "Mail"},
    ),
)
async def test_check_metadata_schema_gate_preserves_legacy_payment_creation(
    check_metadata,
):
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"receivables_check_metadata_gate_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _create_pre_receivables_schema(conn, schema)
        await conn.execute(_receivables_migration_sql(include_check_metadata=False))
        contact_id = uuid4()
        await conn.execute(
            "INSERT INTO contacts (id, business_context_id) "
            "VALUES ($1, 'effingham_maids')",
            contact_id,
        )
        service = ReceivablesService(_SingleConnectionPool(conn, schema))

        assert await service.is_ready() is False
        legacy_payment = await service.create_payment(
            contact_id=contact_id,
            payer_name="Residential Customer",
            total_amount=Decimal("125.00"),
            payment_method="check",
            received_date=date(2026, 8, 12),
            allocations=[],
            idempotency_key="pre-368-legacy-payment",
            recorded_by="Juan Canfield",
            allow_unapplied=True,
            unapplied_contact_context_id="effingham_maids",
        )
        assert legacy_payment["status"] == "received"

        with pytest.raises(
            ReceivablesSchemaUnavailableError,
            match="schema unavailable for check metadata",
        ):
            await service.create_payment(
                contact_id=contact_id,
                payer_name="Residential Customer",
                total_amount=Decimal("125.00"),
                payment_method="check",
                received_date=date(2026, 8, 12),
                allocations=[],
                idempotency_key="pre-368-check-metadata-payment",
                recorded_by="Juan Canfield",
                allow_unapplied=True,
                unapplied_contact_context_id="effingham_maids",
                **check_metadata,
            )

        assert await conn.fetchval("SELECT COUNT(*) FROM customer_payments") == 1
        assert await conn.fetchval("SELECT COUNT(*) FROM payment_events") == 1
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_receipt_required_replay_recovers_legacy_payment_without_outbox_schema():
    """A CRM outage cannot turn a valid old payment retry into a table error."""
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"receivables_receipt_legacy_replay_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _create_pre_receivables_schema(conn, schema)
        await conn.execute(_receivables_migration_sql(include_receipt_outbox=False))
        contact_id = uuid4()
        await conn.execute(
            "INSERT INTO contacts (id, business_context_id) "
            "VALUES ($1, 'effingham_maids')",
            contact_id,
        )
        service = ReceivablesService(_SingleConnectionPool(conn, schema))
        create_args = {
            "contact_id": contact_id,
            "payer_name": "Residential Customer",
            "total_amount": Decimal("125.00"),
            "payment_method": "check",
            "received_date": date(2026, 8, 12),
            "allocations": [],
            "idempotency_key": "legacy-payment-replay-without-outbox-1",
            "recorded_by": "Juan Canfield",
            "allow_unapplied": True,
            "unapplied_contact_context_id": "effingham_maids",
        }
        original = await service.create_payment_with_outcome(**create_args)
        replay = await service.create_payment_with_outcome(
            **{**create_args, "require_receipt_recipient": True}
        )

        assert original.replayed is False
        assert replay.replayed is True
        assert replay.payment["id"] == original.payment["id"]
        assert "receipt_delivery" not in replay.payment
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM customer_payments "
                "WHERE idempotency_key = 'legacy-payment-replay-without-outbox-1'"
            )
            == 1
        )
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_real_migration_backfills_and_adopts_late_rolling_writer_checks():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"receivables_migration_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _create_pre_receivables_schema(conn, schema)
        contact_id = uuid4()
        invoice_a, invoice_b = uuid4(), uuid4()
        await conn.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        await conn.executemany(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, total_amount,
                due_date, status
            ) VALUES ($1, $2, $3, 'Acme', 100, CURRENT_DATE + 30, 'sent')
            """,
            [
                (invoice_a, "INV-MIG-A", contact_id),
                (invoice_b, "INV-MIG-B", contact_id),
            ],
        )
        legacy_a, legacy_b = uuid4(), uuid4()
        await conn.executemany(
            """
            INSERT INTO invoice_payments (
                id, invoice_id, amount, payment_date, payment_method,
                reference, created_at
            ) VALUES ($1, $2, 10, CURRENT_DATE, 'check', 'same-reference', NOW())
            """,
            [(legacy_a, invoice_a), (legacy_b, invoice_b)],
        )
        pre_migration_invoice_payment_columns = {
            row["column_name"]
            for row in await conn.fetch(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'invoice_payments'
                """
            )
        }

        migration_sql = _receivables_migration_sql()
        dispatch_migration_sql = _packaged_migration_sql(
            "378_receivables_payment_receipt_delivery"
        )
        await _apply_receivables_migrations(conn, schema)

        created_tables = set(
            re.findall(r"CREATE TABLE IF NOT EXISTS ([a-z_]+)", migration_sql)
        ) | set(
            re.findall(
                r"CREATE TABLE IF NOT EXISTS ([a-z_]+)", dispatch_migration_sql
            )
        )
        # The receipt outbox is an EOM-only additive capability.  It is
        # intentionally absent from the legacy/full-MCP readiness set, so a
        # mixed rollout cannot make established financial entrypoints claim
        # their schema is incomplete solely because the EOM outbox has not
        # reached that database yet.
        receipt_delivery_tables = set(_RECEIPT_DELIVERY_REQUIRED_COLUMNS) | set(
            _RECEIPT_DISPATCH_REQUIRED_COLUMNS
        )
        assert set(_RECEIVABLES_REQUIRED_COLUMNS) == (
            created_tables - receipt_delivery_tables
        ) | {"invoice_payments"}
        assert "payment_receipt_deliveries" in created_tables

        for table_name, required_columns in _RECEIVABLES_REQUIRED_COLUMNS.items():
            actual_columns = {
                row["column_name"]
                for row in await conn.fetch(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = $1
                    """,
                    table_name,
                )
            }
            if table_name == "invoice_payments":
                actual_columns -= pre_migration_invoice_payment_columns
            assert actual_columns == set(required_columns)

        for table_name, required_columns in _RECEIPT_DISPATCH_REQUIRED_COLUMNS.items():
            actual_columns = {
                row["column_name"]
                for row in await conn.fetch(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = $1
                    """,
                    table_name,
                )
            }
            expected_columns = set(required_columns)
            if table_name == "payment_receipt_deliveries":
                expected_columns |= set(_RECEIPT_DELIVERY_REQUIRED_COLUMNS[table_name])
            assert actual_columns == expected_columns

        backfilled = await conn.fetch(
            """
            SELECT ip.id, ip.payment_id, cp.id AS parent_id, cp.status
            FROM invoice_payments ip
            JOIN customer_payments cp ON cp.id = ip.payment_id
            WHERE ip.id = ANY($1::uuid[])
            ORDER BY ip.id
            """,
            [legacy_a, legacy_b],
        )
        assert len(backfilled) == 2
        assert all(
            row["id"] == row["payment_id"] == row["parent_id"] for row in backfilled
        )
        assert {row["status"] for row in backfilled} == {"legacy"}

        late_check, malformed_check = uuid4(), uuid4()
        await conn.execute(
            """
            INSERT INTO invoice_payments (
                id, invoice_id, amount, payment_date, payment_method,
                reference, recorded_by, created_at, metadata
            ) VALUES
                ($1, $3, 25, CURRENT_DATE, ' Check ', 'late-1001', 'old-node', NOW(), '{}'),
                ($2, $3, 0, CURRENT_DATE, 'Check', 'bad-zero', 'old-node', NOW(), '{}')
            """,
            late_check,
            malformed_check,
            invoice_a,
        )
        adopted = await conn.fetch(
            """
            SELECT ip.id, ip.payment_id, cp.status, cp.payment_method,
                   cp.total_amount, cp.metadata
            FROM invoice_payments ip
            JOIN customer_payments cp ON cp.id = ip.payment_id
            WHERE ip.id = ANY($1::uuid[])
            ORDER BY cp.total_amount DESC
            """,
            [late_check, malformed_check],
        )
        assert adopted[0]["id"] == adopted[0]["payment_id"] == late_check
        assert adopted[0]["status"] == "received"
        assert adopted[0]["payment_method"] == "check"
        assert json.loads(adopted[0]["metadata"])["adopted_rolling_writer"] is True
        assert adopted[1]["status"] == "legacy"

        service = ReceivablesService(_SingleConnectionPool(conn, schema))
        received = await service.list_payments(status="received")
        assert [item["id"] for item in received] == [str(late_check)]
        batch = await service.create_deposit_batch(
            payment_ids=[late_check],
            deposit_date=date.today(),
            bank_reference="late-writer-deposit",
            actor="Migration test",
            idempotency_key="late-writer-batch",
        )
        assert batch["payment_count"] == 1
        assert batch["payments"][0]["id"] == str(late_check)

        assert await service.is_ready() is True
        explicit_migration_index_names = set(
            re.findall(
                r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?"
                r"(?:IF\s+NOT\s+EXISTS\s+)?([a-z_]+)",
                f"{migration_sql}\n{dispatch_migration_sql}",
            )
        )
        migration_table_names = created_tables
        constraint_index_rows = await conn.fetch(
            """
            SELECT
                table_class.relname AS table_name,
                constraint_state.conname AS index_name
            FROM pg_constraint AS constraint_state
            JOIN pg_class AS table_class
              ON table_class.oid = constraint_state.conrelid
            JOIN pg_namespace AS table_namespace
              ON table_namespace.oid = table_class.relnamespace
            WHERE table_namespace.nspname = current_schema()
              AND table_class.relname = ANY($1::text[])
              AND constraint_state.contype IN ('p', 'u')
              AND constraint_state.conindid <> 0
            """,
            sorted(migration_table_names),
        )
        constraint_indexes = {
            row["index_name"]: row["table_name"] for row in constraint_index_rows
        }
        migration_index_names = explicit_migration_index_names | set(
            constraint_indexes
        )
        required_index_names = {
            index_name
            for _table_name, index_name, *_rest in _RECEIVABLES_REQUIRED_INDEXES
        }
        receipt_delivery_index_names = {
            index_name
            for _table_name, index_name, *_rest in _RECEIPT_DELIVERY_REQUIRED_INDEXES
        }
        receipt_dispatch_index_names = {
            index_name
            for _table_name, index_name, *_rest in _RECEIPT_DISPATCH_REQUIRED_INDEXES
        }
        assert (
            required_index_names
            | receipt_delivery_index_names
            | receipt_dispatch_index_names
            == migration_index_names
        )
        for index_name in sorted(required_index_names):
            transaction = conn.transaction()
            await transaction.start()
            try:
                if index_name in constraint_indexes:
                    table_name = constraint_indexes[index_name]
                    await conn.execute(
                        f'ALTER TABLE "{table_name}" '
                        f'DROP CONSTRAINT "{index_name}" CASCADE'
                    )
                else:
                    await conn.execute(f'DROP INDEX "{index_name}"')
                assert await service.is_ready() is False
            finally:
                await transaction.rollback()
            assert await service.is_ready() is True

        for index_name in sorted(receipt_delivery_index_names):
            transaction = conn.transaction()
            await transaction.start()
            try:
                if index_name in constraint_indexes:
                    table_name = constraint_indexes[index_name]
                    await conn.execute(
                        f'ALTER TABLE "{table_name}" '
                        f'DROP CONSTRAINT "{index_name}" CASCADE'
                    )
                else:
                    await conn.execute(f'DROP INDEX "{index_name}"')
                assert await service.is_ready() is True
                assert await service.is_receipt_delivery_ready() is False
            finally:
                await transaction.rollback()
            assert await service.is_receipt_delivery_ready() is True

        await conn.execute(
            "ALTER TABLE payment_deposit_batches "
            "DROP COLUMN clear_request_fingerprint"
        )
        assert await service.is_ready() is False
        await conn.execute(
            "ALTER TABLE payment_deposit_batches "
            "ADD COLUMN clear_request_fingerprint VARCHAR(64)"
        )
        assert await service.is_ready() is True
        await conn.execute(
            "ALTER TABLE invoice_payments DROP COLUMN reversal_reason"
        )
        assert await service.is_ready() is False
        await conn.execute(
            "ALTER TABLE invoice_payments ADD COLUMN reversal_reason TEXT"
        )
        assert await service.is_ready() is True
        await conn.execute("DROP TABLE payment_events")
        assert await service.is_ready() is False
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_real_postgres_customer_ledger_is_bounded_receipt_aware_and_read_only():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"receivables_customer_ledger_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _create_pre_receivables_schema(conn, schema)
        await _apply_receivables_migrations(conn, schema)
        contact_id, other_contact_id = uuid4(), uuid4()
        invoice_a, invoice_b, other_invoice = uuid4(), uuid4(), uuid4()
        await conn.executemany(
            "INSERT INTO contacts (id, business_context_id) VALUES ($1, 'effingham_maids')",
            [(contact_id,), (other_contact_id,)],
        )
        await conn.executemany(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, total_amount,
                issue_date, due_date, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, 'sent')
            """,
            [
                (
                    invoice_a,
                    "INV-LEDGER-ALPHA",
                    contact_id,
                    "Ledger Customer",
                    Decimal("200.00"),
                    date(2026, 8, 1),
                    date(2026, 8, 31),
                ),
                (
                    invoice_b,
                    "INV-LEDGER-BRAVO",
                    contact_id,
                    "Ledger Customer",
                    Decimal("100.00"),
                    date(2026, 8, 2),
                    date(2026, 9, 1),
                ),
                (
                    other_invoice,
                    "INV-OTHER-CUSTOMER",
                    other_contact_id,
                    "Other Customer",
                    Decimal("400.00"),
                    date(2026, 8, 3),
                    date(2026, 9, 2),
                ),
            ],
        )
        service = ReceivablesService(_SingleConnectionPool(conn, schema))

        check = await service.create_payment(
            contact_id=contact_id,
            payer_name="Ledger Check Payer",
            total_amount=Decimal("80.00"),
            payment_method="check",
            received_date=date(2026, 8, 10),
            allocations=[{"invoice_id": invoice_a, "amount": Decimal("50.00")}],
            reference="CHECK_1042",
            idempotency_key="customer-ledger-check-1",
            recorded_by="Ledger test",
            allow_unapplied=True,
            unapplied_contact_context_id="effingham_maids",
            receipt_recipient=PaymentReceiptRecipient(
                contact_id=contact_id,
                customer_name="Ledger Customer",
                customer_type="residential",
                recipient_email="ledger@example.test",
            ),
            require_receipt_recipient=True,
        )
        batch = await service.create_deposit_batch(
            payment_ids=[UUID(check["id"])],
            deposit_date=date(2026, 8, 11),
            bank_reference="ledger-deposit-1",
            actor="Ledger test",
            idempotency_key="customer-ledger-deposit-1",
        )
        await service.clear_deposit_batch(
            batch_id=UUID(batch["id"]),
            actor="Ledger test",
            idempotency_key="customer-ledger-clear-1",
        )
        returned = await service.create_payment(
            contact_id=contact_id,
            payer_name="Ledger ACH Payer",
            total_amount=Decimal("30.00"),
            payment_method="ach",
            received_date=date(2026, 8, 11),
            allocations=[{"invoice_id": invoice_a, "amount": Decimal("30.00")}],
            reference="SQUAREXTXN-9001",
            idempotency_key="customer-ledger-ach-1",
            recorded_by="Ledger test",
        )
        await service.return_payment(
            payment_id=UUID(returned["id"]),
            reason="NSF",
            actor="Ledger test",
            idempotency_key="customer-ledger-return-1",
        )
        square = await service.create_payment(
            contact_id=contact_id,
            payer_name="Ledger Square Payer",
            total_amount=Decimal("20.00"),
            payment_method="square",
            received_date=date(2026, 8, 12),
            allocations=[],
            reference="SQUARE%TXN-9001",
            idempotency_key="customer-ledger-square-1",
            recorded_by="Ledger test",
            allow_unapplied=True,
            unapplied_contact_context_id="effingham_maids",
        )
        other_payment = await service.create_payment(
            contact_id=other_contact_id,
            payer_name="Other Ledger Payer",
            total_amount=Decimal("40.00"),
            payment_method="check",
            received_date=date(2026, 8, 12),
            allocations=[{"invoice_id": other_invoice, "amount": Decimal("40.00")}],
            reference="OTHER-CHECK-1042",
            idempotency_key="customer-ledger-other-payment-1",
            recorded_by="Ledger test",
        )
        backdated = await service.create_payment(
            contact_id=contact_id,
            payer_name=r"Ledger\Backdated Payer",
            total_amount=Decimal("10.00"),
            payment_method="ach",
            received_date=date(2026, 7, 31),
            allocations=[],
            reference="CHECKX1042",
            idempotency_key="customer-ledger-backdated-payment-1",
            recorded_by="Ledger test",
            allow_unapplied=True,
            unapplied_contact_context_id="effingham_maids",
        )
        before = {
            table: await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            for table in (
                "customer_payments",
                "invoice_payments",
                "payment_receipt_deliveries",
                "payment_events",
            )
        }

        receipt_number = check["receipt_delivery"]["receipt_number"]
        receipt_search = await service.list_customer_ledger(
            contact_id=contact_id,
            search=receipt_number,
        )
        [receipt_entry] = receipt_search["entries"]
        assert receipt_entry["entry_type"] == "payment"
        assert receipt_entry["payment"]["id"] == check["id"]
        assert receipt_entry["payment"]["status"] == "cleared"
        assert receipt_entry["payment"]["batch_id"] == batch["id"]
        assert receipt_entry["payment"]["receipt_delivery"] == {
            "receipt_number": receipt_number,
            "recipient_email": "ledger@example.test",
            "status": "pending",
            "skip_reason": None,
        }
        assert receipt_search["balances"] == {
            "open_invoice_balance_cents": 25_000,
            "unapplied_payment_balance_cents": 6_000,
        }

        payer_search = await service.list_customer_ledger(
            contact_id=contact_id,
            search="ledger check payer",
        )
        assert [entry["payment"]["id"] for entry in payer_search["entries"]] == [
            check["id"]
        ]
        check_reference_search = await service.list_customer_ledger(
            contact_id=contact_id,
            search="check_1042",
        )
        assert [
            entry["payment"]["id"]
            for entry in check_reference_search["entries"]
        ] == [check["id"]]
        percent_reference_search = await service.list_customer_ledger(
            contact_id=contact_id,
            search="square%txn-9001",
        )
        assert [
            entry["payment"]["id"]
            for entry in percent_reference_search["entries"]
        ] == [square["id"]]
        backslash_payer_search = await service.list_customer_ledger(
            contact_id=contact_id,
            search=r"ledger\backdated payer",
        )
        assert [
            entry["payment"]["id"]
            for entry in backslash_payer_search["entries"]
        ] == [backdated["id"]]
        receipt_casefold_search = await service.list_customer_ledger(
            contact_id=contact_id,
            search=receipt_number.lower(),
        )
        assert [
            entry["payment"]["id"]
            for entry in receipt_casefold_search["entries"]
        ] == [check["id"]]
        other_payment_search = await service.list_customer_ledger(
            contact_id=contact_id,
            search="other ledger payer",
        )
        assert other_payment_search["entries"] == []
        full_customer_ledger = await service.list_customer_ledger(
            contact_id=contact_id,
        )
        assert other_payment["id"] not in {
            entry["entry_id"] for entry in full_customer_ledger["entries"]
        }
        assert [entry["occurred_date"] for entry in full_customer_ledger["entries"]] == sorted(
            (entry["occurred_date"] for entry in full_customer_ledger["entries"]),
            reverse=True,
        )
        assert full_customer_ledger["entries"][-1]["payment"]["id"] == backdated["id"]

        invoice_search = await service.list_customer_ledger(
            contact_id=contact_id,
            search="inv-ledger-alpha",
        )
        assert {entry["entry_type"] for entry in invoice_search["entries"]} == {
            "invoice",
            "payment",
        }
        assert all(
            entry["entry_id"] != str(other_invoice)
            for entry in invoice_search["entries"]
        )
        assert {
            entry["payment"]["id"]
            for entry in invoice_search["entries"]
            if entry["entry_type"] == "payment"
        } == {
            check["id"],
            returned["id"],
        }

        returned_only = await service.list_customer_ledger(
            contact_id=contact_id,
            payment_status="returned",
        )
        assert len(returned_only["entries"]) == 1
        returned_entry = returned_only["entries"][0]
        assert returned_entry["entry_type"] == "payment"
        assert returned_entry["entry_id"] == returned["id"]
        assert returned_entry["payment"]["return_reason"] == "NSF"
        assert returned_entry["payment"]["unapplied_amount_cents"] == 0

        square_only = await service.list_customer_ledger(
            contact_id=contact_id,
            payment_method="square",
            search="SQUARE%TXN-9001",
            from_date=date(2026, 8, 12),
            to_date=date(2026, 8, 12),
        )
        assert [entry["payment"]["id"] for entry in square_only["entries"]] == [
            square["id"]
        ]

        with pytest.raises(ReceivablesValidationError, match="status cannot be blank"):
            await service.list_customer_ledger(
                contact_id=contact_id,
                payment_status="   ",
            )
        with pytest.raises(ReceivablesValidationError, match="method cannot be blank"):
            await service.list_customer_ledger(
                contact_id=contact_id,
                payment_method="\t",
            )

        first_page = await service.list_customer_ledger(
            contact_id=contact_id,
            limit=1,
        )
        second_page = await service.list_customer_ledger(
            contact_id=contact_id,
            limit=1,
            offset=first_page["next_offset"],
        )
        assert first_page["next_offset"] == 1
        assert (
            second_page["entries"][0]["entry_id"]
            != first_page["entries"][0]["entry_id"]
        )
        assert receipt_search == await service.list_customer_ledger(
            contact_id=contact_id,
            search=receipt_number,
        )
        assert before == {
            table: await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            for table in before
        }

        with pytest.raises(ReceivablesValidationError, match="start date"):
            await service.list_customer_ledger(
                contact_id=contact_id,
                from_date=date(2026, 8, 13),
                to_date=date(2026, 8, 12),
            )
        assert before == {
            table: await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            for table in before
        }

        await conn.execute("DROP TABLE payment_receipt_deliveries CASCADE")
        with pytest.raises(ReceivablesSchemaUnavailableError, match="customer ledger"):
            await service.list_customer_ledger(contact_id=contact_id)
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_real_postgres_customer_ledger_bounds_allocation_history_and_preserves_active_total():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"receivables_ledger_allocation_history_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _create_pre_receivables_schema(conn, schema)
        await _apply_receivables_migrations(conn, schema)
        contact_id, invoice_id, payment_id = uuid4(), uuid4(), uuid4()
        await conn.execute(
            "INSERT INTO contacts (id, business_context_id) VALUES ($1, 'effingham_maids')",
            contact_id,
        )
        await conn.execute(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, total_amount,
                issue_date, due_date, status
            ) VALUES ($1, 'INV-ALLOCATION-HISTORY', $2, 'History Customer', 200,
                      DATE '2026-08-01', DATE '2026-08-31', 'sent')
            """,
            invoice_id,
            contact_id,
        )
        await conn.execute(
            """
            INSERT INTO customer_payments (
                id, contact_id, payer_name, total_amount, payment_method,
                received_date, status, source, idempotency_key,
                request_fingerprint, recorded_by
            ) VALUES ($1, $2, 'Allocation History Payer', 100, 'ach',
                      DATE '2026-08-12', 'received', 'test',
                      'allocation-history-payment', 'allocation-history-fingerprint',
                      'Ledger test')
            """,
            payment_id,
            contact_id,
        )
        historical_at = datetime(2026, 8, 12, tzinfo=timezone.utc)
        history_rows = [
            (
                uuid4(),
                invoice_id,
                payment_id,
                Decimal("1.00"),
                date(2026, 8, 12),
                "ach",
                "ALLOCATION-HISTORY",
                "historical adjustment",
                "Ledger test",
                historical_at if index else None,
                "replaced allocation" if index else None,
            )
            for index in range(101)
        ]
        await conn.executemany(
            """
            INSERT INTO invoice_payments (
                id, invoice_id, payment_id, amount, payment_date,
                payment_method, reference, notes, recorded_by, reversed_at,
                reversal_reason
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            history_rows,
        )
        service = ReceivablesService(_SingleConnectionPool(conn, schema))

        ledger = await service.list_customer_ledger(
            contact_id=contact_id,
            payment_method="ach",
        )

        [entry] = ledger["entries"]
        payment = entry["payment"]
        assert payment["id"] == str(payment_id)
        assert payment["allocated_amount_cents"] == 100
        assert payment["unapplied_amount_cents"] == 9_900
        assert payment["allocation_history_count"] == 101
        assert payment["allocations_truncated"] is True
        assert len(payment["allocations"]) == 100
        assert payment["allocations"][0]["reversed_at"] is None
        assert all(
            "active_allocated_amount" not in allocation
            and "allocation_history_count" not in allocation
            for allocation in payment["allocations"]
        )
        assert all(
            allocation["reversed_at"] is not None
            for allocation in payment["allocations"][1:]
        )
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_real_postgres_customer_ledger_aggregate_balances_exceed_one_row_money_cap():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"receivables_ledger_aggregate_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _create_pre_receivables_schema(conn, schema)
        await _apply_receivables_migrations(conn, schema)
        contact_id = uuid4()
        await conn.execute(
            "INSERT INTO contacts (id, business_context_id) VALUES ($1, 'effingham_maids')",
            contact_id,
        )
        amount = Decimal("5000000000.00")
        await conn.executemany(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, total_amount,
                issue_date, due_date, status
            ) VALUES ($1, $2, $3, 'Aggregate Customer', $4,
                      DATE '2026-08-01', DATE '2026-08-31', 'sent')
            """,
            [
                (uuid4(), "INV-LEDGER-AGGREGATE-A", contact_id, amount),
                (uuid4(), "INV-LEDGER-AGGREGATE-B", contact_id, amount),
            ],
        )
        await conn.executemany(
            """
            INSERT INTO customer_payments (
                id, contact_id, payer_name, total_amount, payment_method,
                received_date, status, source
            ) VALUES ($1, $2, $3, $4, 'ach', DATE '2026-08-02', 'received', 'test')
            """,
            [
                (uuid4(), contact_id, "Aggregate Payer A", amount),
                (uuid4(), contact_id, "Aggregate Payer B", amount),
            ],
        )

        ledger = await ReceivablesService(
            _SingleConnectionPool(conn, schema)
        ).list_customer_ledger(contact_id=contact_id)

        assert ledger["balances"] == {
            "open_invoice_balance_cents": 1_000_000_000_000,
            "unapplied_payment_balance_cents": 1_000_000_000_000,
        }
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_real_postgres_receivables_lifecycle_and_concurrent_replays():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"receivables_e2e_{uuid4().hex}"
    observer = await asyncpg.connect(database_url)
    conn_a = await asyncpg.connect(database_url)
    conn_b = await asyncpg.connect(database_url)
    try:
        await _create_pre_receivables_schema(observer, schema)
        await _apply_receivables_migrations(observer, schema)
        contact_id = uuid4()
        invoice_ids = [uuid4() for _ in range(4)]
        await observer.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        await observer.executemany(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, total_amount,
                due_date, status
            ) VALUES ($1, $2, $3, 'Acme', $4, CURRENT_DATE + 30, 'sent')
            """,
            [
                (invoice_ids[0], "INV-E2E-1", contact_id, Decimal("60")),
                (invoice_ids[1], "INV-E2E-2", contact_id, Decimal("80")),
                (invoice_ids[2], "INV-E2E-3", contact_id, Decimal("10")),
                (invoice_ids[3], "INV-E2E-4", contact_id, Decimal("10")),
            ],
        )
        unapplied_contact_id = uuid4()
        await observer.execute("INSERT INTO contacts (id, business_context_id) VALUES ($1, 'effingham_maids')", unapplied_contact_id)
        service_a = ReceivablesService(_SingleConnectionPool(conn_a, schema))
        service_b = ReceivablesService(_SingleConnectionPool(conn_b, schema))
        unapplied_args = {
            "contact_id": unapplied_contact_id,
            "payer_name": "Residential Customer",
            "total_amount": Decimal("125"),
            "payment_method": "check",
            "received_date": date.today(),
            "allocations": [],
            "idempotency_key": "e2e-unapplied-payment-create",
            "allow_unapplied": True,
            "unapplied_contact_context_id": "effingham_maids",
        }
        unapplied_outcomes = await asyncio.gather(
            service_a.create_payment_with_outcome(**unapplied_args),
            service_b.create_payment_with_outcome(**unapplied_args),
        )
        assert sorted(outcome.replayed for outcome in unapplied_outcomes) == [
            False,
            True,
        ]
        unapplied_first, unapplied_replay = (
            outcome.payment for outcome in unapplied_outcomes
        )
        assert unapplied_first["id"] == unapplied_replay["id"]
        assert unapplied_first["status"] == "received"
        assert unapplied_first["allocated_amount_cents"] == 0
        assert unapplied_first["unapplied_amount_cents"] == 12_500
        assert (
            await observer.fetchval(
                """
                SELECT COUNT(*)
                FROM customer_payments
                WHERE idempotency_key = 'e2e-unapplied-payment-create'
                """
            )
            == 1
        )
        assert (
            await observer.fetchval(
                "SELECT COUNT(*) FROM invoice_payments WHERE payment_id = $1",
                UUID(unapplied_first["id"]),
            )
            == 0
        )
        create_args = {
            "contact_id": contact_id,
            "payer_name": "Acme",
            "total_amount": Decimal("100"),
            "payment_method": "check",
            "received_date": date.today(),
            "allocations": [
                {"invoice_id": invoice_ids[0], "amount": Decimal("50")},
                {"invoice_id": invoice_ids[1], "amount": Decimal("30")},
            ],
            "reference": "e2e-1001",
            "idempotency_key": "e2e-payment-create",
        }
        outcomes = await asyncio.gather(
            service_a.create_payment_with_outcome(**create_args),
            service_b.create_payment_with_outcome(**create_args),
        )
        assert sorted(outcome.replayed for outcome in outcomes) == [False, True]
        first, replay = (outcome.payment for outcome in outcomes)
        assert first["id"] == replay["id"]
        assert first["allocated_amount_cents"] == 8_000
        assert first["unapplied_amount_cents"] == 2_000
        assert (
            await observer.fetchval(
                "SELECT COUNT(*) FROM customer_payments WHERE idempotency_key = 'e2e-payment-create'"
            )
            == 1
        )

        with pytest.raises(ReceivablesConflictError, match="different request"):
            await service_a.create_payment(
                **{**create_args, "total_amount": Decimal("99")}
            )

        payment_id = UUID(first["id"])
        with pytest.raises(ReceivablesConflictError, match="Idempotency key"):
            await service_a.return_payment(
                payment_id=payment_id,
                reason="Cross-operation key reuse",
                actor="E2E admin",
                idempotency_key="e2e-payment-create",
            )
        assert (
            await observer.fetchval(
                "SELECT status FROM customer_payments WHERE id = $1", payment_id
            )
            == "received"
        )

        adjusted = await service_a.adjust_allocations(
            payment_id=payment_id,
            allocations=[
                {"invoice_id": invoice_ids[0], "amount": Decimal("60")},
                {"invoice_id": invoice_ids[1], "amount": Decimal("40")},
            ],
            reason="Apply remainder",
            actor="E2E admin",
            idempotency_key="e2e-adjust",
        )
        assert adjusted["unapplied_amount_cents"] == 0

        with pytest.raises(ReceivablesConflictError, match="different request"):
            await service_a.create_deposit_batch(
                payment_ids=[payment_id],
                deposit_date=date.today(),
                bank_reference="cross-operation",
                actor="E2E admin",
                idempotency_key="e2e-payment-create",
            )
        assert (
            await observer.fetchval(
                "SELECT status FROM customer_payments WHERE id = $1", payment_id
            )
            == "received"
        )

        deposit_a, deposit_replay = await asyncio.gather(
            service_a.create_deposit_batch(
                payment_ids=[payment_id],
                deposit_date=date.today(),
                bank_reference="bank-1",
                actor="E2E admin",
                idempotency_key="e2e-deposit",
            ),
            service_b.create_deposit_batch(
                payment_ids=[payment_id],
                deposit_date=date.today(),
                bank_reference="bank-1",
                actor="E2E admin",
                idempotency_key="e2e-deposit",
            ),
        )
        assert deposit_a["id"] == deposit_replay["id"]
        batch_id = UUID(deposit_a["id"])
        with pytest.raises(ReceivablesConflictError, match="different request"):
            await service_a.clear_deposit_batch(
                batch_id=batch_id,
                actor="E2E admin",
                idempotency_key="e2e-deposit",
            )
        assert (
            await observer.fetchval(
                "SELECT status FROM payment_deposit_batches WHERE id = $1", batch_id
            )
            == "deposited"
        )

        clear_a, clear_replay = await asyncio.gather(
            service_a.clear_deposit_batch(
                batch_id=batch_id,
                actor="E2E admin",
                idempotency_key="e2e-clear",
            ),
            service_b.clear_deposit_batch(
                batch_id=batch_id,
                actor="E2E admin",
                idempotency_key="e2e-clear",
            ),
        )
        assert clear_a["id"] == clear_replay["id"]
        assert clear_a["status"] == "cleared"

        returned = await service_a.return_payment(
            payment_id=payment_id,
            reason="NSF",
            actor="E2E admin",
            idempotency_key="e2e-return",
        )
        assert returned["status"] == "returned"
        balances = await observer.fetch(
            """
            SELECT invoice_number, amount_paid, status
            FROM invoices
            WHERE id = ANY($1::uuid[])
            ORDER BY invoice_number
            """,
            invoice_ids[:2],
        )
        assert [(row["amount_paid"], row["status"]) for row in balances] == [
            (Decimal("0.00"), "sent"),
            (Decimal("0.00"), "sent"),
        ]

        returned_adjustment_replay = await service_b.adjust_allocations(
            payment_id=payment_id,
            allocations=[
                {"invoice_id": invoice_ids[0], "amount": Decimal("60")},
                {"invoice_id": invoice_ids[1], "amount": Decimal("40")},
            ],
            reason="Apply remainder",
            actor="Retrying E2E admin",
            idempotency_key="e2e-adjust",
        )
        assert returned_adjustment_replay["status"] == "returned"
        assert returned_adjustment_replay["allocated_amount_cents"] == 0
        assert returned_adjustment_replay["unapplied_amount_cents"] == 0
        for changed in (
            {
                "allocations": [
                    {"invoice_id": invoice_ids[0], "amount": Decimal("59")},
                    {"invoice_id": invoice_ids[1], "amount": Decimal("40")},
                ],
                "reason": "Apply remainder",
            },
            {
                "allocations": [
                    {"invoice_id": invoice_ids[0], "amount": Decimal("60")},
                    {"invoice_id": invoice_ids[1], "amount": Decimal("40")},
                ],
                "reason": "Changed replay reason",
            },
        ):
            with pytest.raises(ReceivablesConflictError, match="different request"):
                await service_a.adjust_allocations(
                    payment_id=payment_id,
                    allocations=changed["allocations"],
                    reason=changed["reason"],
                    actor="E2E admin",
                    idempotency_key="e2e-adjust",
                )

        void_candidate = await service_a.create_payment(
            contact_id=contact_id,
            payer_name="Acme",
            total_amount=Decimal("20"),
            payment_method="check",
            received_date=date.today(),
            allocations=[{"invoice_id": invoice_ids[0], "amount": Decimal("15")}],
            reference="e2e-void-candidate",
            idempotency_key="e2e-void-create",
        )
        void_payment_id = UUID(void_candidate["id"])
        void_adjustment = {
            "payment_id": void_payment_id,
            "allocations": [{"invoice_id": invoice_ids[0], "amount": Decimal("10")}],
            "reason": "Leave part unapplied",
            "actor": "E2E admin",
            "idempotency_key": "e2e-void-adjust",
        }
        await service_a.adjust_allocations(**void_adjustment)
        await service_a.void_payment(
            payment_id=void_payment_id,
            reason="Entry mistake",
            actor="E2E admin",
            idempotency_key="e2e-void",
        )
        void_adjustment_replay = await service_b.adjust_allocations(
            **{**void_adjustment, "actor": "Retrying E2E admin"}
        )
        assert void_adjustment_replay["status"] == "voided"
        assert void_adjustment_replay["allocated_amount_cents"] == 0
        assert void_adjustment_replay["unapplied_amount_cents"] == 0

        with pytest.raises(ReceivablesConflictError, match="different request"):
            await service_a.adjust_allocations(
                **{
                    **void_adjustment,
                    "allocations": [
                        {"invoice_id": invoice_ids[0], "amount": Decimal("9")}
                    ],
                }
            )
        with pytest.raises(ReceivablesConflictError, match="different request"):
            await service_a.adjust_allocations(
                **{
                    **void_adjustment,
                    "idempotency_key": "e2e-adjust",
                }
            )

        extra_payments = []
        for index in (2, 3):
            extra = await service_a.create_payment(
                contact_id=contact_id,
                payer_name="Acme",
                total_amount=Decimal("10"),
                payment_method="check",
                received_date=date.today(),
                allocations=[
                    {"invoice_id": invoice_ids[index], "amount": Decimal("10")}
                ],
                reference=f"e2e-extra-{index}",
                idempotency_key=f"e2e-extra-{index}",
            )
            extra_payments.append(UUID(extra["id"]))
        with pytest.raises(ReceivablesConflictError, match="different request"):
            await service_a.void_payment(
                payment_id=extra_payments[0],
                reason="Cross-payment key reuse",
                actor="E2E admin",
                idempotency_key="e2e-return",
            )
        assert (
            await observer.fetchval(
                "SELECT status FROM customer_payments WHERE id = $1", extra_payments[0]
            )
            == "received"
        )
        collision = await asyncio.gather(
            service_a.create_deposit_batch(
                payment_ids=[extra_payments[0]],
                deposit_date=date.today(),
                bank_reference="collision-a",
                actor="E2E admin",
                idempotency_key="e2e-deposit-collision",
            ),
            service_b.create_deposit_batch(
                payment_ids=[extra_payments[1]],
                deposit_date=date.today(),
                bank_reference="collision-b",
                actor="E2E admin",
                idempotency_key="e2e-deposit-collision",
            ),
            return_exceptions=True,
        )
        assert sum(isinstance(item, dict) for item in collision) == 1
        assert (
            sum(isinstance(item, ReceivablesConflictError) for item in collision) == 1
        )
        assert await observer.fetchval("""
            SELECT COUNT(*) FROM payment_deposit_batches
            WHERE idempotency_key = 'e2e-deposit-collision'
            """) == 1

        collision_batch = next(item for item in collision if isinstance(item, dict))
        deposited_payment = UUID(collision_batch["payments"][0]["id"])
        remaining_payment = next(
            item for item in extra_payments if item != deposited_payment
        )
        other_batch = await service_a.create_deposit_batch(
            payment_ids=[remaining_payment],
            deposit_date=date.today(),
            bank_reference="collision-other",
            actor="E2E admin",
            idempotency_key="e2e-deposit-other",
        )
        clear_collision = await asyncio.gather(
            service_a.clear_deposit_batch(
                batch_id=UUID(collision_batch["id"]),
                actor="E2E admin",
                idempotency_key="e2e-clear-collision",
            ),
            service_b.clear_deposit_batch(
                batch_id=UUID(other_batch["id"]),
                actor="E2E admin",
                idempotency_key="e2e-clear-collision",
            ),
            return_exceptions=True,
        )
        assert sum(isinstance(item, dict) for item in clear_collision) == 1
        assert (
            sum(isinstance(item, ReceivablesConflictError) for item in clear_collision)
            == 1
        )
    finally:
        await conn_a.close()
        await conn_b.close()
        await observer.execute("SET search_path TO public")
        await observer.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await observer.close()


@pytest.mark.asyncio
async def test_real_postgres_receipt_outbox_failure_rolls_back_new_payment():
    """The receipt enqueue is financial-transaction work, never an email send.

    An injected outbox write failure must leave no payment, event, or orphaned
    delivery row behind.  Retrying after the transient failure clears must
    produce exactly one complete payment/outbox pair under the same key.
    """
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"receivables_receipt_outbox_atomicity_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _create_pre_receivables_schema(conn, schema)
        await _apply_receivables_migrations(conn, schema)
        contact_id = uuid4()
        await conn.execute(
            "INSERT INTO contacts (id, business_context_id) "
            "VALUES ($1, 'effingham_maids')",
            contact_id,
        )
        await conn.execute(
            """
            CREATE FUNCTION fail_payment_receipt_enqueue()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                RAISE EXCEPTION 'injected receipt outbox failure';
            END;
            $$;

            CREATE TRIGGER receipt_outbox_failure
            BEFORE INSERT ON payment_receipt_deliveries
            FOR EACH ROW EXECUTE FUNCTION fail_payment_receipt_enqueue();
            """
        )
        service = ReceivablesService(_SingleConnectionPool(conn, schema))
        create_args = {
            "contact_id": contact_id,
            "payer_name": "Residential Customer",
            "total_amount": Decimal("125.00"),
            "payment_method": "check",
            "received_date": date(2026, 8, 12),
            "reference": "1042",
            "allocations": [],
            "idempotency_key": "receipt-outbox-atomicity-1",
            "recorded_by": "Juan Canfield",
            "allow_unapplied": True,
            "unapplied_contact_context_id": "effingham_maids",
            "receipt_recipient": PaymentReceiptRecipient(
                contact_id=contact_id,
                customer_name="Residential Customer",
                customer_type="residential",
                recipient_email="residential@example.test",
            ),
            "require_receipt_recipient": True,
        }

        with pytest.raises(asyncpg.PostgresError, match="injected receipt outbox"):
            await service.create_payment_with_outcome(**create_args)

        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM customer_payments "
                "WHERE idempotency_key = 'receipt-outbox-atomicity-1'"
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM payment_events "
                "WHERE idempotency_key = 'receipt-outbox-atomicity-1'"
            )
            == 0
        )
        assert (
            await conn.fetchval("SELECT COUNT(*) FROM payment_receipt_deliveries")
            == 0
        )

        await conn.execute(
            "DROP TRIGGER receipt_outbox_failure ON payment_receipt_deliveries"
        )
        recovered = await service.create_payment_with_outcome(**create_args)
        assert recovered.replayed is False
        assert recovered.payment["receipt_delivery"]["status"] == "pending"
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM customer_payments "
                "WHERE idempotency_key = 'receipt-outbox-atomicity-1'"
            )
            == 1
        )
        assert (
            await conn.fetchval("SELECT COUNT(*) FROM payment_receipt_deliveries")
            == 1
        )
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_real_postgres_unapplied_payment_holds_customer_eligibility_lock():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"receivables_customer_lock_{uuid4().hex}"
    observer = await asyncpg.connect(database_url)
    payment_conn = await asyncpg.connect(database_url)
    archive_conn = await asyncpg.connect(database_url)
    contact_locked = asyncio.Event()
    release_contact_lock = asyncio.Event()
    payment_task = None
    archive_task = None
    try:
        await _create_pre_receivables_schema(observer, schema)
        await _apply_receivables_migrations(observer, schema)
        contact_id = uuid4()
        await observer.execute(
            "INSERT INTO contacts (id, business_context_id) "
            "VALUES ($1, 'effingham_maids')",
            contact_id,
        )
        service = ReceivablesService(
            _PauseAfterContactLockPool(
                payment_conn,
                schema,
                contact_locked,
                release_contact_lock,
            )
        )
        payment_task = asyncio.create_task(
            service.create_payment(
                contact_id=contact_id,
                payer_name="Residential Customer",
                total_amount=Decimal("125"),
                payment_method="check",
                received_date=date.today(),
                allocations=[],
                idempotency_key="e2e-unapplied-customer-lock",
                allow_unapplied=True,
                unapplied_contact_context_id="effingham_maids",
            )
        )
        await asyncio.wait_for(contact_locked.wait(), timeout=3)

        async with archive_conn.transaction():
            await archive_conn.execute(f'SET LOCAL search_path TO "{schema}"')
            archive_task = asyncio.create_task(
                archive_conn.execute(
                    "UPDATE contacts SET status = 'inactive' WHERE id = $1",
                    contact_id,
                )
            )
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(archive_task), timeout=0.2)
            assert (
                await observer.fetchval(
                    "SELECT status FROM contacts WHERE id = $1", contact_id
                )
                == "active"
            )

            release_contact_lock.set()
            payment = await asyncio.wait_for(payment_task, timeout=3)
            payment_task = None
            assert payment["status"] == "received"
            assert payment["allocated_amount_cents"] == 0
            assert payment["unapplied_amount_cents"] == 12_500
            assert await asyncio.wait_for(archive_task, timeout=3) == "UPDATE 1"
            archive_task = None

        assert (
            await observer.fetchval(
                "SELECT status FROM contacts WHERE id = $1", contact_id
            )
            == "inactive"
        )
        assert (
            await observer.fetchval(
                "SELECT COUNT(*) FROM invoice_payments WHERE payment_id = $1",
                UUID(payment["id"]),
            )
            == 0
        )
    finally:
        release_contact_lock.set()
        pending_tasks = [
            task for task in (payment_task, archive_task) if task is not None
        ]
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        await payment_conn.close()
        await archive_conn.close()
        await observer.execute("SET search_path TO public")
        await observer.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await observer.close()


@pytest.mark.asyncio
async def test_real_postgres_unapplied_payment_rejects_ineligible_customer_rows():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"receivables_ineligible_customer_{uuid4().hex}"
    observer = await asyncpg.connect(database_url)
    payment_conn = await asyncpg.connect(database_url)
    try:
        await _create_pre_receivables_schema(observer, schema)
        await _apply_receivables_migrations(observer, schema)
        cases = (
            ("foreign", "other_business", "customer", "active"),
            ("lead", "effingham_maids", "lead", "active"),
            ("inactive", "effingham_maids", "customer", "inactive"),
        )
        contacts = [(uuid4(), *case[1:]) for case in cases]
        await observer.executemany(
            """
            INSERT INTO contacts (id, business_context_id, contact_type, status)
            VALUES ($1, $2, $3, $4)
            """,
            contacts,
        )
        service = ReceivablesService(_SingleConnectionPool(payment_conn, schema))

        for (label, *_), (contact_id, *_contact) in zip(cases, contacts):
            key = f"e2e-unapplied-ineligible-{label}"
            with pytest.raises(ReceivablesNotFoundError, match="Customer not found"):
                await service.create_payment(
                    contact_id=contact_id,
                    payer_name="Residential Customer",
                    total_amount=Decimal("125"),
                    payment_method="check",
                    received_date=date.today(),
                    allocations=[],
                    idempotency_key=key,
                    allow_unapplied=True,
                    unapplied_contact_context_id="effingham_maids",
                )
            assert (
                await observer.fetchval(
                    "SELECT COUNT(*) FROM customer_payments "
                    "WHERE idempotency_key = $1",
                    key,
                )
                == 0
            )
            assert (
                await observer.fetchval(
                    "SELECT COUNT(*) FROM payment_events "
                    "WHERE idempotency_key = $1",
                    key,
                )
                == 0
            )
    finally:
        await payment_conn.close()
        await observer.execute("SET search_path TO public")
        await observer.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await observer.close()


@pytest.mark.asyncio
async def test_return_waits_for_concurrent_create_before_balance_snapshot():
    """A return cannot overwrite a concurrently-created active allocation."""
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"receivables_concurrency_{uuid4().hex}"
    contact_id = uuid4()
    invoice_id = uuid4()
    returned_payment_id = uuid4()
    concurrent_payment_id = uuid4()
    conn_a = await asyncpg.connect(database_url)
    conn_b = await asyncpg.connect(database_url)
    observer = await asyncpg.connect(database_url)
    task = None
    tx_b = None
    try:
        await observer.execute(f'CREATE SCHEMA "{schema}"')
        await observer.execute(f'SET search_path TO "{schema}"')
        await observer.execute("""
            CREATE TABLE invoices (
                id uuid PRIMARY KEY, invoice_number varchar NOT NULL,
                contact_id uuid, customer_name varchar NOT NULL,
                total_amount numeric(12,2) NOT NULL,
                amount_paid numeric(12,2) NOT NULL,
                due_date date NOT NULL, status varchar NOT NULL,
                paid_at timestamptz, updated_at timestamptz NOT NULL
            );
            CREATE TABLE customer_payments (
                id uuid PRIMARY KEY, contact_id uuid, payer_name varchar NOT NULL,
                total_amount numeric(12,2) NOT NULL, payment_method varchar NOT NULL,
                reference varchar, received_date date NOT NULL, status varchar NOT NULL,
                source varchar NOT NULL, idempotency_key varchar,
                request_fingerprint varchar, notes text, recorded_by varchar,
                deposited_at timestamptz, cleared_at timestamptz,
                returned_at timestamptz, return_reason text,
                voided_at timestamptz, void_reason text,
                metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                created_at timestamptz NOT NULL, updated_at timestamptz NOT NULL
            );
            CREATE TABLE invoice_payments (
                id uuid PRIMARY KEY, invoice_id uuid NOT NULL,
                payment_id uuid, amount numeric(12,2) NOT NULL,
                payment_date date NOT NULL, payment_method varchar NOT NULL,
                reference varchar, notes text, recorded_by varchar,
                created_at timestamptz NOT NULL,
                metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                reversed_at timestamptz, reversed_by varchar, reversal_reason text
            );
            CREATE TABLE payment_events (
                id uuid PRIMARY KEY, payment_id uuid NOT NULL, event_type varchar NOT NULL,
                previous_status varchar, new_status varchar, effective_date date,
                actor varchar, reason text, idempotency_key varchar,
                request_fingerprint varchar, metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE payment_deposit_items (
                batch_id uuid NOT NULL, payment_id uuid PRIMARY KEY,
                created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """)
        await observer.execute(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, total_amount,
                amount_paid, due_date, status, updated_at
            ) VALUES ($1, 'INV-RACE', $2, 'Acme', 100, 60, CURRENT_DATE, 'partial', NOW())
            """,
            invoice_id,
            contact_id,
        )
        await observer.execute(
            """
            INSERT INTO customer_payments (
                id, contact_id, payer_name, total_amount, payment_method,
                received_date, status, source, idempotency_key,
                request_fingerprint, created_at, updated_at
            ) VALUES ($1, $2, 'Acme', 60, 'check', CURRENT_DATE, 'received',
                      'test', 'old-payment', 'old-fingerprint', NOW(), NOW())
            """,
            returned_payment_id,
            contact_id,
        )
        await observer.execute(
            """
            INSERT INTO invoice_payments (
                id, invoice_id, payment_id, amount, payment_date,
                payment_method, created_at
            ) VALUES ($1, $2, $3, 60, CURRENT_DATE, 'check', NOW())
            """,
            uuid4(),
            invoice_id,
            returned_payment_id,
        )

        tx_b = conn_b.transaction()
        await tx_b.start()
        await conn_b.execute(f'SET LOCAL search_path TO "{schema}"')
        await conn_b.fetch(
            "SELECT id FROM invoices WHERE id = $1 FOR UPDATE", invoice_id
        )
        await conn_b.execute(
            """
            INSERT INTO customer_payments (
                id, contact_id, payer_name, total_amount, payment_method,
                received_date, status, source, idempotency_key,
                request_fingerprint, created_at, updated_at
            ) VALUES ($1, $2, 'Acme', 40, 'ach', CURRENT_DATE, 'cleared',
                      'test', 'concurrent-payment', 'concurrent-fingerprint', NOW(), NOW())
            """,
            concurrent_payment_id,
            contact_id,
        )
        await conn_b.execute(
            """
            INSERT INTO invoice_payments (
                id, invoice_id, payment_id, amount, payment_date,
                payment_method, created_at
            ) VALUES ($1, $2, $3, 40, CURRENT_DATE, 'ach', NOW())
            """,
            uuid4(),
            invoice_id,
            concurrent_payment_id,
        )

        reached_recalculation = asyncio.Event()

        class _SignallingService(ReceivablesService):
            async def _recalculate_invoices(self, conn, invoice_ids):
                reached_recalculation.set()
                return await ReceivablesService._recalculate_invoices(conn, invoice_ids)

        service = _SignallingService(_SingleConnectionPool(conn_a, schema))
        task = asyncio.create_task(
            service.return_payment(
                payment_id=returned_payment_id,
                reason="NSF",
                actor="Concurrency test",
                idempotency_key="return-race-test",
            )
        )
        await asyncio.wait_for(reached_recalculation.wait(), timeout=3)

        for _ in range(300):
            wait_type = await observer.fetchval(
                "SELECT wait_event_type FROM pg_stat_activity WHERE pid = $1",
                conn_a.get_server_pid(),
            )
            if wait_type == "Lock":
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("return transaction did not wait on the invoice lock")

        await tx_b.commit()
        tx_b = None
        await asyncio.wait_for(task, timeout=3)
        task = None

        stored, active = await observer.fetchrow(
            """
            SELECT i.amount_paid AS stored,
                   COALESCE(SUM(ip.amount) FILTER (
                       WHERE ip.reversed_at IS NULL
                         AND (ip.payment_id IS NULL OR cp.status = ANY($2::varchar[]))
                   ), 0) AS active
            FROM invoices i
            LEFT JOIN invoice_payments ip ON ip.invoice_id = i.id
            LEFT JOIN customer_payments cp ON cp.id = ip.payment_id
            WHERE i.id = $1
            GROUP BY i.id
            """,
            invoice_id,
            list(("legacy", "received", "deposited", "cleared")),
        )
        assert stored == Decimal("40.00")
        assert stored == active
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if tx_b is not None:
            await tx_b.rollback()
        await conn_a.close()
        await conn_b.close()
        await observer.execute("SET search_path TO public")
        await observer.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await observer.close()


def test_enabled_receivables_api_rejects_raw_missing_placeholder_and_bad_digests():
    generated = eom_receivables_auth.generate_receivables_service_token()
    for token, digest, message in (
        (generated.token, generated.sha256, "Raw EOM receivables bearer token"),
        ("", "", "digest is required"),
        ("", "0" * 63, "lowercase SHA-256"),
        ("", "0" * 64, "placeholder"),
        ("", eom_receivables_auth._token_sha256("change-me"), "placeholder"),
    ):
        config = SimpleNamespace(
            receivables_api_enabled=True,
            receivables_service_token=token,
            receivables_service_token_sha256=digest,
        )
        with pytest.raises(RuntimeError, match=message):
            receivables_auth.validate_receivables_api_config(config)


@pytest.mark.asyncio
async def test_receivables_api_is_fail_closed():
    generated = eom_receivables_auth.generate_receivables_service_token()
    config = SimpleNamespace(
        receivables_api_enabled=True,
        receivables_service_token="",
        receivables_service_token_sha256=generated.sha256,
    )

    with pytest.raises(HTTPException) as missing:
        await receivables_auth.require_receivables_api("", config=config)
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as invalid:
        await receivables_auth.require_receivables_api(
            "Bearer wrong",
            config=config,
        )
    assert invalid.value.status_code == 401

    assert (
        await receivables_auth.require_receivables_api(
            f"Bearer {generated.token}",
            config=config,
        )
        is None
    )
    with pytest.raises(HTTPException) as disabled:
        await receivables_auth.require_receivables_api(
            "Bearer a-valid-service-token-for-tests",
            config=SimpleNamespace(
                receivables_api_enabled=False,
                receivables_service_token="",
                receivables_service_token_sha256="",
            ),
        )
    assert disabled.value.status_code == 503


def test_legacy_invoicing_route_rejects_well_formed_mismatched_bearer():
    configured = eom_receivables_auth.generate_receivables_service_token()
    presented = eom_receivables_auth.generate_receivables_service_token()
    while presented.token == configured.token:
        presented = eom_receivables_auth.generate_receivables_service_token()
    config = SimpleNamespace(
        receivables_api_enabled=True,
        receivables_service_token="",
        receivables_service_token_sha256=configured.sha256,
    )
    app = FastAPI()
    app.include_router(actions.router)
    app.dependency_overrides[receivables_auth.get_receivables_api_config] = (
        lambda: config
    )

    response = TestClient(app).get(
        "/invoicing/INV-1",
        headers={"Authorization": f"Bearer {presented.token}"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid bearer token"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_name",
    (
        "database-unavailable",
        "ConnectionDoesNotExistError",
        "ClientCannotConnectError",
        "CannotConnectNowError",
        "TooManyConnectionsError",
        "AdminShutdownError",
        "CrashShutdownError",
        "connection-closed",
        "pool-closed",
        "pool-closing",
        "pool-uninitialized",
        "pool-initializing",
        "resource-connection-closed",
        "timeout",
        "connection-reset",
        "network-unreachable",
    ),
)
async def test_receivables_api_maps_database_outage_classes_to_503(error_name):
    asyncpg = pytest.importorskip("asyncpg")
    from atlas_brain.api.invoicing import receivables as routes

    if error_name == "database-unavailable":
        error = DatabaseUnavailableError("receivables ledger")
    elif error_name == "connection-closed":
        error = asyncpg.InterfaceError("connection is closed")
    elif error_name == "pool-closed":
        error = asyncpg.InterfaceError("pool is closed")
    elif error_name == "pool-closing":
        error = asyncpg.InterfaceError("pool is closing")
    elif error_name == "pool-uninitialized":
        error = asyncpg.InterfaceError("pool is not initialized")
    elif error_name == "pool-initializing":
        error = asyncpg.InterfaceError(
            "pool is being initialized, but not yet ready: likely there is a race"
        )
    elif error_name == "resource-connection-closed":
        error = asyncpg.InterfaceError(
            "cannot call Transaction.__aexit__(): the underlying connection is closed"
        )
    elif error_name == "timeout":
        error = TimeoutError("database operation timed out")
    elif error_name == "connection-reset":
        error = ConnectionResetError("database socket reset")
    elif error_name == "network-unreachable":
        error = OSError(errno.ENETUNREACH, "database network unreachable")
    else:
        error = getattr(asyncpg, error_name)("database unavailable")

    async def fail():
        raise error

    with pytest.raises(HTTPException) as exc:
        await routes._call(fail())

    assert exc.value.status_code == 503
    assert exc.value.detail["code"] == "database_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status_code"),
    (
        (ReceivablesValidationError("invalid"), 422),
        (ReceivablesNotFoundError("missing"), 404),
        (ReceivablesConflictError("conflict"), 409),
        (ReceivablesError("other domain error"), 400),
    ),
)
async def test_receivables_api_preserves_domain_error_statuses(error, status_code):
    from atlas_brain.api.invoicing import receivables as routes

    async def fail():
        raise error

    with pytest.raises(HTTPException) as exc:
        await routes._call(fail())

    assert exc.value.status_code == status_code


@pytest.mark.asyncio
async def test_receivables_api_does_not_hide_non_availability_failures():
    asyncpg = pytest.importorskip("asyncpg")
    from atlas_brain.api.invoicing import receivables as routes

    for error in (
        asyncpg.UniqueViolationError("duplicate data"),
        asyncpg.InterfaceError("cannot start; the transaction is already started"),
        asyncpg.InterfaceError(
            "cannot call Transaction.__aexit__(): the underlying connection has "
            "been released back to the pool"
        ),
        OSError(errno.ENOENT, "unrelated file is missing"),
        OSError("unclassified OS failure"),
        RuntimeError("programming defect"),
    ):

        async def fail():
            raise error

        with pytest.raises(type(error)):
            await routes._call(fail())


@pytest.mark.asyncio
async def test_real_postgres_backend_termination_maps_closed_resource_to_503():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    from atlas_brain.api.invoicing import receivables as routes

    victim = await asyncpg.connect(database_url)
    terminator = await asyncpg.connect(database_url)

    async def terminate_inside_transaction():
        async with victim.transaction():
            terminated = await terminator.fetchval(
                "SELECT pg_terminate_backend($1)", victim.get_server_pid()
            )
            assert terminated is True
            await victim.execute("SELECT 1")

    try:
        with pytest.raises(HTTPException) as exc:
            await routes._call(terminate_inside_transaction())
        assert exc.value.status_code == 503
        assert exc.value.detail["code"] == "database_unavailable"
    finally:
        await terminator.close()
        if not victim.is_closed():
            await victim.close()


@pytest.mark.asyncio
async def test_legacy_quick_pay_is_gone():
    with pytest.raises(HTTPException) as exc:
        await actions.mark_paid("INV-1")
    assert exc.value.status_code == 410


def test_legacy_quick_pay_returns_gone_without_feature_flag_or_auth():
    from atlas_brain.api.invoicing import router as invoicing_router

    app = FastAPI()
    app.include_router(invoicing_router)

    response = TestClient(app).post("/invoicing/INV-1/mark-paid")

    assert response.status_code == 410


def test_all_invoicing_action_routes_require_service_auth():
    for route in actions.router.routes:
        dependency_calls = [
            dependency.call for dependency in route.dependant.dependencies
        ]
        assert receivables_auth.require_receivables_api in dependency_calls


@pytest.mark.parametrize(
    "invalid_key",
    ["", "\t", "x" * 129, "operation-" + "z" * 256],
)
@pytest.mark.asyncio
async def test_singular_mcp_rejects_invalid_key_before_repository_access(
    invalid_key,
):
    from atlas_brain.mcp import invoicing_server

    class _NoRepositoryAccess:
        async def get_by_id(self, *_args):
            raise AssertionError("repository must not be queried")

        async def get_by_number(self, *_args):
            raise AssertionError("repository must not be queried")

    async def no_crm(*_args):
        raise AssertionError("CRM must not be called")

    result = json.loads(
        await invoicing_server._record_payment_with_dependencies(
            repo=_NoRepositoryAccess(),
            crm_logger=no_crm,
            invoice_id="INV-1",
            amount=10.0,
            idempotency_key=invalid_key,
        )
    )

    assert result == {
        "success": False,
        "error": "idempotency_key must contain 1 to 128 characters",
    }


@pytest.mark.asyncio
async def test_real_postgres_http_and_mcp_entrypoints_use_supported_dependencies(
    monkeypatch,
):
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    from atlas_brain import main, main_eom
    from atlas_brain.api.invoicing import receivables as routes
    from atlas_brain.eom_api import receivables as eom_routes
    from atlas_brain.mcp import invoicing_server
    from atlas_brain.storage.repositories.invoice import InvoiceRepository

    schema = f"receivables_entrypoints_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _create_pre_receivables_schema(conn, schema)
        await _apply_receivables_migrations(conn, schema)
        contact_id = uuid4()
        invoice_ids = [uuid4() for _ in range(3)]
        await conn.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        await conn.executemany(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, total_amount,
                due_date, status
            ) VALUES ($1, $2, $3, 'Acme', 200, CURRENT_DATE + 30, 'sent')
            """,
            [
                (invoice_ids[0], "INV-HTTP-1", contact_id),
                (invoice_ids[1], "INV-HTTP-2", contact_id),
                (invoice_ids[2], "INV-MCP-1", contact_id),
            ],
        )

        unapplied_contact_id = uuid4()
        await conn.execute(
            "INSERT INTO contacts (id, business_context_id) "
            "VALUES ($1, 'effingham_maids')",
            unapplied_contact_id,
        )

        allocation_contact_id = uuid4()
        older_due_invoice, newer_due_invoice = uuid4(), uuid4()
        await conn.execute(
            "INSERT INTO contacts (id) VALUES ($1)", allocation_contact_id
        )
        await conn.executemany(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, total_amount,
                issue_date, due_date, status
            ) VALUES ($1, $2, $3, $4, 100, $5, $6, 'sent')
            """,
            [
                (
                    older_due_invoice,
                    "INV-ORDER-OLDER",
                    allocation_contact_id,
                    "Zulu Office",
                    date(2026, 6, 1),
                    date(2026, 6, 30),
                ),
                (
                    newer_due_invoice,
                    "INV-ORDER-NEWER",
                    allocation_contact_id,
                    "Alpha Office",
                    date(2026, 7, 1),
                    date(2026, 7, 31),
                ),
            ],
        )

        pool = _SingleConnectionPool(conn, schema)
        service = ReceivablesService(pool)
        repository = InvoiceRepository(pool=pool, receivables_service=service)
        suggestions = await service.suggest_allocations(
            contact_id=allocation_contact_id,
            total_amount=Decimal("50.00"),
        )
        assert [item["invoice_id"] for item in suggestions] == [
            str(older_due_invoice)
        ]
        generated = eom_receivables_auth.generate_receivables_service_token()
        config = SimpleNamespace(
            receivables_api_enabled=True,
            receivables_service_token="",
            receivables_service_token_sha256=generated.sha256,
        )
        app = main.app
        original_dependency_overrides = app.dependency_overrides.copy()
        app.dependency_overrides[receivables_auth.get_receivables_api_config] = (
            lambda: config
        )
        app.dependency_overrides[routes.get_receivables_service] = lambda: service
        slim_app = main_eom.app
        original_slim_dependency_overrides = slim_app.dependency_overrides.copy()
        slim_app.dependency_overrides[
            eom_receivables_auth.get_receivables_api_config
        ] = lambda: config
        slim_app.dependency_overrides[
            eom_routes.get_receivables_service
        ] = lambda: service

        class _CanonicalPool:
            is_initialized = True

        class _SlimPaymentCRM:
            async def get_eom_payment_customer(self, requested_contact_id):
                if requested_contact_id != unapplied_contact_id:
                    return None
                return {
                    "contact_id": unapplied_contact_id,
                    "customer_name": "Residential Customer",
                    "customer_type": "residential",
                    "recipient_email": "residential@example.test",
                }

        class _FullPaymentCRM:
            async def get_eom_payment_customer(self, requested_contact_id):
                if requested_contact_id == contact_id:
                    return {
                        "contact_id": contact_id,
                        "customer_name": "Acme",
                        "customer_type": "commercial",
                        "recipient_email": "billing@example.test",
                    }
                if requested_contact_id == unapplied_contact_id:
                    return {
                        "contact_id": unapplied_contact_id,
                        "customer_name": "Residential Customer",
                        "customer_type": "residential",
                        "recipient_email": "residential@example.test",
                    }
                return None

        monkeypatch.setattr(routes, "get_crm_provider", lambda: _FullPaymentCRM())
        monkeypatch.setattr(
            eom_routes, "get_eom_funnel_db_pool", lambda: _CanonicalPool()
        )
        monkeypatch.setattr(
            slim_app.state,
            "eom_funnel_crm_provider",
            lambda: _SlimPaymentCRM(),
            raising=False,
        )
        body = {
            "contact_id": str(contact_id),
            "payer_name": "Acme",
            "total_amount_cents": 20_000,
            "payment_method": "check",
            "received_date": "2026-07-16",
            "reference": "HTTP-1001",
            "allocations": [
                {"invoice_id": str(invoice_ids[0]), "amount_cents": 12_500},
                {"invoice_id": str(invoice_ids[1]), "amount_cents": 5_000},
            ],
        }
        unapplied_body = {
            "contact_id": str(unapplied_contact_id),
            "payer_name": "Residential Customer",
            "total_amount_cents": 12_500,
            "payment_method": "check",
            "received_date": "2026-08-12",
            "check_date": "2026-08-10",
            "received_through": " Mail ",
            "reference": "1001",
        }
        slim_unapplied_body = {
            **unapplied_body,
            "check_date": "2026-08-11",
            "received_through": "Employee handoff",
        }

        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://receivables.test",
            ) as client:
                unauthorized = await client.post(
                    "/api/v1/receivables/payments", json=body
                )
                assert unauthorized.status_code == 401
                assert (
                    await conn.fetchval("SELECT COUNT(*) FROM customer_payments")
                    == 0
                )

                response = await client.post(
                    "/api/v1/receivables/payments",
                    headers={
                        "Authorization": f"Bearer {generated.token}",
                        "X-EOM-Actor": "Juan Canfield",
                        "Idempotency-Key": "http-payment-1",
                    },
                    json=body,
                )
                unapplied_response = await client.post(
                    "/api/v1/receivables/payments",
                    headers={
                        "Authorization": f"Bearer {generated.token}",
                        "X-EOM-Actor": "Juan Canfield",
                        "Idempotency-Key": "http-unapplied-payment-1",
                    },
                    json=unapplied_body,
                )
                unapplied_replay = await client.post(
                    "/api/v1/receivables/payments",
                    headers={
                        "Authorization": f"Bearer {generated.token}",
                        "X-EOM-Actor": "Juan Canfield",
                        "Idempotency-Key": "http-unapplied-payment-1",
                    },
                    json=unapplied_body,
                )

            slim_transport = httpx.ASGITransport(app=slim_app)
            async with httpx.AsyncClient(
                transport=slim_transport,
                base_url="http://receivables.test",
            ) as client:
                slim_unapplied_response = await client.post(
                    "/api/v1/receivables/payments",
                    headers={
                        "Authorization": f"Bearer {generated.token}",
                        "X-EOM-Actor": "Juan Canfield",
                        "Idempotency-Key": "slim-http-unapplied-payment-1",
                    },
                    json=slim_unapplied_body,
                )
                slim_unapplied_replay = await client.post(
                    "/api/v1/receivables/payments",
                    headers={
                        "Authorization": f"Bearer {generated.token}",
                        "X-EOM-Actor": "Juan Canfield",
                        "Idempotency-Key": "slim-http-unapplied-payment-1",
                    },
                    json=slim_unapplied_body,
                )

            before_ledger_reads = await conn.fetchval(
                "SELECT COUNT(*) FROM customer_payments"
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://receivables.test",
            ) as client:
                unauthenticated_ledger = await client.get(
                    f"/api/v1/receivables/customers/{contact_id}/ledger"
                )
                invalid_ledger_limit = await client.get(
                    f"/api/v1/receivables/customers/{contact_id}/ledger",
                    headers={"Authorization": f"Bearer {generated.token}"},
                    params={"limit": 0},
                )
                blank_ledger_status = await client.get(
                    f"/api/v1/receivables/customers/{contact_id}/ledger",
                    headers={"Authorization": f"Bearer {generated.token}"},
                    params={"payment_status": "   "},
                )
                routable_large_offset = await client.get(
                    f"/api/v1/receivables/customers/{contact_id}/ledger",
                    headers={"Authorization": f"Bearer {generated.token}"},
                    params={"offset": 10_001},
                )
                full_ledger = await client.get(
                    f"/api/v1/receivables/customers/{contact_id}/ledger",
                    headers={"Authorization": f"Bearer {generated.token}"},
                )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=slim_app),
                base_url="http://receivables.test",
            ) as client:
                slim_ledger = await client.get(
                    f"/api/v1/receivables/customers/{unapplied_contact_id}/ledger",
                    headers={"Authorization": f"Bearer {generated.token}"},
                )
                slim_blank_ledger_method = await client.get(
                    f"/api/v1/receivables/customers/{unapplied_contact_id}/ledger",
                    headers={"Authorization": f"Bearer {generated.token}"},
                    params={"payment_method": "\t"},
                )
                slim_routable_large_offset = await client.get(
                    f"/api/v1/receivables/customers/{unapplied_contact_id}/ledger",
                    headers={"Authorization": f"Bearer {generated.token}"},
                    params={"offset": 10_001},
                )

            assert unauthenticated_ledger.status_code == 401
            assert invalid_ledger_limit.status_code == 422
            assert blank_ledger_status.status_code == 422
            assert routable_large_offset.status_code == 200
            assert routable_large_offset.json()["entries"] == []
            assert full_ledger.status_code == 200, full_ledger.text
            full_ledger_payload = full_ledger.json()
            assert full_ledger_payload["contact_id"] == str(contact_id)
            assert any(
                entry["entry_type"] == "invoice"
                and entry["invoice"]["invoice_number"] == "INV-HTTP-1"
                for entry in full_ledger_payload["entries"]
            )
            assert any(
                entry["entry_type"] == "payment"
                and entry["payment"]["id"] == response.json()["id"]
                for entry in full_ledger_payload["entries"]
            )
            assert slim_ledger.status_code == 200, slim_ledger.text
            assert slim_blank_ledger_method.status_code == 422
            assert slim_routable_large_offset.status_code == 200
            assert slim_routable_large_offset.json()["entries"] == []
            slim_ledger_payload = slim_ledger.json()
            assert slim_ledger_payload["contact_id"] == str(unapplied_contact_id)
            assert (
                slim_ledger_payload["entries"][0]["payment"]["receipt_delivery"][
                    "status"
                ]
                == "pending"
            )
            assert (
                await conn.fetchval("SELECT COUNT(*) FROM customer_payments")
                == before_ledger_reads
            )

            assert response.status_code == 201
            assert response.json()["status"] == "received"
            http_payment = await conn.fetchrow("""
                SELECT idempotency_key, total_amount, recorded_by
                FROM customer_payments
                WHERE source = 'eom_admin' AND idempotency_key = 'http-payment-1'
                """)
            assert http_payment["total_amount"] == Decimal("200.00")
            assert http_payment["recorded_by"] == "Juan Canfield"
            http_allocations = await conn.fetch("""
                SELECT amount FROM invoice_payments ip
                JOIN customer_payments cp ON cp.id = ip.payment_id
                WHERE cp.idempotency_key = 'http-payment-1'
                ORDER BY amount DESC
                """)
            assert [row["amount"] for row in http_allocations] == [
                Decimal("125.00"),
                Decimal("50.00"),
            ]

            assert unapplied_response.status_code == 201
            assert unapplied_replay.status_code == 201
            unapplied_payment = unapplied_response.json()
            assert unapplied_replay.json()["id"] == unapplied_payment["id"]
            assert unapplied_payment["status"] == "received"
            assert unapplied_payment["check_date"] == "2026-08-10"
            assert unapplied_payment["received_through"] == "Mail"
            assert unapplied_payment["allocated_amount_cents"] == 0
            assert unapplied_payment["unapplied_amount_cents"] == 12_500
            assert unapplied_payment["receipt_delivery"] == {
                "receipt_number": f"EOM-RCP-{unapplied_payment['id']}",
                "recipient_email": "residential@example.test",
                "status": "pending",
                "skip_reason": None,
            }
            stored_check_metadata = await conn.fetchrow(
                """
                SELECT check_date, received_through
                FROM customer_payments
                WHERE idempotency_key = 'http-unapplied-payment-1'
                """
            )
            assert stored_check_metadata["check_date"] == date(2026, 8, 10)
            assert stored_check_metadata["received_through"] == "Mail"
            full_receipt_delivery = await conn.fetchrow(
                """
                SELECT receipt_number, recipient_email, delivery_status, skip_reason
                FROM payment_receipt_deliveries
                WHERE payment_id = (
                    SELECT id
                    FROM customer_payments
                    WHERE idempotency_key = 'http-unapplied-payment-1'
                )
                """
            )
            assert dict(full_receipt_delivery) == {
                "receipt_number": f"EOM-RCP-{unapplied_payment['id']}",
                "recipient_email": "residential@example.test",
                "delivery_status": "pending",
                "skip_reason": None,
            }
            assert slim_unapplied_response.status_code == 201
            assert slim_unapplied_replay.status_code == 201
            assert (
                slim_unapplied_replay.json()["id"]
                == slim_unapplied_response.json()["id"]
            )
            assert slim_unapplied_response.json()["check_date"] == "2026-08-11"
            assert (
                slim_unapplied_response.json()["received_through"]
                == "Employee handoff"
            )
            stored_slim_check_metadata = await conn.fetchrow(
                """
                SELECT check_date, received_through, recorded_by
                FROM customer_payments
                WHERE idempotency_key = 'slim-http-unapplied-payment-1'
                """
            )
            assert stored_slim_check_metadata["check_date"] == date(2026, 8, 11)
            assert stored_slim_check_metadata["received_through"] == "Employee handoff"
            assert stored_slim_check_metadata["recorded_by"] == "Juan Canfield"
            receipt_delivery = await conn.fetchrow(
                """
                SELECT receipt_number, recipient_email, delivery_status, skip_reason
                FROM payment_receipt_deliveries
                WHERE payment_id = (
                    SELECT id
                    FROM customer_payments
                    WHERE idempotency_key = 'slim-http-unapplied-payment-1'
                )
                """
            )
            assert dict(receipt_delivery) == {
                "receipt_number": (
                    f"EOM-RCP-{slim_unapplied_response.json()['id']}"
                ),
                "recipient_email": "residential@example.test",
                "delivery_status": "pending",
                "skip_reason": None,
            }
            assert (
                await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM payment_receipt_deliveries
                    WHERE payment_id = (
                        SELECT id
                        FROM customer_payments
                        WHERE idempotency_key = 'slim-http-unapplied-payment-1'
                    )
                    """
                )
                == 1
            )
            assert (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM invoice_payments ip "
                    "JOIN customer_payments cp ON cp.id = ip.payment_id "
                    "WHERE cp.idempotency_key = 'http-unapplied-payment-1'"
                )
                == 0
            )
            assert (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM payment_events pe "
                    "JOIN customer_payments cp ON cp.id = pe.payment_id "
                    "WHERE cp.idempotency_key = 'http-unapplied-payment-1' "
                    "AND pe.event_type = 'payment_recorded'"
                )
                == 1
            )
        finally:
            app.dependency_overrides.clear()
            app.dependency_overrides.update(original_dependency_overrides)
            slim_app.dependency_overrides.clear()
            slim_app.dependency_overrides.update(original_slim_dependency_overrides)

        crm_calls = []

        async def record_crm_call(*args):
            crm_calls.append(args)

        singular_first = json.loads(
            await invoicing_server._record_payment_with_dependencies(
                repo=repository,
                crm_logger=record_crm_call,
                invoice_id="INV-MCP-1",
                amount=25.0,
                payment_method="card",
                idempotency_key="mcp-payment-retry-1",
            )
        )
        singular_replay = json.loads(
            await invoicing_server._record_payment_with_dependencies(
                repo=repository,
                crm_logger=record_crm_call,
                invoice_id="INV-MCP-1",
                amount=25.0,
                payment_method="card",
                idempotency_key="mcp-payment-retry-1",
            )
        )
        assert singular_first["success"] is True
        assert (
            singular_first["payment"]["payment_id"]
            == singular_replay["payment"]["payment_id"]
        )
        assert singular_first["payment"]["status"] == "cleared"
        assert "replayed" not in singular_first["payment"]
        assert "replayed" not in singular_replay["payment"]
        assert len(crm_calls) == 1
        received_payment_ids = {
            payment["id"] for payment in await service.list_payments(status="received")
        }
        assert singular_first["payment"]["payment_id"] not in received_payment_ids

        multi_result = json.loads(
            await invoicing_server._record_customer_payment_with_service(
                service=service,
                contact_id=str(contact_id),
                payer_name="Acme",
                total_amount_cents=10_000,
                payment_method="check",
                allocations=[
                    {"invoice_id": str(invoice_ids[0]), "amount_cents": 7_500},
                    {"invoice_id": str(invoice_ids[1]), "amount_cents": 500},
                ],
                idempotency_key="mcp-multi-1",
                reference="1001",
            )
        )
        assert multi_result["success"] is True
        assert multi_result["payment"]["total_amount_cents"] == 10_000
        assert sorted(
            item["amount_cents"] for item in multi_result["payment"]["allocations"]
        ) == [500, 7_500]

        before_invalid = await conn.fetchval("SELECT COUNT(*) FROM customer_payments")
        invalid = json.loads(
            await invoicing_server._record_customer_payment_with_service(
                service=service,
                contact_id=str(contact_id),
                payer_name="Acme",
                total_amount_cents=100.5,
                payment_method="check",
                allocations=[{"invoice_id": str(invoice_ids[0]), "amount_cents": 100}],
                idempotency_key="mcp-multi-invalid",
            )
        )
        assert invalid["success"] is False
        assert (
            await conn.fetchval("SELECT COUNT(*) FROM customer_payments")
            == before_invalid
        )
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_mcp_payment_response_projects_future_eom_check_metadata():
    from atlas_brain.mcp import invoicing_server

    contact_id = uuid4()
    invoice_id = uuid4()
    canonical_payment = {
        "id": "payment-1",
        "contact_id": str(contact_id),
        "payer_name": "Residential Customer",
        "total_amount_cents": 12_500,
        "payment_method": "check",
        "status": "received",
        "allocations": [{"invoice_id": str(invoice_id), "amount_cents": 12_500}],
        "check_date": "2026-08-10",
        "received_through": "mail",
        "unapproved_future_field": "preserved-by-legacy-mcp",
    }

    class _FutureMetadataService:
        def __init__(self) -> None:
            self.kwargs = None

        async def create_payment(self, **kwargs):
            self.kwargs = kwargs
            return canonical_payment

    service = _FutureMetadataService()
    result = json.loads(
        await invoicing_server._record_customer_payment_with_service(
            service=service,
            contact_id=str(contact_id),
            payer_name="Residential Customer",
            total_amount_cents=12_500,
            payment_method="check",
            allocations=[{"invoice_id": str(invoice_id), "amount_cents": 12_500}],
            idempotency_key="mcp-projection-1",
            reference="1001",
            notes="Received in mail",
            payment_date="2026-08-12",
        )
    )

    assert result == {
        "success": True,
        "payment": {
            "id": "payment-1",
            "contact_id": str(contact_id),
            "payer_name": "Residential Customer",
            "total_amount_cents": 12_500,
            "payment_method": "check",
            "status": "received",
            "allocations": [
                {"invoice_id": str(invoice_id), "amount_cents": 12_500}
            ],
            "unapproved_future_field": "preserved-by-legacy-mcp",
        },
    }
    assert service.kwargs["contact_id"] == contact_id
    assert service.kwargs["total_amount"] == Decimal("125")
    assert service.kwargs["received_date"] == date(2026, 8, 12)
    assert service.kwargs["allocations"] == [
        {"invoice_id": invoice_id, "amount": Decimal("125")}
    ]
    assert service.kwargs["reference"] == "1001"
    assert service.kwargs["notes"] == "Received in mail"


@pytest.mark.asyncio
async def test_legacy_mcp_customer_payment_keeps_reference_optional():
    from atlas_brain.mcp import invoicing_server

    contact_id = uuid4()
    invoice_id = uuid4()

    class _LegacyMCPService:
        def __init__(self) -> None:
            self.kwargs = None

        async def create_payment(self, **kwargs):
            self.kwargs = kwargs
            return {"id": "legacy-mcp-payment", "status": "received"}

    service = _LegacyMCPService()
    result = json.loads(
        await invoicing_server._record_customer_payment_with_service(
            service=service,
            contact_id=str(contact_id),
            payer_name="Legacy Customer",
            total_amount_cents=12_500,
            payment_method="check",
            allocations=[{"invoice_id": str(invoice_id), "amount_cents": 12_500}],
            idempotency_key="legacy-mcp-optional-reference-1",
        )
    )

    assert result == {
        "success": True,
        "payment": {"id": "legacy-mcp-payment", "status": "received"},
    }
    assert service.kwargs["reference"] is None
    assert service.kwargs["source"] == "invoicing_mcp"


def test_full_invoicing_mcp_http_refuses_to_start_without_strong_auth():
    from atlas_brain.mcp import invoicing_server
    from atlas_brain.mcp.auth import BearerAuthMiddleware

    app_calls = []

    def app_factory():
        app_calls.append(True)
        return object()

    for token, message in (
        ("", "is required"),
        ("change-me", "placeholder"),
        ("still-too-short", "at least 24"),
    ):
        with pytest.raises(RuntimeError, match=message):
            invoicing_server._authenticated_streamable_http_app(token, app_factory)
    assert app_calls == []

    app = invoicing_server._authenticated_streamable_http_app(
        "strong-invoicing-mcp-token-for-tests",
        app_factory,
    )
    assert isinstance(app, BearerAuthMiddleware)
    assert app_calls == [True]


def test_payment_http_models_reject_coercive_cent_values():
    from atlas_brain.api.invoicing.receivables import CreatePaymentRequest
    from atlas_brain.eom_api.receivables import AdjustAllocationsRequest
    from atlas_brain.eom_api.receivables import (
        CreatePaymentRequest as EOMCreatePaymentRequest,
    )

    base = {
        "contact_id": str(uuid4()),
        "payer_name": "Acme",
        "total_amount_cents": 100,
        "payment_method": "check",
        "received_date": "2026-07-16",
        "reference": "1001",
        "allocations": [{"invoice_id": str(uuid4()), "amount_cents": 100}],
    }
    assert CreatePaymentRequest.model_validate(base).total_amount_cents == 100
    for request_model in (CreatePaymentRequest, EOMCreatePaymentRequest):
        assert (
            request_model.model_validate(
                {key: value for key, value in base.items() if key != "allocations"}
            ).allocations
            == []
        )
        assert request_model.model_validate({**base, "allocations": []}).allocations == []
    with pytest.raises(ValueError):
        AdjustAllocationsRequest.model_validate(
            {"allocations": [], "reason": "Cannot leave adjustment empty"}
        )

    for invalid in (True, False, 100.0, 100.5, "100", 0, -1):
        with pytest.raises(ValueError):
            CreatePaymentRequest.model_validate({**base, "total_amount_cents": invalid})
        with pytest.raises(ValueError):
            CreatePaymentRequest.model_validate(
                {
                    **base,
                    "allocations": [
                        {"invoice_id": str(uuid4()), "amount_cents": invalid}
                    ],
                }
            )


@pytest.mark.parametrize("payment_method", ("check", "ach", "square"))
def test_eom_payment_http_models_require_trimmed_reference(payment_method):
    from atlas_brain.api.invoicing.receivables import CreatePaymentRequest
    from atlas_brain.eom_api.receivables import (
        CreatePaymentRequest as EOMCreatePaymentRequest,
    )

    base = {
        "contact_id": str(uuid4()),
        "payer_name": "Receipt Customer",
        "total_amount_cents": 12_500,
        "payment_method": payment_method,
        "received_date": "2026-08-15",
        "reference": "  receipt-reference-1001  ",
    }
    invalid_references = (None, "", " \t ", 1001, False, {}, [], "x" * 257)

    for request_model in (CreatePaymentRequest, EOMCreatePaymentRequest):
        assert request_model.model_validate(base).reference == "receipt-reference-1001"

        with pytest.raises(ValueError):
            request_model.model_validate(
                {key: value for key, value in base.items() if key != "reference"}
            )

        for reference in invalid_references:
            with pytest.raises(ValueError):
                request_model.model_validate({**base, "reference": reference})


def test_payment_http_models_preserve_optional_check_metadata():
    from atlas_brain.api.invoicing.receivables import CreatePaymentRequest
    from atlas_brain.eom_api.receivables import (
        CreatePaymentRequest as EOMCreatePaymentRequest,
    )

    base = {
        "contact_id": str(uuid4()),
        "payer_name": "Residential Customer",
        "total_amount_cents": 12_500,
        "payment_method": "check",
        "received_date": "2026-08-12",
        "reference": "1001",
    }
    for request_model in (CreatePaymentRequest, EOMCreatePaymentRequest):
        omitted = request_model.model_validate(base)
        assert omitted.check_date is None
        assert omitted.received_through is None

        with_metadata = request_model.model_validate(
            {
                **base,
                "check_date": "2026-08-10",
                "received_through": "Mail",
            }
        )
        assert with_metadata.check_date == date(2026, 8, 10)
        assert with_metadata.received_through == "Mail"
        assert (
            request_model.model_validate(
                {**base, "received_through": "x" * 128}
            ).received_through
            == "x" * 128
        )

        with pytest.raises(ValueError):
            request_model.model_validate({**base, "check_date": "not-a-date"})
        with pytest.raises(ValueError):
            request_model.model_validate({**base, "received_through": "x" * 129})


@pytest.mark.asyncio
async def test_payment_routes_forward_omitted_allocations_as_empty_list(monkeypatch):
    from atlas_brain.api.invoicing import receivables as full_routes
    from atlas_brain.api.invoicing.receivables import (
        CreatePaymentRequest as FullCreatePaymentRequest,
    )
    from atlas_brain.api.invoicing.receivables import create_payment as full_create_payment
    from atlas_brain.eom_api.receivables import (
        CreatePaymentRequest as EOMCreatePaymentRequest,
    )
    from atlas_brain.eom_api import receivables as eom_routes
    from atlas_brain.eom_api.receivables import create_payment as eom_create_payment

    class _RecordingService:
        def __init__(self) -> None:
            self.kwargs = None

        async def create_payment(self, **kwargs):
            self.kwargs = kwargs
            return {"id": "payment-1"}

    class _CRM:
        async def get_eom_payment_customer(self, contact_id):
            return {
                "contact_id": contact_id,
                "customer_name": "Residential Customer",
                "customer_type": "residential",
                "recipient_email": "residential@example.test",
            }

    monkeypatch.setattr(full_routes, "get_crm_provider", lambda: _CRM())

    for request_model, create_route in (
        (FullCreatePaymentRequest, full_create_payment),
        (EOMCreatePaymentRequest, eom_create_payment),
    ):
        body = request_model.model_validate(
            {
                "contact_id": str(uuid4()),
                "payer_name": "Residential Customer",
                "total_amount_cents": 12_500,
                "payment_method": "check",
                "received_date": "2026-08-12",
                "reference": "1001",
            }
        )
        service = _RecordingService()

        route_args = {
            "actor": "Juan Canfield",
            "idempotency_key": "route-unapplied-payment-1",
            "service": service,
        }
        if create_route is eom_create_payment:
            class _CanonicalPool:
                is_initialized = True

            monkeypatch.setattr(
                eom_routes, "get_eom_funnel_db_pool", lambda: _CanonicalPool()
            )
            route_args["request"] = SimpleNamespace(
                app=SimpleNamespace(
                    state=SimpleNamespace(eom_funnel_crm_provider=lambda: _CRM())
                )
            )

        result = await create_route(body, **route_args)

        assert result == {"id": "payment-1"}
        assert service.kwargs["allocations"] == []
        assert service.kwargs["allow_unapplied"] is True
        assert service.kwargs["unapplied_contact_context_id"] == "effingham_maids"
        assert service.kwargs["recorded_by"] == "Juan Canfield"
        assert service.kwargs["idempotency_key"] == "route-unapplied-payment-1"
        assert service.kwargs["check_date"] is None
        assert service.kwargs["received_through"] is None
        assert service.kwargs["reference"] == "1001"
        assert service.kwargs["require_receipt_recipient"] is True
        assert service.kwargs["receipt_recipient"].contact_id == body.contact_id

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "check_metadata",
    (
        {"check_date": "2026-08-10"},
        {"received_through": "Mail"},
    ),
)
async def test_full_and_slim_payment_entrypoints_reject_check_metadata_when_schema_is_unready(
    check_metadata,
    monkeypatch,
):
    from atlas_brain import main, main_eom
    from atlas_brain.api.invoicing import auth as full_auth
    from atlas_brain.api.invoicing import receivables as full_routes
    from atlas_brain.eom_api import auth as slim_auth
    from atlas_brain.eom_api import receivables as slim_routes

    generated = eom_receivables_auth.generate_receivables_service_token()
    config = SimpleNamespace(
        receivables_api_enabled=True,
        receivables_service_token="",
        receivables_service_token_sha256=generated.sha256,
    )
    body = {
        "contact_id": str(uuid4()),
        "payer_name": "Residential Customer",
        "total_amount_cents": 12_500,
        "payment_method": "check",
        "received_date": "2026-08-12",
        "reference": "1001",
        **check_metadata,
    }
    headers = {
        "Authorization": f"Bearer {generated.token}",
        "X-EOM-Actor": "Juan Canfield",
        "Idempotency-Key": "asgi-unready-check-metadata-1",
    }

    class _CanonicalPool:
        is_initialized = True

    class _CRM:
        async def get_eom_payment_customer(self, contact_id):
            return {
                "contact_id": contact_id,
                "customer_name": "Residential Customer",
                "customer_type": "residential",
                "recipient_email": "residential@example.test",
            }

    monkeypatch.setattr(full_routes, "get_crm_provider", lambda: _CRM())
    monkeypatch.setattr(
        slim_routes, "get_eom_funnel_db_pool", lambda: _CanonicalPool()
    )
    monkeypatch.setattr(
        main_eom.app.state,
        "eom_funnel_crm_provider",
        lambda: _CRM(),
        raising=False,
    )

    for app, config_dependency, service_dependency in (
        (main.app, full_auth.get_receivables_api_config, full_routes.get_receivables_service),
        (
            main_eom.app,
            slim_auth.get_receivables_api_config,
            slim_routes.get_receivables_service,
        ),
    ):
        service = ReceivablesService()
        readiness_calls = 0

        async def unready() -> bool:
            nonlocal readiness_calls
            readiness_calls += 1
            return False

        service.is_ready = unready
        original_overrides = app.dependency_overrides.copy()
        app.dependency_overrides[config_dependency] = lambda: config
        app.dependency_overrides[service_dependency] = lambda: service
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://receivables.test",
            ) as client:
                response = await client.post(
                    "/api/v1/receivables/payments",
                    headers=headers,
                    json=body,
                )
            assert response.status_code == 503
            assert response.json()["detail"]["code"] == "schema_unavailable"
            assert readiness_calls == 1
        finally:
            app.dependency_overrides.clear()
            app.dependency_overrides.update(original_overrides)


@pytest.mark.asyncio
@pytest.mark.parametrize("payment_method", ("check", "ach", "square"))
@pytest.mark.parametrize(
    "reference_body",
    (
        {},
        {"reference": None},
        {"reference": ""},
        {"reference": " \t "},
        {"reference": 1001},
    ),
)
async def test_full_and_slim_payment_entrypoints_reject_invalid_reference_before_service(
    payment_method,
    reference_body,
):
    from atlas_brain import main, main_eom
    from atlas_brain.api.invoicing import auth as full_auth
    from atlas_brain.api.invoicing import receivables as full_routes
    from atlas_brain.eom_api import auth as slim_auth
    from atlas_brain.eom_api import receivables as slim_routes

    generated = eom_receivables_auth.generate_receivables_service_token()
    config = SimpleNamespace(
        receivables_api_enabled=True,
        receivables_service_token="",
        receivables_service_token_sha256=generated.sha256,
    )
    body = {
        "contact_id": str(uuid4()),
        "payer_name": "Receipt Customer",
        "total_amount_cents": 12_500,
        "payment_method": payment_method,
        "received_date": "2026-08-15",
        **reference_body,
    }
    headers = {
        "Authorization": f"Bearer {generated.token}",
        "X-EOM-Actor": "Juan Canfield",
        "Idempotency-Key": "invalid-reference-before-service-1",
    }

    class _NoPaymentWriteService:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def create_payment(self, **kwargs):
            self.calls.append(kwargs)
            raise AssertionError("invalid payment reference must not reach service")

    for app, config_dependency, service_dependency in (
        (
            main.app,
            full_auth.get_receivables_api_config,
            full_routes.get_receivables_service,
        ),
        (
            main_eom.app,
            slim_auth.get_receivables_api_config,
            slim_routes.get_receivables_service,
        ),
    ):
        service = _NoPaymentWriteService()
        original_overrides = app.dependency_overrides.copy()
        app.dependency_overrides[config_dependency] = lambda: config
        app.dependency_overrides[service_dependency] = lambda: service
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://receivables.test",
            ) as client:
                response = await client.post(
                    "/api/v1/receivables/payments",
                    headers=headers,
                    json=body,
                )

            assert response.status_code == 422, response.text
            assert any(
                error["loc"][-1] == "reference" for error in response.json()["detail"]
            )
            assert service.calls == []
        finally:
            app.dependency_overrides.clear()
            app.dependency_overrides.update(original_overrides)


def test_multi_invoice_mcp_is_registered_with_bounded_strict_schema():
    probe = r'''
import asyncio
from uuid import uuid4

from mcp.server.fastmcp.exceptions import ToolError

from atlas_brain.mcp import invoicing_server


async def main():
    tools = await invoicing_server.mcp.list_tools()
    matching = [tool for tool in tools if tool.name == "record_customer_payment"]
    assert len(matching) == 1
    properties = matching[0].inputSchema["properties"]
    assert properties["total_amount_cents"]["type"] == "integer"
    assert properties["total_amount_cents"]["exclusiveMinimum"] == 0
    assert properties["allocations"]["minItems"] == 1
    assert properties["allocations"]["maxItems"] == 100
    assert properties["payer_name"]["maxLength"] == 256
    assert properties["idempotency_key"]["maxLength"] == 128
    assert "reference" not in matching[0].inputSchema.get("required", [])

    singular = [tool for tool in tools if tool.name == "record_payment"]
    assert len(singular) == 1
    singular_key = singular[0].inputSchema["properties"]["idempotency_key"]
    singular_string = next(
        branch for branch in singular_key["anyOf"] if branch.get("type") == "string"
    )
    assert singular_string["minLength"] == 1
    assert singular_string["maxLength"] == 128

    base = {
        "contact_id": str(uuid4()),
        "payer_name": "Acme",
        "total_amount_cents": 100,
        "payment_method": "check",
        "allocations": [{"invoice_id": str(uuid4()), "amount_cents": 100}],
        "idempotency_key": str(uuid4()),
        "reference": "1001",
    }

    async def assert_rejected(candidate, expected_field):
        try:
            await invoicing_server.mcp.call_tool(
                "record_customer_payment", candidate
            )
        except ToolError as exc:
            message = str(exc)
            assert "1 validation error" in message, message
            assert f"\n{expected_field}\n" in message, message
        else:
            raise AssertionError(f"MCP accepted invalid arguments: {candidate!r}")

    for invalid in (True, False, 100.0, 100.5, "100", 0, -1):
        await assert_rejected(
            {**base, "total_amount_cents": invalid},
            "total_amount_cents",
        )
        await assert_rejected(
            {
                **base,
                "allocations": [
                    {"invoice_id": str(uuid4()), "amount_cents": invalid}
                ],
            },
            "allocations.0.amount_cents",
        )

    for field, value in (
        ("payer_name", "x" * 257),
        ("idempotency_key", "x" * 129),
        ("reference", "x" * 257),
        ("allocations", []),
        (
            "allocations",
            [{"invoice_id": str(uuid4()), "amount_cents": 1}] * 101,
        ),
    ):
        await assert_rejected({**base, field: value}, field)


asyncio.run(main())
'''
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        "clean-process MCP contract probe failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
