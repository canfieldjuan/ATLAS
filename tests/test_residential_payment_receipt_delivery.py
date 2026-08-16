"""Contract tests for explicit, durable residential payment receipt delivery.

All Gmail interactions in this module use a recording fake or an in-process
HTTP transport.  No test reads real mailbox credentials or sends customer mail.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from email import message_from_bytes
from itertools import product
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest

from atlas_brain.services.residential_payment_receipt_delivery import (
    _DeliveryContext,
    ResidentialPaymentReceiptDeliveryConflictError,
    ResidentialPaymentReceiptDeliveryService,
    ResidentialPaymentReceiptDeliveryUnavailableError,
)
from atlas_brain.storage.migrations import run_migrations
from atlas_brain.tools.gmail import GmailSendError, GmailTransport


class _SchemaPool:
    is_initialized = True

    def __init__(self, connection, schema: str) -> None:
        self.connection = connection
        self.schema = schema

    @asynccontextmanager
    async def transaction(self):
        async with self.connection.transaction():
            await self.connection.execute(f'SET LOCAL search_path TO "{self.schema}"')
            yield self.connection


class _MigrationPool:
    is_initialized = True

    def __init__(self, connection, schema: str) -> None:
        self.connection = connection
        self.schema = schema

    async def acquire(self):
        await self.connection.execute(f'SET search_path TO "{self.schema}"')
        return self.connection

    async def release(self, released) -> None:
        assert released is self.connection


class _ReceiptGateway:
    def __init__(self) -> None:
        self.send_calls: list[dict] = []
        self.lookup_calls: list[str] = []
        self.sent_messages: dict[str, dict] = {}
        self.send_error: Exception | None = None
        self.lookup_error: Exception | None = None
        self.record_before_error = False
        self.record_before_error_recipient: str | None = None

    async def send(
        self,
        to: list[str],
        subject: str,
        body: str,
        *,
        headers: dict[str, str],
    ) -> dict:
        call = {
            "to": list(to),
            "subject": subject,
            "body": body,
            "headers": dict(headers),
        }
        self.send_calls.append(call)
        if self.record_before_error:
            self.record_sent(
                rfc_message_id=headers["Message-ID"],
                receipt_delivery_id=UUID(headers["X-Atlas-EOM-Payment-Receipt"]),
                recipient_email=self.record_before_error_recipient or to[0],
            )
        if self.send_error is not None:
            raise self.send_error
        return {
            "id": f"gmail-message-{len(self.send_calls)}",
            "threadId": f"gmail-thread-{len(self.send_calls)}",
        }

    async def find_sent_message_by_rfc_message_id(
        self, rfc_message_id: str
    ) -> dict | None:
        self.lookup_calls.append(rfc_message_id)
        if self.lookup_error is not None:
            raise self.lookup_error
        message = self.sent_messages.get(rfc_message_id)
        return json.loads(json.dumps(message)) if message is not None else None

    def record_sent(
        self,
        *,
        rfc_message_id: str,
        receipt_delivery_id: UUID,
        recipient_email: str,
        internal_date: str = "1775088000123",
    ) -> None:
        self.sent_messages[rfc_message_id] = {
            "id": "sent-message-1",
            "threadId": "sent-thread-1",
            "labelIds": ["SENT"],
            "internalDate": internal_date,
            "headers": [
                {"name": "Message-ID", "value": rfc_message_id},
                {
                    "name": "X-Atlas-EOM-Payment-Receipt",
                    "value": str(receipt_delivery_id),
                },
                {"name": "To", "value": recipient_email},
            ],
        }


class _BlockingReceiptGateway(_ReceiptGateway):
    def __init__(self) -> None:
        super().__init__()
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()

    async def send(self, *args, **kwargs) -> dict:
        call = {
            "to": list(args[0]),
            "subject": args[1],
            "body": args[2],
            "headers": dict(kwargs["headers"]),
        }
        self.send_calls.append(call)
        self.send_started.set()
        await self.release_send.wait()
        return {
            "id": f"gmail-message-{len(self.send_calls)}",
            "threadId": f"gmail-thread-{len(self.send_calls)}",
        }


class _RawSentProofGateway:
    """Return a producer-shaped response without JSON normalizing containers."""

    def __init__(self) -> None:
        self.response: object = None

    async def find_sent_message_by_rfc_message_id(self, _rfc_message_id: str):
        return self.response


@asynccontextmanager
async def _receipt_database():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")
    schema = f"residential_receipt_delivery_{uuid4().hex}"
    connection = await asyncpg.connect(database_url)
    migrations = Path(__file__).parents[1] / "atlas_brain/storage/migrations"
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        await connection.execute("CREATE TABLE contacts (id UUID PRIMARY KEY)")
        await connection.execute("CREATE TABLE customer_payments (id UUID PRIMARY KEY)")
        await run_migrations(
            _MigrationPool(connection, schema),
            migrations_dir=migrations,
            only=(
                "369_receivables_payment_receipt_outbox",
                "378_receivables_payment_receipt_delivery",
            ),
        )
        yield connection, schema
    finally:
        await connection.execute("SET search_path TO public")
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


async def _seed_receipt_delivery(
    connection,
    *,
    recipient_email: str | None = "riley@example.test",
    delivery_status: str = "pending",
) -> dict:
    payment_id = uuid4()
    contact_id = uuid4()
    delivery_id = uuid4()
    rfc_message_id = (
        f"<atlas-eom-payment-receipt-{delivery_id}@effinghamofficemaids.com>"
    )
    await connection.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
    await connection.execute(
        "INSERT INTO customer_payments (id) VALUES ($1)", payment_id
    )
    await connection.execute(
        """
        INSERT INTO payment_receipt_deliveries (
            id, payment_id, contact_id, receipt_number, recipient_email,
            delivery_status, skip_reason, subject, body, rfc_message_id,
            created_at, updated_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $11)
        """,
        delivery_id,
        payment_id,
        contact_id,
        f"EOM-RCP-{payment_id}",
        recipient_email,
        delivery_status,
        "no_email" if delivery_status == "skipped" else None,
        "Payment received - receipt",
        "Thank you. Your payment has been received, not cleared.",
        rfc_message_id,
        datetime(2026, 8, 15, tzinfo=timezone.utc),
    )
    return {
        "contact_id": contact_id,
        "delivery_id": delivery_id,
        "payment_id": payment_id,
        "rfc_message_id": rfc_message_id,
    }


def _service(
    connection,
    schema: str,
    gateway: _ReceiptGateway,
    *,
    now: datetime = datetime(2026, 8, 15, 18, 30, tzinfo=timezone.utc),
):
    return ResidentialPaymentReceiptDeliveryService(
        pool=_SchemaPool(connection, schema),
        gmail_gateway=gateway,
        now=lambda: now,
    )


@pytest.mark.asyncio
async def test_migration_preserves_legacy_receipt_writer_and_records_stable_identity():
    async with _receipt_database() as (connection, _schema):
        payment_id, contact_id, delivery_id = uuid4(), uuid4(), uuid4()
        await connection.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        await connection.execute(
            "INSERT INTO customer_payments (id) VALUES ($1)", payment_id
        )
        row = await connection.fetchrow(
            """
            INSERT INTO payment_receipt_deliveries (
                id, payment_id, contact_id, receipt_number, recipient_email,
                delivery_status, skip_reason, subject, body
            )
            VALUES ($1, $2, $3, $4, $5, 'pending', NULL, 'Receipt', 'Body')
            RETURNING rfc_message_id
            """,
            delivery_id,
            payment_id,
            contact_id,
            f"EOM-RCP-{payment_id}",
            "riley@example.test",
        )

        assert row["rfc_message_id"].startswith("<atlas-eom-payment-receipt-")
        assert row["rfc_message_id"].endswith("@effinghamofficemaids.com>")
        assert await connection.fetchval(
            "SELECT to_regclass('payment_receipt_delivery_operations') IS NOT NULL"
        ) is True


@pytest.mark.asyncio
async def test_migration_retry_rebuilds_an_invalid_concurrent_rfc_message_id_index():
    """A partial concurrent build must not be mistaken for the uniqueness guard."""

    async with _receipt_database() as (connection, schema):
        first = await _seed_receipt_delivery(connection)
        second = await _seed_receipt_delivery(connection)
        migration_name = "378_receivables_payment_receipt_delivery"
        migrations = Path(__file__).parents[1] / "atlas_brain/storage/migrations"

        await connection.execute(
            "DELETE FROM schema_migrations WHERE name = $1", migration_name
        )
        await connection.execute(
            "DROP INDEX CONCURRENTLY idx_payment_receipt_deliveries_rfc_message_id"
        )
        await connection.execute(
            """
            UPDATE payment_receipt_deliveries
            SET rfc_message_id = $2
            WHERE id = $1
            """,
            second["delivery_id"],
            first["rfc_message_id"],
        )
        asyncpg = pytest.importorskip("asyncpg")
        with pytest.raises(asyncpg.UniqueViolationError):
            await connection.execute(
                """
                CREATE UNIQUE INDEX CONCURRENTLY
                    idx_payment_receipt_deliveries_rfc_message_id
                ON payment_receipt_deliveries (rfc_message_id)
                """
            )
        assert await connection.fetchval(
            """
            SELECT index_state.indisvalid
            FROM pg_index AS index_state
            JOIN pg_class AS index_class ON index_class.oid = index_state.indexrelid
            WHERE index_class.relname = 'idx_payment_receipt_deliveries_rfc_message_id'
              AND index_class.relnamespace = current_schema()::regnamespace
            """
        ) is False

        await connection.execute(
            """
            UPDATE payment_receipt_deliveries
            SET rfc_message_id = $2
            WHERE id = $1
            """,
            second["delivery_id"],
            second["rfc_message_id"],
        )
        await run_migrations(
            _MigrationPool(connection, schema),
            migrations_dir=migrations,
            only=(migration_name,),
        )

        assert await connection.fetchval(
            """
            SELECT index_state.indisvalid
            FROM pg_index AS index_state
            JOIN pg_class AS index_class ON index_class.oid = index_state.indexrelid
            WHERE index_class.relname = 'idx_payment_receipt_deliveries_rfc_message_id'
              AND index_class.relnamespace = current_schema()::regnamespace
            """
        ) is True
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM schema_migrations WHERE name = $1", migration_name
        ) == 1


@pytest.mark.asyncio
async def test_receipt_dispatch_sends_once_and_replays_without_financial_mutation():
    async with _receipt_database() as (connection, schema):
        seed = await _seed_receipt_delivery(connection)
        gateway = _ReceiptGateway()
        service = _service(connection, schema, gateway)
        first_key, second_key = "one", "two"

        first = await service.dispatch(
            payment_id=seed["payment_id"], idempotency_key=first_key, actor="Juan"
        )
        replay = await service.dispatch(
            payment_id=seed["payment_id"], idempotency_key=first_key, actor="Mayra"
        )
        reused = await service.dispatch(
            payment_id=seed["payment_id"], idempotency_key=second_key, actor="Mayra"
        )

        assert first["receipt_delivery"]["status"] == "sent"
        assert first["operation"] == {
            "state": "completed",
            "outcome": "sent",
            "requested_at": "2026-08-15T18:30:00+00:00",
            "completed_at": "2026-08-15T18:30:00+00:00",
        }
        assert first["replayed"] is False
        assert first["reused"] is False
        assert replay["replayed"] is True
        assert replay["reused"] is True
        assert reused["replayed"] is False
        assert reused["reused"] is True
        assert reused["operation"]["outcome"] == "already_sent"
        assert len(gateway.send_calls) == 1
        assert gateway.send_calls[0]["to"] == ["riley@example.test"]
        assert gateway.send_calls[0]["headers"] == {
            "Message-ID": seed["rfc_message_id"],
            "X-Atlas-EOM-Payment-Receipt": str(seed["delivery_id"]),
        }
        assert await connection.fetchval("SELECT COUNT(*) FROM customer_payments") == 1
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM payment_receipt_delivery_operations"
        ) == 2
        delivery = await connection.fetchrow(
            """
            SELECT delivery_status, gmail_message_id, gmail_thread_id, sent_at,
                   recovery_required_at
            FROM payment_receipt_deliveries
            WHERE id = $1
            """,
            seed["delivery_id"],
        )
        assert dict(delivery) == {
            "delivery_status": "sent",
            "gmail_message_id": "gmail-message-1",
            "gmail_thread_id": "gmail-thread-1",
            "sent_at": datetime(2026, 8, 15, 18, 30, tzinfo=timezone.utc),
            "recovery_required_at": None,
        }


@pytest.mark.asyncio
async def test_preflight_sent_proof_completes_a_prepared_operation_without_sending():
    async with _receipt_database() as (connection, schema):
        seed = await _seed_receipt_delivery(connection)
        gateway = _ReceiptGateway()
        gateway.record_sent(
            rfc_message_id=seed["rfc_message_id"],
            receipt_delivery_id=seed["delivery_id"],
            recipient_email="riley@example.test",
        )
        service = _service(connection, schema, gateway)

        result = await service.dispatch(
            payment_id=seed["payment_id"],
            idempotency_key="receipt-preflight-proof-1",
            actor="Juan",
        )
        replay = await service.dispatch(
            payment_id=seed["payment_id"],
            idempotency_key="receipt-preflight-proof-1",
            actor="Mayra",
        )

        assert result["receipt_delivery"]["status"] == "sent"
        assert result["operation"]["state"] == "completed"
        assert result["operation"]["outcome"] == "already_sent"
        assert replay["receipt_delivery"]["status"] == "sent"
        assert replay["replayed"] is True
        assert gateway.send_calls == []
        operation = await connection.fetchrow(
            """
            SELECT state, outcome, attempt_started_at, result_delivery_status,
                   result_sent_at
            FROM payment_receipt_delivery_operations
            WHERE idempotency_key = 'receipt-preflight-proof-1'
            """
        )
        assert dict(operation) == {
            "state": "completed",
            "outcome": "already_sent",
            "attempt_started_at": None,
            "result_delivery_status": "sent",
            "result_sent_at": datetime.fromtimestamp(1775088000.123, tz=timezone.utc),
        }


@pytest.mark.asyncio
async def test_skipped_receipt_without_an_email_never_loads_or_sends_gmail():
    async with _receipt_database() as (connection, schema):
        seed = await _seed_receipt_delivery(
            connection,
            recipient_email=None,
            delivery_status="skipped",
        )
        gateway = _ReceiptGateway()

        with pytest.raises(ResidentialPaymentReceiptDeliveryConflictError):
            await _service(connection, schema, gateway).dispatch(
                payment_id=seed["payment_id"],
                idempotency_key="receipt-skipped-1",
                actor="Juan",
            )

        assert gateway.send_calls == []
        assert gateway.lookup_calls == []


@pytest.mark.asyncio
async def test_unavailable_sent_lookup_fails_closed_before_a_new_gmail_send():
    async with _receipt_database() as (connection, schema):
        seed = await _seed_receipt_delivery(connection)
        gateway = _ReceiptGateway()
        gateway.lookup_error = RuntimeError("synthetic Gmail lookup outage")

        with pytest.raises(ResidentialPaymentReceiptDeliveryUnavailableError):
            await _service(connection, schema, gateway).dispatch(
                payment_id=seed["payment_id"],
                idempotency_key="receipt-lookup-outage-1",
                actor="Juan",
            )

        assert gateway.lookup_calls == [seed["rfc_message_id"]]
        assert gateway.send_calls == []


@pytest.mark.asyncio
async def test_mismatched_sent_mail_recipient_is_not_accepted_as_receipt_evidence():
    async with _receipt_database() as (connection, schema):
        seed = await _seed_receipt_delivery(connection)
        gateway = _ReceiptGateway()
        gateway.record_sent(
            rfc_message_id=seed["rfc_message_id"],
            receipt_delivery_id=seed["delivery_id"],
            recipient_email="wrong-recipient@example.test",
        )

        with pytest.raises(ResidentialPaymentReceiptDeliveryConflictError):
            await _service(connection, schema, gateway).dispatch(
                payment_id=seed["payment_id"],
                idempotency_key="one",
                actor="Juan",
            )

        assert gateway.send_calls == []


@pytest.mark.asyncio
async def test_sent_mail_proof_grammar_uses_a_spec_derived_oracle():
    """Exercise tokens x containers x key families at the Gmail proof choke point.

    Gmail's metadata response is producer-supplied and therefore open.  The
    independent oracle below admits only the documented JSON-list shape with
    one exact member from each server-owned proof-header family; every other
    grammar product must reject the evidence before a receipt can be recorded
    as sent. Header-name casing is intentionally neutral because RFC headers
    are case-insensitive.
    """

    delivery_id = uuid4()
    rfc_message_id = f"<atlas-eom-payment-receipt-{delivery_id}@example.test>"
    context = _DeliveryContext(
        delivery_id=delivery_id,
        payment_id=uuid4(),
        recipient_email="riley@example.test",
        receipt_number="EOM-RCP-proof-grammar",
        rfc_message_id=rfc_message_id,
        subject="Receipt",
        body="Body",
    )
    gateway = _RawSentProofGateway()
    service = ResidentialPaymentReceiptDeliveryService(gmail_gateway=gateway)

    # Grammar axes: tokens x containers x key families. Names are not the
    # oracle: HTTP/Gmail header names are case-insensitive by specification.
    name_form_builders = {
        "canonical": lambda name: name,
        "lowercase": lambda name: name.lower(),
    }
    header_container_builders = {
        "json_list": lambda items: list(items),
        "tuple": lambda items: tuple(items),
        "mapping": lambda items: {"items": list(items)},
    }
    header_key_family_builders = {
        "all_required_once": lambda items: items,
        "missing_receipt": lambda items: items[:1] + items[2:],
        "duplicate_message_id": lambda items: [items[0], *items],
    }
    message_id_value_builders = {
        "exact": lambda value: value,
        "affixed": lambda value: f"x{value}",
        "whitespace": lambda value: f"{value} ",
    }

    for (
        name_form,
        container_name,
        key_family,
        message_id_form,
    ) in product(
        name_form_builders,
        header_container_builders,
        header_key_family_builders,
        message_id_value_builders,
    ):
        name = name_form_builders[name_form]
        message_id_value = message_id_value_builders[message_id_form](
            rfc_message_id
        )
        exact_headers = [
            {"name": name("Message-ID"), "value": message_id_value},
            {
                "name": name("X-Atlas-EOM-Payment-Receipt"),
                "value": str(delivery_id),
            },
            {"name": name("To"), "value": "riley@example.test"},
        ]
        headers = header_container_builders[container_name](
            header_key_family_builders[key_family](exact_headers)
        )
        gateway.response = {
            "id": "sent-message-grammar",
            "threadId": "sent-thread-grammar",
            "labelIds": ["SENT"],
            "internalDate": "1775088000123",
            "headers": headers,
        }

        expected_by_spec_oracle = (
            container_name == "json_list"
            and key_family == "all_required_once"
            and message_id_form == "exact"
        )
        if expected_by_spec_oracle:
            proof = await service._lookup_sent(context)
            assert proof.gmail_message_id == "sent-message-grammar"
        else:
            with pytest.raises(ResidentialPaymentReceiptDeliveryConflictError):
                await service._lookup_sent(context)


@pytest.mark.asyncio
async def test_definite_gmail_rejection_is_failed_and_a_new_key_can_retry():
    async with _receipt_database() as (connection, schema):
        seed = await _seed_receipt_delivery(connection)
        gateway = _ReceiptGateway()
        gateway.send_error = GmailSendError(
            "synthetic rejection", definitely_not_sent=True
        )
        service = _service(connection, schema, gateway)

        failed = await service.dispatch(
            payment_id=seed["payment_id"], idempotency_key="receipt-rejected-1", actor="Juan"
        )
        gateway.send_error = None
        recovered = await service.dispatch(
            payment_id=seed["payment_id"], idempotency_key="receipt-rejected-2", actor="Juan"
        )
        failed_replay = await service.dispatch(
            payment_id=seed["payment_id"], idempotency_key="receipt-rejected-1", actor="Mayra"
        )

        assert failed["receipt_delivery"]["status"] == "failed"
        assert failed["operation"]["outcome"] == "failed"
        assert recovered["receipt_delivery"]["status"] == "sent"
        assert recovered["operation"]["outcome"] == "sent"
        assert failed_replay["receipt_delivery"] == {
            "receipt_number": f"EOM-RCP-{seed['payment_id']}",
            "recipient_email": "riley@example.test",
            "status": "failed",
            "skip_reason": None,
            "sent_at": None,
            "recovery_required_at": None,
        }
        assert failed_replay["operation"]["outcome"] == "failed"
        assert failed_replay["replayed"] is True
        assert len(gateway.send_calls) == 2
        stored = await connection.fetchrow(
            """
            SELECT delivery_status, last_failure_code, last_failure_at
            FROM payment_receipt_deliveries WHERE id = $1
            """,
            seed["delivery_id"],
        )
        assert dict(stored) == {
            "delivery_status": "sent",
            "last_failure_code": "gmail_rejected",
            "last_failure_at": datetime(2026, 8, 15, 18, 30, tzinfo=timezone.utc),
        }


@pytest.mark.asyncio
async def test_unknown_gmail_result_never_resends_and_can_later_reconcile_sent_evidence():
    async with _receipt_database() as (connection, schema):
        seed = await _seed_receipt_delivery(connection)
        gateway = _ReceiptGateway()
        gateway.send_error = GmailSendError("synthetic timeout", definitely_not_sent=False)
        service = _service(connection, schema, gateway)

        uncertain = await service.dispatch(
            payment_id=seed["payment_id"], idempotency_key="receipt-unknown-1", actor="Juan"
        )

        assert uncertain["receipt_delivery"]["status"] == "pending"
        assert uncertain["receipt_delivery"]["recovery_required_at"] is not None
        assert uncertain["operation"]["state"] == "recovery_required"
        assert len(gateway.send_calls) == 1
        with pytest.raises(ResidentialPaymentReceiptDeliveryConflictError):
            await service.dispatch(
                payment_id=seed["payment_id"],
                idempotency_key="receipt-unknown-new-key",
                actor="Mayra",
            )
        assert len(gateway.send_calls) == 1

        gateway.record_sent(
            rfc_message_id=seed["rfc_message_id"],
            receipt_delivery_id=seed["delivery_id"],
            recipient_email="riley@example.test",
        )
        reconciled = await _service(connection, schema, gateway).reconcile(
            payment_id=seed["payment_id"], actor="Mayra"
        )

        assert reconciled["receipt_delivery"]["status"] == "sent"
        assert reconciled["receipt_delivery"]["recovery_required_at"] is None
        assert reconciled["operation"]["outcome"] == "sent"
        assert len(gateway.send_calls) == 1
        reconciliation_event = await connection.fetchrow(
            """
            SELECT actor, outcome, reconciled_at
            FROM payment_receipt_delivery_reconciliation_events
            WHERE receipt_delivery_id = $1 AND outcome = 'sent'
            ORDER BY reconciled_at DESC, id DESC
            LIMIT 1
            """,
            seed["delivery_id"],
        )
        assert dict(reconciliation_event) == {
            "actor": "Mayra",
            "outcome": "sent",
            "reconciled_at": datetime(2026, 8, 15, 18, 30, tzinfo=timezone.utc),
        }


@pytest.mark.asyncio
async def test_invalid_sent_proof_after_ambiguous_send_requires_recovery_without_resend():
    async with _receipt_database() as (connection, schema):
        seed = await _seed_receipt_delivery(connection)
        gateway = _ReceiptGateway()
        gateway.record_before_error = True
        gateway.record_before_error_recipient = "wrong-recipient@example.test"
        gateway.send_error = GmailSendError("synthetic timeout", definitely_not_sent=False)

        result = await _service(connection, schema, gateway).dispatch(
            payment_id=seed["payment_id"],
            idempotency_key="receipt-invalid-proof-after-send-1",
            actor="Juan",
        )

        assert result["receipt_delivery"]["status"] == "pending"
        assert result["receipt_delivery"]["recovery_required_at"] is not None
        assert result["operation"]["state"] == "recovery_required"
        assert len(gateway.send_calls) == 1


@pytest.mark.asyncio
async def test_no_sent_reconciliation_attempt_is_actor_audited_without_a_second_send():
    async with _receipt_database() as (connection, schema):
        seed = await _seed_receipt_delivery(connection)
        gateway = _ReceiptGateway()
        gateway.send_error = GmailSendError("synthetic timeout", definitely_not_sent=False)
        service = _service(connection, schema, gateway)

        await service.dispatch(
            payment_id=seed["payment_id"], idempotency_key="receipt-audit-1", actor="Juan"
        )
        reconciled = await service.reconcile(
            payment_id=seed["payment_id"], actor="Mayra"
        )

        assert reconciled["receipt_delivery"]["status"] == "pending"
        assert reconciled["operation"]["state"] == "recovery_required"
        assert len(gateway.send_calls) == 1
        events = await connection.fetch(
            """
            SELECT actor, outcome
            FROM payment_receipt_delivery_reconciliation_events
            WHERE receipt_delivery_id = $1
            """,
            seed["delivery_id"],
        )
        assert {(event["actor"], event["outcome"]) for event in events} == {
            ("Juan", "recovery_required"),
            ("Mayra", "recovery_required"),
        }


@pytest.mark.asyncio
async def test_reconcile_refuses_queued_receipt_without_reaching_gmail():
    async with _receipt_database() as (connection, schema):
        seed = await _seed_receipt_delivery(connection)
        gateway = _ReceiptGateway()

        with pytest.raises(ResidentialPaymentReceiptDeliveryConflictError, match="ambiguous"):
            await _service(connection, schema, gateway).reconcile(
                payment_id=seed["payment_id"], actor="Juan"
            )

        assert gateway.send_calls == []
        assert gateway.lookup_calls == []


@pytest.mark.asyncio
async def test_unknown_gmail_result_with_immediate_sent_proof_is_recorded_without_duplicate_send():
    async with _receipt_database() as (connection, schema):
        seed = await _seed_receipt_delivery(connection)
        gateway = _ReceiptGateway()
        gateway.record_before_error = True
        gateway.send_error = GmailSendError("synthetic timeout", definitely_not_sent=False)

        result = await _service(connection, schema, gateway).dispatch(
            payment_id=seed["payment_id"], idempotency_key="receipt-proven-1", actor="Juan"
        )

        assert result["receipt_delivery"]["status"] == "sent"
        assert result["operation"]["outcome"] == "sent"
        assert len(gateway.send_calls) == 1


@pytest.mark.asyncio
async def test_interrupted_attempt_recovers_by_lookup_without_a_second_send():
    async with _receipt_database() as (connection, schema):
        seed = await _seed_receipt_delivery(connection)
        gateway = _BlockingReceiptGateway()
        initial_now = datetime(2026, 8, 15, 18, 30, tzinfo=timezone.utc)
        in_flight = asyncio.create_task(
            _service(connection, schema, gateway, now=initial_now).dispatch(
                payment_id=seed["payment_id"],
                idempotency_key="receipt-interrupted-1",
                actor="Juan",
            )
        )
        await gateway.send_started.wait()
        in_flight.cancel()
        with pytest.raises(asyncio.CancelledError):
            await in_flight

        recovered = await _service(
            connection,
            schema,
            gateway,
            now=datetime(2026, 8, 15, 18, 36, tzinfo=timezone.utc),
        ).dispatch(
            payment_id=seed["payment_id"],
            idempotency_key="receipt-interrupted-1",
            actor="Mayra",
        )

        assert recovered["receipt_delivery"]["status"] == "pending"
        assert recovered["receipt_delivery"]["recovery_required_at"] is not None
        assert recovered["operation"]["state"] == "recovery_required"
        assert len(gateway.send_calls) == 1


@pytest.mark.asyncio
async def test_interrupted_attempt_with_invalid_sent_proof_requires_recovery_without_a_second_send():
    async with _receipt_database() as (connection, schema):
        seed = await _seed_receipt_delivery(connection)
        gateway = _BlockingReceiptGateway()
        initial_now = datetime(2026, 8, 15, 18, 30, tzinfo=timezone.utc)
        in_flight = asyncio.create_task(
            _service(connection, schema, gateway, now=initial_now).dispatch(
                payment_id=seed["payment_id"],
                idempotency_key="bad-proof",
                actor="Juan",
            )
        )
        await gateway.send_started.wait()
        in_flight.cancel()
        with pytest.raises(asyncio.CancelledError):
            await in_flight
        gateway.record_sent(
            rfc_message_id=seed["rfc_message_id"],
            receipt_delivery_id=seed["delivery_id"],
            recipient_email="wrong-recipient@example.test",
        )

        recovered = await _service(
            connection,
            schema,
            gateway,
            now=datetime(2026, 8, 15, 18, 36, tzinfo=timezone.utc),
        ).dispatch(
            payment_id=seed["payment_id"],
            idempotency_key="bad-proof",
            actor="Mayra",
        )

        assert recovered["receipt_delivery"]["status"] == "pending"
        assert recovered["receipt_delivery"]["recovery_required_at"] is not None
        assert recovered["operation"]["state"] == "recovery_required"
        assert len(gateway.send_calls) == 1


@pytest.mark.asyncio
async def test_concurrent_same_key_requests_issue_one_gmail_send():
    async with _receipt_database() as (connection, schema):
        seed = await _seed_receipt_delivery(connection)
        gateway = _BlockingReceiptGateway()
        asyncpg = pytest.importorskip("asyncpg")
        first_connection = await asyncpg.connect(
            os.environ["ATLAS_RECEIVABLES_TEST_DATABASE_URL"]
        )
        second_connection = await asyncpg.connect(
            os.environ["ATLAS_RECEIVABLES_TEST_DATABASE_URL"]
        )
        first_task = None
        try:
            first_service = _service(first_connection, schema, gateway)
            second_service = _service(second_connection, schema, gateway)

            first_task = asyncio.create_task(
                first_service.dispatch(
                    payment_id=seed["payment_id"],
                    idempotency_key="receipt-concurrent-1",
                    actor="Juan",
                )
            )
            await gateway.send_started.wait()
            second = await second_service.dispatch(
                payment_id=seed["payment_id"],
                idempotency_key="receipt-concurrent-1",
                actor="Mayra",
            )
            gateway.release_send.set()
            first = await first_task
        finally:
            gateway.release_send.set()
            if first_task is not None and not first_task.done():
                with suppress(Exception):
                    await first_task
            await first_connection.close()
            await second_connection.close()

        assert first["receipt_delivery"]["status"] == "sent"
        assert second["operation"]["state"] == "attempting"
        assert second["replayed"] is True
        assert len(gateway.send_calls) == 1


@pytest.mark.asyncio
async def test_concurrent_distinct_keys_serialize_on_the_delivery_without_a_second_send():
    async with _receipt_database() as (connection, schema):
        seed = await _seed_receipt_delivery(connection)
        gateway = _BlockingReceiptGateway()
        asyncpg = pytest.importorskip("asyncpg")
        first_connection = await asyncpg.connect(
            os.environ["ATLAS_RECEIVABLES_TEST_DATABASE_URL"]
        )
        second_connection = await asyncpg.connect(
            os.environ["ATLAS_RECEIVABLES_TEST_DATABASE_URL"]
        )
        first_task = None
        try:
            first_service = _service(first_connection, schema, gateway)
            second_service = _service(second_connection, schema, gateway)
            first_key, second_key = "one", "two"
            first_task = asyncio.create_task(
                first_service.dispatch(
                    payment_id=seed["payment_id"],
                    idempotency_key=first_key,
                    actor="Juan",
                )
            )
            await gateway.send_started.wait()
            with pytest.raises(ResidentialPaymentReceiptDeliveryConflictError):
                await asyncio.wait_for(
                    second_service.dispatch(
                        payment_id=seed["payment_id"],
                        idempotency_key=second_key,
                        actor="Mayra",
                    ),
                    timeout=5,
                )
            gateway.release_send.set()
            first = await first_task
        finally:
            gateway.release_send.set()
            if first_task is not None and not first_task.done():
                with suppress(Exception):
                    await first_task
            await first_connection.close()
            await second_connection.close()

        assert first["receipt_delivery"]["status"] == "sent"
        assert len(gateway.send_calls) == 1


@pytest.mark.asyncio
async def test_same_key_cannot_be_reused_for_a_different_payment():
    async with _receipt_database() as (connection, schema):
        first_seed = await _seed_receipt_delivery(connection)
        second_seed = await _seed_receipt_delivery(connection)
        gateway = _ReceiptGateway()
        service = _service(connection, schema, gateway)

        await service.dispatch(
            payment_id=first_seed["payment_id"], idempotency_key="receipt-shared-key", actor="Juan"
        )
        with pytest.raises(ResidentialPaymentReceiptDeliveryConflictError):
            await service.dispatch(
                payment_id=second_seed["payment_id"],
                idempotency_key="receipt-shared-key",
                actor="Juan",
            )

        assert len(gateway.send_calls) == 1
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM payment_receipt_delivery_operations"
        ) == 1


@pytest.mark.asyncio
async def test_gmail_transport_send_preserves_immutable_headers_and_classifies_outcomes():
    requests: list[httpx.Request] = []

    async def accepted(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": "message-1", "threadId": "thread-1"})

    transport = GmailTransport()
    transport._access_token = "test-token"
    transport._token_expires = time.time() + 600
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(accepted))
    try:
        result = await transport.send(
            to=["riley@example.test"],
            subject="Receipt",
            body="Body",
            headers={
                "Message-ID": "<atlas-eom-payment-receipt-test@example.test>",
                "X-Atlas-EOM-Payment-Receipt": "receipt-1",
            },
        )
    finally:
        await transport.close()

    assert result == {"id": "message-1", "threadId": "thread-1"}
    assert requests[0].url.path.endswith("/users/me/messages/send")
    raw = json.loads(requests[0].content)["raw"]
    raw_bytes = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    message = message_from_bytes(raw_bytes)
    assert message["To"] == "riley@example.test"
    assert message["Message-ID"] == "<atlas-eom-payment-receipt-test@example.test>"
    assert message["X-Atlas-EOM-Payment-Receipt"] == "receipt-1"

    async def rejected(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "busy"}})

    transport = GmailTransport()
    transport._access_token = "test-token"
    transport._token_expires = time.time() + 600
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(rejected))
    try:
        with pytest.raises(GmailSendError) as rejected_error:
            await transport.send(to=["riley@example.test"], subject="Receipt", body="Body")
    finally:
        await transport.close()
    assert rejected_error.value.definitely_not_sent is True

    async def uncertain(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "retry"}})

    transport = GmailTransport()
    transport._access_token = "test-token"
    transport._token_expires = time.time() + 600
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(uncertain))
    try:
        with pytest.raises(GmailSendError) as uncertain_error:
            await transport.send(to=["riley@example.test"], subject="Receipt", body="Body")
    finally:
        await transport.close()
    assert uncertain_error.value.definitely_not_sent is False
