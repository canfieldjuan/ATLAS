from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from atlas_brain.api.comms import webhooks
from atlas_brain.comms import sms_intelligence
from atlas_brain.services import crm_provider
from atlas_brain.storage import database as storage_database
from atlas_brain.storage.repositories import sms_message


class _ClaimingSMSRepo:
    def __init__(self, *, claim_result: bool, claim_exc: Exception | None = None):
        self.claim_result = claim_result
        self.claim_exc = claim_exc
        self.claimed_ids = []
        self.linked_contacts = []
        self.retry_pending = []
        self.completed = []
        self.owner_tokens = {}
        self.touched = []
        self.owner_status_updates = []
        self.rows = {}
        self.auto_reply_rows = {}
        self.auto_reply_sending = []
        self.auto_reply_sent = []
        self.complete_exc = None
        self.complete_result = True
        self.mark_auto_reply_sent_exc = None
        self.mark_auto_reply_sent_result = True

    async def claim_contact_processing(self, sms_id, *, owner_token=None):
        self.claimed_ids.append(sms_id)
        if self.claim_exc is not None:
            raise self.claim_exc
        if self.claim_result and owner_token is not None:
            self.owner_tokens[sms_id] = owner_token
            row = dict(self.rows.get(sms_id, {"id": sms_id, "contact_id": None}))
            row.update(
                {
                    "status": "processing",
                    "error_message": owner_token,
                    "_claim_acquired": True,
                }
            )
            self.rows[sms_id] = row
            return row
        row = dict(self.rows.get(sms_id, {"id": sms_id, "contact_id": None}))
        row.setdefault("status", "processing")
        row["_claim_acquired"] = False
        return row

    async def owns_contact_processing(self, sms_id, owner_token):
        return self.owner_tokens.get(sms_id) == owner_token

    async def touch_contact_processing_owner(self, sms_id, owner_token):
        self.touched.append((sms_id, owner_token))
        return self.owner_tokens.get(sms_id) == owner_token

    async def link_contact(self, sms_id, contact_id):
        self.linked_contacts.append((sms_id, contact_id))

    async def mark_contact_processing_retry_pending(
        self,
        sms_id,
        error_message=None,
        *,
        owner_token=None,
    ):
        self.retry_pending.append((sms_id, error_message, owner_token))

    async def mark_contact_processing_complete(self, sms_id, *, owner_token=None):
        if self.complete_exc is not None:
            raise self.complete_exc
        self.completed.append((sms_id, owner_token))
        return self.complete_result

    async def get_by_id(self, sms_id):
        return self.rows.get(sms_id)

    async def has_auto_reply_for_inbound(self, inbound_sms_id):
        return inbound_sms_id in self.auto_reply_rows

    async def get_auto_reply_for_inbound(self, inbound_sms_id):
        return self.auto_reply_rows.get(inbound_sms_id)

    async def reserve_auto_reply_for_inbound(
        self,
        *,
        inbound_sms_id,
        from_number,
        to_number,
        body,
        business_context_id,
    ):
        if inbound_sms_id in self.auto_reply_rows:
            return None
        row = {
            "id": uuid4(),
            "message_sid": f"auto_reply_{inbound_sms_id}",
            "from_number": from_number,
            "to_number": to_number,
            "body": body,
            "business_context_id": business_context_id,
            "status": "pending",
            "source": "auto_reply",
            "source_ref": str(inbound_sms_id),
        }
        self.auto_reply_rows[inbound_sms_id] = row
        return row

    async def mark_auto_reply_sending(self, auto_reply_sms_id):
        self.auto_reply_sending.append(auto_reply_sms_id)
        for row in self.auto_reply_rows.values():
            if row["id"] == auto_reply_sms_id and row["status"] == "pending":
                row["status"] = "sending"
                return True
        return False

    async def mark_auto_reply_sent(self, auto_reply_sms_id, *, provider_message_id=None):
        if self.mark_auto_reply_sent_exc is not None:
            raise self.mark_auto_reply_sent_exc
        self.auto_reply_sent.append((auto_reply_sms_id, provider_message_id))
        for row in self.auto_reply_rows.values():
            if row["id"] == auto_reply_sms_id:
                if row["status"] not in {"sending", "sent"}:
                    return False
                row["status"] = "sent"
                if provider_message_id:
                    row["message_sid"] = provider_message_id
                row["error_message"] = None
                return self.mark_auto_reply_sent_result
        return False

    async def update_contact_processing_status(
        self,
        sms_id,
        status,
        *,
        owner_token,
        error_message=None,
        clear_owner=False,
    ):
        self.owner_status_updates.append(
            (sms_id, status, owner_token, error_message, clear_owner)
        )
        if self.owner_tokens.get(sms_id) != owner_token:
            return False
        if clear_owner:
            self.owner_tokens.pop(sms_id, None)
        return True


class _FormRequest:
    async def form(self):
        return {}


class _SlowFormRequest:
    async def form(self):
        await asyncio.sleep(1)
        return {}


class _Provider:
    def __init__(self, *, send_delay: float = 0.0, send_exc: Exception | None = None):
        self.sent_sms = []
        self.send_delay = send_delay
        self.send_exc = send_exc

    async def handle_incoming_sms(self, **kwargs):
        return SimpleNamespace(context_id=None)

    async def send_sms(self, **kwargs):
        if self.send_delay:
            await asyncio.sleep(self.send_delay)
        if self.send_exc is not None:
            raise self.send_exc
        self.sent_sms.append(kwargs)
        return SimpleNamespace(provider_message_id=f"SM-out-{len(self.sent_sms)}")


class _Router:
    def __init__(self, context=None):
        self.context = context or _sms_context()

    def get_context_for_number(self, number):
        return self.context

    def get_context(self, context_id):
        return self.context if context_id == self.context.id else None


class _WebhookRepo(_ClaimingSMSRepo):
    def __init__(self, *, existing=None):
        super().__init__(claim_result=True)
        self.existing = existing
        self.created = []

    async def get_by_message_sid(self, message_sid):
        return self.existing

    async def create(self, **kwargs):
        row = {"id": uuid4(), **kwargs}
        self.created.append(row)
        return row


def _sms_context() -> SimpleNamespace:
    return SimpleNamespace(
        id="effingham_maids",
        sms_auto_reply=False,
        sms_enabled=False,
    )


class _ClaimPool:
    is_initialized = True

    def __init__(self):
        self.calls = []

    async def fetchrow(self, sql, *args):
        self.calls.append((sql, args))
        return {
            "id": args[0],
            "claim_acquired": True,
            "media_urls": [],
            "extracted_data": {},
        }


class _RetryPendingPool:
    is_initialized = True

    def __init__(self):
        self.calls = []

    async def execute(self, sql, *args):
        self.calls.append((sql, args))
        return "UPDATE 1"


class _LiveSMSPool:
    is_initialized = True

    def __init__(self, pool):
        self.pool = pool

    async def fetchrow(self, sql, *args):
        return await self.pool.fetchrow(sql, *args)

    async def execute(self, sql, *args):
        return await self.pool.execute(sql, *args)


class _AttrReplacements:
    def __init__(self):
        self._originals = []

    def replace(self, obj, name, value):
        self._originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def restore(self):
        while self._originals:
            obj, name, value = self._originals.pop()
            setattr(obj, name, value)


@pytest.mark.asyncio
async def test_real_repository_claim_uses_reclaimable_processing_predicate():
    pool = _ClaimPool()
    sms_id = uuid4()
    replacements = _AttrReplacements()
    replacements.replace(sms_message, "get_db_pool", lambda: pool)

    try:
        claimed = await sms_message.SMSMessageRepository().claim_contact_processing(sms_id)
    finally:
        replacements.restore()

    assert claimed["_claim_acquired"] is True
    sql, args = pool.calls[0]
    assert "status IN ('received', 'retry_pending')" in sql
    assert "status = 'processing'" in sql
    assert "CURRENT_TIMESTAMP" in sql
    assert "processed_at IS NULL" in sql
    assert "CURRENT_TIMESTAMP - ($2 * INTERVAL '1 second')" in sql
    assert "error_message = $3" in sql
    assert "contact_id IS NULL" not in sql
    assert args[0] == sms_id
    assert args[1] == sms_message.SMS_CONTACT_PROCESSING_LEASE_SECONDS


@pytest.mark.asyncio
async def test_ownerless_retry_pending_cannot_preempt_live_processing_rows():
    pool = _RetryPendingPool()
    sms_id = uuid4()
    replacements = _AttrReplacements()
    replacements.replace(sms_message, "get_db_pool", lambda: pool)

    try:
        await sms_message.SMSMessageRepository().mark_contact_processing_retry_pending(
            sms_id,
            "handoff failed before claim",
        )
    finally:
        replacements.restore()

    sql, args = pool.calls[0]
    assert "AND contact_id IS NULL" in sql
    assert "status IN ('received', 'retry_pending')" in sql
    assert "status = 'processing'" not in sql
    assert args[0] == sms_id


@pytest.mark.asyncio
async def test_owned_retry_pending_can_release_linked_processing_rows():
    pool = _RetryPendingPool()
    sms_id = uuid4()
    replacements = _AttrReplacements()
    replacements.replace(sms_message, "get_db_pool", lambda: pool)

    try:
        await sms_message.SMSMessageRepository().mark_contact_processing_retry_pending(
            sms_id,
            "handoff failed after link",
            owner_token="owner-token",
        )
    finally:
        replacements.restore()

    sql, args = pool.calls[0]
    assert "AND status = 'processing'" in sql
    assert "AND error_message = $3" in sql
    assert "contact_id IS NULL" not in sql
    assert args[0] == sms_id
    assert args[2] == "owner-token"


@pytest.mark.asyncio
async def test_live_postgres_sms_claim_fences_one_winner_and_stale_reclaim():
    database_url = os.getenv("ATLAS_EOM_SMS_RETRY_POSTGRES_URL")
    if not database_url:
        pytest.skip("Set ATLAS_EOM_SMS_RETRY_POSTGRES_URL to run live SMS claim proof")

    asyncpg = pytest.importorskip("asyncpg")
    pool = await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=4)
    repo = sms_message.SMSMessageRepository()
    message_sid = f"SM-live-claim-{uuid4()}"
    replacements = _AttrReplacements()
    replacements.replace(sms_message, "get_db_pool", lambda: _LiveSMSPool(pool))

    try:
        await pool.execute(
            """
            CREATE TABLE IF NOT EXISTS sms_messages (
                id UUID PRIMARY KEY,
                message_sid VARCHAR(128) NOT NULL UNIQUE,
                from_number VARCHAR(32) NOT NULL,
                to_number VARCHAR(32) NOT NULL,
                direction VARCHAR(10) NOT NULL DEFAULT 'inbound',
                body TEXT NOT NULL DEFAULT '',
                media_urls JSONB DEFAULT '[]'::jsonb,
                business_context_id VARCHAR(64),
                conversation_id UUID,
                intent VARCHAR(32),
                extracted_data JSONB DEFAULT '{}'::jsonb,
                summary TEXT,
                contact_id UUID,
                status VARCHAR(32) NOT NULL DEFAULT 'received',
                error_message TEXT,
                notified BOOLEAN DEFAULT FALSE,
                source VARCHAR(64),
                source_ref VARCHAR(256),
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                delivered_at TIMESTAMPTZ,
                processed_at TIMESTAMPTZ
            )
            """
        )
        row = await repo.create(
            message_sid=message_sid,
            from_number="+12175550101",
            to_number="+12175550102",
            direction="inbound",
            body="live claim proof",
            media_urls=[],
            business_context_id="effingham_maids",
            source="test",
        )
        sms_id = row["id"]

        first_claims = await asyncio.gather(
            repo.claim_contact_processing(sms_id, owner_token="owner-a"),
            repo.claim_contact_processing(sms_id, owner_token="owner-b"),
        )
        first_claim_flags = [bool(row and row.get("_claim_acquired")) for row in first_claims]
        assert sorted(first_claim_flags) == [False, True]
        winning_owner = "owner-a" if first_claim_flags[0] else "owner-b"
        losing_owner = "owner-b" if first_claim_flags[0] else "owner-a"

        live_row = await pool.fetchrow(
            "SELECT status, contact_id, processed_at, error_message FROM sms_messages WHERE id = $1",
            sms_id,
        )
        assert live_row["status"] == "processing"
        assert live_row["contact_id"] is None
        assert live_row["processed_at"] is not None
        assert live_row["error_message"] == winning_owner
        assert await repo.owns_contact_processing(sms_id, winning_owner) is True
        assert await repo.owns_contact_processing(sms_id, losing_owner) is False
        owner_c_row = await repo.claim_contact_processing(sms_id, owner_token="owner-c")
        assert owner_c_row["_claim_acquired"] is False
        assert owner_c_row["status"] == "processing"

        reclaim_owner = "reclaim"
        followup_owner = "followup"
        stale_processed_at = datetime.now(timezone.utc) - timedelta(
            seconds=sms_message.SMS_CONTACT_PROCESSING_LEASE_SECONDS + 5
        )
        await pool.execute(
            "UPDATE sms_messages SET processed_at = $2 WHERE id = $1",
            sms_id,
            stale_processed_at,
        )
        reclaim_row = await repo.claim_contact_processing(sms_id, owner_token=reclaim_owner)
        assert reclaim_row["_claim_acquired"] is True
        assert await repo.owns_contact_processing(sms_id, winning_owner) is False
        assert await repo.owns_contact_processing(sms_id, reclaim_owner) is True

        await repo.mark_contact_processing_complete(sms_id, owner_token=winning_owner)
        assert await repo.owns_contact_processing(sms_id, reclaim_owner) is True

        await pool.execute(
            "UPDATE sms_messages SET contact_id = $2 WHERE id = $1",
            sms_id,
            uuid4(),
        )
        await repo.mark_contact_processing_retry_pending(
            sms_id,
            "wrong owner must not release",
            owner_token=winning_owner,
        )
        live_row = await pool.fetchrow(
            "SELECT status, contact_id, error_message FROM sms_messages WHERE id = $1",
            sms_id,
        )
        assert live_row["status"] == "processing"
        assert live_row["contact_id"] is not None
        assert live_row["error_message"] == reclaim_owner

        await repo.mark_contact_processing_retry_pending(
            sms_id,
            "owned retry",
            owner_token=reclaim_owner,
        )
        live_row = await pool.fetchrow(
            "SELECT status, contact_id, error_message FROM sms_messages WHERE id = $1",
            sms_id,
        )
        assert live_row["status"] == "retry_pending"
        assert live_row["contact_id"] is not None
        assert live_row["error_message"] == "owned retry"

        assert await repo.claim_contact_processing(sms_id, owner_token=followup_owner)
        await pool.execute(
            "UPDATE sms_messages SET contact_id = NULL WHERE id = $1",
            sms_id,
        )
        await repo.mark_contact_processing_retry_pending(
            sms_id,
            "ownerless must not preempt live processing",
        )
        live_row = await pool.fetchrow(
            "SELECT status, contact_id, error_message FROM sms_messages WHERE id = $1",
            sms_id,
        )
        assert live_row["status"] == "processing"
        assert live_row["error_message"] == followup_owner

        stale_update = await repo.update_contact_processing_status(
            sms_id,
            "notified",
            owner_token=reclaim_owner,
            error_message="stale owner",
            clear_owner=True,
        )
        assert stale_update is False
        live_row = await pool.fetchrow(
            "SELECT status, error_message FROM sms_messages WHERE id = $1",
            sms_id,
        )
        assert live_row["status"] == "processing"
        assert live_row["error_message"] == followup_owner

        owner_update = await repo.update_contact_processing_status(
            sms_id,
            "notified",
            owner_token=followup_owner,
            clear_owner=False,
        )
        assert owner_update is True
        live_row = await pool.fetchrow(
            "SELECT status, error_message FROM sms_messages WHERE id = $1",
            sms_id,
        )
        assert live_row["status"] == "notified"
        assert live_row["error_message"] == followup_owner

        await pool.execute(
            "UPDATE sms_messages SET status = 'processing' WHERE id = $1",
            sms_id,
        )
        await repo.mark_contact_processing_complete(sms_id, owner_token=followup_owner)
        owner_d_row = await repo.claim_contact_processing(sms_id, owner_token="owner-d")
        assert owner_d_row["_claim_acquired"] is False
        assert owner_d_row["status"] == "ready"

        auto_reply = await repo.reserve_auto_reply_for_inbound(
            inbound_sms_id=sms_id,
            from_number="+12175550102",
            to_number="+12175550101",
            body="Persisted live reply",
            business_context_id="effingham_maids",
        )
        assert auto_reply["status"] == "pending"
        assert auto_reply["message_sid"] == f"auto_reply_{sms_id}"
        assert await repo.mark_auto_reply_sending(auto_reply["id"]) is True
        provider_sid = f"SM-live-auto-reply-{uuid4()}"
        assert await repo.mark_auto_reply_sent(
            auto_reply["id"],
            provider_message_id=provider_sid,
        ) is True
        provider_row = await repo.get_by_message_sid(provider_sid)
        assert provider_row is not None
        assert provider_row["id"] == auto_reply["id"]
        assert provider_row["source_ref"] == str(sms_id)
        assert provider_row["body"] == "Persisted live reply"
        assert provider_row["status"] == "sent"
        assert provider_row["error_message"] is None
    finally:
        await pool.execute(
            "DELETE FROM sms_messages WHERE message_sid = $1 OR source_ref = $2",
            message_sid,
            str(sms_id) if "sms_id" in locals() else None,
        )
        replacements.restore()
        await pool.close()


@pytest.mark.asyncio
async def test_inbound_sms_handler_processes_new_row_before_ack():
    repo = _WebhookRepo()
    calls = []

    replacements = _AttrReplacements()
    replacements.replace(webhooks, "get_context_router", lambda: _Router())
    replacements.replace(
        webhooks,
        "get_comms_service",
        lambda: SimpleNamespace(provider=_Provider()),
    )
    replacements.replace(sms_message, "get_sms_message_repo", lambda: repo)

    async def processor(*args, **kwargs):
        calls.append((args, kwargs))

    replacements.replace(webhooks, "_process_inbound_sms", processor)

    try:
        response = await webhooks.handle_inbound_sms(
            _FormRequest(),
            MessageSid="SM-new",
            From="+12175550101",
            To="+12175550102",
            Body="hello",
            NumMedia="0",
        )
    finally:
        replacements.restore()

    assert response.status_code == 200
    assert repo.created[0]["message_sid"] == "SM-new"
    assert repo.claimed_ids == []
    assert calls[0][1]["claim_processing"] is True
    assert calls[0][1]["retry_pending_recorder"] == "caller"


@pytest.mark.asyncio
async def test_inbound_sms_handler_returns_503_when_before_ack_processing_needs_retry():
    repo = _WebhookRepo()

    replacements = _AttrReplacements()
    replacements.replace(webhooks, "get_context_router", lambda: _Router())
    replacements.replace(
        webhooks,
        "get_comms_service",
        lambda: SimpleNamespace(provider=_Provider()),
    )
    replacements.replace(sms_message, "get_sms_message_repo", lambda: repo)

    async def processor(*args, **kwargs):
        return "retry_pending"

    replacements.replace(webhooks, "_process_inbound_sms", processor)

    try:
        response = await webhooks.handle_inbound_sms(
            _FormRequest(),
            MessageSid="SM-new-retry",
            From="+12175550101",
            To="+12175550102",
            Body="hello",
            NumMedia="0",
        )
    finally:
        replacements.restore()

    assert response.status_code == 503
    assert repo.created[0]["message_sid"] == "SM-new-retry"


@pytest.mark.asyncio
async def test_inbound_sms_handler_returns_503_while_duplicate_processing_is_leased():
    repo = _WebhookRepo(
        existing={
            "id": uuid4(),
            "message_sid": "SM-active-lease",
            "from_number": "+12175550101",
            "to_number": "+12175550102",
            "body": "hello",
            "media_urls": [],
            "contact_id": None,
            "status": "processing",
        }
    )

    replacements = _AttrReplacements()
    replacements.replace(webhooks, "get_context_router", lambda: _Router())
    replacements.replace(
        webhooks,
        "get_comms_service",
        lambda: SimpleNamespace(provider=_Provider()),
    )
    replacements.replace(sms_message, "get_sms_message_repo", lambda: repo)

    async def processor(*args, **kwargs):
        return "already_processing"

    replacements.replace(webhooks, "_process_inbound_sms", processor)

    try:
        response = await webhooks.handle_inbound_sms(
            _FormRequest(),
            MessageSid="SM-active-lease",
            From="+12175550101",
            To="+12175550102",
            Body="hello",
            NumMedia="0",
        )
    finally:
        replacements.restore()

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_inbound_sms_handler_returns_503_when_processing_exceeds_ack_budget():
    repo = _WebhookRepo()

    replacements = _AttrReplacements()
    replacements.replace(webhooks, "get_context_router", lambda: _Router())
    replacements.replace(
        webhooks,
        "get_comms_service",
        lambda: SimpleNamespace(provider=_Provider()),
    )
    replacements.replace(sms_message, "get_sms_message_repo", lambda: repo)
    replacements.replace(webhooks, "SMS_INBOUND_BEFORE_ACK_TIMEOUT_SECONDS", 0.01)

    async def processor(*args, **kwargs):
        await asyncio.sleep(1)
        return "complete"

    replacements.replace(webhooks, "_process_inbound_sms", processor)

    try:
        response = await webhooks.handle_inbound_sms(
            _FormRequest(),
            MessageSid="SM-timeout",
            From="+12175550101",
            To="+12175550102",
            Body="hello",
            NumMedia="0",
        )
    finally:
        replacements.restore()

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_inbound_sms_handler_allows_slow_but_configured_processing_before_ack():
    repo = _WebhookRepo()

    replacements = _AttrReplacements()
    replacements.replace(webhooks, "get_context_router", lambda: _Router())
    replacements.replace(
        webhooks,
        "get_comms_service",
        lambda: SimpleNamespace(provider=_Provider()),
    )
    replacements.replace(sms_message, "get_sms_message_repo", lambda: repo)
    replacements.replace(webhooks, "SMS_INBOUND_BEFORE_ACK_TIMEOUT_SECONDS", 0.25)

    async def processor(*args, **kwargs):
        await asyncio.sleep(0.05)
        return "complete"

    replacements.replace(webhooks, "_process_inbound_sms", processor)

    try:
        response = await webhooks.handle_inbound_sms(
            _FormRequest(),
            MessageSid="SM-slow-valid",
            From="+12175550101",
            To="+12175550102",
            Body="hello",
            NumMedia="0",
        )
    finally:
        replacements.restore()

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_inbound_sms_handler_applies_ack_budget_to_media_parsing():
    replacements = _AttrReplacements()
    replacements.replace(webhooks, "get_context_router", lambda: _Router())
    replacements.replace(webhooks, "SMS_INBOUND_BEFORE_ACK_TIMEOUT_SECONDS", 0.01)

    try:
        response = await webhooks.handle_inbound_sms(
            _SlowFormRequest(),
            MessageSid="SM-form-timeout",
            From="+12175550101",
            To="+12175550102",
            Body="hello",
            NumMedia="1",
        )
    finally:
        replacements.restore()

    assert response.status_code == 503


def test_sms_before_ack_budget_covers_all_admitted_sequential_stages():
    from atlas_brain.config import settings

    replacements = _AttrReplacements()
    replacements.replace(
        webhooks,
        "SMS_INBOUND_BEFORE_ACK_TIMEOUT_SECONDS",
        webhooks._DEFAULT_SMS_INBOUND_BEFORE_ACK_TIMEOUT_SECONDS,
    )
    try:
        budget = webhooks._sms_before_ack_timeout_seconds()
    finally:
        replacements.restore()

    expected = (
        float(settings.sms_intelligence.llm_timeout)
        + float(settings.call_intelligence.llm_timeout)
        + 10.0
        + max(float(settings.sms_intelligence.auto_reply_timeout), 15.0)
        + 15.0
    )
    assert budget >= expected


@pytest.mark.asyncio
async def test_inbound_sms_duplicate_stale_persisted_context_returns_503():
    repo = _WebhookRepo(
        existing={
            "id": uuid4(),
            "message_sid": "SM-stale-context",
            "from_number": "+12175550101",
            "to_number": "+12175550102",
            "body": "hello",
            "media_urls": [],
            "contact_id": None,
            "status": "received",
            "business_context_id": "deleted-context",
        }
    )

    replacements = _AttrReplacements()
    replacements.replace(webhooks, "get_context_router", lambda: _Router())
    replacements.replace(
        webhooks,
        "get_comms_service",
        lambda: SimpleNamespace(provider=_Provider()),
    )
    replacements.replace(sms_message, "get_sms_message_repo", lambda: repo)

    try:
        response = await webhooks.handle_inbound_sms(
            _FormRequest(),
            MessageSid="SM-stale-context",
            From="+12175550101",
            To="+12175550102",
            Body="hello",
            NumMedia="0",
        )
    finally:
        replacements.restore()

    assert response.status_code == 503
    assert repo.claimed_ids == []


@pytest.mark.asyncio
async def test_unlinked_duplicate_sms_resumes_before_ack_with_persisted_values():
    sms_id = uuid4()
    existing = {
        "id": sms_id,
        "message_sid": "SM-retry",
        "from_number": "+12175550101",
        "to_number": "+12175550102",
        "body": "",
        "media_urls": [],
        "contact_id": None,
        "status": "received",
    }
    repo = _ClaimingSMSRepo(claim_result=True)
    context = _sms_context()
    calls = []

    async def processor(*args, **kwargs):
        calls.append((args, kwargs))
        return "complete"

    replacements = _AttrReplacements()
    replacements.replace(webhooks, "_process_inbound_sms", processor)
    try:
        outcome = await webhooks._resume_duplicate_inbound_sms_before_ack(
            existing,
            fallback_from="+19999999999",
            fallback_to="+18888888888",
            fallback_body="provider retry body must not replace persisted empty body",
            fallback_media_urls=["https://example.test/retry-media.jpg"],
            context=context,
            provider_message_id="SM-retry",
            sms_repo=repo,
        )
    finally:
        replacements.restore()

    assert outcome == "complete"
    assert calls == [
        (
            (
                sms_id,
                "+12175550101",
                "+12175550102",
                "",
                context,
                [],
            ),
            {
                "provider_message_id": "SM-retry",
                "claim_processing": True,
                "retry_pending_recorder": "caller",
            },
        )
    ]


@pytest.mark.asyncio
async def test_unlinked_duplicate_sms_runs_worker_claim_for_owned_rows_before_ack():
    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=False)
    calls = []

    async def processor(*args, **kwargs):
        calls.append((args, kwargs))
        return "already_processing"

    replacements = _AttrReplacements()
    replacements.replace(webhooks, "_process_inbound_sms", processor)
    try:
        outcome = await webhooks._resume_duplicate_inbound_sms_before_ack(
            {
                "id": sms_id,
                "from_number": "+12175550101",
                "to_number": "+12175550102",
                "body": "already processing",
                "media_urls": [],
                "contact_id": None,
                "status": "retry_pending",
            },
            fallback_from="+19999999999",
            fallback_to="+18888888888",
            fallback_body="retry body",
            fallback_media_urls=[],
            context=_sms_context(),
            provider_message_id="SM-racing-retry",
            sms_repo=repo,
        )
    finally:
        replacements.restore()

    assert outcome == "already_processing"
    assert calls[0][1]["claim_processing"] is True


@pytest.mark.asyncio
async def test_linked_received_duplicate_sms_resumes_until_terminal_completion():
    repo = _ClaimingSMSRepo(claim_result=True)
    calls = []

    async def processor(*args, **kwargs):
        calls.append((args, kwargs))
        return "complete"

    replacements = _AttrReplacements()
    replacements.replace(webhooks, "_process_inbound_sms", processor)
    try:
        outcome = await webhooks._resume_duplicate_inbound_sms_before_ack(
            {
                "id": uuid4(),
                "from_number": "+12175550101",
                "to_number": "+12175550102",
                "body": "already linked but not complete",
                "media_urls": [],
                "contact_id": uuid4(),
                "status": "received",
            },
            fallback_from="+19999999999",
            fallback_to="+18888888888",
            fallback_body="retry body",
            fallback_media_urls=[],
            context=_sms_context(),
            provider_message_id="SM-linked",
            sms_repo=repo,
        )
    finally:
        replacements.restore()

    assert outcome == "complete"
    assert calls[0][1]["claim_processing"] is True


@pytest.mark.asyncio
async def test_completed_unlinked_duplicate_sms_does_not_resume():
    repo = _ClaimingSMSRepo(claim_result=True)

    outcome = await webhooks._resume_duplicate_inbound_sms_before_ack(
        {
            "id": uuid4(),
            "from_number": "+12175550101",
            "to_number": "+12175550102",
            "body": "stop",
            "media_urls": [],
            "contact_id": None,
            "status": "ready",
        },
        fallback_from="+19999999999",
        fallback_to="+18888888888",
        fallback_body="retry body",
        fallback_media_urls=[],
        context=_sms_context(),
        provider_message_id="SM-ready",
        sms_repo=repo,
    )

    assert outcome == "complete"
    assert repo.claimed_ids == []


@pytest.mark.asyncio
async def test_processing_duplicate_sms_is_resumable_before_ack():
    calls = []

    async def processor(*args, **kwargs):
        calls.append((args, kwargs))
        return "complete"

    replacements = _AttrReplacements()
    replacements.replace(webhooks, "_process_inbound_sms", processor)
    try:
        outcome = await webhooks._resume_duplicate_inbound_sms_before_ack(
            {
                "id": uuid4(),
                "from_number": "+12175550101",
                "to_number": "+12175550102",
                "body": "worker may have died",
                "media_urls": [],
                "contact_id": None,
                "status": "processing",
            },
            fallback_from="+19999999999",
            fallback_to="+18888888888",
            fallback_body="retry body",
            fallback_media_urls=[],
            context=_sms_context(),
            provider_message_id="SM-processing",
            sms_repo=_ClaimingSMSRepo(claim_result=True),
        )
    finally:
        replacements.restore()

    assert outcome == "complete"
    assert calls[0][1]["claim_processing"] is True


def test_duplicate_sms_context_uses_persisted_context_id_before_number_routing():
    seen_numbers = []
    seen_context_ids = []

    class Router:
        def get_context(self, context_id):
            seen_context_ids.append(context_id)
            return SimpleNamespace(id=f"context-id:{context_id}")

        def get_context_for_number(self, number):
            seen_numbers.append(number)
            return SimpleNamespace(id=f"context:{number}")

    context = webhooks._context_for_persisted_sms(
        {"business_context_id": "effingham_maids", "to_number": "+12175550102"},
        fallback_to="+18888888888",
        context_router=Router(),
    )

    assert seen_context_ids == ["effingham_maids"]
    assert seen_numbers == []
    assert context.id == "context-id:effingham_maids"


def test_duplicate_sms_context_falls_back_to_persisted_destination_for_legacy_rows():
    seen_numbers = []

    class Router:
        def get_context(self, context_id):
            return None

        def get_context_for_number(self, number):
            seen_numbers.append(number)
            return SimpleNamespace(id=f"context:{number}")

    context = webhooks._context_for_persisted_sms(
        {"to_number": "+12175550102"},
        fallback_to="+18888888888",
        context_router=Router(),
    )

    assert seen_numbers == ["+12175550102"]
    assert context.id == "context:+12175550102"


def test_duplicate_sms_context_fails_closed_for_stale_persisted_context_id():
    class Router:
        def get_context(self, context_id):
            return None

        def get_context_for_number(self, number):  # pragma: no cover
            raise AssertionError("number routing must not run for stored context ids")

    with pytest.raises(ValueError, match="business context no longer resolves"):
        webhooks._context_for_persisted_sms(
            {"business_context_id": "deleted-context", "to_number": "+12175550102"},
            fallback_to="+18888888888",
            context_router=Router(),
        )


@pytest.mark.asyncio
async def test_process_inbound_sms_claims_before_running_intelligence():
    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=True)
    intelligence_calls = []

    async def intelligence_runner(**kwargs):
        intelligence_calls.append(kwargs)
        await repo.link_contact(kwargs["sms_id"], "contact-1")

    outcome = await webhooks._process_inbound_sms(
        sms_id,
        "+12175550101",
        "+12175550102",
        "hello",
        _sms_context(),
        [],
        provider_message_id="SM-process",
        sms_repo=repo,
        intelligence_runner=intelligence_runner,
    )

    assert repo.claimed_ids == [sms_id]
    assert repo.linked_contacts == [(sms_id, "contact-1")]
    assert outcome == "complete"
    assert repo.completed[0][0] == sms_id
    assert repo.completed[0][1] == repo.owner_tokens[sms_id]
    assert len(intelligence_calls) == 1
    assert intelligence_calls[0]["provider_message_id"] == "SM-process"


@pytest.mark.asyncio
async def test_process_inbound_sms_resumes_linked_row_with_existing_contact_id():
    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=True)
    repo.rows[sms_id] = {
        "id": sms_id,
        "status": "processing",
        "contact_id": uuid4(),
    }
    intelligence_calls = []

    async def intelligence_runner(**kwargs):
        intelligence_calls.append(kwargs)
        return "complete"

    outcome = await webhooks._process_inbound_sms(
        sms_id,
        "+12175550101",
        "+12175550102",
        "hello",
        _sms_context(),
        [],
        provider_message_id="SM-linked-resume",
        sms_repo=repo,
        intelligence_runner=intelligence_runner,
    )

    assert outcome == "complete"
    assert repo.claimed_ids == [sms_id]
    assert intelligence_calls[0]["existing_contact_id"] == str(repo.rows[sms_id]["contact_id"])
    assert repo.completed == [(sms_id, repo.owner_tokens[sms_id])]


@pytest.mark.asyncio
async def test_sms_intelligence_preserves_processing_owner_token_before_crm_link():
    class Pool:
        is_initialized = True

    class CRM:
        async def find_or_create_contact(self, **kwargs):
            return {"id": "contact-1", "full_name": kwargs["full_name"]}

        async def log_interaction(self, **kwargs):
            return {"id": "interaction-1"}

    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=True)
    owner_token = "owner-token"
    repo.owner_tokens[sms_id] = owner_token
    replacements = _AttrReplacements()
    replacements.replace(sms_intelligence, "get_sms_message_repo", lambda: repo)
    replacements.replace(storage_database, "get_db_pool", lambda: Pool())
    replacements.replace(crm_provider, "get_crm_provider", lambda: CRM())

    async def extract(body, business_context):
        return (
            "Inbound SMS from customer",
            {"customer_phone": "+12175550101", "customer_name": "Customer"},
            "inquiry",
        )

    replacements.replace(sms_intelligence, "_extract_sms_data", extract)

    try:
        outcome = await sms_intelligence.process_inbound_sms(
            sms_id=sms_id,
            from_number="+12175550101",
            to_number="+12175550102",
            body="hello",
            business_context_id="non_eom_context",
            processing_owner_token=owner_token,
            stop_after_crm=True,
        )
    finally:
        replacements.restore()

    assert outcome == "crm_linked"
    assert repo.linked_contacts == [(sms_id, "contact-1")]
    assert repo.owner_tokens[sms_id] == owner_token
    assert repo.touched


@pytest.mark.asyncio
async def test_sms_intelligence_stop_intent_leaves_leased_terminal_transition_to_webhook():
    class Pool:
        is_initialized = True

    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=True)
    owner_token = "owner-token"
    repo.owner_tokens[sms_id] = owner_token
    replacements = _AttrReplacements()
    replacements.replace(sms_intelligence, "get_sms_message_repo", lambda: repo)
    replacements.replace(storage_database, "get_db_pool", lambda: Pool())

    async def extract(body, business_context):
        return (
            "Customer asked to stop",
            {"customer_phone": "+12175550101"},
            "stop",
        )

    replacements.replace(sms_intelligence, "_extract_sms_data", extract)

    try:
        outcome = await sms_intelligence.process_inbound_sms(
            sms_id=sms_id,
            from_number="+12175550101",
            to_number="+12175550102",
            body="STOP",
            business_context_id="non_eom_context",
            processing_owner_token=owner_token,
        )
    finally:
        replacements.restore()

    assert outcome == "terminal_no_contact"
    assert repo.owner_status_updates == []
    assert repo.owner_tokens[sms_id] == owner_token


@pytest.mark.asyncio
async def test_sms_intelligence_existing_contact_id_skips_crm_link():
    class Pool:
        is_initialized = True

    class CRM:
        async def find_or_create_contact(self, **kwargs):  # pragma: no cover
            raise AssertionError("existing linked contacts must not be recreated")

        async def log_interaction(self, **kwargs):  # pragma: no cover
            raise AssertionError("existing linked contacts must not create another CRM interaction")

    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=True)
    replacements = _AttrReplacements()
    replacements.replace(sms_intelligence, "get_sms_message_repo", lambda: repo)
    replacements.replace(storage_database, "get_db_pool", lambda: Pool())
    replacements.replace(crm_provider, "get_crm_provider", lambda: CRM())

    async def extract(body, business_context):
        return (
            "Inbound SMS from customer",
            {"customer_phone": "+12175550101", "customer_name": "Customer"},
            "inquiry",
        )

    replacements.replace(sms_intelligence, "_extract_sms_data", extract)

    try:
        outcome = await sms_intelligence.process_inbound_sms(
            sms_id=sms_id,
            from_number="+12175550101",
            to_number="+12175550102",
            body="hello",
            business_context_id="non_eom_context",
            existing_contact_id="contact-1",
            stop_after_crm=True,
        )
    finally:
        replacements.restore()

    assert outcome == "crm_linked"
    assert repo.linked_contacts == []


@pytest.mark.asyncio
async def test_sms_crm_database_unavailable_is_retryable_not_complete():
    class Pool:
        is_initialized = False

    repo = _ClaimingSMSRepo(claim_result=True)
    sms_id = uuid4()
    replacements = _AttrReplacements()
    replacements.replace(sms_intelligence, "get_sms_message_repo", lambda: repo)
    replacements.replace(storage_database, "get_db_pool", lambda: Pool())

    async def extract(body, business_context):
        return (
            "Inbound SMS from customer",
            {"customer_phone": "+12175550101", "customer_name": "Customer"},
            "inquiry",
        )

    replacements.replace(sms_intelligence, "_extract_sms_data", extract)

    try:
        outcome = await sms_intelligence.process_inbound_sms(
            sms_id=sms_id,
            from_number="+12175550101",
            to_number="+12175550102",
            body="hello",
            business_context_id="non_eom_context",
        )
    finally:
        replacements.restore()

    assert outcome == "retry_pending"


@pytest.mark.asyncio
async def test_process_inbound_sms_does_not_run_intelligence_when_claim_is_owned():
    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=False)
    intelligence_calls = []

    async def intelligence_runner(**kwargs):
        intelligence_calls.append(kwargs)

    outcome = await webhooks._process_inbound_sms(
        sms_id,
        "+12175550101",
        "+12175550102",
        "hello",
        _sms_context(),
        [],
        provider_message_id="SM-owned",
        sms_repo=repo,
        intelligence_runner=intelligence_runner,
    )

    assert repo.claimed_ids == [sms_id]
    assert outcome == "already_processing"
    assert intelligence_calls == []


@pytest.mark.asyncio
async def test_process_inbound_sms_marks_retry_pending_when_claim_raises():
    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=False, claim_exc=RuntimeError("db unavailable"))
    intelligence_calls = []

    async def intelligence_runner(**kwargs):
        intelligence_calls.append(kwargs)

    outcome = await webhooks._process_inbound_sms(
        sms_id,
        "+12175550101",
        "+12175550102",
        "hello",
        _sms_context(),
        [],
        provider_message_id="SM-claim-fails",
        sms_repo=repo,
        intelligence_runner=intelligence_runner,
    )

    assert repo.claimed_ids == [sms_id]
    assert outcome == "retry_pending"
    assert intelligence_calls == []
    assert repo.retry_pending == [(sms_id, "SMS processing claim failed: db unavailable", None)]


@pytest.mark.asyncio
async def test_process_inbound_sms_marks_retry_pending_for_crm_handoff_failure():
    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=True)

    async def intelligence_runner(**kwargs):
        return "retry_pending"

    outcome = await webhooks._process_inbound_sms(
        sms_id,
        "+12175550101",
        "+12175550102",
        "hello",
        _sms_context(),
        [],
        provider_message_id="SM-crm-fails",
        sms_repo=repo,
        intelligence_runner=intelligence_runner,
    )

    assert repo.claimed_ids == [sms_id]
    assert outcome == "retry_pending"
    assert repo.retry_pending == [
        (sms_id, "SMS CRM handoff failed", repo.owner_tokens[sms_id])
    ]
    assert repo.completed == []


@pytest.mark.asyncio
async def test_process_inbound_sms_releases_owned_lease_when_caller_records_retry():
    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=True)

    async def intelligence_runner(**kwargs):
        return "retry_pending"

    outcome = await webhooks._process_inbound_sms(
        sms_id,
        "+12175550101",
        "+12175550102",
        "hello",
        _sms_context(),
        [],
        provider_message_id="SM-crm-fails-before-ack",
        sms_repo=repo,
        intelligence_runner=intelligence_runner,
        retry_pending_recorder="caller",
    )

    assert outcome == "retry_pending"
    assert repo.retry_pending == [
        (sms_id, "SMS CRM handoff failed", repo.owner_tokens[sms_id])
    ]
    assert repo.completed == []


@pytest.mark.asyncio
async def test_process_inbound_sms_treats_terminal_no_contact_as_success():
    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=True)
    context = SimpleNamespace(
        id="effingham_maids",
        sms_auto_reply=True,
        sms_enabled=True,
    )

    async def intelligence_runner(**kwargs):
        return "terminal_no_contact"

    outcome = await webhooks._process_inbound_sms(
        sms_id,
        "+12175550101",
        "+12175550102",
        "STOP",
        context,
        [],
        provider_message_id="SM-stop",
        sms_repo=repo,
        intelligence_runner=intelligence_runner,
    )

    assert outcome == "complete"
    assert repo.retry_pending == []
    assert repo.completed == [(sms_id, repo.owner_tokens[sms_id])]


@pytest.mark.asyncio
async def test_process_inbound_sms_skipped_intelligence_still_auto_replies_and_completes():
    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=True)
    context = SimpleNamespace(
        id="effingham_maids",
        sms_auto_reply=True,
        sms_enabled=True,
    )
    provider = _Provider()
    replacements = _AttrReplacements()
    replacements.replace(
        webhooks,
        "get_comms_service",
        lambda: SimpleNamespace(provider=provider),
    )

    async def generate_reply(body, ctx):
        return "Thanks!"

    replacements.replace(webhooks, "_generate_sms_reply", generate_reply)

    async def intelligence_runner(**kwargs):
        return "skipped_intelligence"

    try:
        outcome = await webhooks._process_inbound_sms(
            sms_id,
            "+12175550101",
            "+12175550102",
            "hello",
            context,
            [],
            provider_message_id="SM-skip-intel",
            sms_repo=repo,
            intelligence_runner=intelligence_runner,
        )
    finally:
        replacements.restore()

    assert outcome == "complete"
    assert len(provider.sent_sms) == 1
    assert repo.auto_reply_rows[sms_id]["status"] == "sent"
    assert len(repo.auto_reply_sent) == 1
    assert repo.completed == [(sms_id, repo.owner_tokens[sms_id])]


@pytest.mark.asyncio
async def test_process_inbound_sms_pending_auto_reply_sends_persisted_body_and_completes():
    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=True)
    pending_row = {
        "id": uuid4(),
        "message_sid": f"auto_reply_{sms_id}",
        "from_number": "+12175550102",
        "to_number": "+12175550101",
        "body": "Persisted thanks!",
        "business_context_id": "effingham_maids",
        "status": "pending",
        "source": "auto_reply",
        "source_ref": str(sms_id),
    }
    repo.auto_reply_rows[sms_id] = pending_row
    context = SimpleNamespace(
        id="effingham_maids",
        sms_auto_reply=True,
        sms_enabled=True,
    )
    provider = _Provider()
    replacements = _AttrReplacements()
    replacements.replace(
        webhooks,
        "get_comms_service",
        lambda: SimpleNamespace(provider=provider),
    )

    async def generate_reply(body, ctx):
        return "Regenerated thanks!"

    replacements.replace(webhooks, "_generate_sms_reply", generate_reply)

    async def intelligence_runner(**kwargs):
        return "complete"

    try:
        outcome = await webhooks._process_inbound_sms(
            sms_id,
            "+12175550101",
            "+12175550102",
            "hello",
            context,
            [],
            provider_message_id="SM-auto-retry",
            sms_repo=repo,
            intelligence_runner=intelligence_runner,
        )
    finally:
        replacements.restore()

    assert outcome == "complete"
    assert len(provider.sent_sms) == 1
    assert provider.sent_sms[0]["body"] == "Persisted thanks!"
    assert repo.auto_reply_sending == [pending_row["id"]]
    assert repo.auto_reply_sent == [(pending_row["id"], "SM-out-1")]
    assert repo.auto_reply_rows[sms_id]["status"] == "sent"
    assert repo.auto_reply_rows[sms_id]["message_sid"] == "SM-out-1"
    assert repo.auto_reply_rows[sms_id]["error_message"] is None
    assert repo.completed == [(sms_id, repo.owner_tokens[sms_id])]


@pytest.mark.asyncio
async def test_process_inbound_sms_sent_auto_reply_skips_duplicate_send_and_completes():
    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=True)
    repo.auto_reply_rows[sms_id] = {
        "id": uuid4(),
        "message_sid": f"auto_reply_{sms_id}",
        "from_number": "+12175550102",
        "to_number": "+12175550101",
        "body": "Thanks!",
        "business_context_id": "effingham_maids",
        "status": "sent",
        "source": "auto_reply",
        "source_ref": str(sms_id),
    }
    context = SimpleNamespace(
        id="effingham_maids",
        sms_auto_reply=True,
        sms_enabled=True,
    )
    provider = _Provider()
    replacements = _AttrReplacements()
    replacements.replace(
        webhooks,
        "get_comms_service",
        lambda: SimpleNamespace(provider=provider),
    )

    async def generate_reply(body, ctx):
        return "Thanks!"

    replacements.replace(webhooks, "_generate_sms_reply", generate_reply)

    async def intelligence_runner(**kwargs):
        return "complete"

    try:
        outcome = await webhooks._process_inbound_sms(
            sms_id,
            "+12175550101",
            "+12175550102",
            "hello",
            context,
            [],
            provider_message_id="SM-auto-retry-sent",
            sms_repo=repo,
            intelligence_runner=intelligence_runner,
        )
    finally:
        replacements.restore()

    assert outcome == "complete"
    assert provider.sent_sms == []
    assert repo.auto_reply_sending == []
    assert repo.auto_reply_sent == []
    assert repo.completed == [(sms_id, repo.owner_tokens[sms_id])]


@pytest.mark.asyncio
async def test_process_inbound_sms_sending_auto_reply_skips_ambiguous_duplicate_and_completes():
    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=True)
    repo.auto_reply_rows[sms_id] = {
        "id": uuid4(),
        "message_sid": f"auto_reply_{sms_id}",
        "from_number": "+12175550102",
        "to_number": "+12175550101",
        "body": "Thanks!",
        "business_context_id": "effingham_maids",
        "status": "sending",
        "source": "auto_reply",
        "source_ref": str(sms_id),
    }
    context = SimpleNamespace(
        id="effingham_maids",
        sms_auto_reply=True,
        sms_enabled=True,
    )
    provider = _Provider()
    replacements = _AttrReplacements()
    replacements.replace(
        webhooks,
        "get_comms_service",
        lambda: SimpleNamespace(provider=provider),
    )

    async def generate_reply(body, ctx):  # pragma: no cover
        raise AssertionError("existing sending outbox rows must not regenerate replies")

    replacements.replace(webhooks, "_generate_sms_reply", generate_reply)

    async def intelligence_runner(**kwargs):
        return "complete"

    try:
        outcome = await webhooks._process_inbound_sms(
            sms_id,
            "+12175550101",
            "+12175550102",
            "hello",
            context,
            [],
            provider_message_id="SM-auto-retry-sending",
            sms_repo=repo,
            intelligence_runner=intelligence_runner,
        )
    finally:
        replacements.restore()

    assert outcome == "complete"
    assert provider.sent_sms == []
    assert repo.auto_reply_sending == []
    assert repo.auto_reply_sent == []
    assert repo.completed == [(sms_id, repo.owner_tokens[sms_id])]


@pytest.mark.asyncio
async def test_process_inbound_sms_auto_reply_send_failure_retries_without_completion():
    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=True)
    context = SimpleNamespace(
        id="effingham_maids",
        sms_auto_reply=True,
        sms_enabled=True,
    )
    provider = _Provider(send_exc=RuntimeError("provider down"))
    replacements = _AttrReplacements()
    replacements.replace(
        webhooks,
        "get_comms_service",
        lambda: SimpleNamespace(provider=provider),
    )

    async def generate_reply(body, ctx):
        return "Thanks!"

    replacements.replace(webhooks, "_generate_sms_reply", generate_reply)

    async def intelligence_runner(**kwargs):
        return "complete"

    try:
        outcome = await webhooks._process_inbound_sms(
            sms_id,
            "+12175550101",
            "+12175550102",
            "hello",
            context,
            [],
            provider_message_id="SM-auto-send-fail",
            sms_repo=repo,
            intelligence_runner=intelligence_runner,
        )
    finally:
        replacements.restore()

    assert outcome == "retry_pending"
    assert repo.auto_reply_rows[sms_id]["status"] == "sending"
    assert repo.auto_reply_sending == [repo.auto_reply_rows[sms_id]["id"]]
    assert repo.completed == []


@pytest.mark.asyncio
async def test_process_inbound_sms_final_completion_failure_retries():
    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=True)
    repo.complete_exc = RuntimeError("db down")
    context = SimpleNamespace(
        id="effingham_maids",
        sms_auto_reply=False,
        sms_enabled=True,
    )

    async def intelligence_runner(**kwargs):
        return "complete"

    outcome = await webhooks._process_inbound_sms(
        sms_id,
        "+12175550101",
        "+12175550102",
        "hello",
        context,
        [],
        provider_message_id="SM-complete-fail",
        sms_repo=repo,
        intelligence_runner=intelligence_runner,
    )

    assert outcome == "retry_pending"
    assert repo.completed == []


@pytest.mark.asyncio
async def test_process_inbound_sms_existing_notified_row_skips_duplicate_notification():
    sms_id = uuid4()
    repo = _ClaimingSMSRepo(claim_result=True)
    repo.rows[sms_id] = {
        "id": sms_id,
        "status": "received",
        "contact_id": uuid4(),
        "notified": True,
    }
    context = SimpleNamespace(
        id="effingham_maids",
        sms_auto_reply=False,
        sms_enabled=True,
    )
    intelligence_calls = []

    async def intelligence_runner(**kwargs):
        intelligence_calls.append(kwargs)
        return "complete"

    outcome = await webhooks._process_inbound_sms(
        sms_id,
        "+12175550101",
        "+12175550102",
        "hello",
        context,
        [],
        provider_message_id="SM-notified-retry",
        sms_repo=repo,
        intelligence_runner=intelligence_runner,
    )

    assert outcome == "complete"
    assert intelligence_calls[0]["notification_already_sent"] is True
    assert repo.completed == [(sms_id, repo.owner_tokens[sms_id])]


@pytest.mark.asyncio
async def test_sms_crm_post_link_interaction_failure_does_not_retry_or_unlink():
    class Pool:
        is_initialized = True

    class CRM:
        async def find_or_create_contact(self, **kwargs):
            return {"id": "contact-1", "full_name": kwargs["full_name"]}

        async def log_interaction(self, **kwargs):
            raise RuntimeError("interaction store unavailable")

    repo = _ClaimingSMSRepo(claim_result=True)
    sms_id = uuid4()
    replacements = _AttrReplacements()
    replacements.replace(storage_database, "get_db_pool", lambda: Pool())
    replacements.replace(crm_provider, "get_crm_provider", lambda: CRM())

    try:
        contact_id, is_new = await sms_intelligence._link_to_crm(
            repo,
            sms_id,
            "+12175550101",
            "non_eom_context",
            {"customer_phone": "+12175550101", "customer_name": "Customer"},
            "Inbound SMS from customer",
            provider_message_id="SM-post-link-log-fails",
        )
    finally:
        replacements.restore()

    assert contact_id == "contact-1"
    assert is_new is False
    assert repo.linked_contacts == [(sms_id, "contact-1")]
