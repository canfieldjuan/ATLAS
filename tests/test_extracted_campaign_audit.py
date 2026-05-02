from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest

from extracted_content_pipeline.autonomous.tasks.campaign_audit import log_campaign_event


class _Pool:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    async def execute(self, query, *args):
        self.calls.append((query, args))
        if self.fail:
            raise RuntimeError("database unavailable")


@pytest.mark.asyncio
async def test_log_campaign_event_inserts_normalized_audit_row() -> None:
    campaign_id = uuid4()
    sequence_id = uuid4()
    pool = _Pool()

    await log_campaign_event(
        pool,
        event_type="sequence_sent",
        source="scheduler",
        campaign_id=str(campaign_id),
        sequence_id=sequence_id,
        step_number=2,
        subject="Follow-up",
        body="Body",
        recipient_email="buyer@example.com",
        esp_message_id="msg-1",
        error_detail=None,
        metadata={"variant": "A"},
    )

    assert len(pool.calls) == 1
    query, args = pool.calls[0]
    assert "INSERT INTO campaign_audit_log" in query
    assert args[0] == campaign_id
    assert isinstance(args[0], UUID)
    assert args[1] == sequence_id
    assert args[2:10] == (
        "sequence_sent",
        2,
        "Follow-up",
        "Body",
        "buyer@example.com",
        "msg-1",
        None,
        "scheduler",
    )
    assert json.loads(args[10]) == {"variant": "A"}


@pytest.mark.asyncio
async def test_log_campaign_event_defaults_metadata_to_empty_object() -> None:
    pool = _Pool()

    await log_campaign_event(pool, event_type="created")

    _, args = pool.calls[0]
    assert args[0] is None
    assert args[1] is None
    assert json.loads(args[10]) == {}


@pytest.mark.asyncio
async def test_log_campaign_event_never_raises_on_insert_failure() -> None:
    pool = _Pool(fail=True)

    await log_campaign_event(pool, event_type="failed", sequence_id=uuid4())

    assert len(pool.calls) == 1
