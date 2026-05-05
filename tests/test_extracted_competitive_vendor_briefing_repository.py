from __future__ import annotations

from typing import Any

import pytest

from extracted_competitive_intelligence.services.b2b import (
    vendor_briefing_repository as repository,
)


class RecordingPool:
    def __init__(
        self,
        *,
        fetchval_result: Any = None,
        fetchrow_result: dict[str, Any] | None = None,
        execute_result: str = "UPDATE 1",
    ) -> None:
        self.fetchval_result = fetchval_result
        self.fetchrow_result = fetchrow_result
        self.execute_result = execute_result
        self.fetchval_calls: list[tuple[Any, ...]] = []
        self.fetchrow_calls: list[tuple[Any, ...]] = []
        self.execute_calls: list[tuple[Any, ...]] = []

    async def fetchval(self, *args: Any) -> Any:
        self.fetchval_calls.append(args)
        return self.fetchval_result

    async def fetchrow(self, *args: Any) -> dict[str, Any] | None:
        self.fetchrow_calls.append(args)
        return self.fetchrow_result

    async def execute(self, *args: Any) -> str:
        self.execute_calls.append(args)
        return self.execute_result


@pytest.mark.asyncio
async def test_prior_and_recent_briefing_checks_share_deliverable_status_filter() -> None:
    prior_pool = RecordingPool(fetchval_result=1)
    recent_pool = RecordingPool(fetchval_result=True)

    assert await repository.has_prior_deliverable_briefing(prior_pool, "Acme")
    assert await repository.has_recent_deliverable_briefing(recent_pool, "Acme", 7)

    prior_query = prior_pool.fetchval_calls[0][0]
    recent_query = recent_pool.fetchval_calls[0][0]
    assert "status NOT IN ('failed', 'suppressed', 'rejected')" in prior_query
    assert "status NOT IN ('failed', 'suppressed', 'rejected')" in recent_query


@pytest.mark.asyncio
async def test_delivery_record_helpers_write_expected_status_rows() -> None:
    pool = RecordingPool()

    await repository.insert_suppressed_briefing_record(
        pool,
        vendor_name="Acme",
        recipient_email="buyer@example.com",
        subject="Suppressed",
        briefing_data={"vendor": "Acme"},
    )
    await repository.insert_delivery_briefing_record(
        pool,
        vendor_name="Acme",
        recipient_email="buyer@example.com",
        subject="Sent",
        briefing_data={"vendor": "Acme"},
        resend_id="resend-1",
        status="sent",
    )
    await repository.insert_pending_approval_briefing_record(
        pool,
        vendor_name="Acme",
        recipient_email="buyer@example.com",
        subject="Pending",
        briefing_data={"vendor": "Acme"},
        briefing_html="<p>briefing</p>",
        target_mode="vendor_retention",
    )

    queries = [call[0] for call in pool.execute_calls]
    assert "'suppressed'" in queries[0]
    assert "resend_id, status" in queries[1]
    assert "'pending_approval'" in queries[2]


@pytest.mark.asyncio
async def test_fetch_pending_approval_briefing_parses_json_payload() -> None:
    pool = RecordingPool(
        fetchrow_result={
            "vendor_name": "Acme",
            "recipient_email": "buyer@example.com",
            "subject": "Stored Subject",
            "briefing_data": "{\"vendor\": \"Acme\"}",
            "briefing_html": "<p>briefing</p>",
        }
    )

    pending = await repository.fetch_pending_approval_briefing(pool, "briefing-1")

    assert pending == repository.PendingVendorBriefing(
        vendor_name="Acme",
        recipient_email="buyer@example.com",
        subject="Stored Subject",
        briefing_data={"vendor": "Acme"},
        briefing_html="<p>briefing</p>",
    )


@pytest.mark.asyncio
async def test_approval_and_rejection_updates_preserve_existing_status_contract() -> None:
    pool = RecordingPool()
    missing_pool = RecordingPool(execute_result="UPDATE 0")

    await repository.mark_pending_briefing_sent(
        pool,
        briefing_id="briefing-1",
        resend_id="resend-1",
    )
    await repository.mark_pending_briefing_failed(pool, "briefing-2")
    rejected = await repository.reject_pending_briefing(
        pool,
        briefing_id="briefing-3",
        reason="bad fit",
    )
    missing = await repository.reject_pending_briefing(
        missing_pool,
        briefing_id="briefing-4",
    )

    assert rejected is True
    assert missing is False
    assert "approved_at = NOW()" in pool.execute_calls[0][0]
    assert "SET status = $1" in pool.execute_calls[1][0]
    assert "rejected_at = NOW()" in pool.execute_calls[2][0]
