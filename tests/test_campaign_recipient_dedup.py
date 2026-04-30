"""Integration tests for assign_recipient_to_sequence (Gap 2).

Verifies cross-sequence person-level deduplication:
- Two active sequences for the same email -> second is superseded.
- Sequence in a non-active terminal status (e.g. 'replied') does not block
  enrollment for a fresh sequence with the same email.
- Empty / whitespace email returns assigned=False with reason='empty_email'.
- The audit log captures supersede events with the conflicting sequence id.
"""

from uuid import uuid4

import pytest
import pytest_asyncio

from atlas_brain.autonomous.tasks.campaign_suppression import (
    assign_recipient_to_sequence,
)


@pytest_asyncio.fixture
async def cleanup_sequences(db_pool):
    """Track created sequences for cleanup after each test."""
    created: list = []
    yield created
    if created:
        await db_pool.execute(
            "DELETE FROM campaign_audit_log WHERE sequence_id = ANY($1::uuid[])",
            created,
        )
        await db_pool.execute(
            "DELETE FROM campaign_sequences WHERE id = ANY($1::uuid[])",
            created,
        )


async def _create_sequence(pool, batch_id: str, recipient_email: str | None = None) -> str:
    """Insert a campaign_sequences row and return its id as a string."""
    row = await pool.fetchrow(
        """
        INSERT INTO campaign_sequences
            (company_name, batch_id, recipient_email, status)
        VALUES ($1, $2, $3, 'active')
        RETURNING id
        """,
        f"test-co-{uuid4().hex[:8]}",
        batch_id,
        recipient_email,
    )
    return str(row["id"])


@pytest.mark.asyncio
async def test_assigns_recipient_when_no_conflict(db_pool, cleanup_sequences):
    batch_id = f"dedup-test-{uuid4().hex[:8]}"
    seq_id = await _create_sequence(db_pool, batch_id)
    cleanup_sequences.append(seq_id)

    result = await assign_recipient_to_sequence(
        db_pool, seq_id, "alice@example.com",
    )

    assert result.assigned is True
    assert result.conflict_with_sequence_id is None

    row = await db_pool.fetchrow(
        "SELECT recipient_email, status FROM campaign_sequences WHERE id = $1",
        seq_id,
    )
    assert row["recipient_email"] == "alice@example.com"
    assert row["status"] == "active"


@pytest.mark.asyncio
async def test_supersedes_second_sequence_with_same_email(db_pool, cleanup_sequences):
    batch_id = f"dedup-test-{uuid4().hex[:8]}"
    email = f"bob-{uuid4().hex[:6]}@example.com"

    first = await _create_sequence(db_pool, batch_id, recipient_email=email)
    second = await _create_sequence(db_pool, batch_id)
    cleanup_sequences.extend([first, second])

    result = await assign_recipient_to_sequence(db_pool, second, email)

    assert result.assigned is False
    assert result.reason == "active_sequence_exists_for_recipient"
    assert str(result.conflict_with_sequence_id) == first

    row = await db_pool.fetchrow(
        "SELECT recipient_email, status FROM campaign_sequences WHERE id = $1",
        second,
    )
    assert row["recipient_email"] is None
    assert row["status"] == "superseded"

    audit = await db_pool.fetchrow(
        """
        SELECT event_type, recipient_email, metadata
        FROM campaign_audit_log
        WHERE sequence_id = $1 AND event_type = 'recipient_superseded'
        """,
        second,
    )
    assert audit is not None
    assert audit["recipient_email"] == email


@pytest.mark.asyncio
async def test_replied_sequence_does_not_block_new_enrollment(db_pool, cleanup_sequences):
    batch_id = f"dedup-test-{uuid4().hex[:8]}"
    email = f"carol-{uuid4().hex[:6]}@example.com"

    replied = await _create_sequence(db_pool, batch_id, recipient_email=email)
    await db_pool.execute(
        "UPDATE campaign_sequences SET status = 'replied' WHERE id = $1",
        replied,
    )
    fresh = await _create_sequence(db_pool, batch_id)
    cleanup_sequences.extend([replied, fresh])

    result = await assign_recipient_to_sequence(db_pool, fresh, email)

    assert result.assigned is True
    assert result.conflict_with_sequence_id is None

    row = await db_pool.fetchrow(
        "SELECT recipient_email, status FROM campaign_sequences WHERE id = $1",
        fresh,
    )
    assert row["recipient_email"] == email
    assert row["status"] == "active"


@pytest.mark.asyncio
async def test_case_insensitive_email_match(db_pool, cleanup_sequences):
    batch_id = f"dedup-test-{uuid4().hex[:8]}"

    first = await _create_sequence(db_pool, batch_id, recipient_email="Dan@Example.COM")
    second = await _create_sequence(db_pool, batch_id)
    cleanup_sequences.extend([first, second])

    result = await assign_recipient_to_sequence(db_pool, second, "DAN@example.com")

    assert result.assigned is False
    assert str(result.conflict_with_sequence_id) == first


@pytest.mark.asyncio
async def test_empty_email_returns_no_assignment(db_pool, cleanup_sequences):
    batch_id = f"dedup-test-{uuid4().hex[:8]}"
    seq_id = await _create_sequence(db_pool, batch_id)
    cleanup_sequences.append(seq_id)

    result = await assign_recipient_to_sequence(db_pool, seq_id, "   ")

    assert result.assigned is False
    assert result.reason == "empty_email"

    row = await db_pool.fetchrow(
        "SELECT recipient_email, status FROM campaign_sequences WHERE id = $1",
        seq_id,
    )
    assert row["recipient_email"] is None
    assert row["status"] == "active"


@pytest.mark.asyncio
async def test_strips_whitespace_from_email(db_pool, cleanup_sequences):
    batch_id = f"dedup-test-{uuid4().hex[:8]}"
    seq_id = await _create_sequence(db_pool, batch_id)
    cleanup_sequences.append(seq_id)

    result = await assign_recipient_to_sequence(
        db_pool, seq_id, "  eve@example.com  ",
    )

    assert result.assigned is True
    row = await db_pool.fetchrow(
        "SELECT recipient_email FROM campaign_sequences WHERE id = $1",
        seq_id,
    )
    assert row["recipient_email"] == "eve@example.com"
