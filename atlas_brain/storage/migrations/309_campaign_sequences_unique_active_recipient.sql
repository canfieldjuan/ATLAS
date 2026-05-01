-- Enforce cross-sequence person dedup at the database level.
--
-- Migration 307 added a non-unique partial index for the conflict probe in
-- assign_recipient_to_sequence(). That left a race window: two workers can
-- both observe "no conflict" via SELECT and both commit recipient_email,
-- silently violating the one-active-sequence-per-recipient invariant.
--
-- This migration:
--   1. Backfill: marks all-but-most-recent duplicate active rows as
--      'superseded' so the UNIQUE index creation does not fail on legacy
--      data. Ranking is by updated_at DESC, created_at DESC, id ASC so the
--      most recently touched sequence wins.
--   2. Cascade: cancels any draft/approved/queued b2b_campaigns rows
--      attached to those just-superseded sequences. Without this,
--      campaign_send.py's queue picker (which selects by
--      b2b_campaigns.status='queued' without filtering on sequence status)
--      could ship pre-existing queued campaigns from the loser side of a
--      duplicate pair right after this migration runs -- breaking the new
--      dedup invariant during rollout. This mirrors the runtime supersede
--      behavior in _supersede_sequence() (campaign_suppression.py).
--   3. Audit log: records one 'recipient_superseded' row per superseded
--      sequence with source='migration_309' and the cancelled-campaign
--      count, so an operator can see exactly what the rollout did.
--   4. Drops the non-unique index from migration 307.
--   5. Recreates the index as UNIQUE so the database rejects concurrent
--      attempts to assign the same email to two active sequences.
--
-- Steps 1-3 run as one CTE chain so all three modifications execute
-- against the same snapshot and we know which sequences we just
-- superseded (vs ones that were already 'superseded' from a prior run).
--
-- The application-side helper (assign_recipient_to_sequence in
-- campaign_suppression.py) catches the resulting UniqueViolationError and
-- converts the loser of the race to a 'superseded' result -- so the
-- constraint is the source of truth and the helper is the recovery path.

WITH ranked AS (
    SELECT
        id,
        recipient_email,
        ROW_NUMBER() OVER (
            PARTITION BY LOWER(recipient_email)
            ORDER BY updated_at DESC, created_at DESC, id ASC
        ) AS rn
    FROM campaign_sequences
    WHERE status = 'active' AND recipient_email IS NOT NULL
),
to_supersede AS (
    SELECT id, recipient_email FROM ranked WHERE rn > 1
),
superseded_seqs AS (
    UPDATE campaign_sequences cs
    SET status = 'superseded', updated_at = NOW()
    FROM to_supersede ts
    WHERE cs.id = ts.id
    RETURNING cs.id, ts.recipient_email
),
cancelled_campaigns AS (
    UPDATE b2b_campaigns bc
    SET status = 'cancelled', updated_at = NOW()
    FROM superseded_seqs ss
    WHERE bc.sequence_id = ss.id
      AND bc.status IN ('draft', 'approved', 'queued')
    RETURNING bc.sequence_id
),
cancel_counts AS (
    SELECT sequence_id, count(*)::int AS n
    FROM cancelled_campaigns
    GROUP BY sequence_id
)
INSERT INTO campaign_audit_log
    (sequence_id, event_type, recipient_email, source, metadata)
SELECT
    ss.id,
    'recipient_superseded',
    ss.recipient_email,
    'migration_309',
    jsonb_build_object(
        'reason', 'migration_309_backfill_duplicate_active_recipient',
        'cancelled_campaigns', COALESCE(cc.n, 0)
    )
FROM superseded_seqs ss
LEFT JOIN cancel_counts cc ON cc.sequence_id = ss.id;

DROP INDEX IF EXISTS idx_campaign_seq_active_email;

CREATE UNIQUE INDEX idx_campaign_seq_active_email
    ON campaign_sequences (LOWER(recipient_email))
    WHERE status = 'active' AND recipient_email IS NOT NULL;
