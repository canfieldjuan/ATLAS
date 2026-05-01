-- Enforce cross-sequence person dedup at the database level.
--
-- Migration 307 added a non-unique partial index for the conflict probe in
-- assign_recipient_to_sequence(). That left a race window: two workers can
-- both observe "no conflict" via SELECT and both commit recipient_email,
-- silently violating the one-active-sequence-per-recipient invariant.
--
-- This migration:
--   1. Marks all-but-most-recent duplicate active rows as 'superseded' so
--      the UNIQUE index creation does not fail on legacy data. Ranking is
--      by updated_at DESC, created_at DESC, id ASC so the most recently
--      touched sequence wins.
--   2. Drops the non-unique index from migration 307.
--   3. Recreates the index as UNIQUE so the database rejects concurrent
--      attempts to assign the same email to two active sequences.
--
-- The application-side helper (assign_recipient_to_sequence in
-- atlas_brain/autonomous/tasks/campaign_suppression.py) catches the
-- resulting UniqueViolationError and converts the loser of the race to a
-- 'superseded' result -- so the constraint is the source of truth and the
-- helper is the recovery path.

WITH ranked AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY LOWER(recipient_email)
            ORDER BY updated_at DESC, created_at DESC, id ASC
        ) AS rn
    FROM campaign_sequences
    WHERE status = 'active' AND recipient_email IS NOT NULL
)
UPDATE campaign_sequences
SET status = 'superseded', updated_at = NOW()
WHERE id IN (SELECT id FROM ranked WHERE rn > 1);

DROP INDEX IF EXISTS idx_campaign_seq_active_email;

CREATE UNIQUE INDEX idx_campaign_seq_active_email
    ON campaign_sequences (LOWER(recipient_email))
    WHERE status = 'active' AND recipient_email IS NOT NULL;
