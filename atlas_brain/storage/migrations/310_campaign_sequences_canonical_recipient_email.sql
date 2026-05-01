-- Canonicalize recipient_email so the dedup invariant cannot be bypassed
-- by leading/trailing whitespace.
--
-- Migration 309 enforced UNIQUE on LOWER(recipient_email) WHERE
-- status='active'. Codex review (PR #36) caught that the uniqueness key
-- treats 'alice@example.com' and 'alice@example.com ' as distinct,
-- letting two active sequences for the same person coexist if any
-- writer (including legacy data) leaves edge whitespace on the email.
--
-- This migration:
--   1. Backfill: BTRIM existing recipient_email values so storage
--      becomes canonical. Mirrors the supersede + campaign-cancel +
--      audit cascade from migration 309 for any post-trim duplicates
--      that newly collide.
--   2. Drops the prior partial unique index from migration 309 and
--      recreates it on LOWER(BTRIM(recipient_email)) so even if a
--      future non-helper writer inserts a whitespace-padded value, the
--      canonical form still gates uniqueness.
--   3. Adds a CHECK constraint requiring recipient_email = BTRIM(...)
--      for non-NULL values, rejecting future whitespace-padded writes
--      at the database boundary so the invariant survives even when a
--      caller bypasses the application-side helpers.
--
-- After this migration, storage canonicalization, index canonicalization,
-- and constraint canonicalization all agree -- whitespace can no longer
-- create a phantom second active sequence for the same recipient.

WITH trimmed AS (
    SELECT id, BTRIM(recipient_email) AS new_email, recipient_email AS old_email
    FROM campaign_sequences
    WHERE recipient_email IS NOT NULL
      AND recipient_email <> BTRIM(recipient_email)
)
UPDATE campaign_sequences cs
SET recipient_email = t.new_email, updated_at = NOW()
FROM trimmed t
WHERE cs.id = t.id;

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
    'migration_310',
    jsonb_build_object(
        'reason', 'migration_310_canonicalize_recipient_email',
        'cancelled_campaigns', COALESCE(cc.n, 0)
    )
FROM superseded_seqs ss
LEFT JOIN cancel_counts cc ON cc.sequence_id = ss.id;

DROP INDEX IF EXISTS idx_campaign_seq_active_email;

CREATE UNIQUE INDEX idx_campaign_seq_active_email
    ON campaign_sequences (LOWER(BTRIM(recipient_email)))
    WHERE status = 'active' AND recipient_email IS NOT NULL;

ALTER TABLE campaign_sequences
DROP CONSTRAINT IF EXISTS campaign_sequences_recipient_email_canonical;

ALTER TABLE campaign_sequences
ADD CONSTRAINT campaign_sequences_recipient_email_canonical
CHECK (recipient_email IS NULL OR recipient_email = BTRIM(recipient_email));
