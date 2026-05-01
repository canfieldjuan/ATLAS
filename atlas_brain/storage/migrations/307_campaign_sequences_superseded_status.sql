-- Cross-sequence person dedup support.
--
-- Adds 'superseded' as a terminal status for campaign_sequences. When a
-- prospect is matched to a vendor outreach but already has an active sequence
-- against a different vendor, the new sequence is marked 'superseded' instead
-- of being enrolled. This prevents one human from receiving concurrent
-- sequences across different vendor targets.
--
-- The partial index on LOWER(recipient_email) speeds the conflict probe issued
-- by assign_recipient_to_sequence (atlas_brain/autonomous/tasks/campaign_suppression.py).

ALTER TABLE campaign_sequences DROP CONSTRAINT IF EXISTS campaign_sequences_status_check;
ALTER TABLE campaign_sequences ADD CONSTRAINT campaign_sequences_status_check
    CHECK (status IN (
        'active',
        'paused',
        'completed',
        'replied',
        'bounced',
        'unsubscribed',
        'superseded'
    ));

CREATE INDEX IF NOT EXISTS idx_campaign_seq_active_email
    ON campaign_sequences (LOWER(recipient_email))
    WHERE status = 'active' AND recipient_email IS NOT NULL;
