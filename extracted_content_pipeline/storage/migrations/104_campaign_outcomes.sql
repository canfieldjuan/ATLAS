-- Migration 104: Campaign outcome tracking
-- Adds business outcome columns to campaign_sequences (orthogonal to delivery status).
-- outcome_history JSONB tracks stage progression (pending -> meeting -> deal -> won).

ALTER TABLE campaign_sequences
    ADD COLUMN IF NOT EXISTS outcome TEXT NOT NULL DEFAULT 'pending'
        CHECK (outcome IN (
            'pending', 'meeting_booked', 'deal_opened',
            'deal_won', 'deal_lost', 'no_opportunity', 'disqualified'
        )),
    ADD COLUMN IF NOT EXISTS outcome_recorded_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS outcome_recorded_by TEXT,
    ADD COLUMN IF NOT EXISTS outcome_notes TEXT,
    ADD COLUMN IF NOT EXISTS outcome_revenue NUMERIC(12,2),
    ADD COLUMN IF NOT EXISTS outcome_history JSONB NOT NULL DEFAULT '[]';

-- Partial index: only non-pending outcomes (most rows stay pending)
CREATE INDEX IF NOT EXISTS idx_campaign_seq_outcome
    ON campaign_sequences (outcome) WHERE outcome != 'pending';

-- Recency index for outcome queries
CREATE INDEX IF NOT EXISTS idx_campaign_seq_outcome_date
    ON campaign_sequences (outcome_recorded_at DESC)
    WHERE outcome_recorded_at IS NOT NULL;

-- Composite index for signal effectiveness GROUP BY queries
-- Covers the JOIN from b2b_campaigns to campaign_sequences grouped by signal dimensions
CREATE INDEX IF NOT EXISTS idx_b2b_campaigns_signal_outcome
    ON b2b_campaigns (sequence_id, opportunity_score, buying_stage, role_type, target_mode)
    WHERE sequence_id IS NOT NULL;
