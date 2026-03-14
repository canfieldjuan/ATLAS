-- Migration 106: Score calibration weights derived from campaign outcomes.
--
-- Stores learned weight adjustments per signal dimension (role_type, buying_stage,
-- urgency_bucket, seat_bucket, context_keyword). The calibration job computes
-- positive outcome rates and derives adjustments relative to baseline.
--
-- The scoring function blends these with static defaults using a configurable alpha.

CREATE TABLE IF NOT EXISTS score_calibration_weights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Which dimension this weight applies to
    dimension TEXT NOT NULL,
    -- CHECK: role_type, buying_stage, urgency_bucket, seat_bucket, context_keyword
    dimension_value TEXT NOT NULL,

    -- Observed outcome data
    total_sequences INT NOT NULL DEFAULT 0,
    positive_outcomes INT NOT NULL DEFAULT 0,       -- meeting_booked + deal_opened + deal_won
    deals_won INT NOT NULL DEFAULT 0,
    total_revenue NUMERIC(12,2) NOT NULL DEFAULT 0,

    -- Derived calibration
    positive_rate NUMERIC(5,4) NOT NULL DEFAULT 0,  -- positive_outcomes / total_sequences
    baseline_rate NUMERIC(5,4) NOT NULL DEFAULT 0,  -- overall positive rate across all sequences
    lift NUMERIC(6,4) NOT NULL DEFAULT 1.0,         -- positive_rate / baseline_rate (1.0 = no lift)
    weight_adjustment NUMERIC(5,2) NOT NULL DEFAULT 0, -- points to add/subtract from static score

    -- Static default (the hardcoded value being calibrated)
    static_default NUMERIC(5,2) NOT NULL DEFAULT 0,

    -- Metadata
    calibrated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sample_window_days INT NOT NULL DEFAULT 90,
    model_version INT NOT NULL DEFAULT 1,

    UNIQUE (dimension, dimension_value, model_version)
);

DO $$ BEGIN
    ALTER TABLE score_calibration_weights ADD CONSTRAINT chk_calibration_dimension
        CHECK (dimension IN ('role_type', 'buying_stage', 'urgency_bucket', 'seat_bucket', 'context_keyword'));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE INDEX IF NOT EXISTS idx_calibration_active
    ON score_calibration_weights (dimension, model_version DESC);

CREATE INDEX IF NOT EXISTS idx_calibration_latest
    ON score_calibration_weights (calibrated_at DESC);
