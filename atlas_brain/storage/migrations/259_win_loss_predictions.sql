-- Win/Loss Predictor persistence
-- Stores every prediction for tenant-scoped history and revisiting.

CREATE TABLE IF NOT EXISTS b2b_win_loss_predictions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id      TEXT NOT NULL,
    vendor_name     TEXT NOT NULL,
    company_size    TEXT,
    industry        TEXT,
    win_probability NUMERIC(4,3) NOT NULL,
    confidence      TEXT NOT NULL,
    verdict         TEXT,
    is_gated        BOOLEAN NOT NULL DEFAULT FALSE,
    recommended_approach TEXT,
    lead_with       JSONB,
    talking_points  JSONB,
    timing_advice   TEXT,
    risk_factors    JSONB,
    factors         JSONB NOT NULL DEFAULT '[]',
    data_gates      JSONB NOT NULL DEFAULT '[]',
    switching_triggers JSONB NOT NULL DEFAULT '[]',
    proof_quotes    JSONB NOT NULL DEFAULT '[]',
    objections      JSONB NOT NULL DEFAULT '[]',
    displacement_targets JSONB NOT NULL DEFAULT '[]',
    segment_match   JSONB,
    data_coverage   JSONB NOT NULL DEFAULT '{}',
    weights_source  TEXT NOT NULL DEFAULT 'static',
    calibration_version INT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_wl_pred_account_created
    ON b2b_win_loss_predictions (account_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_wl_pred_vendor
    ON b2b_win_loss_predictions (vendor_name, created_at DESC);
