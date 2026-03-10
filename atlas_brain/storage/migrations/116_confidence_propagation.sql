-- Migration 116: Propagate confidence_score to remaining entity tables
-- Sprint 3: Close the confidence gap on company_signals, churn_signals, product_profiles

ALTER TABLE b2b_company_signals
    ADD COLUMN IF NOT EXISTS confidence_score NUMERIC(3,2) DEFAULT 0;

ALTER TABLE b2b_churn_signals
    ADD COLUMN IF NOT EXISTS confidence_score NUMERIC(3,2) DEFAULT 0;

ALTER TABLE b2b_product_profiles
    ADD COLUMN IF NOT EXISTS confidence_score NUMERIC(3,2) DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_bcs_confidence ON b2b_company_signals (confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_bchs_confidence ON b2b_churn_signals (confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_bpp_confidence ON b2b_product_profiles (confidence_score DESC);
