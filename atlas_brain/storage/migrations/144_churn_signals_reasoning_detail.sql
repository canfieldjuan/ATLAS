-- Persist the full stratified reasoning contract into b2b_churn_signals
-- so follow-up tasks (reports, battle cards) can reconstruct reasoning_lookup
-- without re-running the reasoner.

ALTER TABLE b2b_churn_signals
    ADD COLUMN IF NOT EXISTS reasoning_risk_level TEXT,
    ADD COLUMN IF NOT EXISTS reasoning_executive_summary TEXT,
    ADD COLUMN IF NOT EXISTS reasoning_key_signals JSONB,
    ADD COLUMN IF NOT EXISTS reasoning_uncertainty_sources JSONB;
