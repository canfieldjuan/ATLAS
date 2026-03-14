-- Add pressure score and complaint rate columns to b2b_vendor_snapshots
ALTER TABLE b2b_vendor_snapshots
    ADD COLUMN IF NOT EXISTS pressure_score    NUMERIC,
    ADD COLUMN IF NOT EXISTS dm_churn_rate     NUMERIC,
    ADD COLUMN IF NOT EXISTS price_complaint_rate NUMERIC;
