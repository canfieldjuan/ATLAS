-- Migration 117: Add pressure_score and dm_churn_rate to vendor snapshots
-- Sprint 4: Enable pressure_score_spike and dm_churn_spike change-event detection

ALTER TABLE b2b_vendor_snapshots
    ADD COLUMN IF NOT EXISTS pressure_score NUMERIC(5,1) DEFAULT 0;

ALTER TABLE b2b_vendor_snapshots
    ADD COLUMN IF NOT EXISTS dm_churn_rate NUMERIC(5,4);

ALTER TABLE b2b_vendor_snapshots
    ADD COLUMN IF NOT EXISTS price_complaint_rate NUMERIC(5,4);
