-- Account intelligence: canonical per-vendor account-level signals consumed by
-- challenger briefs, accounts-in-motion, and battle cards (segment_playbook).
-- Keyed by vendor + date + schema version (aggregates all company signals per vendor).

CREATE TABLE IF NOT EXISTS b2b_account_intelligence (
    vendor_name          text        NOT NULL,
    as_of_date           date        NOT NULL,
    analysis_window_days int         NOT NULL,
    schema_version       text        NOT NULL,
    accounts             jsonb       NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (vendor_name, as_of_date, analysis_window_days, schema_version)
);

CREATE INDEX IF NOT EXISTS idx_b2b_account_intelligence_vendor
    ON b2b_account_intelligence (vendor_name);
CREATE INDEX IF NOT EXISTS idx_b2b_account_intelligence_date
    ON b2b_account_intelligence (as_of_date DESC);
