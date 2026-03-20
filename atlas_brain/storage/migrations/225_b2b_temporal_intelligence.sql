-- Temporal intelligence: canonical per-vendor timing and trend objects consumed by
-- battle cards (timing_intelligence), churn reports, and market landscapes.
-- Same composite-key history pattern as b2b_evidence_vault.

CREATE TABLE IF NOT EXISTS b2b_temporal_intelligence (
    vendor_name          text        NOT NULL,
    as_of_date           date        NOT NULL,
    analysis_window_days int         NOT NULL,
    schema_version       text        NOT NULL,
    temporal             jsonb       NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (vendor_name, as_of_date, analysis_window_days, schema_version)
);

CREATE INDEX IF NOT EXISTS idx_b2b_temporal_intelligence_vendor
    ON b2b_temporal_intelligence (vendor_name);
CREATE INDEX IF NOT EXISTS idx_b2b_temporal_intelligence_date
    ON b2b_temporal_intelligence (as_of_date DESC);
