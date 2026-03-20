-- Segment intelligence: canonical per-vendor buyer segment objects consumed by
-- battle cards (segment_playbook), challenger briefs, and churn reports.
-- Same composite-key history pattern as b2b_evidence_vault.

CREATE TABLE IF NOT EXISTS b2b_segment_intelligence (
    vendor_name          text        NOT NULL,
    as_of_date           date        NOT NULL,
    analysis_window_days int         NOT NULL,
    schema_version       text        NOT NULL,
    segments             jsonb       NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (vendor_name, as_of_date, analysis_window_days, schema_version)
);

CREATE INDEX IF NOT EXISTS idx_b2b_segment_intelligence_vendor
    ON b2b_segment_intelligence (vendor_name);
CREATE INDEX IF NOT EXISTS idx_b2b_segment_intelligence_date
    ON b2b_segment_intelligence (as_of_date DESC);
