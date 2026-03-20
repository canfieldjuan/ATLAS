-- Evidence vault: canonical per-vendor intelligence objects consumed by
-- battle cards, challenger briefs, churn reports, and all downstream products.
-- Composite key allows historical snapshots per schema version.

CREATE TABLE IF NOT EXISTS b2b_evidence_vault (
    vendor_name          text        NOT NULL,
    as_of_date           date        NOT NULL,
    analysis_window_days int         NOT NULL,
    schema_version       text        NOT NULL,
    vault                jsonb       NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (vendor_name, as_of_date, analysis_window_days, schema_version)
);

CREATE INDEX IF NOT EXISTS idx_b2b_evidence_vault_vendor
    ON b2b_evidence_vault (vendor_name);
CREATE INDEX IF NOT EXISTS idx_b2b_evidence_vault_date
    ON b2b_evidence_vault (as_of_date DESC);
CREATE INDEX IF NOT EXISTS idx_b2b_evidence_vault_version
    ON b2b_evidence_vault (schema_version);
