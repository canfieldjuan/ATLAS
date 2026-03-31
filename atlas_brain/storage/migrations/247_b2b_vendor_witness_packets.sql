-- Witness-backed reasoning packet artifacts.
--
-- These tables persist the deterministic packet assembled before the Stage 5
-- synthesis call so we can inspect, cache, diff, and reuse witness-backed
-- evidence without rerunning the whole reasoning path.

CREATE TABLE IF NOT EXISTS b2b_vendor_reasoning_packets (
    vendor_name          text        NOT NULL,
    as_of_date           date        NOT NULL,
    analysis_window_days int         NOT NULL,
    schema_version       text        NOT NULL,
    evidence_hash        text        NOT NULL,
    packet               jsonb       NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (vendor_name, as_of_date, analysis_window_days, schema_version)
);

CREATE INDEX IF NOT EXISTS idx_b2b_vendor_reasoning_packets_hash
    ON b2b_vendor_reasoning_packets (evidence_hash);

CREATE TABLE IF NOT EXISTS b2b_vendor_witnesses (
    vendor_name          text        NOT NULL,
    as_of_date           date        NOT NULL,
    analysis_window_days int         NOT NULL,
    schema_version       text        NOT NULL,
    evidence_hash        text        NOT NULL,
    witness_id           text        NOT NULL,
    review_id            text,
    witness_type         text        NOT NULL,
    excerpt_text         text        NOT NULL,
    source               text,
    reviewed_at          timestamptz,
    reviewer_company     text,
    reviewer_title       text,
    pain_category        text,
    competitor           text,
    salience_score       numeric(8,2) NOT NULL DEFAULT 0,
    selection_reason     text,
    signal_tags          jsonb       NOT NULL DEFAULT '[]'::jsonb,
    source_id            text        NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (
        vendor_name, as_of_date, analysis_window_days, schema_version, witness_id
    )
);

CREATE INDEX IF NOT EXISTS idx_b2b_vendor_witnesses_vendor
    ON b2b_vendor_witnesses (vendor_name, as_of_date DESC);

CREATE INDEX IF NOT EXISTS idx_b2b_vendor_witnesses_hash
    ON b2b_vendor_witnesses (evidence_hash);
