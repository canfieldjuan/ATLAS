-- Persist per-vendor evidence diffs from the differential reasoning engine.
-- Tracks what changed between reasoning runs (confirmed, contradicted, novel,
-- missing) and the decision (reconstitute vs full_reason).

CREATE TABLE IF NOT EXISTS reasoning_evidence_diffs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name         TEXT NOT NULL,
    computed_date       DATE NOT NULL DEFAULT CURRENT_DATE,
    confirmed_count     INT NOT NULL DEFAULT 0,
    contradicted_count  INT NOT NULL DEFAULT 0,
    novel_count         INT NOT NULL DEFAULT 0,
    missing_count       INT NOT NULL DEFAULT 0,
    total_fields        INT NOT NULL DEFAULT 0,
    diff_ratio          FLOAT NOT NULL DEFAULT 0,
    weighted_diff_ratio FLOAT NOT NULL DEFAULT 0,
    has_core_contradiction BOOLEAN NOT NULL DEFAULT FALSE,
    decision            TEXT NOT NULL DEFAULT 'full_reason',  -- recall | reconstitute | full_reason
    compared            BOOLEAN NOT NULL DEFAULT TRUE,       -- false for recall/cold full-reason (no old evidence to diff)
    contradicted_fields JSONB DEFAULT '[]',                  -- truncated to 20 items
    novel_fields        JSONB DEFAULT '[]',                  -- truncated to 20 items
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (vendor_name, computed_date)
);

CREATE INDEX IF NOT EXISTS idx_red_vendor_date
    ON reasoning_evidence_diffs (vendor_name, computed_date DESC);
CREATE INDEX IF NOT EXISTS idx_red_date
    ON reasoning_evidence_diffs (computed_date DESC);
