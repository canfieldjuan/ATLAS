-- Cross-vendor reasoning synthesis: canonical table for synthesis-produced
-- cross-vendor conclusions (battles, category councils, resource asymmetry).
-- Runs alongside legacy b2b_cross_vendor_conclusions during migration.

CREATE TABLE IF NOT EXISTS b2b_cross_vendor_reasoning_synthesis (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_type   TEXT NOT NULL,              -- pairwise_battle, category_council, resource_asymmetry
    vendors         TEXT[] NOT NULL,             -- sorted, canonicalized vendor names
    category        TEXT,                        -- for category_council only
    as_of_date      DATE NOT NULL,
    analysis_window_days INT NOT NULL,
    schema_version  TEXT NOT NULL,
    evidence_hash   TEXT NOT NULL,
    synthesis       JSONB NOT NULL,
    tokens_used     INT NOT NULL DEFAULT 0,
    llm_model       TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Pairwise battle / asymmetry: one per (type, vendor pair, date, window, schema)
CREATE UNIQUE INDEX IF NOT EXISTS uq_xv_synth_pairwise
    ON b2b_cross_vendor_reasoning_synthesis (analysis_type, vendors, as_of_date, analysis_window_days, schema_version)
    WHERE analysis_type IN ('pairwise_battle', 'resource_asymmetry');

-- Category council: one per (type, category, date, window, schema)
CREATE UNIQUE INDEX IF NOT EXISTS uq_xv_synth_council
    ON b2b_cross_vendor_reasoning_synthesis (analysis_type, category, as_of_date, analysis_window_days, schema_version)
    WHERE analysis_type = 'category_council';

CREATE INDEX IF NOT EXISTS idx_xv_synth_type_date
    ON b2b_cross_vendor_reasoning_synthesis (analysis_type, as_of_date DESC);

CREATE INDEX IF NOT EXISTS idx_xv_synth_vendors
    ON b2b_cross_vendor_reasoning_synthesis USING GIN (vendors);
